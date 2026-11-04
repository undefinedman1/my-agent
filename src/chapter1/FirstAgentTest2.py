import requests
import os
import re
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置API密钥
API_KEY =  os.environ.get("API_KEY")  # 或者直接在代码中写入
BASE_URL = os.environ.get("BASE_URL")
MODEL_ID = os.environ.get("MODEL_ID")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

WEATHER_CODE_MAP = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "毛毛雨",
    56: "冻毛毛雨",
    57: "冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷暴伴冰雹",
    99: "雷暴伴冰雹",
}

# 系统提示词
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    # 1. 城市名转坐标（支持中文城市名）
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
    geo_data = requests.get(geo_url).json()
    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]

    # 2. 查当前天气
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        f"&timezone=auto"
    )
    weather_data = requests.get(weather_url).json() 
    
    try:
        # 发起网络请求
        # 提取当前天气状况
        current_condition = weather_data['current']
        weather_desc = current_condition["weather_code"]
        temp_c = current_condition["temperature_2m"]
        weather = WEATHER_CODE_MAP.get(weather_desc, "未知")

        # 格式化成自然语言返回
        return f"{city}当前天气：{weather}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key:
        return "错误：未配置TAVILY_API_KEY。"

    # 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # Tavily返回的结果已经非常干净，可以直接使用
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
print("✅ 工具函数定义完成!")

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"

class TravelAssistant:
    """
    智能旅行助手类
    """
    def __init__(self):
        self.llm = OpenAICompatibleClient(
            model=MODEL_ID,
            api_key=API_KEY,
            base_url=BASE_URL
        )
        self.prompt_history = []
    
    def reset(self):
        """重置对话历史"""
        self.prompt_history = []
    
    def add_user_message(self, message: str):
        """添加用户消息到历史"""
        self.prompt_history.append(f"用户请求: {message}")
    
    def add_assistant_message(self, message: str):
        """添加助手消息到历史"""
        self.prompt_history.append(message)
    
    def add_observation(self, observation: str):
        """添加观察结果到历史"""
        self.prompt_history.append(f"Observation: {observation}")
print("✅ 智能助手类定义完成!")

def display_conversation(history):
    """美观地显示对话历史"""
    print("\n" + "="*60)
    print("📝 对话历史")
    print("="*60)
    
    for i, message in enumerate(history, 1):
        if message.startswith("用户请求:"):
            print(f"\n👤 用户 [{i}]: {message[5:]}")
        elif message.startswith("Thought:"):
            print(f"\n🤔 思考 [{i}]: {message[8:].strip()}")
        elif message.startswith("Action:"):
            print(f"🛠️  行动 [{i}]: {message[7:].strip()}")
        elif message.startswith("Observation:"):
            print(f"📊 观察 [{i}]: {message[12:].strip()}")
        else:
            print(f"💬 消息 [{i}]: {message}")
    
    print("="*60 + "\n")

def parse_action(action_str):
    """解析行动字符串"""
    if action_str.startswith("Finish"):
        match = re.match(r"\w+\[(.*)\]", action_str)
        if match:
            return "finish", {"answer": match.group(1)}
        return "finish", {"answer": "任务完成"}
    
    tool_name_match = re.search(r"(\w+)\(", action_str)
    if not tool_name_match:
        return None, {}
    
    tool_name = tool_name_match.group(1)
    args_match = re.search(r"\((.*)\)", action_str)
    if args_match:
        args_str = args_match.group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    else:
        kwargs = {}
    
    return tool_name, kwargs
print("✅ 显示函数定义完成!")

def run_assistant(user_input, max_iterations=5, display=True):
    """
    运行旅行助手的主函数
    
    Args:
        user_input: 用户输入的问题
        max_iterations: 最大循环次数
        display: 是否显示对话历史
    
    Returns:
        tuple: (最终答案, 完整的对话历史)
    """
    assistant = TravelAssistant()
    assistant.add_user_message(user_input)
    
    if display:
        print(f"👤 用户输入: {user_input}")
        print("="*50)
    
    for i in range(max_iterations):
        if display:
            print(f"\n🔄 循环 {i+1}/{max_iterations}")
        
        # 构建完整prompt并调用LLM
        full_prompt = "\n".join(assistant.prompt_history)
        llm_output = assistant.llm.generate(full_prompt, AGENT_SYSTEM_PROMPT)
        # 模型可能会输出多余的Thought-Action，需要截断
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("⚠️ 已截断多余的 Thought-Action 对")
        
        assistant.add_assistant_message(llm_output)
        
        if display:
            print(f"🤖 模型输出:\n{llm_output}")
        
        # 解析行动
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            assistant.prompt_history.append(observation_str)
            continue
            
        action_str = action_match.group(1).strip()
        tool_name, kwargs = parse_action(action_str)
        
        # 处理完成行动
        if tool_name == "finish":
            final_answer = kwargs.get("answer", "任务完成")
            if display:
                print(f"🎉 任务完成!")
                print(f"📋 最终答案: {final_answer}")
            return final_answer, assistant.prompt_history
        
        # 处理工具调用
        if tool_name in available_tools:
            if display:
                print(f"🛠️  调用工具: {tool_name}({kwargs})")
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"
        
        # 记录观察结果
        if display:
            print(f"📊 观察结果: {observation}")
            print("="*50)
        
        assistant.add_observation(observation)
    
    # 如果达到最大循环次数仍未完成
    timeout_answer = "抱歉，经过多次尝试仍未完成您的请求。请尝试简化您的问题或稍后重试。"
    if display:
        print(f"⏰ 达到最大循环次数: {timeout_answer}")
    
    return timeout_answer, assistant.prompt_history

# 测试示例
def test_basic_example():
    """测试北京天气+景点推荐的示例"""
    print("🚀 开始测试北京天气+景点推荐示例")
    user_input = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    
    final_answer, history = run_assistant(user_input, display=True)
    
    print("\n" + "="*60)
    print("📊 测试完成!")
    print("="*60)
    print(f"最终答案: {final_answer}")
    
    # 显示完整对话历史
    display_conversation(history)
    
    return final_answer, history

# 运行测试示例
final_answer, history = test_basic_example()