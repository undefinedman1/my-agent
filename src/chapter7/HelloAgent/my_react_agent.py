MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动：
"""

import re
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry

class MyReActAgent(ReActAgent):
    """
    重写的ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            prompt = self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(),
                question=input_text,
                history="\n".join(self.current_history),
            )

            response = self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)
            response_text = response.content if hasattr(response, "content") else str(response)
            thought, action = self._parse_output(response_text)

            if action and action.lower().startswith("finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            if thought:
                self.current_history.append(f"Thought: {thought}")

            if not action:
                self.current_history.append("Action: 缺少有效格式的 Action")
                continue

            try:
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                observation_text = self._tool_result_text(observation)
            except Exception as exc:
                observation_text = f"工具执行失败：{exc}"

            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation_text}")

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _parse_output(self, text: str) -> Tuple[str, Optional[str]]:
        thought_match = re.search(
            r"Thought:\s*(.*?)(?:\n\s*Action:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        action_match = re.search(
            r"Action:\s*(.+?)(?:\n|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action_input(self, action: str) -> str:
        match = re.match(r"Finish\[(.*)]", action, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else action

    def _parse_action(self, action: str) -> Tuple[str, str]:
        match = re.match(r"([^\[]+)\[(.*)]", action, re.DOTALL)
        if not match:
            raise ValueError(f"无法解析 Action: {action}")
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _tool_result_text(result) -> str:
        if hasattr(result, "to_model_text"):
            return result.to_model_text()
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)
