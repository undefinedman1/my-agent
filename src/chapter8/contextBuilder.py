from datetime import datetime
from pathlib import Path

from hello_agents.context import (
    ContextBuilder,
    ContextConfig,
    MemoryContextProvider,
    RetrievalContextProvider,
)
from hello_agents.core.message import Message
from hello_agents.memory import MemoryStore
from hello_agents.retrieval import RAGStore

# 将示例数据库放在脚本目录，避免依赖当前工作目录。
data_dir = Path(__file__).resolve().parent
memory_store = MemoryStore(
    path=str(data_dir / "chapter8_memory.db"),
    user_id="user123",
    task_id="memory-demo",
)
rag_store = RAGStore(path=str(data_dir / "chapter8_rag.db"))

# 使用框架支持的上下文预算配置。
config = ContextConfig(
    max_tokens=3000,
    reserve_ratio=0.2,
    min_relevance=0.05,
    enable_mmr=True,
    mmr_lambda=0.7,
)

builder = ContextBuilder(config=config)

conversation_history = [
    Message(content="我正在开发一个数据分析工具", role="user", timestamp=datetime.now()),
    Message(content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?", role="assistant", timestamp=datetime.now()),
    Message(content="我打算使用Python和Pandas,已经完成了CSV读取模块", role="user", timestamp=datetime.now()),
    Message(content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。", role="assistant", timestamp=datetime.now()),
]

# 添加示例记忆，重复运行时避免创建重复记录。
demo_memories = [
    ("用户正在开发数据分析工具，使用Python和Pandas", "fact"),
    ("已完成CSV读取模块的开发", "episode"),
]
_, existing_memories = memory_store.snapshot()
existing_contents = {record.content for record in existing_memories}
for content, kind in demo_memories:
    if content not in existing_contents:
        memory_store.add(
            content=content,
            source="chapter8 context builder demo",
            kind=kind,
        )

# 相同文档 ID、版本和原文可安全地重复导入。
rag_store.add_document(
    text=(
        "Pandas内存优化建议：读取CSV时为列指定合适的数据类型；"
        "将低基数文本列转换为category；使用更小的整数或浮点类型；"
        "分块读取大文件，并及时释放不再使用的中间对象。"
    ),
    source="chapter8-pandas-memory-optimization",
    version="1",
    document_id="pandas-memory-optimization",
)

user_query = "如何优化Pandas的内存占用?"
memory_provider = MemoryContextProvider(
    store=memory_store,
    limit=10,
    use_query=True,
)
retrieval_provider = RetrievalContextProvider(store=rag_store, limit=5)
additional_packets = [
    *memory_provider.get_context(user_query),
    *retrieval_provider.get_context(user_query),
]

context = builder.build(
    user_query=user_query,
    conversation_history=conversation_history,
    system_instructions="你是一位资深的Python数据工程顾问。你的回答需要:1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例",
    additional_packets=additional_packets,
)

print("=" * 80)
print("构建的上下文:")
print("=" * 80)
print(context)
print("=" * 80)
print("上下文预算诊断:")
print(builder.last_diagnostics)
