from pathlib import Path

from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.context import (
    ContextBuilder,
    ContextConfig,
    MemoryContextProvider,
    RetrievalContextProvider,
)
from hello_agents.core.message import Message
from hello_agents.memory import MemoryStore
from hello_agents.retrieval import RAGStore


class ContextAwareAgent(SimpleAgent):
    """具有上下文感知能力的 Agent。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str = "",
        user_id: str = "default",
        knowledge_base_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        max_context_tokens: int = 4000,
    ) -> None:
        super().__init__(name=name, llm=llm, system_prompt=system_prompt)

        data_dir = Path(__file__).resolve().parent
        self.memory_store = MemoryStore(
            path=str(memory_path or data_dir / "chapter8_memory.db"),
            user_id=user_id,
            task_id=name,
        )
        self.rag_store = RAGStore(
            path=str(knowledge_base_path or data_dir / "chapter8_rag.db")
        )

        self.memory_provider = MemoryContextProvider(
            store=self.memory_store,
            limit=10,
            use_query=True,
        )
        self.retrieval_provider = RetrievalContextProvider(
            store=self.rag_store,
            limit=5,
        )
        self.context_builder = ContextBuilder(
            config=ContextConfig(max_tokens=max_context_tokens)
        )

    @staticmethod
    def _response_text(response: object) -> str:
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    def run(self, user_input: str) -> str:
        """结合相关记忆、知识库和对话历史生成回复。"""
        additional_packets = [
            *self.memory_provider.get_context(user_input),
            *self.retrieval_provider.get_context(user_input),
        ]
        optimized_context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self._history,
            system_instructions=self.system_prompt,
            additional_packets=additional_packets,
        )

        response = self._response_text(
            self.llm.invoke(
                [
                    {"role": "system", "content": optimized_context},
                    {"role": "user", "content": user_input},
                ]
            )
        )

        self.add_message(Message(user_input, "user"))
        self.add_message(Message(response, "assistant"))
        self.memory_store.add(
            content=f"Q: {user_input}\nA: {response[:500]}",
            source=f"conversation with {self.name}",
            kind="episode",
        )
        return response


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    agent = ContextAwareAgent(
        name="数据分析顾问",
        llm=HelloAgentsLLM(),
        system_prompt="你是一位资深的Python数据工程顾问。",
        user_id="user123",
    )
    print(agent.run("如何优化Pandas的内存占用?"))


if __name__ == "__main__":
    main()
