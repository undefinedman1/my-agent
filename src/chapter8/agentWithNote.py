"""基于 HelloAgents 上下文、记忆和检索组件的项目助手。"""

import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
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
from hello_agents.tools import MemoryTool, ToolResponse, ToolStatus


class ProjectAssistant(SimpleAgent):
    """结合项目笔记、对话记忆和知识库提供建议。"""

    def __init__(
        self,
        name: str,
        project_name: str,
        llm: HelloAgentsLLM | None = None,
        system_prompt: str | None = None,
        max_context_tokens: int = 4000,
        **kwargs: Any,
    ) -> None:
        self.project_name = project_name
        self.project_id = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip(
            "._"
        ) or "project"
        self.note_source_prefix = f"project:{self.project_id}:note:"

        super().__init__(
            name=name,
            llm=llm or HelloAgentsLLM(),
            system_prompt=system_prompt or self._build_system_instructions(),
            **kwargs,
        )

        data_dir = Path(__file__).resolve().parent / "project_data" / self.project_id
        data_dir.mkdir(parents=True, exist_ok=True)

        self.memory_store = MemoryStore(
            path=str(data_dir / "memory.db"),
            user_id=self.project_id,
            task_id=name,
        )
        self.rag_store = RAGStore(path=str(data_dir / "knowledge.db"))
        self.memory_tool = MemoryTool(store=self.memory_store)
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
    def _check_tool_response(response: ToolResponse, tool_name: str) -> None:
        if response.status == ToolStatus.ERROR:
            raise RuntimeError(f"{tool_name} 调用失败：{response.text}")

    @staticmethod
    def _response_text(response: object) -> str:
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    def run(self, user_input: str, note_as_action: bool = False) -> str:
        """结合相关项目记录生成回答，可选择将本轮内容保存为笔记。"""
        if not user_input.strip():
            raise ValueError("user_input 不能为空")

        additional_packets = [
            *self.memory_provider.get_context(user_input),
            *self.retrieval_provider.get_context(user_input),
        ]
        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self._history,
            system_instructions=self.system_prompt,
            additional_packets=additional_packets,
        )
        response = self._response_text(
            self.llm.invoke(
                [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input},
                ]
            )
        )

        self._save_interaction(user_input, response)
        if note_as_action:
            self._save_as_note(user_input, response)
        self._update_history(user_input, response)
        return response

    def _save_interaction(self, user_input: str, response: str) -> None:
        tool_response = self.memory_tool.run(
            {
                "action": "add",
                "content": f"用户：{user_input}\n助手：{response[:500]}",
                "source": f"project:{self.project_id}:conversation",
                "kind": "episode",
            }
        )
        self._check_tool_response(tool_response, "MemoryTool")

    def _save_as_note(self, user_input: str, response: str) -> None:
        if any(keyword in user_input for keyword in ("问题", "阻塞", "冲突")):
            note_type = "blocker"
        elif any(keyword in user_input for keyword in ("计划", "下一步")):
            note_type = "action"
        else:
            note_type = "conclusion"

        title = user_input.strip().replace("\n", " ")[:30]
        tool_response = self.memory_tool.run(
            {
                "action": "add",
                "content": (
                    f"[{note_type}] {title}\n"
                    f"## 用户输入\n{user_input}\n\n## 助手分析\n{response}"
                ),
                "source": f"{self.note_source_prefix}{note_type}",
                "kind": "fact",
            }
        )
        self._check_tool_response(tool_response, "MemoryTool")

    def get_note_summary(self) -> str:
        """列出当前项目已保存的笔记。"""
        _, records = self.memory_store.snapshot()
        notes = [
            record
            for record in records
            if record.source.startswith(self.note_source_prefix)
        ]
        if not notes:
            return "当前项目还没有保存笔记。"

        return "\n\n".join(
            f"- [{record.source.removeprefix(self.note_source_prefix)}] "
            f"{record.content}"
            for record in notes
        )

    def _build_system_instructions(self) -> str:
        return f"""你是 {self.project_name} 项目的长期助手。

请基于检索到的项目笔记、对话记忆和知识库回答，优先关注 blocker 类型的记录，
说明建议的依据，并给出具体、可执行的下一步。若上下文中没有相关依据，请明确说明。"""

    def _update_history(self, user_input: str, response: str) -> None:
        self.add_message(Message(user_input, "user"))
        self.add_message(Message(response, "assistant"))
        self._history = self._history[-10:]


def main() -> None:
    load_dotenv()
    assistant = ProjectAssistant(
        name="项目助手",
        project_name="data_pipeline_refactoring",
    )
    response = assistant.run(
        "我们已完成数据模型层重构，下一步计划重构业务逻辑层。",
        note_as_action=True,
    )
    # 第二次交互:提出问题
    response = assistant.run(
        "在重构业务逻辑层时,我遇到了依赖版本冲突的问题,该如何解决?"
    )
    print(response)
    print("\n项目笔记：")
    print(assistant.get_note_summary())


if __name__ == "__main__":
    main()
