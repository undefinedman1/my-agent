"""基于 HelloAgents 的代码库维护助手。"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from hello_agents import HelloAgentsLLM, SimpleAgent, ToolRegistry
from hello_agents.context import ContextBuilder, ContextConfig, MemoryContextProvider
from hello_agents.core.message import Message
from hello_agents.memory import MemoryStore
from hello_agents.tools import MemoryTool, ToolResponse, ToolStatus


class CodebaseMaintainer:
    """结合只读代码库工具、长期记忆和上下文管理的项目助手。"""

    _IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    _TEXT_SUFFIXES = {
        ".c",
        ".cfg",
        ".css",
        ".go",
        ".h",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
    _MAX_FILE_BYTES = 200_000
    _MAX_RESULTS = 50

    def __init__(
        self,
        project_name: str,
        codebase_path: str | Path,
        llm: HelloAgentsLLM | None = None,
        max_context_tokens: int = 4000,
    ) -> None:
        self.project_name = project_name
        self.project_root = Path(codebase_path).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"代码库目录不存在：{self.project_root}")

        self.project_id = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip(
            "._"
        ) or "project"
        self.note_source_prefix = f"project:{self.project_id}:note:"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        data_dir = Path(__file__).resolve().parent / "project_data" / self.project_id
        data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store = MemoryStore(
            path=str(data_dir / "memory.db"),
            user_id=self.project_id,
            task_id=self.session_id,
        )
        self.memory_tool = MemoryTool(store=self.memory_store)
        self.memory_provider = MemoryContextProvider(
            store=self.memory_store,
            limit=10,
            use_query=True,
        )
        self.context_builder = ContextBuilder(
            config=ContextConfig(max_tokens=max_context_tokens)
        )

        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(self.memory_tool)
        self.tool_registry.register_function(
            name="list_project_files",
            description="列出代码库目录中的文件；只读且不能访问代码库以外的路径。",
            func=self.list_project_files,
        )
        self.tool_registry.register_function(
            name="read_project_file",
            description="读取代码库中的文本文件；不能读取代码库以外的路径或 .env 文件。",
            func=self.read_project_file,
        )
        self.tool_registry.register_function(
            name="search_project",
            description="在代码库文本文件中搜索关键词；只读且不能访问代码库以外的路径。",
            func=self.search_project,
        )

        self.agent = SimpleAgent(
            name="CodebaseMaintainer",
            llm=llm or HelloAgentsLLM(),
            system_prompt=self._build_base_system_prompt(),
            tool_registry=self.tool_registry,
        )
        self.conversation_history: list[Message] = []
        self.stats: dict[str, Any] = {
            "session_start": datetime.now(),
            "file_operations": 0,
            "notes_created": 0,
            "tool_calls": 0,
        }

    def run(self, user_input: str, mode: str = "auto") -> str:
        """检索相关上下文并让 HelloAgents Agent 回答用户。"""
        if not user_input.strip():
            raise ValueError("user_input 不能为空")

        context = self.context_builder.build(
            user_query=user_input,
            conversation_history=self.conversation_history,
            system_instructions=self._build_system_instructions(mode),
            additional_packets=self.memory_provider.get_context(user_input),
        )
        self.agent.system_prompt = context
        response = self._response_text(self.agent.run(user_input))

        self._save_interaction(user_input, response)
        self._update_history(user_input, response)
        return response

    def _build_base_system_prompt(self) -> str:
        return f"""你是 {self.project_name} 项目的代码库维护助手。

你可以使用以下工具：
- list_project_files：查看代码库目录结构。
- read_project_file：读取代码库文本文件。
- search_project：搜索代码库中的文本内容。
- MemoryTool：保存或检索项目记忆和笔记。

代码库根目录：{self.project_root}
当前会话 ID：{self.session_id}

只使用只读文件工具探索代码，不要尝试执行 shell 命令或访问代码库之外的路径。
结论应基于工具返回的信息；缺少依据时请明确说明。"""

    def _build_system_instructions(self, mode: str) -> str:
        mode_hints = {
            "explore": "优先查看目录结构和关键文档，并总结项目组织方式。",
            "analyze": "优先搜索 TODO、FIXME 等标记，再读取相关文件分析。",
            "plan": "结合相关项目记忆和当前进度，给出有依据的下一步计划。",
            "auto": "根据用户需求灵活选择是否使用代码库或记忆工具。",
        }
        return f"{self._build_base_system_prompt()}\n\n{mode_hints.get(mode, mode_hints['auto'])}"

    @staticmethod
    def _response_text(response: object) -> str:
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _check_tool_response(response: ToolResponse, tool_name: str) -> None:
        if response.status == ToolStatus.ERROR:
            raise RuntimeError(f"{tool_name} 调用失败：{response.text}")

    def _resolve_project_path(self, relative_path: str) -> Path:
        path = (self.project_root / relative_path).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("路径必须位于代码库目录内") from exc
        if any(part.lower() in self._IGNORED_PARTS for part in path.parts):
            raise ValueError("不允许访问依赖或缓存目录")
        if path.name.lower() == ".env" or path.name.lower().startswith(".env."):
            raise ValueError("不允许读取环境变量文件")
        return path

    def list_project_files(self, relative_path: str = ".") -> str:
        """列出指定目录下的代码库文件（最多 200 个）。"""
        directory = self._resolve_project_path(relative_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"目录不存在：{relative_path}")

        files = []
        for path in sorted(directory.rglob("*")):
            try:
                resolved = self._resolve_project_path(str(path.relative_to(self.project_root)))
            except ValueError:
                continue
            if resolved.is_file():
                files.append(resolved.relative_to(self.project_root).as_posix())
                if len(files) >= 200:
                    break

        self._record_file_tool_call()
        return "\n".join(files) if files else "目录中没有可列出的文件。"

    def read_project_file(self, relative_path: str) -> str:
        """读取一个代码库文本文件，单文件最多读取 200 KB。"""
        path = self._resolve_project_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{relative_path}")
        if path.suffix.lower() not in self._TEXT_SUFFIXES:
            raise ValueError(f"不支持读取此类文件：{relative_path}")
        if path.stat().st_size > self._MAX_FILE_BYTES:
            raise ValueError("文件超过 200 KB，拒绝读取")

        content = path.read_text(encoding="utf-8")
        self._record_file_tool_call()
        return content

    def search_project(self, query: str, relative_path: str = ".") -> str:
        """在代码库文本文件中搜索关键词并返回最多 50 处匹配。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        directory = self._resolve_project_path(relative_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"目录不存在：{relative_path}")

        matches = []
        needle = query.casefold()
        for path in sorted(directory.rglob("*")):
            try:
                resolved = self._resolve_project_path(str(path.relative_to(self.project_root)))
            except ValueError:
                continue
            if (
                not resolved.is_file()
                or resolved.suffix.lower() not in self._TEXT_SUFFIXES
                or resolved.stat().st_size > self._MAX_FILE_BYTES
            ):
                continue

            try:
                lines = resolved.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line.casefold():
                    relative = resolved.relative_to(self.project_root).as_posix()
                    matches.append(f"{relative}:{line_number}: {line.strip()}")
                    if len(matches) >= self._MAX_RESULTS:
                        break
            if len(matches) >= self._MAX_RESULTS:
                break

        self._record_file_tool_call()
        return "\n".join(matches) if matches else f"未找到匹配内容：{query}"

    def _record_file_tool_call(self) -> None:
        self.stats["file_operations"] += 1
        self.stats["tool_calls"] += 1

    def _save_interaction(self, user_input: str, response: str) -> None:
        result = self.memory_tool.run(
            {
                "action": "add",
                "content": f"用户：{user_input}\n助手：{response[:500]}",
                "source": f"project:{self.project_id}:conversation",
                "kind": "episode",
            }
        )
        self._check_tool_response(result, "MemoryTool")

    def _retrieve_relevant_notes(
        self, query: str = "", limit: int = 10
    ) -> list[dict[str, str]]:
        _, records = self.memory_store.snapshot()
        notes = []
        for record in records:
            if not record.source.startswith(self.note_source_prefix):
                continue
            if query and query.casefold() not in record.content.casefold():
                continue
            notes.append(
                {
                    "type": record.source.removeprefix(self.note_source_prefix),
                    "content": record.content,
                }
            )
        return notes[:limit]

    def _update_history(self, user_input: str, response: str) -> None:
        self.conversation_history.extend(
            [
                Message(content=user_input, role="user", timestamp=datetime.now()),
                Message(content=response, role="assistant", timestamp=datetime.now()),
            ]
        )
        self.conversation_history = self.conversation_history[-20:]

    def explore(self, target: str = ".") -> str:
        return self.run(f"请探索 {target} 的代码结构，了解项目组织方式", mode="explore")

    def analyze(self, focus: str = "") -> str:
        query = "请分析代码质量"
        if focus:
            query += f"，重点关注{focus}"
        return self.run(query, mode="analyze")

    def plan_next_steps(self) -> str:
        return self.run("根据之前的分析和当前进度，规划下一步任务", mode="plan")

    def create_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: list[str] | None = None,
    ) -> str:
        """将项目笔记保存到 HelloAgents 长期记忆。"""
        note_content = f"{title}\n{content}"
        if tags:
            note_content += f"\n标签：{', '.join(tags)}"
        result = self.memory_tool.run(
            {
                "action": "add",
                "content": note_content,
                "source": f"{self.note_source_prefix}{note_type}",
                "kind": "fact",
            }
        )
        self._check_tool_response(result, "MemoryTool")
        self.stats["notes_created"] += 1
        return result.text

    def get_stats(self) -> dict[str, Any]:
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        notes = self._retrieve_relevant_notes()
        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration,
            },
            "activity": {
                "file_operations": self.stats["file_operations"],
                "notes_created": self.stats["notes_created"],
                "tool_calls": self.stats["tool_calls"],
            },
            "notes": {"count": len(notes), "items": notes},
        }

    def generate_report(self, save_to_file: bool = True) -> dict[str, Any]:
        report = self.get_stats()
        if save_to_file:
            report_file = Path.cwd() / f"maintainer_report_{self.session_id}.json"
            report_file.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            report["report_file"] = str(report_file)
        return report


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    project_root = Path(__file__).resolve().parents[2]
    maintainer = CodebaseMaintainer(
        project_name=project_root.name,
        codebase_path=project_root,
    )
    print(maintainer.explore())
    print(maintainer.analyze())
    print(maintainer.plan_next_steps())
    print(json.dumps(maintainer.generate_report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
