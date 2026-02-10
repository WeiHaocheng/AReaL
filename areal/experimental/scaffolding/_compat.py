"""Compatibility layer for optional tensorrt_llm.scaffolding dependency.

Provides imports from tensorrt_llm.scaffolding when available, or minimal stub
classes when not installed. The stubs allow the scaffolding module to be imported
without crashing, but actual usage requires tensorrt_llm to be installed.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

_INSTALL_MSG = (
    "tensorrt_llm is required for the scaffolding framework but is not installed. "
    "See https://github.com/NVIDIA/TensorRT-LLM for installation instructions."
)

try:
    from tensorrt_llm.scaffolding import (
        NativeGenerationController,
        ScaffoldingLlm,
    )
    from tensorrt_llm.scaffolding.controller import Controller
    from tensorrt_llm.scaffolding.result import ScaffoldingOutput
    from tensorrt_llm.scaffolding.task import (
        AssistantMessage,
        ChatTask,
        GenerationTask,
        Task,
        TaskStatus,
    )
    from tensorrt_llm.scaffolding.task_collection import (
        TaskCollection,
        with_task_collection,
    )
    from tensorrt_llm.scaffolding.worker import OpenaiWorker, Worker

    HAS_TENSORRT_LLM = True

except ImportError:
    HAS_TENSORRT_LLM = False

    # ---- Stub base classes ----
    # These allow subclass definitions and isinstance checks to succeed at
    # import time. Instantiation of classes that depend on the real
    # tensorrt_llm will raise ImportError with a clear message.

    class Controller:
        """Stub for tensorrt_llm.scaffolding.controller.Controller."""

        def process(self, tasks: list, **kwargs) -> Any:
            raise ImportError(_INSTALL_MSG)

    @dataclass
    class Task:
        """Stub for tensorrt_llm.scaffolding.task.Task."""

        worker_tag: Any = None

    @dataclass
    class GenerationTask(Task):
        """Stub for tensorrt_llm.scaffolding.task.GenerationTask."""

        input_str: str | None = None
        output_str: str | None = None
        input_tokens: list | None = None
        output_tokens: list | None = None
        logprobs: Any = None
        finish_reason: str | None = None
        perf_metrics: Any = None
        customized_result_fields: dict = field(default_factory=dict)

    @dataclass
    class ChatTask(Task):
        """Stub for tensorrt_llm.scaffolding.task.ChatTask."""

        messages: list = field(default_factory=list)
        completion: Any = None
        tools: list | None = None
        finish_reason: str | None = None
        input_tokens: list | None = None
        output_tokens: list | None = None
        enable_token_counting: bool = False
        prompt_tokens_num: int = 0
        completion_tokens_num: int = 0
        reasoning_tokens_num: int = 0
        perf_metrics: Any = None

        @staticmethod
        def create_from_prompt(prompt: str) -> ChatTask:
            return ChatTask(messages=[{"role": "user", "content": prompt}])

        def messages_to_dict_content(self) -> list:
            return self.messages

    class TaskStatus(enum.Enum):
        """Stub for tensorrt_llm.scaffolding.task.TaskStatus."""

        SUCCESS = "success"
        WORKER_EXECEPTION = "worker_exception"  # noqa: S105 (matches upstream typo)

    class AssistantMessage:
        """Stub for tensorrt_llm.scaffolding.task.AssistantMessage."""

        def __init__(
            self,
            content: str | None = None,
            reasoning: str | None = None,
            reasoning_content: str | None = None,
            tool_calls: list | None = None,
        ):
            self.content = content
            self.reasoning = reasoning
            self.reasoning_content = reasoning_content
            self.tool_calls = tool_calls

    class TaskCollection:
        """Stub for tensorrt_llm.scaffolding.task_collection.TaskCollection."""

        def before_yield(self, tasks: list) -> None:
            pass

        def after_yield(self, tasks: list) -> None:
            pass

    def with_task_collection(name: str, collection_cls: type):
        """Stub for tensorrt_llm.scaffolding.task_collection.with_task_collection."""

        def decorator(cls):
            if not hasattr(cls, "task_collections"):
                cls.task_collections = {}
            cls.task_collections[name] = collection_cls()
            return cls

        return decorator

    class Worker:
        """Stub for tensorrt_llm.scaffolding.worker.Worker."""

    class OpenaiWorker(Worker):
        """Stub for tensorrt_llm.scaffolding.worker.OpenaiWorker."""

        def __init__(self, async_client: Any = None, model: str = "", **kwargs):
            self.async_client = async_client
            self.model = model

        def convert_task_params(self, task: Any) -> dict:
            return {}

    @dataclass
    class ScaffoldingOutput:
        """Stub for tensorrt_llm.scaffolding.result.ScaffoldingOutput."""

        text: str = ""
        token_ids: list = field(default_factory=list)

    class NativeGenerationController(Controller):
        """Stub for tensorrt_llm.scaffolding.NativeGenerationController."""

        class WorkerTag(enum.Enum):
            GENERATION = "generation"

    class ScaffoldingLlm:
        """Stub for tensorrt_llm.scaffolding.ScaffoldingLlm."""

        def __init__(self, *args, **kwargs):
            raise ImportError(_INSTALL_MSG)


__all__ = [
    "HAS_TENSORRT_LLM",
    "AssistantMessage",
    "ChatTask",
    "Controller",
    "GenerationTask",
    "NativeGenerationController",
    "OpenaiWorker",
    "ScaffoldingLlm",
    "ScaffoldingOutput",
    "Task",
    "TaskCollection",
    "TaskStatus",
    "Worker",
    "with_task_collection",
]
