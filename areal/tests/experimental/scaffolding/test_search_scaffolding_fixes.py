from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.experimental.scaffolding._compat import (
    AssistantMessage,
    ChatTask,
    Controller,
    Task,
)
from areal.experimental.scaffolding.controllers import TraceTrajectoryMaker
from areal.experimental.scaffolding.worker import SGLangWorker
from examples.scaffolding.search_agent_controller import SearchAgentController
from examples.scaffolding.search_scaffolding import SearchScaffoldingWorkflow


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        return [101, 102, 103]

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids):
        return "decoded prompt"


class _FakeGenerationController(Controller):
    def __init__(self, responses: list[str]):
        super().__init__()
        self.responses = responses
        self.index = 0
        self.sampling_params = {"max_tokens": 32}

    def process(self, tasks: list[Task], **kwargs) -> Generator[list[Task], None, None]:
        for task in tasks:
            assert isinstance(task, ChatTask)
            task.messages.append(AssistantMessage(self.responses[self.index]))
        self.index += 1
        yield tasks


class _AwaitableResult:
    def __init__(self, output):
        self.outputs = [output]

    def __await__(self):
        async def _done():
            return self

        return _done().__await__()


@pytest.mark.asyncio
async def test_search_agent_controller_executes_tool_calls_inside_event_loop():
    responses = [
        '<tool_call>{"name": "search", "arguments": {"query": ["weather"]}}</tool_call>',
        "<answer>sunny</answer>",
    ]
    controller = SearchAgentController(
        generation_controller=_FakeGenerationController(responses),
        tokenizer=_FakeTokenizer(),
        max_turns=2,
        max_total_tokens=1024,
        messages=[{"role": "user", "content": "weather?"}],
        input_tokens=[11, 12],
    )

    async def _fake_execute_tool(tool_name: str, tool_args: dict) -> str:
        assert tool_name == "search"
        assert tool_args == {"query": ["weather"]}
        return "tool result"

    controller._execute_tool = _fake_execute_tool

    yielded = []
    for batch in controller.process([]):
        yielded.append(batch)

    assert len(yielded) == 2
    chat_task = yielded[-1][0]
    assert isinstance(chat_task, ChatTask)
    assert any(
        getattr(message, "content", "") == "<tool_response>\ntool result\n</tool_response>"
        for message in chat_task.messages
    )


@pytest.mark.asyncio
async def test_sglang_worker_chat_handler_populates_output_tokens():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="hello",
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                    token_ids=[1, 2, 3],
                ),
            )
        ],
        usage=None,
    )
    async_client = MagicMock()
    async_client.base_url = "http://test/v1"
    async_client.chat.completions.create = AsyncMock(return_value=response)

    worker = SGLangWorker(
        async_client=async_client,
        model="default",
        engine=MagicMock(),
    )
    task = ChatTask.create_from_prompt("hello")

    status = await worker.chat_handler(task)

    assert status.value == "success"
    assert task.output_tokens == [1, 2, 3]


@pytest.mark.asyncio
async def test_search_workflow_returns_full_trace_results():
    workflow = SearchScaffoldingWorkflow(
        reward_fn=lambda *args, **kwargs: 1.0,
        gconfig=MagicMock(new_with_stop_and_pad_token_ids=lambda tokenizer: MagicMock()),
        tokenizer=_FakeTokenizer(),
    )
    workflow.worker = MagicMock()
    workflow.search_controller = MagicMock()
    workflow.reward_controller = MagicMock()

    first = InteractionWithTokenLogpReward()
    first._cache = {"input_ids": MagicMock()}
    second = InteractionWithTokenLogpReward()
    second._cache = {"input_ids": MagicMock()}
    trace_results = {"a": first, "b": second}
    workflow.scaffolding_llm = MagicMock(
        generate_async=MagicMock(
            return_value=_AwaitableResult(
                SimpleNamespace(data=trace_results, text="ignored")
            )
        )
    )

    result = await workflow.arun_episode(
        engine=MagicMock(),
        data={"question": "q", "answer": "a"},
    )

    assert result == trace_results


def test_trace_trajectory_maker_uses_instance_task_collection():
    first = TraceTrajectoryMaker(MagicMock(), MagicMock())
    second = TraceTrajectoryMaker(MagicMock(), MagicMock())

    assert first.task_collections["chat_tracer"] is not second.task_collections["chat_tracer"]
