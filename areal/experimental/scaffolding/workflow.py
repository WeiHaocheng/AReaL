"""
ScaffoldingWorkflow - RolloutWorkflow implementation using Scaffolding controllers.

This module provides the ScaffoldingWorkflow class that uses AReaL's engine
for generation and scaffolding controllers for reward computation.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.core import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("ScaffoldingWorkflow")


class ScaffoldingWorkflow(RolloutWorkflow):
    """RolloutWorkflow using scaffolding controllers for reward computation.

    Uses AReaL's engine for generation (to get proper token IDs and logprobs)
    and scaffolding's RLVRRewardController for reward computation.

    This serves as a scaffolding-compatible base workflow that can be extended
    with custom controllers for multi-step or multi-turn scenarios.

    Parameters
    ----------
    reward_fn : Callable | str
        The reward function, or an importable string path.
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer or path to load it.
    enable_thinking : bool
        Whether to enable thinking tokens.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking

        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Compute reward using the scaffolding reward function."""
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )
        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward."""
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single episode using engine generation + scaffolding reward.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine for generating responses.
        data : dict[str, Any]
            Input data containing messages and ground truth.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensors for PPO training.
        """
        # Lazy-resolve string reward_fn
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # Tokenize prompt
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        input_ids = list(input_ids)

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        prompt_str = self.tokenizer.decode(input_ids)

        # Generate + compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
