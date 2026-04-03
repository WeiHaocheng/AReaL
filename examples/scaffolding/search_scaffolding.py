"""
Search Agent Scaffolding Example.

This example demonstrates multi-turn search-based RL training using
the scaffolding framework.  A ``SearchAgentController`` drives a
tool-calling loop (search + visit) expressed as a scaffolding
``Controller``, while ``TraceTrajectoryMaker`` traces each LLM call
for PPO training.

The example uses real web search (via Serper API, requires SERPER_KEY_ID
env var) and basic HTTP fetching for page visits.  An LLM judge is used
for reward computation (the same inference engine is used for both agent
generation and judging).

Usage:
    python examples/scaffolding/search_scaffolding.py \\
        --config examples/scaffolding/search_scaffolding.yaml \\
        +scheduler.type=local experiment_name=areal trial_name=search_scaffolding
"""

import datetime
import sys
from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.dataset import get_custom_dataset
from areal.experimental.scaffolding._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from areal.experimental.scaffolding.controllers import (
    LLMJudgeController,
    TraceTrajectoryMaker,
)
from areal.experimental.scaffolding.workflow import ScaffoldingWorkflow
from areal.trainer import PPOTrainer
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

from .search_agent_controller import SearchAgentController

logger = logging.getLogger("SearchScaffoldingWorkflow")

# Reuse the system prompt from tongyi_deepresearch (search + visit only).
SYSTEM_PROMPT = (
    "You are a deep research assistant. Your core function is to conduct "
    "thorough, multi-source investigations into any topic. You must handle "
    "both broad, open-domain inquiries and queries within specialized academic "
    "fields. For every request, synthesize information from credible, diverse "
    "sources to deliver a comprehensive, accurate, and objective response. "
    "When you have gathered sufficient information and are ready to provide "
    "the definitive response, you must enclose the entire final answer within "
    "<answer></answer> tags.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "search", "description": '
    '"Perform Google web searches then returns a string of the top search '
    'results. Accepts multiple queries.", "parameters": {"type": "object", '
    '"properties": {"query": {"type": "array", "items": {"type": "string", '
    '"description": "The search query."}, "minItems": 1, "description": '
    '"The list of search queries."}}, "required": ["query"]}}}\n'
    '{"type": "function", "function": {"name": "visit", "description": '
    '"Visit webpage(s) and return the summary of the content.", "parameters": '
    '{"type": "object", "properties": {"url": {"type": "array", "items": '
    '{"type": "string"}, "description": "The URL(s) of the webpage(s) to '
    'visit. Can be a single URL or an array of URLs."}, "goal": {"type": '
    '"string", "description": "The specific information goal for visiting '
    'webpage(s)."}}, "required": ["url", "goal"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Current date: "
)


def _resolve_context_length(config: GRPOConfig) -> int:
    """Pick the active inference context window from config."""
    if getattr(config, "sglang", None) and config.sglang.context_length is not None:
        return config.sglang.context_length
    if getattr(config, "vllm", None) and config.vllm.max_model_len is not None:
        return config.vllm.max_model_len
    return 8192


def _bounded_completion_tokens(max_total_tokens: int, requested_max_tokens: int) -> int:
    """Keep per-request completion length well within the context budget."""
    return max(128, min(requested_max_tokens, max_total_tokens // 4))


def _bounded_judge_tokens(max_total_tokens: int) -> int:
    """Reserve a smaller budget for judge calls to avoid context overflows."""
    return max(128, min(512, max_total_tokens // 4))


class SearchScaffoldingWorkflow(ScaffoldingWorkflow):
    """ScaffoldingWorkflow for multi-turn search-agent RL training.

    The episode loop delegates to ``SearchAgentController`` (multi-turn
    tool calling) composed with ``TraceTrajectoryMaker`` (trajectory
    tracing) and ``LLMJudgeController`` (LLM-as-judge reward).

    Parameters
    ----------
    reward_fn : Callable | str
        Fallback reward function or importable path (used by parent class
        for non-LLM-judge scenarios; the LLM judge is the primary reward).
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer or path.
    enable_thinking : bool
        Whether to enable thinking tokens.
    max_turns : int
        Maximum number of LLM calls per episode.
    max_total_tokens : int
        Soft token budget for the conversation.
    max_judge_tokens : int
        Maximum tokens for the LLM judge response.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        max_turns: int = 20,
        max_total_tokens: int = 32768,
        max_judge_tokens: int = 8192,
    ):
        super().__init__(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.max_judge_tokens = max_judge_tokens

    # ------------------------------------------------------------------
    # Scaffolding construction
    # ------------------------------------------------------------------

    def build_scaffolding_llm(self, engine: InferenceEngine) -> ScaffoldingLlm:
        """Build ``ScaffoldingLlm`` with ``SearchAgentController`` + ``TraceTrajectoryMaker``.

        Uses ``LLMJudgeController`` as the reward controller so that
        answer correctness is determined by the same LLM (via a judge
        prompt) rather than a deterministic string-matching function.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (worker already initialised by parent).

        Returns
        -------
        ScaffoldingLlm
        """
        max_completion_tokens = _bounded_completion_tokens(
            self.max_total_tokens, self.gconfig.max_new_tokens
        )
        stop_strings = ["\n<tool_response>", "<tool_response>"]
        sampling_params: dict[str, Any] = {
            "max_tokens": max_completion_tokens,
            "temperature": self.gconfig.temperature or 1.0,
            "stop": stop_strings,
        }

        self.gen_controller = NativeGenerationController(
            sampling_params=sampling_params,
        )
        self.reward_controller = LLMJudgeController(
            max_judge_tokens=self.max_judge_tokens,
        )

        self.search_controller = SearchAgentController(
            generation_controller=self.gen_controller,
            tokenizer=self.tokenizer,
            max_turns=self.max_turns,
            max_total_tokens=self.max_total_tokens,
        )

        self.trajectory_maker = TraceTrajectoryMaker(
            rollout_controller=self.search_controller,
            reward_controller=self.reward_controller,
        )

        return ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )

    def _ensure_trace_tokens(self, trace_results: dict[str, Any]) -> dict[str, Any]:
        """Backfill output tokens for traced chat turns when the API omits them."""
        for interaction in trace_results.values():
            resp = getattr(interaction, "model_response", None)
            if resp is None or resp.output_tokens:
                continue

            output_text = ""
            completion = getattr(interaction, "completion", None)
            if completion is not None and completion.choices:
                output_text = completion.choices[0].message.content or ""

            if not output_text:
                continue

            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            resp.output_tokens = list(output_tokens)
            resp.output_logprobs = [0.0] * len(output_tokens)
            resp.output_versions = [-1] * len(output_tokens)
        return trace_results

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single search-agent episode.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine.
        data : dict[str, Any]
            Must contain ``"question"`` and ``"answer"`` keys.

        Returns
        -------
        dict[str, Any]
            Full traced interactions for PPO training.
        """
        if self.worker is None:
            self._lazy_init_scaffolding(engine)

        # Build messages: system prompt + user question
        system_prompt = SYSTEM_PROMPT + datetime.date.today().strftime("%Y-%m-%d")
        question = data.get("question", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Tokenize the original prompt
        input_ids = list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )
        prompt_str = self.tokenizer.decode(input_ids)

        # Configure per-episode state on the search controller and reward controller.
        # ScaffoldingLlm.clone() deep-copies the controllers for each request,
        # so the per-episode data is isolated across concurrent episodes.
        self.search_controller.messages = messages
        self.search_controller.input_tokens = input_ids
        self.reward_controller.task_data = data

        # Run the full pipeline (generation + LLM judge reward)
        result = self.scaffolding_llm.generate_async(prompt_str)
        await result

        # Extract trace results
        scaffolding_output = result.outputs[0]
        trace_results = scaffolding_output.data

        if trace_results:
            return self._ensure_trace_tokens(trace_results)

        return {}


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(args):
    """Main entry point for search scaffolding training."""
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    max_total_tokens = _resolve_context_length(config)
    max_judge_tokens = _bounded_judge_tokens(max_total_tokens)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        reward_fn="examples.scaffolding.search_reward.search_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
        max_turns=10,
        max_total_tokens=max_total_tokens,
        max_judge_tokens=max_judge_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.scaffolding.search_scaffolding.SearchScaffoldingWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.scaffolding.search_scaffolding.SearchScaffoldingWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
