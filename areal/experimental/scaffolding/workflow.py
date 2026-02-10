"""
ScaffoldingWorkflow - RolloutWorkflow implementation using TensorRT-LLM Scaffolding.

This module provides the ScaffoldingWorkflow class that wraps a ScaffoldingLlm
instance to be used as a RolloutWorkflow in AReaL's training pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.experimental.scaffolding._compat import ScaffoldingLlm

logger = logging.getLogger("ScaffoldingWorkflow")


class ScaffoldingWorkflow(RolloutWorkflow):
    """RolloutWorkflow implementation using TensorRT-LLM Scaffolding framework.

    This workflow wraps a ScaffoldingLlm instance and uses it for inference
    and reward computation instead of the standard InferenceEngine. The
    scaffolding_llm handles the full pipeline including:
    - Text generation via NativeGenerationController
    - Reward computation via RLVRRewardController
    - Trajectory creation via PipelineTrajectoryMaker

    Parameters
    ----------
    scaffolding_llm : ScaffoldingLlm
        The configured ScaffoldingLlm instance that orchestrates controllers
        and workers for the RLVR pipeline.

    Example
    -------
    ```python
    from tensorrt_llm.scaffolding import NativeGenerationController, ScaffoldingLlm

    # Create worker from AReaL engine
    rollout_worker = CreateWorkerFromEngine(engine)

    # Create controllers
    rollout_controller = NativeGenerationController()
    reward_controller = RLVRRewardController(gsm8k_reward_fn)
    trajectory_maker = PipelineTrajectoryMaker(rollout_controller, reward_controller)

    # Create ScaffoldingLlm
    scaffolding_llm = ScaffoldingLlm(
        trajectory_maker,
        {NativeGenerationController.WorkerTag.GENERATION: rollout_worker},
    )

    # Create ScaffoldingWorkflow
    workflow = ScaffoldingWorkflow(scaffolding_llm)
    ```
    """

    def __init__(self, scaffolding_llm: ScaffoldingLlm):
        """Initialize the ScaffoldingWorkflow.

        Parameters
        ----------
        scaffolding_llm : ScaffoldingLlm
            The configured ScaffoldingLlm instance for inference and rewards.
        """
        self.scaffolding_llm = scaffolding_llm

    async def arun_episode(
        self,
        engine: InferenceEngine,  # noqa: ARG002 - Not used, using self.scaffolding_llm instead
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode using the scaffolding framework.

        This method uses self.scaffolding_llm for inference and reward
        computation instead of the provided engine parameter. The scaffolding
        framework handles the full RLVR pipeline internally.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (not used - scaffolding_llm handles inference).
        data : dict[str, Any]
            Input data for the workflow episode, typically containing:
            - "messages": The chat messages/prompt
            - "answer": The ground truth answer for reward computation
            - Other task-specific fields

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping interaction IDs to their completion results
            including token IDs, log probabilities, and rewards.
        """
        # Extract prompt from data
        # The data format follows AReaL's dataset conventions
        prompt = self._extract_prompt(data)

        # Run the scaffolding pipeline
        # The scaffolding_llm.generate() returns a ScaffoldingResult
        # The PipelineTrajectoryMaker controller produces InteractionWithTokenLogpReward
        result = await self._run_scaffolding_inference(prompt, data)

        return result

    def _extract_prompt(self, data: dict[str, Any]) -> str:
        """Extract the prompt string from input data.

        Parameters
        ----------
        data : dict[str, Any]
            Input data containing messages or prompt.

        Returns
        -------
        str
            The extracted prompt string.
        """
        # Handle different data formats
        if "messages" in data:
            # Chat format - messages will be processed by the scaffolding pipeline
            # Return raw data for scaffolding to handle
            return data
        elif "prompt" in data:
            return data["prompt"]
        else:
            raise ValueError(
                f"Data must contain 'messages' or 'prompt' key. Got keys: {data.keys()}"
            )

    async def _run_scaffolding_inference(
        self,
        prompt: str | dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward]:
        """Run inference through the scaffolding pipeline asynchronously.

        Uses the async interface of ScaffoldingLlm (generate_async) to run
        inference without blocking. The ScaffoldingResult supports async
        iteration and can be awaited directly.

        Parameters
        ----------
        prompt : str | dict[str, Any]
            The prompt string or data dict for generation.
        data : dict[str, Any]
            Full input data including ground truth for reward computation.

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary of interaction results with rewards.

        See Also
        --------
        TensorRT-LLM examples/scaffolding/run_basic_generation.py : test_async function
        """
        # Use the async interface of ScaffoldingLlm
        # generate_async returns a ScaffoldingResult that supports:
        # 1. async iteration: async for result in llm.generate_async(prompt)
        # 2. direct await: await llm.generate_async(prompt)
        #
        # We await the result directly to get the final output
        # The ScaffoldingResult.__await__ calls aresult() which waits until done
        scaffolding_result = await self.scaffolding_llm.generate_async(prompt)

        # The result from PipelineTrajectoryMaker is already in the expected format
        # dict[str, InteractionWithTokenLogpReward]
        # Access the trajectory data from the scaffolding result
        return scaffolding_result

    def shutdown(self):
        """Shutdown the scaffolding LLM and release resources."""
        if self.scaffolding_llm is not None:
            self.scaffolding_llm.shutdown()
