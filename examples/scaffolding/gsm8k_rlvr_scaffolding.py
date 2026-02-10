"""
RLVR (Reinforcement Learning with Verifiable Rewards) Example using Scaffolding Framework.

This example demonstrates how to use the scaffolding framework for RLVR training
on the GSM8K math dataset. The scaffolding framework provides a modular and
extensible way to compose inference-time compute methods with RL training.

Key components:
- NativeGenerationController: Handles text generation from scaffolding
- RLVRRewardController: Computes rewards for generated samples
- PipelineTrajectoryMaker: Creates trajectories for RL training
- CreateWorkerFromEngine: Creates a Worker from AReaL's InferenceEngine
- ScaffoldingLlm: Orchestrates controllers and workers
- ScaffoldingWorkflow: Wraps ScaffoldingLlm as a RolloutWorkflow

Usage:
    python examples/scaffolding/gsm8k_rlvr_scaffolding.py --config examples/scaffolding/gsm8k_rlvr_scaffolding.yaml
"""

import sys

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.scaffolding import (
    CreateWorkerFromEngine,
    PipelineTrajectoryMaker,
    RLVRRewardController,
    ScaffoldingWorkflow,
)
from areal.experimental.scaffolding._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from areal.experimental.trainer import PPOTrainer
from areal.reward.gsm8k import gsm8k_reward_fn
from areal.utils.hf_utils import load_hf_tokenizer


def create_rlvr_scaffolding_workflow(
    engine: RemoteSGLangEngine,
) -> ScaffoldingWorkflow:
    """Create the RLVR scaffolding workflow with all required controllers.

    This function sets up the scaffolding pipeline following the RFC pattern:
    1. Create Worker from the SGLang engine
    2. Create NativeGenerationController for text generation
    3. Create RLVRRewardController for reward computation
    4. Create PipelineTrajectoryMaker to compose controllers
    5. Create ScaffoldingLlm to orchestrate the pipeline
    6. Create ScaffoldingWorkflow as the final RolloutWorkflow

    Args:
        engine: The RemoteSGLangEngine for model inference.

    Returns:
        ScaffoldingWorkflow: The configured workflow ready for training.
    """
    # Step 1: Create Worker from the SGLang engine
    # This wraps AReaL's InferenceEngine into a scaffolding-compatible Worker
    rollout_worker = CreateWorkerFromEngine(engine)

    # Step 2: Create the generation controller from scaffolding
    # This handles the core text generation logic
    rollout_controller = NativeGenerationController()

    # Step 3: Create the RLVR reward controller
    # This computes rewards for generated samples using verifiable reward functions
    reward_controller = RLVRRewardController(gsm8k_reward_fn)

    # Step 4: Create the trajectory maker
    # This composes the generation and reward controllers into a pipeline
    # that produces training trajectories
    trajectory_maker = PipelineTrajectoryMaker(rollout_controller, reward_controller)

    # Step 5: Create ScaffoldingLlm
    # This orchestrates the trajectory maker with the worker
    scaffolding_llm = ScaffoldingLlm(
        trajectory_maker,
        {NativeGenerationController.WorkerTag.GENERATION: rollout_worker},
    )

    # Step 6: Create ScaffoldingWorkflow
    # This wraps the ScaffoldingLlm as a RolloutWorkflow for AReaL training
    scaffolding_workflow = ScaffoldingWorkflow(scaffolding_llm)

    return scaffolding_workflow


def main(args):
    """Main entry point for RLVR training with scaffolding."""
    # Load configuration
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load datasets
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

    # Start training with PPOTrainer
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        # Get the rollout engine from the trainer's parallel strategy
        # The engine is initialized by the trainer
        rollout_engine = trainer.get_rollout_engine()

        # Create the scaffolding workflow with the engine
        scaffolding_workflow = create_rlvr_scaffolding_workflow(rollout_engine)

        # Train using the scaffolding workflow
        trainer.train(
            workflow=scaffolding_workflow,
            eval_workflow=scaffolding_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
