#!/usr/bin/env python3
"""
Example training script for RLtoolbox framework.

This script demonstrates how to use the RLtoolbox framework to train
an agent on the CartPole environment using different configurations.
"""

import sys
import argparse
from pathlib import Path

# Add src to path so we can import rltoolbox
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rltoolbox import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train RL agent using RLtoolbox")
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Enable development mode (skip version validation)"
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=0,
        help="Number of evaluation episodes to run after training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to save final checkpoint"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Path to load checkpoint from"
    )

    args = parser.parse_args()

    try:
        # Create trainer
        print(f"Loading configuration from: {args.config}")
        trainer = RLTrainer(config_path=args.config)

        # Enable development mode if requested
        if args.development:
            trainer.registry.set_development_mode(True)

        # Load checkpoint if specified
        if args.load_checkpoint:
            print(f"Loading checkpoint from: {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)

        # Train the agent
        print("Starting training...")
        trainer.train()

        # Run evaluation if requested
        if args.evaluate > 0:
            print(f"\nRunning evaluation for {args.evaluate} episodes...")
            eval_results = trainer.evaluate(num_episodes=args.evaluate)

            print("Evaluation Results:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")

        # Save checkpoint if requested
        if args.checkpoint:
            print(f"Saving checkpoint to: {args.checkpoint}")
            trainer.save_checkpoint(args.checkpoint)

        # Print final metrics
        metrics = trainer.get_metrics()
        if metrics["episode_rewards"]:
            print(f"\nTraining completed!")
            print(f"Total episodes: {len(metrics['episode_rewards'])}")
            print(f"Final mean reward: {sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards'])):.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
