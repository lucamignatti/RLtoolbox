#!/usr/bin/env python3
"""
Standalone training script for RLtoolbox.

This script loads a JSON configuration, runs training, and performs evaluation at the end.
No wrapping code is required—simply run:

    python scripts/train.py path/to/config.json

Evaluation is performed automatically after training (default: 10 episodes).
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import rltoolbox
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rltoolbox import RLTrainer

def main():
    parser = argparse.ArgumentParser(description="Standalone RL training with RLtoolbox")
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=10,
        help="Number of evaluation episodes to run after training (default: 10)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to save final checkpoint (optional)"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Path to load checkpoint from (optional)"
    )

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    trainer = RLTrainer(config_path=args.config)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

    # Train the agent
    print("Starting training...")
    trainer.train()

    # Always run evaluation at the end
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
    if metrics.get("episode_rewards"):
        print(f"\nTraining completed!")
        print(f"Total episodes: {len(metrics['episode_rewards'])}")
        print(f"Final mean reward: {sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards'])):.2f}")

if __name__ == "__main__":
    main()
