#!/usr/bin/env python3
"""
Command-line interface for RLtoolbox.

Provides standalone training commands that load JSON configurations and execute
training with automatic evaluation, requiring no wrapping code.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.trainer import RLTrainer


def main():
    """Main CLI entry point for rltoolbox-train command."""
    parser = argparse.ArgumentParser(
        description="Standalone RL training with RLtoolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rltoolbox-train config.json
  rltoolbox-train config.json --checkpoint model.ckpt
  rltoolbox-train config.json --load-checkpoint prev_model.ckpt
        """
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration JSON file"
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Skip evaluation even if configured in JSON"
    )

    args = parser.parse_args()

    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            sys.exit(1)

        # Create trainer
        if args.verbose:
            print(f"Loading configuration from: {args.config}")

        trainer = RLTrainer(config_path=str(config_path), skip_automatic_evaluation=args.no_evaluation)

        # Load checkpoint if specified
        if args.load_checkpoint:
            checkpoint_path = Path(args.load_checkpoint)
            if not checkpoint_path.exists():
                print(f"Error: Checkpoint file not found: {args.load_checkpoint}")
                sys.exit(1)

            if args.verbose:
                print(f"Loading checkpoint from: {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)

        # Train the agent
        if args.verbose:
            print("Starting training...")
        trainer.train()

        # Run evaluation if configured and not disabled
        if not args.no_evaluation:
            evaluation_config = trainer.config.get("evaluation", {})
            if evaluation_config.get("enabled", False):
                num_episodes = evaluation_config.get("num_episodes", 10)

                if args.verbose:
                    print(f"\nRunning CLI evaluation for {num_episodes} episodes...")

                eval_results = trainer.evaluate(num_episodes=num_episodes)

                print("CLI Evaluation Results:")
                print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")

                # Store evaluation results in trainer context for potential hooks
                trainer.context["final_evaluation"] = eval_results
        else:
            if args.verbose:
                print("\nSkipping evaluation due to --no-evaluation flag")

        # Save checkpoint if requested
        if args.checkpoint:
            if args.verbose:
                print(f"Saving checkpoint to: {args.checkpoint}")
            trainer.save_checkpoint(args.checkpoint)

        # Print final metrics
        metrics = trainer.get_metrics()
        if metrics.get("episode_rewards"):
            if args.verbose:
                print(f"\nTraining completed!")
                print(f"Total episodes: {len(metrics['episode_rewards'])}")
                recent_rewards = metrics['episode_rewards'][-10:]
                print(f"Final mean reward (last 10): {sum(recent_rewards) / len(recent_rewards):.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
