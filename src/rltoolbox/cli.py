import argparse
import sys
from pathlib import Path

# Add src to path so we can import rltoolbox
sys.path.insert(0, str(Path(__file__).parent.parent))

from rltoolbox.core.trainer import RLTrainer
from rltoolbox.core.registry import get_registry


def train(args):
    print(f"Loading configuration from: {args.config}")
    trainer = RLTrainer(config_path=args.config)

    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

    print("Starting training...")
    trainer.train()

    if args.evaluate:
        print(f"\nRunning evaluation for {args.evaluate} episodes...")
        eval_results = trainer.evaluate(num_episodes=args.evaluate)
        print("Evaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} \u00b1 {eval_results['std_reward']:.2f}")
        print(f"  Mean Length: {eval_results['mean_length']:.2f} \u00b1 {eval_results['std_length']:.2f}")

    if args.checkpoint:
        print(f"Saving checkpoint to: {args.checkpoint}")
        trainer.save_checkpoint(args.checkpoint)

    metrics = trainer.get_metrics()
    if metrics.get("episode_rewards"):
        print(f"\nTraining completed!")
        print(f"Total episodes: {len(metrics['episode_rewards'])}")
        print(f"Final mean reward: {sum(metrics['episode_rewards'][-10:]) / min(10, len(metrics['episode_rewards'])):.2f}")


def evaluate(args):
    print(f"Loading checkpoint from: {args.checkpoint}")
    trainer = RLTrainer(config_path=args.config)
    trainer.load_checkpoint(args.checkpoint)

    print(f"\nRunning evaluation for {args.num_episodes} episodes...")
    eval_results = trainer.evaluate(num_episodes=args.num_episodes)

    print("Evaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} \u00b1 {eval_results['std_reward']:.2f}")
    print(f"  Mean Length: {eval_results['mean_length']:.2f} \u00b1 {eval_results['std_length']:.2f}")


def list_components(args):
    registry = get_registry()
    # The registry needs to be populated with packages from a config file.
    # This is a placeholder for how it would work.
    print("Available components:")
    for component in registry.list_components():
        print(f"- {component}")


def new_component(args):
    package_name = args.name
    package_path = Path(package_name)

    if package_path.exists():
        print(f"Error: Directory '{package_name}' already exists.")
        return

    # Create directory structure
    src_path = package_path / "src" / package_name
    src_path.mkdir(parents=True)

    # Create __init__.py
    (src_path / "__init__.py").touch()

    # Create component file
    component_name = "MyComponent"
    component_file_content = f"""
from typing import Dict, Any

from rltoolbox.core.base import RLComponent


class {component_name}(RLComponent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    # Implement the required hooks here

"""
    (src_path / f"{component_name.lower()}.py").write_text(component_file_content)

    # Create setup.py
    setup_py_content = f"""
from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={{"": "src"}},
    install_requires=[
        "rltoolbox",
    ],
)
"""
    (package_path / "setup.py").write_text(setup_py_content)

    # Create README.md
    readme_content = f"""
# {package_name}

This package contains custom components for RLtoolbox.
"""
    (package_path / "README.md").write_text(readme_content)

    # Initialize Git repository
    try:
        import subprocess
        subprocess.run(["git", "init"], cwd=package_path, check=True, capture_output=True)
        print(f"Initialized Git repository in {package_path}")
    except (ImportError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: Could not initialize Git repository. {e}")

    print(f"Successfully created new component package: {package_name}")


def main():
    parser = argparse.ArgumentParser(description="RLtoolbox CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("config", type=str, help="Path to configuration JSON file")
    train_parser.add_argument("--evaluate", type=int, help="Number of evaluation episodes to run after training")
    train_parser.add_argument("--checkpoint", type=str, help="Path to save final checkpoint")
    train_parser.add_argument("--load-checkpoint", type=str, help="Path to load checkpoint from")
    train_parser.set_defaults(func=train)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("config", type=str, help="Path to configuration JSON file")
    eval_parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    eval_parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    eval_parser.set_defaults(func=evaluate)

    # List components command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.set_defaults(func=list_components)

    # New component command
    new_parser = subparsers.add_parser("new", help="Create a new component template")
    new_parser.add_argument("name", type=str, help="Name of the new component")
    new_parser.set_defaults(func=new_component)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()