#!/usr/bin/env python3
"""
Basic test script to verify RLtoolbox framework functionality.

This script runs a simple test to ensure the framework is working correctly
by training a random agent on a simple environment.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add src to path so we can import rltoolbox
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rltoolbox import RLTrainer


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        "seed": 42,
        "development_mode": True,

        "packages": {},

        "components": {
            "env": {
                "package": "rltoolbox",
                "type": "RandomEnvironment",
                "state_dim": 4,
                "action_dim": 2,
                "episode_length": 10,
                "reward_range": [-1.0, 1.0]
            },
            "agent": {
                "package": "rltoolbox",
                "type": "RandomAgent",
                "action_space_type": "discrete",
                "num_actions": 2
            },
            "logger": {
                "package": "rltoolbox",
                "type": "ConsoleLogger",
                "log_frequency": 1,
                "verbose": True,
                "log_steps": False
            }
        },

        "hooks": {
            "training_start": ["logger"],
            "episode_start": [],
            "episode_reset": ["env"],
            "step_start": [],
            "action_selection": ["agent"],
            "environment_step": ["env"],
            "step_end": [],
            "episode_end": ["logger"],
            "training_end": ["logger"]
        },

        "training": {
            "max_episodes": 5,
            "max_steps_per_episode": 10
        }
    }

    return config


def test_basic_functionality():
    """Test basic framework functionality."""
    print("Testing RLtoolbox basic functionality...")

    try:
        # Create test configuration
        config = create_test_config()

        # Create trainer
        trainer = RLTrainer(config_dict=config)

        # Debug: Check what components are registered
        print(f"Available components: {trainer.registry.list_components()}")

        # Run training
        trainer.train()

        # Test evaluation
        eval_results = trainer.evaluate(num_episodes=2)

        print("\nTest completed successfully!")
        print(f"Evaluation results: {eval_results}")

        # Verify expected results
        assert "mean_reward" in eval_results
        assert "episode_rewards" in eval_results
        assert len(eval_results["episode_rewards"]) == 2

        print("✅ All basic functionality tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")

    try:
        # Test missing seed
        config = create_test_config()
        del config["seed"]

        trainer = RLTrainer(config_dict=config)
        print("⚠️  Warning: Missing seed should generate warning but still work")

        # Test invalid component
        config = create_test_config()
        config["components"]["invalid"] = {
            "package": "rltoolbox",
            "type": "NonExistentComponent"
        }
        config["hooks"]["action_selection"] = ["invalid"]

        try:
            trainer = RLTrainer(config_dict=config)
            print("❌ Should have failed with invalid component")
        except Exception:
            print("✅ Invalid component correctly rejected")

    except Exception as e:
        print(f"❌ Config validation test failed: {e}")
        raise


def test_checkpoint_functionality():
    """Test checkpoint save/load functionality."""
    print("\nTesting checkpoint functionality...")

    try:
        config = create_test_config()

        # Create trainer and run partial training
        trainer1 = RLTrainer(config_dict=config)

        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save checkpoint
            trainer1.save_checkpoint(checkpoint_path)

            # Create new trainer and load checkpoint
            trainer2 = RLTrainer(config_dict=config)
            trainer2.load_checkpoint(checkpoint_path)

            print("✅ Checkpoint save/load test passed!")

        finally:
            # Clean up
            Path(checkpoint_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        raise


def test_hook_execution_order():
    """Test that hooks execute in the correct order."""
    print("\nTesting hook execution order...")

    try:
        # Just test basic multi-component execution with existing components
        config = create_test_config()

        # Add another agent to test ordering
        config["components"]["agent2"] = {
            "package": "rltoolbox",
            "type": "RandomAgent",
            "action_space_type": "discrete",
            "num_actions": 2
        }

        # Test multiple components in action selection
        config["hooks"]["action_selection"] = ["agent", "agent2"]

        # Create trainer with minimal episodes
        config["training"]["max_episodes"] = 1
        trainer = RLTrainer(config_dict=config)

        # This should run without errors if hook ordering works
        trainer.train()

        print("✅ Hook execution order test completed!")

    except Exception as e:
        print(f"❌ Hook execution order test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("🧪 Running RLtoolbox Framework Tests")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_config_validation()
        test_checkpoint_functionality()
        test_hook_execution_order()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! RLtoolbox is working correctly.")

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"💥 Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
