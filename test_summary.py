#!/usr/bin/env python3
"""
Test summary script for RLtoolbox framework.

This script runs all tests and provides a comprehensive summary of the framework's
functionality and test coverage.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            if result.stdout:
                print(result.stdout.split('\n')[-3])  # Show summary line
            return True
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    print("🚀 RLtoolbox Framework Test Suite")
    print("="*60)

    total_tests = 0
    passed_tests = 0

    # Test categories
    tests = [
        ("python -m pytest tests/unit/test_base.py -v --tb=short", "Base Component Tests"),
        ("python -m pytest tests/unit/test_registry.py -v --tb=short", "Component Registry Tests"),
        ("python -m pytest tests/unit/test_hooks.py -v --tb=short", "Hook Executor Tests"),
        ("python -m pytest tests/integration/test_trainer.py -v --tb=short", "Integration Tests"),
        ("python examples/test_framework.py", "Framework Functionality Test"),
        ("python examples/train_cartpole.py configs/cartpole_random.json", "Random Agent Example"),
        ("python examples/train_cartpole.py configs/cartpole_epsilon_greedy.json --evaluate 3", "Epsilon-Greedy Example"),
    ]

    # Run all tests
    for cmd, description in tests:
        total_tests += 1
        if run_command(cmd, description):
            passed_tests += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total test categories: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! RLtoolbox is ready for use.")
        print(f"\n📚 Quick Start:")
        print(f"   python examples/train_cartpole.py configs/cartpole_random.json")
        print(f"\n📖 Documentation:")
        print(f"   See README.md for detailed usage instructions")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test categories failed.")
        print(f"   Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
