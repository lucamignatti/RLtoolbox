# Makefile for RLtoolbox development tasks

.PHONY: help install install-dev test test-unit test-integration test-coverage clean lint format check docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package in development mode with all dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black"
	@echo "  check          Run all checks (lint + format check)"
	@echo "  clean          Clean up temporary files"
	@echo "  docs           Build documentation"
	@echo "  example        Run example training script"

# Installation targets
install:
	pip install .

install-dev:
	pip install -e ".[dev,examples]"
	pip install -r requirements.txt

# Testing targets
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v -m unit

test-integration:
	python -m pytest tests/integration/ -v -m integration

test-coverage:
	python -m pytest tests/ --cov=rltoolbox --cov-report=html --cov-report=term-missing

test-fast:
	python -m pytest tests/ -v -x --disable-warnings

# Code quality targets
lint:
	flake8 src/rltoolbox tests examples
	mypy src/rltoolbox --ignore-missing-imports

format:
	black src/rltoolbox tests examples

format-check:
	black --check src/rltoolbox tests examples

check: lint format-check

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/
	rm -rf dist/

# Documentation targets
docs:
	@echo "Documentation build not yet implemented"

# Example targets
example:
	python examples/train_cartpole.py configs/cartpole_random.json

example-epsilon:
	python examples/train_cartpole.py configs/cartpole_epsilon_greedy.json

# Development workflow targets
dev-setup: install-dev
	@echo "Development environment setup complete"

quick-test:
	python examples/test_framework.py

# CI/CD targets
ci-test: install-dev test-coverage lint

# Package building
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*
