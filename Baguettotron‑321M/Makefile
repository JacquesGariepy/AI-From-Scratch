# Baguettotron Makefile
# Clean, modern build automation for Baguettotron

.PHONY: help install dev-install test clean lint format dataset-wiki dataset-demo train generate docs

# Default target
help:
	@echo "Baguettotron - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package in production mode"
	@echo "  make dev-install      Install package in development mode with dev dependencies"
	@echo ""
	@echo "Datasets:"
	@echo "  make dataset-setup    Interactive dataset setup"
	@echo "  make dataset-wiki     Prepare Wikipedia Simple dataset (200MB)"
	@echo "  make dataset-demo     Prepare demo dataset for testing"
	@echo "  make dataset-list     List available datasets"
	@echo "  make dataset-validate Validate prepared datasets"
	@echo ""
	@echo "Training:"
	@echo "  make train-tiny       Quick training with tiny model"
	@echo "  make train-321m       Full 321M model training"
	@echo ""
	@echo "Code Quality:"
	@echo "  make test             Run test suite"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Run linting checks"
	@echo "  make format           Auto-format code with black"
	@echo "  make typecheck        Run type checking with mypy"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Clean build artifacts and cache"
	@echo "  make docs             Build documentation"
	@echo "  make structure        Show project structure"
	@echo ""

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pip install black mypy pytest pytest-cov

# Dataset preparation
dataset-setup:
	python scripts/setup_dataset.py

dataset-wiki:
	python scripts/prepare_wikipedia_data.py --lang simple --tokenize

dataset-demo:
	python scripts/prepare_demo_data.py --num-samples 5000

dataset-list:
	python scripts/validate_dataset.py --list

dataset-validate:
	python scripts/validate_dataset.py data/*.json

# Training
train-tiny:
	python scripts/train.py \
		--train-data data/train_tokens.json \
		--model-config tiny \
		--epochs 10 \
		--batch-size 32

train-321m:
	python scripts/train.py \
		--train-data data/wikipedia_simple_tokens.json \
		--model-config 321m \
		--epochs 100 \
		--batch-size 64 \
		--mixed-precision \
		--compile

# Code quality
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/baguettotron --cov-report=html --cov-report=term

lint:
	pylint src/baguettotron --disable=C0111,R0913

format:
	black src/ tests/ scripts/

typecheck:
	mypy src/baguettotron --ignore-missing-imports

# Utilities
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache htmlcov
	rm -rf **/__pycache__ **/*.pyc
	rm -f .coverage coverage.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Documentation available in docs/"
	@echo "- README.md: Main documentation"
	@echo "- DATASET_GUIDE.md: Complete dataset guide"
	@echo "- QUICK_START_DATASETS.md: Quick start guide"

structure:
	@echo "Baguettotron Project Structure:"
	@tree -L 2 -I '__pycache__|*.pyc|.pytest_cache|htmlcov|.mypy_cache|*.egg-info|.venv' || ls -R

# Development workflow
dev: dev-install dataset-demo train-tiny

# CI/CD workflow
ci: lint typecheck test

# Production setup
prod: install dataset-wiki

.DEFAULT_GOAL := help
