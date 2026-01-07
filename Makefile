# ML Template - Development Commands
# ===================================

.PHONY: help install install-dev lint format test train docs clean docker

# Default target
help:
	@echo "ML Template - Available Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install all dependencies including dev tools"
	@echo ""
	@echo "Development:"
	@echo "  make lint         Run linter and type checker"
	@echo "  make format       Format code with ruff"
	@echo "  make test         Run test suite"
	@echo "  make test-cov     Run tests with coverage report"
	@echo ""
	@echo "Training:"
	@echo "  make train        Run training with default config"
	@echo "  make train-debug  Quick debug training run"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Build Sphinx documentation"
	@echo "  make docs-serve   Build and serve docs locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker       Build Docker image"
	@echo "  make docker-dev   Build development Docker image"
	@echo "  make docker-up    Start services with docker-compose"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Remove build artifacts"
	@echo "  make export       Export best model to ONNX"

# ===================================
# Setup
# ===================================

install:
	uv sync --no-dev

install-dev:
	uv sync
	uv run pre-commit install

# ===================================
# Development
# ===================================

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src --ignore-missing-imports

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest tests -v

test-cov:
	uv run pytest tests -v --cov=src/ml_template --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

# ===================================
# Training
# ===================================

train:
	uv run python -m ml_template.train

train-debug:
	uv run python -m ml_template.train experiment=debug

train-full:
	uv run python -m ml_template.train experiment=full_train

# ===================================
# Documentation
# ===================================

docs:
	uv run sphinx-build -b html docs/source docs/_build/html

docs-serve: docs
	@echo "Serving docs at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

# ===================================
# Docker
# ===================================

docker:
	docker build -f docker/Dockerfile -t ml-template .

docker-dev:
	docker build -f docker/Dockerfile --target dev -t ml-template:dev .

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down

# ===================================
# Export
# ===================================

export:
	@if [ -f "checkpoints/last.ckpt" ]; then \
		uv run python -m ml_template.export checkpoints/last.ckpt -o model.onnx --benchmark; \
	else \
		echo "No checkpoint found. Run 'make train' first."; \
	fi

# ===================================
# Utilities
# ===================================

clean:
	rm -rf dist build *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf docs/_build
	rm -rf outputs logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

