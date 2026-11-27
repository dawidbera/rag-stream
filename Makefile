.PHONY: help install lint format typecheck test run docker-build docker-up docker-down clean

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
UVICORN ?= .venv/bin/uvicorn
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

help:
	@echo "Available targets:"
	@echo "  install      - Install production and development dependencies"
	@echo "  lint         - Run ruff linter checks"
	@echo "  format       - Run ruff code formatter"
	@echo "  typecheck    - Run mypy static type checker"
	@echo "  test         - Run unit and integration tests with pytest"
	@echo "  run          - Run FastAPI application locally with hot-reload"
	@echo "  docker-build - Build application Docker image"
	@echo "  docker-up    - Start full stack (app + Qdrant) via docker-compose"
	@echo "  docker-down  - Stop docker-compose services"
	@echo "  clean        - Remove build artifacts and caches"

install:
	$(PIP) install -e .[dev]

lint:
	$(RUFF) check src tests

format:
	$(RUFF) format src tests

typecheck:
	$(MYPY) src

test:
	$(PYTEST) tests/

run:
	$(UVICORN) rag_stream.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t rag-stream:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
