# ============================================================
# DocsQuery Makefile
# ============================================================

.PHONY: test lint format check run \
        docker-build docker-up docker-down docker-logs

# ============================================================
# Python development
# ============================================================

# Run the complete test suite.
test:
	pytest -v

# Run Ruff linting.
lint:
	ruff check .

# Format Python code and verify formatting.
format:
	ruff format .
	ruff format --check .

# Run the checks used during development/CI.
check:
	ruff check .
	ruff format --check .
	pytest tests/unit -v

# Start the FastAPI development server with auto-reload.
run:
	python -m uvicorn app.main:app --reload

# ============================================================
# Docker
# ============================================================

# Build the production Docker image from scratch.
docker-build:
	docker compose build --no-cache

# Start the local API + Qdrant stack in the background.
docker-up:
	docker compose up -d

# Stop the local API + Qdrant stack.
docker-down:
	docker compose down

# Follow API container logs.
docker-logs:
	docker compose logs -f api