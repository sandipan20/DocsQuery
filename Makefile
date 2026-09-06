# DocsQuery development commands

# Run all tests.
test:
	pytest -v

# Check Python code for linting errors.
lint:
	ruff check .

# Format Python code.
format:
	ruff format .
	ruff format --check .

# Run linting and tests together.
check:
	ruff check .
# 	pytest -v
	ruff format --check .
	pytest tests/unit -v

# Start the FastAPI development server.
run:
	python -m uvicorn app.main:app --reload

