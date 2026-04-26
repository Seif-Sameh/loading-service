.PHONY: install install-dev install-rl test lint fmt run benchmark clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-rl:
	pip install -r requirements-rl.txt

test:
	pytest

lint:
	ruff check app tests

fmt:
	ruff check --fix app tests
	black app tests

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8009 --reload

benchmark:
	python -m scripts.run_benchmarks

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
