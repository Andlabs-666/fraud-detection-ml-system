.PHONY: install download validate train evaluate serve test lint docker-build docker-run clean

install:
	pip install -e .

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	python scripts/serve.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

docker-build:
	docker build -t fraud-ml-service .

docker-run:
	docker-compose up

mlflow-ui:
	mlflow ui --backend-store-uri mlruns/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache