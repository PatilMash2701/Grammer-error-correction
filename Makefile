.PHONY: install train serve test lint format docker-build docker-up clean

# ============================================
# Development
# ============================================

install:
	pip install -r requirements.txt
	cd frontend && npm install

install-dev:
	pip install -r requirements.txt
	pip install black isort mypy ruff pytest pytest-asyncio httpx

# ============================================
# Training
# ============================================

train:
	python -m src.training.train \
		--model_name facebook/bart-base \
		--dataset bea2019 \
		--epochs 10 \
		--batch_size 16 \
		--lr 5e-5

evaluate:
	python -m src.training.evaluate \
		--model_path ./checkpoints/best_model \
		--dataset conll2014

# ============================================
# API Server
# ============================================

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ============================================
# Frontend
# ============================================

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ============================================
# Testing
# ============================================

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ============================================
# Code Quality
# ============================================

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# ============================================
# Docker
# ============================================

docker-build:
	docker-compose -f src/deployment/docker-compose.yml build

docker-up:
	docker-compose -f src/deployment/docker-compose.yml up

docker-down:
	docker-compose -f src/deployment/docker-compose.yml down

# ============================================
# Cleanup
# ============================================

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .mypy_cache
