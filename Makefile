# Funding Rate Arbitrage System Makefile

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check security clean build docker-build docker-run start-dev start-prod setup-pre-commit

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt

install-prod: ## Install production dependencies with performance optimizations
	pip install -r requirements-prod.txt

# Testing targets
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	pytest-watch

# Code quality targets
lint: ## Run all linters
	flake8 src tests
	pylint src
	bandit -r src

format: ## Format code with black and isort
	black src tests
	isort src tests

format-check: ## Check code formatting
	black --check src tests
	isort --check-only src tests

type-check: ## Run type checking with mypy
	mypy src

security: ## Run security checks
	bandit -r src
	safety check

quality: format-check lint type-check security ## Run all code quality checks

# Development targets
clean: ## Clean up build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .tox/ .coverage .coverage.*

build: clean ## Build the package
	python -m build

setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# Application targets
start-dev: ## Start development server
	python main.py

start-prod: ## Start production server
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker targets
docker-build: ## Build Docker image
	docker build -t funding-rate-arb:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 funding-rate-arb:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# Database targets
db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migration
	alembic downgrade -1

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(MESSAGE)"

# Monitoring targets
metrics: ## View Prometheus metrics
	curl http://localhost:9090/metrics

health: ## Check application health
	curl http://localhost:8000/health

# Environment setup
setup-dev: install-dev setup-pre-commit ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "1. Copy config/main.yaml.template to config/main.yaml"
	@echo "2. Configure your exchange API credentials"
	@echo "3. Start Redis and PostgreSQL services"
	@echo "4. Run 'make start-dev' to start the application"

setup-prod: install-prod ## Setup production environment
	@echo "Production environment setup complete!"

# Documentation targets
docs: ## Generate documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

# Utility targets
version: ## Show version information
	@python -c "import src; print(f'Version: {src.__version__}')"

requirements-update: ## Update requirements files
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in
	pip-compile --upgrade requirements-prod.in

# CI/CD targets
ci-test: install-dev test quality ## Run CI test pipeline
	@echo "CI test pipeline completed successfully!"

ci-build: clean build ## Run CI build pipeline
	@echo "CI build pipeline completed successfully!"

# Kubernetes targets
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f deployment/k8s/

k8s-delete: ## Delete Kubernetes deployment
	kubectl delete -f deployment/k8s/

k8s-logs: ## View Kubernetes logs
	kubectl logs -f deployment/funding-arb-app

# Load testing
load-test: ## Run load tests
	locust -f tests/load/locustfile.py --host=http://localhost:8000