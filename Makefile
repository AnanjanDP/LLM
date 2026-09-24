.PHONY: install run-backend run-frontend test build docker-build docker-run help

help:
	@echo "Available commands:"
	@echo "  make install      - Install backend & frontend dependencies"
	@echo "  make run-backend  - Start FastAPI backend dev server"
	@echo "  make run-frontend - Start Vite frontend dev server"
	@echo "  make test         - Run backend pytest test suite"
	@echo "  make build        - Build production frontend bundle"
	@echo "  make docker-build - Build Docker container"
	@echo "  make docker-run   - Run containerized application via Docker Compose"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

run-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

run-frontend:
	cd frontend && npm run dev

test:
	cd backend && python -m pytest

build:
	cd frontend && npm run build

docker-build:
	docker build -t rag-llm-platform .

docker-run:
	docker-compose up -d --build
