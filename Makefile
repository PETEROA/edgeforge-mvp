.PHONY: setup run worker test clean

# First-time setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env
	python scripts/init_db.py
	@echo ""
	@echo "✓ Setup complete. Run 'make run' to start the server."

# Start the API server
run:
	. venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Start Celery worker (run in separate terminal)
worker:
	. venv/bin/activate && celery -A app.tasks.worker worker --loglevel=info

# Run tests
test:
	. venv/bin/activate && pytest tests/ -v

# Start everything with Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Clean up
clean:
	rm -f edgeforge.db
	rm -rf storage/models/* storage/optimized/* storage/reports/*
	rm -rf __pycache__ app/__pycache__ app/**/__pycache__
	@echo "✓ Cleaned."
