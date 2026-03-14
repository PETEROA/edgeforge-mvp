# EdgeForge MVP

Edge MLOps platform — upload a model, define hardware constraints, get back an optimized model with benchmarks.

## Tech Stack

- **Backend**: FastAPI + Uvicorn, Python 3.12
- **Database**: PostgreSQL (via `DATABASE_URL` env var)
- **ORM**: SQLAlchemy
- **Auth**: JWT (python-jose + passlib/bcrypt)
- **Task Queue**: Celery + Redis (optional, required for optimization jobs)
- **ML**: PyTorch, ONNX, ONNX Runtime, HuggingFace Transformers

## Project Structure

```
app/
├── main.py              # FastAPI entry point
├── config.py            # Settings (reads from env vars)
├── api/
│   ├── routes_auth.py   # /v1/auth/* - Register, login, token
│   ├── routes_models.py # /v1/models/* - Upload/list ML models
│   ├── routes_jobs.py   # /v1/jobs/* - Optimization jobs
│   └── routes_devices.py# /v1/devices/* - Device profiles
├── core/
│   ├── auth.py          # JWT auth helpers
│   └── database.py      # SQLAlchemy models + session
├── models/
│   └── schemas.py       # Pydantic request/response schemas
├── services/
│   ├── optimizer.py     # Core optimization pipeline
│   ├── quantizer.py     # INT8 quantization
│   ├── pruner.py        # Structured pruning
│   └── benchmarker.py   # Benchmarking + reporting
└── tasks/
    └── worker.py        # Celery task definitions
```

## Running the App

The app runs via the "Start application" workflow:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs available at `/docs` (Swagger UI) and `/redoc`.

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string (auto-set by Replit)
- `REDIS_URL` - Redis URL for Celery (default: `redis://localhost:6379/0`)
- `JWT_SECRET_KEY` - JWT signing secret
- `HF_TOKEN` - HuggingFace API token (optional)

## Database

Initialized with `python init_db.py`. Uses PostgreSQL in Replit (via `DATABASE_URL` env var). SQLite fallback if env var not set.

## Notes

- CORS is set to allow all origins for development
- Celery/Redis needed only for actual optimization job processing
- ML packages (torch, transformers, etc.) are heavy — install only if needed
