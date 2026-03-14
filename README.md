# EdgeForge MVP

Edge MLOps platform — upload a model, define hardware constraints, get back an optimized model with benchmarks.

## Quick Start

```bash
# 1. Clone and setup
cd edgeforge-mvp
python -m venv venv
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Start Redis (needed for task queue)
# Option A: If you have Docker
docker run -d -p 6379:6379 redis:7-alpine

# Option B: If not, install Redis
# brew install redis && redis-server

# 5. Run database migrations
python scripts/init_db.py

# 6. Start the API server
uvicorn app.main:app --reload --port 8000

# 7. Start the Celery worker (separate terminal)
celery -A app.tasks.worker worker --loglevel=info

# 8. Visit docs
open http://localhost:8000/docs
```

## Project Structure

```
edgeforge-mvp/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py             # Settings & environment vars
│   ├── api/
│   │   ├── routes_models.py  # Model upload endpoints
│   │   ├── routes_jobs.py    # Optimization job endpoints
│   │   ├── routes_devices.py # Device profile endpoints
│   │   └── routes_auth.py    # Auth endpoints
│   ├── core/
│   │   ├── auth.py           # JWT authentication
│   │   └── database.py       # Database connection
│   ├── models/
│   │   └── schemas.py        # Pydantic models & DB schemas
│   ├── services/
│   │   ├── optimizer.py      # Core optimization pipeline
│   │   ├── quantizer.py      # Quantization module
│   │   ├── pruner.py         # Pruning module
│   │   └── benchmarker.py    # Benchmark & reporting
│   └── tasks/
│       └── worker.py         # Celery task definitions
├── device_profiles/          # JSON device profile configs
├── scripts/
│   └── init_db.py            # Database initialization
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 8-Week MVP Plan (1-2 hrs/day)

### Week 1-2: Core API + Model Upload
- FastAPI running with auth
- Model upload endpoint (file + HuggingFace ID)
- PostgreSQL storing model metadata
- File storage (local → S3 later)

### Week 3-4: Optimization Pipeline
- Quantization module (INT8 PTQ)
- Structured pruning module
- Celery job queue processing
- Job status tracking

### Week 5-6: Distillation + Benchmarks
- Knowledge distillation (response-based)
- Benchmark service with device profiles
- HTML/PDF report generation

### Week 7-8: Polish + Launch Prep
- Frontend dashboard (basic React)
- HuggingFace model hub integration
- Error handling, logging, tests
- Deploy to production
