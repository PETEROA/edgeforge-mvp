from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import routes_auth, routes_models, routes_jobs, routes_devices

settings = get_settings()

app = FastAPI(
    title="EdgeForge API",
    description="Edge MLOps platform — optimize ML models for any hardware target.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local frontend dev
        "https://edgeforge.dev",  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(routes_auth.router)
app.include_router(routes_models.router)
app.include_router(routes_jobs.router)
app.include_router(routes_devices.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}
