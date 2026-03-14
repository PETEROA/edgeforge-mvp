"""
Basic API tests for EdgeForge.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import Base, engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# ---- Health ----

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "EdgeForge"
    assert data["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---- Auth ----

def test_signup():
    response = client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
        "name": "Test User",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@edgeforge.dev"
    assert data["tier"] == "free"
    assert data["quota_limit"] == 3


def test_signup_duplicate():
    client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    response = client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "different",
    })
    assert response.status_code == 400


def test_login():
    # Signup first
    client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    # Login
    response = client.post("/v1/auth/login", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password():
    client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    response = client.post("/v1/auth/login", json={
        "email": "test@edgeforge.dev",
        "password": "wrong",
    })
    assert response.status_code == 401


# ---- Helpers ----

def get_auth_header():
    """Signup, login, and return auth header."""
    client.post("/v1/auth/signup", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    response = client.post("/v1/auth/login", json={
        "email": "test@edgeforge.dev",
        "password": "testpass123",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---- Device Profiles ----

def test_list_devices():
    response = client.get("/v1/devices/")
    assert response.status_code == 200
    data = response.json()
    assert len(data["profiles"]) >= 10


def test_get_device_profile():
    response = client.get("/v1/devices/android-mid")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Android Mid-Range"


def test_get_device_profile_not_found():
    response = client.get("/v1/devices/nonexistent")
    assert response.status_code == 404


# ---- Models ----

def test_list_models_unauthorized():
    response = client.get("/v1/models/")
    assert response.status_code == 403


def test_list_models_empty():
    headers = get_auth_header()
    response = client.get("/v1/models/", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


def test_import_huggingface_model():
    headers = get_auth_header()
    response = client.post("/v1/models/from-huggingface", json={
        "model_id": "microsoft/resnet-50",
    }, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "resnet-50"
    assert data["format"] == "huggingface"


# ---- Jobs ----

def test_create_job():
    headers = get_auth_header()

    # Create a model first
    model_resp = client.post("/v1/models/from-huggingface", json={
        "model_id": "microsoft/resnet-50",
    }, headers=headers)
    model_id = model_resp.json()["id"]

    # Create optimization job
    response = client.post("/v1/jobs/optimize", json={
        "model_id": model_id,
        "device_profile": "android-mid",
        "enable_quantization": True,
        "enable_pruning": True,
        "enable_distillation": False,
    }, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "queued"
    assert data["device_profile"] == "android-mid"
