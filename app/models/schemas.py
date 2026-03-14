from pydantic import BaseModel, EmailStr
from datetime import datetime


# ---- Auth ----

class UserCreate(BaseModel):
    email: str
    password: str
    name: str | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    tier: str
    quota_used: int
    quota_limit: int
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: str
    password: str


# ---- Models ----

class ModelUploadResponse(BaseModel):
    id: str
    name: str
    format: str
    size_bytes: int | None
    param_count: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class ModelFromHuggingFace(BaseModel):
    model_id: str  # e.g., "microsoft/resnet-50"
    name: str | None = None


# ---- Jobs ----

class JobCreate(BaseModel):
    model_id: str
    device_profile: str
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_distillation: bool = False
    target_size_mb: float | None = None
    max_accuracy_drop: float = 0.02


class JobResponse(BaseModel):
    id: str
    model_id: str
    device_profile: str
    status: str
    enable_quantization: bool
    enable_pruning: bool
    enable_distillation: bool

    # Results (null until completed)
    original_size_mb: float | None
    optimized_size_mb: float | None
    compression_ratio: float | None
    original_latency_ms: float | None
    optimized_latency_ms: float | None
    speedup_factor: float | None
    accuracy_original: float | None
    accuracy_optimized: float | None

    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


# ---- Device Profiles ----

class DeviceProfile(BaseModel):
    id: str
    name: str
    cpu: str
    ram_mb: int
    gpu: str | None
    target_runtime: str
    max_model_size_mb: float
    description: str


class DeviceProfileList(BaseModel):
    profiles: list[DeviceProfile]
