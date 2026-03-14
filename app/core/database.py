from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import enum
import uuid

from app.config import get_settings

settings = get_settings()

# Use connect_args for SQLite only
connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- Enums ----

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelFormat(str, enum.Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"


# ---- Database Models ----

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=True)
    tier = Column(String, default="free")  # free, pro, enterprise
    api_key_hash = Column(String, nullable=True)
    quota_used = Column(Integer, default=0)
    quota_limit = Column(Integer, default=3)  # 3 for free tier
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    models = relationship("MLModel", back_populates="owner")
    jobs = relationship("OptimizationJob", back_populates="owner")


class MLModel(Base):
    __tablename__ = "models"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    format = Column(String, nullable=False)  # pytorch, onnx, huggingface
    file_path = Column(String, nullable=True)
    hf_model_id = Column(String, nullable=True)  # e.g., "microsoft/resnet-50"
    size_bytes = Column(Integer, nullable=True)
    param_count = Column(Integer, nullable=True)
    file_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="models")
    jobs = relationship("OptimizationJob", back_populates="model")


class OptimizationJob(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    device_profile = Column(String, nullable=False)
    status = Column(String, default=JobStatus.PENDING)
    pipeline_config = Column(Text, nullable=True)  # JSON string

    # Optimization settings
    enable_quantization = Column(Boolean, default=True)
    enable_pruning = Column(Boolean, default=True)
    enable_distillation = Column(Boolean, default=False)
    target_size_mb = Column(Float, nullable=True)
    max_accuracy_drop = Column(Float, default=0.02)  # 2% default

    # Results
    optimized_model_path = Column(String, nullable=True)
    benchmark_report_path = Column(String, nullable=True)

    # Metrics (populated after completion)
    original_size_mb = Column(Float, nullable=True)
    optimized_size_mb = Column(Float, nullable=True)
    compression_ratio = Column(Float, nullable=True)
    original_latency_ms = Column(Float, nullable=True)
    optimized_latency_ms = Column(Float, nullable=True)
    speedup_factor = Column(Float, nullable=True)
    accuracy_original = Column(Float, nullable=True)
    accuracy_optimized = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error info
    error_message = Column(Text, nullable=True)

    # Celery task ID for tracking
    celery_task_id = Column(String, nullable=True)

    owner = relationship("User", back_populates="jobs")
    model = relationship("MLModel", back_populates="jobs")
