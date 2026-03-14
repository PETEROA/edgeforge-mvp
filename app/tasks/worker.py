"""
Celery worker for processing optimization jobs asynchronously.

Start with: celery -A app.tasks.worker worker --loglevel=info
"""

from celery import Celery
from datetime import datetime, timezone
import logging

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "edgeforge",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Re-queue if worker crashes
    worker_prefetch_multiplier=1,  # One job at a time per worker
)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_optimization(self, job_id: str):
    """
    Execute an optimization job.

    This runs in a separate Celery worker process with access to GPU if available.
    """
    from app.core.database import SessionLocal, OptimizationJob, MLModel
    from app.services.optimizer import OptimizationPipeline

    db = SessionLocal()

    try:
        # Load job from DB
        job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"status": "error", "message": "Job not found"}

        # Update status
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # Load model info
        model = db.query(MLModel).filter(MLModel.id == job.model_id).first()
        if not model:
            raise ValueError(f"Model {job.model_id} not found")

        # Run optimization pipeline
        pipeline = OptimizationPipeline(
            model_path=model.file_path,
            hf_model_id=model.hf_model_id,
            device_profile=job.device_profile,
            enable_quantization=job.enable_quantization,
            enable_pruning=job.enable_pruning,
            enable_distillation=job.enable_distillation,
            target_size_mb=job.target_size_mb,
            max_accuracy_drop=job.max_accuracy_drop,
        )

        result = pipeline.run()

        # Update job with results
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.optimized_model_path = result.optimized_model_path
        job.benchmark_report_path = result.benchmark_report_path
        job.original_size_mb = result.original_size_mb
        job.optimized_size_mb = result.optimized_size_mb
        job.compression_ratio = result.compression_ratio
        job.original_latency_ms = result.original_latency_ms
        job.optimized_latency_ms = result.optimized_latency_ms
        job.speedup_factor = result.speedup_factor
        job.accuracy_original = result.accuracy_original
        job.accuracy_optimized = result.accuracy_optimized
        db.commit()

        logger.info(f"Job {job_id} completed: {result.compression_ratio:.1f}x compression")
        return {
            "status": "completed",
            "job_id": job_id,
            "compression_ratio": result.compression_ratio,
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")

        # Update job status
        job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

        # Retry if retries remaining
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {"status": "failed", "job_id": job_id, "error": str(e)}

    finally:
        db.close()
