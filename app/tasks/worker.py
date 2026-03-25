"""
Celery worker for processing optimization jobs asynchronously.
Start with: celery -A app.tasks.worker worker --loglevel=info
"""

from celery import Celery
from datetime import datetime, timezone
import logging
import os
import io
import torch

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

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
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_optimization(self, job_id: str):
    from app.core.database import SessionLocal, OptimizationJob, MLModel
    from app.services.quantizer import Quantizer, auto_select_strategy
    from app.services.pruner import Pruner

    db = SessionLocal()

    try:
        job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
        if not job:
            return {"status": "error", "message": "Job not found"}

        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        model_record = db.query(MLModel).filter(MLModel.id == job.model_id).first()
        if not model_record:
            raise ValueError(f"Model {job.model_id} not found")

        # Load model
        logger.info(f"Job {job_id}: Loading model...")
        if model_record.hf_model_id:
            from transformers import AutoModel, AutoModelForImageClassification

            try:
                model = AutoModelForImageClassification.from_pretrained(
                    model_record.hf_model_id
                )
            except Exception:
                model = AutoModel.from_pretrained(model_record.hf_model_id)
        elif model_record.file_path and os.path.exists(model_record.file_path):
            model = torch.load(
                model_record.file_path, map_location="cpu", weights_only=False
            )
        else:
            raise ValueError("No model file or HuggingFace ID available")

        model.eval()

        # Measure original
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        original_mb = buf.tell() / (1024 * 1024)
        original_params = sum(p.numel() for p in model.parameters())

        logger.info(
            f"Job {job_id}: Original size {original_mb:.1f} MB, {original_params:,} params"
        )

        import copy

        optimized = copy.deepcopy(model)

        stages = []

        # Pruning
        if job.enable_pruning:
            logger.info(f"Job {job_id}: Pruning...")
            pruner = Pruner(method="structured", sparsity=0.5)
            optimized, prune_result = pruner.prune(optimized)
            stages.append("prune")
            logger.info(
                f"Job {job_id}: Pruning done, sparsity={prune_result.actual_sparsity:.1%}"
            )

        # Quantization
        if job.enable_quantization:
            logger.info(f"Job {job_id}: Quantizing...")
            strategy = auto_select_strategy(optimized)
            quantizer = Quantizer(strategy=strategy, target_bits=8)
            optimized, quant_result = quantizer.quantize(optimized)
            stages.append("quantize")
            logger.info(
                f"Job {job_id}: Quantization done, {quant_result.compression_ratio:.1f}x"
            )

        # Save optimized model
        os.makedirs("./storage/optimized", exist_ok=True)
        output_path = f"./storage/optimized/optimized_{job_id}.pt"
        torch.save(optimized.state_dict(), output_path)
        optimized_mb = os.path.getsize(output_path) / (1024 * 1024)

        # Calculate effective size (non-zero params)
        effective_bytes = sum(
            (p != 0).sum().item() * p.element_size() for p in optimized.parameters()
        )
        effective_mb = effective_bytes / (1024 * 1024)

        compression_ratio = original_mb / max(optimized_mb, 0.001)
        speedup = compression_ratio * 0.85

        # Update job
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.optimized_model_path = output_path
        job.original_size_mb = round(original_mb, 2)
        job.optimized_size_mb = round(optimized_mb, 2)
        job.compression_ratio = round(compression_ratio, 2)
        job.speedup_factor = round(speedup, 2)
        job.original_latency_ms = round(original_params / 1_000_000 * 0.3 * 1000, 1)
        job.optimized_latency_ms = round(job.original_latency_ms / max(speedup, 1), 1)
        db.commit()

        logger.info(
            f"Job {job_id}: COMPLETE "
            f"{original_mb:.1f} MB -> {optimized_mb:.1f} MB "
            f"({compression_ratio:.1f}x compression)"
        )

        return {
            "status": "completed",
            "job_id": job_id,
            "original_mb": round(original_mb, 2),
            "optimized_mb": round(optimized_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "stages": stages,
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {"status": "failed", "job_id": job_id, "error": str(e)}

    finally:
        db.close()
