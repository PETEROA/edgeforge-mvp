from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db, User, MLModel, OptimizationJob
from app.core.auth import get_current_user
from app.models.schemas import JobCreate, JobResponse, JobListResponse

router = APIRouter(prefix="/v1/jobs", tags=["Optimization Jobs"])


@router.post("/optimize", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_optimization_job(
    job_data: JobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit a new model optimization job."""
    # Check quota
    if current_user.quota_used >= current_user.quota_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly quota exceeded ({current_user.quota_limit} optimizations). Upgrade to Pro for unlimited.",
        )

    # Verify model exists and belongs to user
    model = db.query(MLModel).filter(
        MLModel.id == job_data.model_id,
        MLModel.user_id == current_user.id,
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create job
    job = OptimizationJob(
        user_id=current_user.id,
        model_id=job_data.model_id,
        device_profile=job_data.device_profile,
        enable_quantization=job_data.enable_quantization,
        enable_pruning=job_data.enable_pruning,
        enable_distillation=job_data.enable_distillation,
        target_size_mb=job_data.target_size_mb,
        max_accuracy_drop=job_data.max_accuracy_drop,
        status="queued",
    )
    db.add(job)

    # Increment quota
    current_user.quota_used += 1
    db.commit()
    db.refresh(job)

    # Dispatch to Celery worker
    from app.tasks.worker import run_optimization
    task = run_optimization.delay(job.id)

    # Store celery task ID
    job.celery_task_id = task.id
    db.commit()

    return job


@router.get("/", response_model=JobListResponse)
def list_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
):
    """List all optimization jobs for the current user."""
    query = db.query(OptimizationJob).filter(OptimizationJob.user_id == current_user.id)
    total = query.count()
    jobs = query.order_by(OptimizationJob.created_at.desc()).offset(offset).limit(limit).all()
    return JobListResponse(jobs=jobs, total=total)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the status and details of an optimization job."""
    job = db.query(OptimizationJob).filter(
        OptimizationJob.id == job_id,
        OptimizationJob.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/download")
def download_optimized_model(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the optimized model artifact."""
    job = db.query(OptimizationJob).filter(
        OptimizationJob.id == job_id,
        OptimizationJob.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job.status})")
    if not job.optimized_model_path:
        raise HTTPException(status_code=404, detail="Optimized model not found")

    return FileResponse(
        path=job.optimized_model_path,
        filename=f"optimized_{job.id}.onnx",
        media_type="application/octet-stream",
    )
