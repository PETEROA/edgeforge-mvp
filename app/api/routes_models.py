import os
import hashlib
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.database import get_db, User, MLModel
from app.core.auth import get_current_user
from app.models.schemas import ModelUploadResponse, ModelFromHuggingFace

settings = get_settings()
router = APIRouter(prefix="/v1/models", tags=["Models"])

# Ensure storage directory exists
UPLOAD_DIR = os.path.join(settings.storage_local_path, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=ModelUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a model file (PyTorch .pt/.pth, ONNX .onnx)."""
    # Validate file extension
    allowed_extensions = {".pt", ".pth", ".onnx", ".bin", ".safetensors"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Determine format
    format_map = {
        ".pt": "pytorch", ".pth": "pytorch",
        ".onnx": "onnx",
        ".bin": "pytorch", ".safetensors": "huggingface",
    }
    model_format = format_map.get(ext, "pytorch")

    # Read and save file
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate
    existing = db.query(MLModel).filter(
        MLModel.user_id == current_user.id,
        MLModel.file_hash == file_hash,
    ).first()
    if existing:
        return existing

    # Save to disk
    file_path = os.path.join(UPLOAD_DIR, f"{file_hash}{ext}")
    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    model = MLModel(
        user_id=current_user.id,
        name=file.filename,
        format=model_format,
        file_path=file_path,
        size_bytes=len(content),
        file_hash=file_hash,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.post("/from-huggingface", response_model=ModelUploadResponse, status_code=status.HTTP_201_CREATED)
def import_from_huggingface(
    data: ModelFromHuggingFace,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Import a model from HuggingFace by model ID (e.g., 'microsoft/resnet-50')."""
    model = MLModel(
        user_id=current_user.id,
        name=data.name or data.model_id.split("/")[-1],
        format="huggingface",
        hf_model_id=data.model_id,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.get("/", response_model=list[ModelUploadResponse])
def list_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all models uploaded by the current user."""
    models = db.query(MLModel).filter(MLModel.user_id == current_user.id).all()
    return models


@router.get("/{model_id}", response_model=ModelUploadResponse)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get details of a specific model."""
    model = db.query(MLModel).filter(
        MLModel.id == model_id,
        MLModel.user_id == current_user.id,
    ).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
