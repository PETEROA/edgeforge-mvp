from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    app_name: str = "EdgeForge"
    app_env: str = "development"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./edgeforge.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret_key: str = "change-this-to-a-random-secret-key-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # Storage
    storage_backend: str = "local"
    storage_local_path: str = "./storage"

    # HuggingFace
    hf_token: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
