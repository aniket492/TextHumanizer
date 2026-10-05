"""Application configuration, loaded from environment variables / backend/.env."""
from __future__ import annotations

import logging
import secrets
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# The ML package lives at the project root (``ml/``); make it importable from the backend.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "AI Humanizer"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://humanizer:humanizer@localhost:5433/humanizer"

    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    api_base_url: str = "http://localhost:8000"

    # --- Model configuration ---
    # MODEL_BACKEND: "auto" (try transformers, fall back to rule baseline),
    # "transformers" (model required), or "baseline" (rule-based only, no download).
    model_backend: str = "auto"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Optional local directory with a fine-tuned model or LoRA adapter (overrides MODEL_NAME).
    model_path: str = ""
    model_device: str = "auto"
    model_max_new_tokens: int = 384
    model_timeout_seconds: float = 90.0
    model_num_candidates: int = 1
    model_preload: bool = False
    # Sentence-Transformers model for semantic similarity; empty -> TF-IDF cosine fallback.
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Input limits ---
    max_input_words: int = 1500
    max_input_chars: int = 12000
    min_input_words: int = 3

    @field_validator("model_backend")
    @classmethod
    def _check_backend(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"auto", "transformers", "baseline"}:
            raise ValueError("MODEL_BACKEND must be one of: auto, transformers, baseline")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if not settings.jwt_secret:
        if settings.environment.lower() == "production":
            raise RuntimeError("JWT_SECRET must be set in production.")
        # Development convenience: ephemeral secret. Tokens are invalidated on restart.
        settings.jwt_secret = secrets.token_urlsafe(48)
        logging.getLogger("humanizer").warning(
            "JWT_SECRET not set; using an ephemeral development secret."
        )
    return settings


settings = get_settings()
