"""All settings come from environment variables (Vercel project settings in production).
This is the only module that reads GROQ_API_KEY."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    groq_api_key: str | None
    groq_model: str
    model_dir: str | None
    model_repo: str | None


def sqlalchemy_url(url: str) -> str:
    """Neon gives postgres:// or postgresql:// URLs; SQLAlchemy needs the psycopg 3 driver named."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def get_settings() -> Settings:
    return Settings(
        database_url=sqlalchemy_url(os.environ.get("DATABASE_URL", "sqlite:///./local.db")),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        model_dir=os.environ.get("MODEL_DIR", "models/resume-ner"),
        model_repo=os.environ.get("MODEL_REPO") or None,
    )
