"""FastAPI dependencies. Each heavy thing (DB engine, ONNX model, Groq client) is built once
per server process and reused. Tests replace these with app.dependency_overrides."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .config import get_settings
from .db import init_db
from .llm import GroqClient
from .ner import WordTagger, load_tagger


@lru_cache
def _engine() -> Engine:
    url = get_settings().database_url
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    init_db(engine)
    return engine


def get_engine() -> Engine:
    return _engine()


def get_db(engine: Engine = Depends(get_engine)):
    with Session(engine) as session:
        yield session


@lru_cache
def _tagger() -> WordTagger:
    s = get_settings()
    return load_tagger(s.model_dir, s.model_repo)


def get_tagger() -> WordTagger:
    try:
        return _tagger()
    except Exception as e:  # missing files, failed download
        raise HTTPException(503, f"NER model is not available: {e}") from e


@lru_cache
def _llm() -> GroqClient | None:
    s = get_settings()
    return GroqClient(s.groq_api_key, s.groq_model) if s.groq_api_key else None


def get_llm():
    llm = _llm()
    if llm is None:
        raise HTTPException(503, "GROQ_API_KEY is not set on the server")
    return llm
