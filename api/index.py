"""Vercel entry point: Vercel runs the ASGI `app` found in this file for /api/* requests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from resume_screener.main import app  # noqa: E402,F401
