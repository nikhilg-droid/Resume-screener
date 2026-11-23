"""Tables: jobs, candidates, entities, fit_reports. Postgres (Neon) in production, SQLite in tests."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint, DateTime
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    required_skills: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    candidates: Mapped[list["Candidate"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Candidate.id"
    )


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(10), default="paste")  # paste | pdf
    resume_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    job: Mapped[Job] = relationship(back_populates="candidates")
    entities: Mapped[list["Entity"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", order_by="Entity.start"
    )
    fit_reports: Mapped[list["FitReport"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan", order_by="FitReport.id"
    )


class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(String(300))
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    candidate: Mapped[Candidate] = relationship(back_populates="entities")


class FitReport(Base):
    """Cached LLM answer. input_hash covers the model and the full prompt."""
    __tablename__ = "fit_reports"
    __table_args__ = (UniqueConstraint("candidate_id", "input_hash"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"), index=True)
    input_hash: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(100))
    summary: Mapped[str] = mapped_column(Text)
    questions: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    candidate: Mapped[Candidate] = relationship(back_populates="fit_reports")


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
