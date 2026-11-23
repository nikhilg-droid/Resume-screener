"""HTTP API. Every route is under /api so Vercel can send /api/* to this function
and everything else to the React build."""
from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import deps
from .db import Candidate, Entity, FitReport, Job
from .llm import LLMError, build_fit_messages, fit_input_hash, parse_fit_report
from .matching import match_skills, normalize_skill, parse_years
from .ner import WordTagger, extract_entities
from .pdf import PdfError, pdf_to_text
from .text import LABELS

app = FastAPI(title="Resume Screener API")

MAX_TEXT = 30_000


# ---------- request bodies ----------

def _not_blank(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be blank")
    return v


class JobIn(BaseModel):
    title: str = Field(max_length=200)
    description: str = Field(max_length=MAX_TEXT)
    required_skills: list[str] | None = None

    _strip = field_validator("title", "description")(_not_blank)


class JobPatch(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    required_skills: list[str] | None = None


class CandidateIn(BaseModel):
    resume_text: str = Field(max_length=MAX_TEXT)
    label: str | None = Field(default=None, max_length=100)

    _strip = field_validator("resume_text")(_not_blank)


# ---------- helpers ----------

def dedupe_skills(skills: list[str]) -> list[str]:
    out, seen = [], set()
    for s in skills:
        s = s.strip()
        key = normalize_skill(s)
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def entity_groups(candidate: Candidate) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {label: [] for label in LABELS}
    for e in candidate.entities:
        values = groups.setdefault(e.label, [])
        if all(normalize_skill(v) != normalize_skill(e.text) for v in values):
            values.append(e.text)
    return groups


def years_experience(groups: dict[str, list[str]]) -> float | None:
    years = [y for y in (parse_years(p) for p in groups.get("YEARS_EXP", [])) if y is not None]
    return max(years) if years else None


def job_out(job: Job, candidate_count: int) -> dict:
    return {
        "id": job.id,
        "title": job.title,
        "description": job.description,
        "required_skills": job.required_skills,
        "created_at": job.created_at.isoformat(),
        "candidate_count": candidate_count,
    }


def fit_report_out(r: FitReport, cached: bool | None = None) -> dict:
    out = {"summary": r.summary, "questions": r.questions, "model": r.model,
           "created_at": r.created_at.isoformat()}
    if cached is not None:
        out["cached"] = cached
    return out


def candidate_summary(c: Candidate) -> dict:
    groups = entity_groups(c)
    m = match_skills(c.job.required_skills, groups["SKILL"])
    return {
        "id": c.id,
        "job_id": c.job_id,
        "label": c.label,
        "source": c.source,
        "created_at": c.created_at.isoformat(),
        "entity_groups": groups,
        "years_experience": years_experience(groups),
        "match": {"score": m.score, "matched": m.matched, "missing": m.missing},
        "has_fit_report": bool(c.fit_reports),
    }


def candidate_detail(c: Candidate) -> dict:
    return {
        **candidate_summary(c),
        "resume_text": c.resume_text,
        "entities": [{"label": e.label, "text": e.text, "start": e.start, "end": e.end} for e in c.entities],
        "fit_report": fit_report_out(c.fit_reports[-1]) if c.fit_reports else None,
    }


def get_job_or_404(db: Session, job_id: int) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def get_candidate_or_404(db: Session, candidate_id: int) -> Candidate:
    c = db.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(404, "candidate not found")
    return c


def save_candidate(db: Session, job: Job, text: str, label: str | None, source: str,
                   tagger: WordTagger) -> Candidate:
    spans = extract_entities(text, tagger)
    count = db.scalar(select(func.count()).select_from(Candidate).where(Candidate.job_id == job.id))
    c = Candidate(job=job, resume_text=text, source=source,
                  label=(label or "").strip() or f"Candidate {count + 1}")
    c.entities = [Entity(label=s.label, text=s.text_in(text)[:300], start=s.start, end=s.end) for s in spans]
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


# ---------- routes ----------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs", status_code=201)
def create_job(body: JobIn, db: Session = Depends(deps.get_db),
               tagger: WordTagger = Depends(deps.get_tagger)):
    if body.required_skills is not None:
        skills = dedupe_skills(body.required_skills)
    else:
        spans = extract_entities(body.description, tagger)
        skills = dedupe_skills([s.text_in(body.description) for s in spans if s.label == "SKILL"])
    job = Job(title=body.title, description=body.description, required_skills=skills)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job_out(job, 0)


@app.get("/api/jobs")
def list_jobs(db: Session = Depends(deps.get_db)):
    rows = db.execute(
        select(Job, func.count(Candidate.id))
        .outerjoin(Candidate, Candidate.job_id == Job.id)
        .group_by(Job.id)
        .order_by(Job.id.desc())
    ).all()
    return [job_out(job, n) for job, n in rows]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(deps.get_db)):
    job = get_job_or_404(db, job_id)
    return job_out(job, len(job.candidates))


@app.patch("/api/jobs/{job_id}")
def update_job(job_id: int, body: JobPatch, db: Session = Depends(deps.get_db)):
    job = get_job_or_404(db, job_id)
    if body.title is not None and body.title.strip():
        job.title = body.title.strip()
    if body.required_skills is not None:
        job.required_skills = dedupe_skills(body.required_skills)
    db.commit()
    return job_out(job, len(job.candidates))


@app.get("/api/jobs/{job_id}/candidates")
def shortlist(job_id: int, db: Session = Depends(deps.get_db)):
    job = get_job_or_404(db, job_id)
    rows = [candidate_summary(c) for c in job.candidates]
    return sorted(rows, key=lambda r: (-r["match"]["score"], r["id"]))


@app.post("/api/jobs/{job_id}/candidates", status_code=201)
def add_candidate(job_id: int, body: CandidateIn, db: Session = Depends(deps.get_db),
                  tagger: WordTagger = Depends(deps.get_tagger)):
    job = get_job_or_404(db, job_id)
    return candidate_detail(save_candidate(db, job, body.resume_text, body.label, "paste", tagger))


@app.post("/api/jobs/{job_id}/candidates/pdf", status_code=201)
async def upload_candidate_pdf(job_id: int, file: UploadFile = File(...),
                               db: Session = Depends(deps.get_db),
                               tagger: WordTagger = Depends(deps.get_tagger)):
    job = get_job_or_404(db, job_id)
    try:
        text = pdf_to_text(await file.read())
    except PdfError as e:
        raise HTTPException(400, str(e)) from e
    label = (file.filename or "").rsplit(".", 1)[0][:100] or None
    return candidate_detail(save_candidate(db, job, text[:MAX_TEXT], label, "pdf", tagger))


@app.get("/api/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: Session = Depends(deps.get_db)):
    return candidate_detail(get_candidate_or_404(db, candidate_id))


@app.delete("/api/candidates/{candidate_id}", status_code=204)
def delete_candidate(candidate_id: int, db: Session = Depends(deps.get_db)):
    db.delete(get_candidate_or_404(db, candidate_id))
    db.commit()
    return Response(status_code=204)


@app.post("/api/candidates/{candidate_id}/fit-report")
def create_fit_report(candidate_id: int, db: Session = Depends(deps.get_db), llm=Depends(deps.get_llm)):
    c = get_candidate_or_404(db, candidate_id)
    groups = entity_groups(c)
    match = match_skills(c.job.required_skills, groups["SKILL"])
    messages = build_fit_messages(c.job.title, c.job.description, c.resume_text, groups, match)
    model = getattr(llm, "model", "unknown")
    key = fit_input_hash(model, messages)

    cached = db.scalar(select(FitReport).where(FitReport.candidate_id == c.id, FitReport.input_hash == key))
    if cached:
        return fit_report_out(cached, cached=True)

    try:
        report = parse_fit_report(llm.chat_json(messages))
    except LLMError as e:
        raise HTTPException(503, str(e)) from e
    row = FitReport(candidate=c, input_hash=key, model=model, **report)
    db.add(row)
    db.commit()
    db.refresh(row)
    return fit_report_out(row, cached=False)
