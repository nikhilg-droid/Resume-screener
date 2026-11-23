"""Fit report from the Groq LLM: a short summary and five interview questions for the gaps.

The API key is used only in the Authorization header here, on the server.
"""
from __future__ import annotations

import hashlib
import json

import httpx

from .matching import SkillMatch

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RESUME_CHARS = 6000
MAX_JD_CHARS = 3000


class LLMError(Exception):
    pass


class GroqClient:
    def __init__(self, api_key: str, model: str, transport: httpx.BaseTransport | None = None):
        self.model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30, transport=transport
        )

    def chat_json(self, messages: list[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            r = self._client.post(GROQ_URL, json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"could not reach Groq: {e.__class__.__name__}") from e
        if r.status_code == 429:
            raise LLMError("Groq rate limit reached, try again in a minute")
        if r.status_code >= 400:
            raise LLMError(f"Groq returned HTTP {r.status_code}")
        try:
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("Groq reply was not valid JSON") from e


SYSTEM_PROMPT = """You help a recruiter screen resumes for one job.
Reply with JSON only, in this shape:
{"summary": "<3 to 4 sentences: how well the candidate fits the job, citing evidence from the resume>",
 "questions": ["<q1>", "<q2>", "<q3>", "<q4>", "<q5>"]}
Write exactly five interview questions. Aim them at the missing skills and at claims in the
resume that need checking. Do not invent facts that are not in the resume."""


def build_fit_messages(
    job_title: str,
    job_description: str,
    resume_text: str,
    entities: dict[str, list[str]],
    match: SkillMatch,
) -> list[dict]:
    found = "\n".join(f"- {label}: {', '.join(vals)}" for label, vals in entities.items() if vals) or "- none"
    user = (
        f"JOB TITLE: {job_title}\n\n"
        f"JOB DESCRIPTION:\n{job_description[:MAX_JD_CHARS]}\n\n"
        f"RESUME:\n{resume_text[:MAX_RESUME_CHARS]}\n\n"
        f"DETAILS PULLED FROM THE RESUME BY OUR NER MODEL:\n{found}\n\n"
        f"SKILL MATCH: {match.score}%\n"
        f"Matched skills: {', '.join(match.matched) or 'none'}\n"
        f"Missing skills: {', '.join(match.missing) or 'none'}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_fit_report(raw: dict) -> dict:
    summary = raw.get("summary")
    questions = raw.get("questions")
    if not isinstance(summary, str) or not summary.strip():
        raise LLMError("fit report has no summary")
    if not isinstance(questions, list):
        raise LLMError("fit report questions are not a list")
    questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
    if len(questions) < 5:
        raise LLMError("fit report has fewer than five questions")
    return {"summary": summary.strip(), "questions": questions[:5]}


def fit_input_hash(model: str, messages: list[dict]) -> str:
    """Cache key: the same model and the same prompt never go to Groq twice."""
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
