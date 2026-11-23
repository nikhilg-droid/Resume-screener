import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fakes import DictTagger, FakeLLM
from resume_screener import deps
from resume_screener.db import init_db
from resume_screener.llm import LLMError
from resume_screener.main import app

VOCAB = {"Python": "SKILL", "SQL": "SKILL", "AWS": "SKILL", "Docker": "SKILL",
         "Infosys": "COMPANY", "MBA": "DEGREE", "TAPMI": "COLLEGE"}

JD = "We need a Data Analyst with Python, SQL and AWS."
RESUME_A = "MBA from TAPMI. Worked at Infosys. Skills: Python, SQL, Docker"
RESUME_B = "Skills: Docker"


@pytest.fixture
def env():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    init_db(engine)
    tagger, llm = DictTagger(VOCAB), FakeLLM()
    app.dependency_overrides[deps.get_engine] = lambda: engine
    app.dependency_overrides[deps.get_tagger] = lambda: tagger
    app.dependency_overrides[deps.get_llm] = lambda: llm
    yield TestClient(app), tagger, llm
    app.dependency_overrides.clear()


@pytest.fixture
def client(env):
    return env[0]


def new_job(client, **kw):
    body = {"title": "Data Analyst", "description": JD, **kw}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def add_candidate(client, job_id, text, **kw):
    r = client.post(f"/api/jobs/{job_id}/candidates", json={"resume_text": text, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_create_job_pulls_required_skills_from_description(client):
    job = new_job(client)
    assert job["required_skills"] == ["Python", "SQL", "AWS"]
    assert client.get(f"/api/jobs/{job['id']}").json()["title"] == "Data Analyst"


def test_create_job_uses_given_skills_when_provided(client):
    job = new_job(client, required_skills=["Excel", " ", "Excel"])
    assert job["required_skills"] == ["Excel"]


def test_create_job_rejects_empty_description(client):
    assert client.post("/api/jobs", json={"title": "x", "description": "  "}).status_code == 422


def test_edit_required_skills(client):
    job = new_job(client)
    r = client.patch(f"/api/jobs/{job['id']}", json={"required_skills": ["Docker"]})
    assert r.json()["required_skills"] == ["Docker"]


def test_list_jobs_with_candidate_counts(client):
    job = new_job(client)
    add_candidate(client, job["id"], RESUME_A)
    jobs = client.get("/api/jobs").json()
    assert [(j["id"], j["candidate_count"]) for j in jobs] == [(job["id"], 1)]


def test_missing_job_is_404(client):
    assert client.get("/api/jobs/999").status_code == 404
    assert client.post("/api/jobs/999/candidates", json={"resume_text": "x"}).status_code == 404


def test_add_candidate_extracts_entities_and_scores_match(client):
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_A)
    assert c["label"] == "Candidate 1"
    labels = {(e["label"], e["text"]) for e in c["entities"]}
    assert {("DEGREE", "MBA"), ("COLLEGE", "TAPMI"), ("COMPANY", "Infosys"), ("SKILL", "Docker")} <= labels
    for e in c["entities"]:
        assert RESUME_A[e["start"]:e["end"]] == e["text"]
    assert c["match"] == {"score": 67, "matched": ["Python", "SQL"], "missing": ["AWS"]}


def test_shortlist_is_sorted_by_match_score(client):
    job = new_job(client)
    b = add_candidate(client, job["id"], RESUME_B, label="Resume B")
    a = add_candidate(client, job["id"], RESUME_A)
    rows = client.get(f"/api/jobs/{job['id']}/candidates").json()
    assert [(r["id"], r["match"]["score"]) for r in rows] == [(a["id"], 67), (b["id"], 0)]
    assert rows[1]["label"] == "Resume B"


def test_scores_follow_edited_required_skills(client):
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_B)
    client.patch(f"/api/jobs/{job['id']}", json={"required_skills": ["Docker"]})
    assert client.get(f"/api/candidates/{c['id']}").json()["match"]["score"] == 100


def test_add_candidate_rejects_blank_resume(client):
    job = new_job(client)
    assert client.post(f"/api/jobs/{job['id']}/candidates", json={"resume_text": " "}).status_code == 422


def test_delete_candidate(client):
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_A)
    assert client.delete(f"/api/candidates/{c['id']}").status_code == 204
    assert client.get(f"/api/candidates/{c['id']}").status_code == 404


def test_fit_report_is_generated_once_then_served_from_cache(env):
    client, _, llm = env
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_A)
    first = client.post(f"/api/candidates/{c['id']}/fit-report")
    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert len(first.json()["questions"]) == 5
    second = client.post(f"/api/candidates/{c['id']}/fit-report").json()
    assert second["cached"] is True
    assert second["summary"] == first.json()["summary"]
    assert len(llm.calls) == 1
    assert client.get(f"/api/candidates/{c['id']}").json()["fit_report"]["summary"] == second["summary"]


def test_fit_report_prompt_names_missing_skills(env):
    client, _, llm = env
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_A)
    client.post(f"/api/candidates/{c['id']}/fit-report")
    assert "Missing skills: AWS" in llm.calls[0][-1]["content"]


def test_fit_report_llm_failure_is_503_and_not_cached(env):
    client, _, llm = env
    job = new_job(client)
    c = add_candidate(client, job["id"], RESUME_A)

    def boom(messages):
        raise LLMError("Groq rate limit reached, try again in a minute")

    llm.chat_json = boom
    r = client.post(f"/api/candidates/{c['id']}/fit-report")
    assert r.status_code == 503
    assert "rate limit" in r.json()["detail"]
    assert client.get(f"/api/candidates/{c['id']}").json()["fit_report"] is None


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with one line of text, built by hand."""
    stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_upload_pdf_resume(client):
    job = new_job(client)
    files = {"file": ("resume.pdf", make_pdf("Skills: Python and AWS"), "application/pdf")}
    r = client.post(f"/api/jobs/{job['id']}/candidates/pdf", files=files)
    assert r.status_code == 201, r.text
    assert r.json()["source"] == "pdf"
    assert r.json()["match"]["matched"] == ["Python", "AWS"]


def test_upload_rejects_non_pdf(client):
    job = new_job(client)
    files = {"file": ("resume.txt", b"hello", "text/plain")}
    assert client.post(f"/api/jobs/{job['id']}/candidates/pdf", files=files).status_code == 400
