"""Make labelled NER data with the Groq LLM (a big model teaches a small one).

Each call asks the LLM to write ONE made-up resume (or job description) and list its entities.
We find each entity in the text, convert to BIO tags, cut into BERT-sized chunks and save JSONL.

  python ml/generate_data.py --n 400 --out data/train.jsonl --seed 1
  python ml/generate_data.py --n 120 --out data/test_raw.jsonl --seed 999 --messy   # for the test set

Needs GROQ_API_KEY (or SARVAM_API_KEY with --provider sarvam) in the environment. Safe to stop and re-run: it appends and skips done ids.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

from datautils import ROOT, align_entities, make_record, read_jsonl, write_jsonl  # noqa: F401

sys.path.insert(0, str(ROOT / "backend"))
from resume_screener.llm import GroqClient, LLMError  # noqa: E402
from resume_screener.text import Span, chunk_text  # noqa: E402

ROLES = [
    "data analyst", "data scientist", "backend developer", "frontend developer", "full-stack developer",
    "DevOps engineer", "machine learning engineer", "business analyst", "product manager",
    "marketing manager", "digital marketing executive", "financial analyst", "investment banking analyst",
    "HR generalist", "talent acquisition specialist", "supply chain analyst", "operations manager",
    "sales executive", "key account manager", "chartered accountant", "QA automation engineer",
    "cloud architect", "cybersecurity analyst", "UI/UX designer", "mechanical design engineer",
    "civil site engineer", "content writer", "customer success manager", "management trainee",
    "credit risk analyst",
]
SENIORITY = ["fresher (internships only)", "1-2 years", "3-5 years", "6-10 years", "12+ years"]
STYLES = [
    "classic sections with headings (SUMMARY, EDUCATION, EXPERIENCE, SKILLS) and bullet points",
    "compact one-page style with short lines and pipes, e.g. 'Skills: Python | SQL | Tableau'",
    "a narrative profile paragraph followed by a short work history",
    "a table-like layout with 'Year - Degree - Institute' lines and dated job lines",
    "a LinkedIn 'About' style text in first person",
    "an academic CV of a fresh graduate with projects and certifications",
]
MESSY = [
    "copied from a PDF: some words in ALL CAPS, odd line breaks, inconsistent bullets",
    "typed casually in lowercase with abbreviations (yrs, exp, b.tech, mgmt)",
    "mixed Indian English phrasing with long run-on lines",
]

SYSTEM = """You create realistic but completely made-up training data for a resume NER model.
Reply with JSON only: {"text": "<the document>", "entities": [{"text": "...", "label": "..."}]}

Labels:
- SKILL: a technical or business skill, tool, language, framework or method
  (Python, SQL, Power BI, financial modelling, SEO, negotiation, AutoCAD, Agile)
- DEGREE: a degree or qualification, with its subject if written next to it
  (B.Tech in Computer Science, MBA, M.Sc Statistics, CA, Class XII)
- COLLEGE: a university, college, institute or school name (IIT Bombay, Christ University)
- COMPANY: an organisation the person worked or interned at (Infosys, HDFC Bank, a startup's name)
- YEARS_EXP: a phrase giving an amount of work experience (3 years, 5+ years, 18 months)

Rules:
- List EVERY mention in the order it appears, repeats included. Each entity text must be copied
  exactly from the document, character for character.
- Do not label job titles, dates, cities, certifications bodies or project names.
- Use a made-up person name, the email name@example.com and the phone +91-XXXXXXXXXX only; never
  label them. Never use a real person's name."""


def user_prompt(rng: random.Random, messy: bool) -> tuple[str, str]:
    role, level = rng.choice(ROLES), rng.choice(SENIORITY)
    if rng.random() < 0.2:
        kind = "job description"
        prompt = (f"Write a job description (120-220 words) for a {role} role ({level} experience) "
                  f"at an Indian or global company. List required and nice-to-have skills, a degree "
                  f"requirement and the years of experience needed.")
    else:
        kind = "resume"
        style = rng.choice(MESSY if messy else STYLES)
        prompt = (f"Write the resume of a {role} with {level} of experience. Format: {style}. "
                  f"Length 150-300 words. Mix Indian and global colleges and employers.")
    return kind, prompt


def generate(client: GroqClient, rng: random.Random, messy: bool) -> tuple[str, str, list[Span]]:
    kind, prompt = user_prompt(rng, messy)
    raw = client.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    text = str(raw.get("text", "")).strip()
    ents = raw.get("entities", [])
    if not text or not isinstance(ents, list):
        raise LLMError("bad reply shape")
    return kind, text, align_entities(text, ents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of documents to generate")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--messy", action="store_true", help="use messy formats (good for the test set)")
    ap.add_argument("--provider", choices=["groq", "sarvam"], default="groq")
    ap.add_argument("--model", default=None, help="default: llama-3.3-70b-versatile (groq), sarvam-105b (sarvam)")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between calls (rate limits)")
    args = ap.parse_args()

    if args.provider == "sarvam":
        client = GroqClient(os.environ.get("SARVAM_API_KEY", "proxy-injected"), args.model or "sarvam-105b",
                            url="https://api.sarvam.ai/v1/chat/completions", auth_header="api-subscription-key")
    else:
        client = GroqClient(os.environ["GROQ_API_KEY"], args.model or "llama-3.3-70b-versatile")
    rng = random.Random(args.seed)
    out = ROOT / args.out
    rows = read_jsonl(out) if out.exists() else []
    done_docs = {r["doc_id"] for r in rows}

    for i in range(args.n):
        doc_id = f"s{args.seed}-{i}"
        if doc_id in done_docs:
            rng.random(); rng.random()  # keep the random stream roughly aligned
            continue
        for attempt in range(3):
            try:
                kind, text, spans = generate(client, rng, args.messy)
                break
            except LLMError as e:
                print(f"{doc_id}: {e}; retrying", file=sys.stderr)
                time.sleep(20 if "rate" in str(e) else 2)
        else:
            continue
        for j, chunk in enumerate(chunk_text(text, max_words=200)):
            local = [Span(s.start - chunk.offset, s.end - chunk.offset, s.label) for s in spans
                     if s.start >= chunk.offset and s.end <= chunk.offset + len(chunk.text)]
            rec = make_record(f"{doc_id}-{j}", chunk.text, local)
            rec.update(doc_id=doc_id, kind=kind)
            rows.append(rec)
        write_jsonl(out, rows)
        print(f"{i + 1}/{args.n} {kind}: {len(spans)} entities, {len(rows)} chunks total")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
