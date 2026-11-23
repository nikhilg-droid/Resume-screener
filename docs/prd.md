# PRD: Resume Screener

## Problem
A recruiter has around 200 resumes for one opening. Reading each one to find skills, degree,
college, past employers and experience takes hours. Comparing each against the job description is
inconsistent from one resume to the next.

## User
One recruiter (the demo user). No login.

## Goals
1. Pull **skills, degree, college, companies and years of experience** out of each resume with a
   fine-tuned DistilBERT NER model that runs inside our back end.
2. Rank candidates for a job by a transparent **match score**: the share of required skills found.
3. Write a **fit report** for any candidate with the Groq LLM: a short summary and five interview
   questions aimed at the gaps.
4. Show, with numbers, whether the fine-tuned BERT or a zero-shot LLM is better at the extraction.

## Non-goals
Login, multiple users, emailing candidates, extracting names or contact details, OCR of scanned PDFs.

## User stories
1. As a recruiter, I create a job from a title and a JD, and the app pulls the required skills out of the JD.
2. As a recruiter, I can edit the required skills, and every candidate's score updates.
3. As a recruiter, I paste a resume, or upload a text PDF, and see the details pulled out of it,
   highlighted in the resume text.
4. As a recruiter, I see a shortlist sorted by match score, with matched and missing skills, years and degree.
5. As a recruiter, I generate a fit report for a candidate. Asking again gives the same report
   instantly, without a second LLM call.
6. As a recruiter, I can delete a candidate.

## Design decisions
| Decision | Choice | Why |
|---|---|---|
| Labels | SKILL, DEGREE, COLLEGE, COMPANY, YEARS_EXP (BIO, 11 tags) | The five details in the brief; no personal data |
| Base model | `distilbert-base-cased` | Small and fast; capital letters help with org names |
| Training data | Made-up resumes and JDs written and labelled by a large Groq model | Public resume NER sets contain real names and emails |
| Test set | 150–200 chunks from a separate, messier generation run, checked by hand | Never trained on; tests robustness to messy formats |
| Long resumes | Split into chunks of ≤ 200 words along line breaks | BERT reads at most 512 sub-word tokens |
| Serving | ONNX int8 + `onnxruntime` + `tokenizers`, model files downloaded from the HF Hub at cold start | No PyTorch; stays under Vercel's 500 MB |
| Match score | Python set overlap with alias normalisation, computed when read | Explainable, testable, never stale after skill edits |
| LLM | Groq `llama-3.1-8b-instant`, JSON mode | Fast, within free rate limits |
| LLM cache | `fit_reports` table, unique on (candidate, SHA-256 of model + full prompt) | Same question never sent twice; edited skills give a new report |
| Database | Neon Postgres via SQLAlchemy; SQLite in tests | One-click on Vercel; fast offline tests |
| Junk entities | Post-filter for punctuation, stopwords, number-less experience | Small model slips should not reach the UI; evaluation uses raw tags |

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness |
| POST | `/api/jobs` | Create job; extracts required skills if none given |
| GET | `/api/jobs` | Jobs with candidate counts |
| GET/PATCH | `/api/jobs/{id}` | Read, or edit title / required skills |
| GET | `/api/jobs/{id}/candidates` | Shortlist, sorted by match score |
| POST | `/api/jobs/{id}/candidates` | Add a pasted resume |
| POST | `/api/jobs/{id}/candidates/pdf` | Add a PDF resume (≤ 4 MB, text PDFs) |
| GET/DELETE | `/api/candidates/{id}` | Candidate detail with entities and latest fit report / delete |
| POST | `/api/candidates/{id}/fit-report` | Generate or return the cached fit report |

## Evaluation
On the hand-checked test set: fine-tuned BERT against the Groq LLM zero-shot. Report token accuracy,
entity-level macro F1 (seqeval), per-label F1 and median time per text (`ml/evaluate.py` → `docs/evaluation.md`).

## Risks
- LLM-made labels can be wrong → the test set is checked by hand, and the overlap rule in `spans_to_bio` tolerates off-by-one offsets.
- Cold start downloads about 65 MB → model files cached in `/tmp`, one download per warm instance.
- Groq rate limits → small model, cache, clear 503 message in the UI.
