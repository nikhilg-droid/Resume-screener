# Resume Screener — rules for the coding agent

ADS 6703 NLP mini project. A recruiter pastes a job description and resumes; a fine-tuned
DistilBERT (ONNX) pulls out skills, degrees, colleges, companies and years of experience;
a Groq LLM writes a fit summary and five interview questions.

## Stack
- `frontend/` React + Vite. Talks only to `/api/*`. Never holds a secret.
- `backend/resume_screener/` FastAPI app. `api/index.py` is the Vercel entry point.
- Postgres on Neon in production (`DATABASE_URL`), SQLite in tests.
- `ml/` data generation, training notebook, evaluation. Not deployed.

## Rules
- Back end is built test-first: write a failing test in `backend/tests/`, commit it,
  then make it pass, then tidy. Run `pytest` before every commit.
- No PyTorch or `transformers` in `requirements.txt`. Inference uses `onnxruntime` + `tokenizers` only
  (Vercel function limit is 500 MB).
- `GROQ_API_KEY` is read only in `backend/resume_screener/config.py`. Never log it, never send it to the browser.
- Every LLM answer is cached in the `fit_reports` table, keyed by a hash of its inputs.
- Tests never call Groq or download the model: use the fakes in `backend/tests/fakes.py`.
- No personal data: no real names, emails or phone numbers in data, tests or demo.
- Use the words in `CONTEXT.md` (job, candidate, entity, fit report…) in code and UI.

## Commands
- `pytest` — back-end tests
- `uvicorn resume_screener.main:app --app-dir backend --reload` — API on :8000
- `cd frontend && npm run dev` — UI on :5173 (proxies /api to :8000)
