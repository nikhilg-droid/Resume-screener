# Resume Screener

ADS 6703 NLP mini project (topic 3). A recruiter pastes a job description and resumes.
A **fine-tuned DistilBERT** (NER, run as ONNX) pulls out skills, degrees, colleges, companies and years
of experience. Candidates are ranked by match score, and a **Groq LLM** writes a fit summary and five
interview questions for the gaps.

React (Vite) · FastAPI · Neon Postgres · Vercel · onnxruntime · Groq

```
api/index.py                 Vercel entry point (imports the FastAPI app)
backend/resume_screener/     text.py (words, BIO, chunks) · ner.py · matching.py · llm.py · db.py · main.py
backend/tests/               pytest, written test-first (see git log)
frontend/                    React + Vite UI
ml/                          generate_data.py · annotate.py · train_ner.ipynb · evaluate.py
docs/                        prd.md · issues.md · evaluation.md · grilling.md · review.md
CONTEXT.md                   the words the app uses
```

## Run locally
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
pytest                                            # all tests, offline
cp .env.example .env                              # fill in GROQ_API_KEY; DATABASE_URL can stay unset (SQLite)
set -a; . ./.env; set +a
uvicorn resume_screener.main:app --app-dir backend --reload --port 8000
```
```bash
cd frontend && npm install && npm run dev         # http://localhost:5173, /api proxied to :8000
```
The API needs the model files in `models/resume-ner/` (`MODEL_DIR`), or `MODEL_REPO` set to download them.

## Build the model
1. **Training data** (Groq writes and labels made-up resumes and JDs):
   `python ml/generate_data.py --n 400 --out data/train.jsonl --seed 1`
2. **Test set** (separate, messier run, then checked by hand):
   `python ml/generate_data.py --n 120 --out data/test_raw.jsonl --seed 999 --messy`
   `python ml/annotate.py export` → edit `data/test_review.txt`, mark each checked block `✓` → `python ml/annotate.py import`
3. **Train**: open `ml/train_ner.ipynb` in Colab (T4 GPU), set `REPO_URL` and `HF_REPO`, run all.
   It uploads `model.onnx`, `tokenizer.json` and `labels.json` to the Hugging Face Hub.
4. **Evaluate**: download those three files to `models/resume-ner/`, then `python ml/evaluate.py` → `docs/evaluation.md`.

## Deploy to Vercel
1. Push to GitHub and import the repo in Vercel (framework preset: Other; `vercel.json` sets the build).
2. Storage → Marketplace → **Neon** → connect it to the project (this sets `DATABASE_URL`).
3. Settings → Environment Variables: `GROQ_API_KEY`, `MODEL_REPO` (e.g. `you/resume-ner-distilbert-onnx`),
   optionally `GROQ_MODEL`. `HF_TOKEN` is only needed if the model repo is private.
4. Deploy. Tables are created on the first request. The first request after a cold start downloads the model (~65 MB) to `/tmp`.

The Groq key is read only in `backend/resume_screener/config.py` and never reaches the browser.
