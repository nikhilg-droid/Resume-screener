# Issues

Created on GitHub with `bash scripts/create_issues.sh` (needs `gh auth login`).
Each issue is closed by the commits that make its tests pass.

1. **Word tokenizer, BIO tags and chunking.** Tech words stay whole (C++, Node.js); spans become BIO tags and back; resumes split into chunks of ≤ 200 words with exact offsets.
2. **Skill match score.** Normalise skills (case, aliases such as JS = JavaScript), score = % of required skills found, matched and missing lists.
3. **NER inference with ONNX.** First-sub-token decoding, chunked extraction with offsets in the original text, model download from the HF Hub.
4. **Fit report via Groq.** Prompt with JD, resume, entities and missing skills; check the reply has a summary and five questions; cache key = hash of model + prompt.
5. **Jobs API.** Create a job (skills from the JD if not given), list, read, edit required skills.
6. **Candidates API and shortlist.** Paste a resume → entities + match; shortlist sorted by score; delete.
7. **Cached fit-report endpoint.** Second request served from the DB; Groq failure → 503, nothing cached.
8. **PDF upload (stretch).** Text PDFs ≤ 4 MB; clear errors for non-PDF and scanned PDFs.
9. **React UI.** Jobs page, job page with shortlist and add-candidate form, candidate page with highlighted resume and fit report.
10. **Training data and test set.** Generate data with Groq; hand-check 150–200 test chunks.
11. **Fine-tune, export, quantise.** Colab notebook → `model.onnx` (int8), `tokenizer.json`, `labels.json` on the HF Hub.
12. **Evaluation.** BERT vs Groq zero-shot: token accuracy, macro F1, per-label F1, time per text.
13. **Deploy to Vercel.** Neon from the Marketplace, `GROQ_API_KEY` and `MODEL_REPO` as environment variables.
14. **Filter implausible entities.** Drop punctuation, stopwords and number-less experience before the UI.
