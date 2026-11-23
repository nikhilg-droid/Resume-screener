# Context: Resume Screener

The words this app uses, in code, UI and docs. One meaning each.

| Term | Meaning | In code |
|---|---|---|
| **Recruiter** | The one demo user. Opens a job, adds candidates, reads fit reports. | — |
| **Job** | One opening: a title and a job description (JD) text. | `Job`, `jobs` table |
| **Job description (JD)** | The free text of a job. | `Job.description` |
| **Required skills** | The skills a job asks for. Pulled from the JD by the NER model, editable by the recruiter. | `Job.required_skills` |
| **Candidate** | One resume submitted against one job. The same person applying to two jobs is two candidates. | `Candidate` |
| **Resume text** | Plain text of the resume, pasted or extracted from a PDF. | `Candidate.resume_text` |
| **Entity** | A span of resume text with a label. | `Entity` |
| **Label** | One of `SKILL`, `DEGREE`, `COLLEGE`, `COMPANY`, `YEARS_EXP`. | `LABELS` |
| **Tag** | A per-word BIO tag the model predicts (`B-SKILL`, `I-SKILL`, `O`…). Tags become entities. | `bio_to_spans` |
| **Chunk** | A piece of resume text short enough for BERT (≤ 512 tokens). | `chunk_text` |
| **Match score** | Share of required skills found among the candidate's SKILL entities, 0–100. Computed in Python, not by the LLM. | `match_skills` |
| **Matched / missing skills** | Required skills the candidate has / does not have. | `SkillMatch` |
| **Fit report** | LLM output for one candidate against its job: a short summary and exactly five interview questions aimed at the missing skills. Cached. | `FitReport` |
| **Shortlist** | A job's candidates sorted by match score, highest first. | `GET /api/jobs/{id}/candidates` |

## Not in scope
Names, emails, phone numbers and locations are never extracted or stored as entities.
