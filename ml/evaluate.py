"""Compare the fine-tuned BERT (ONNX, exactly as the app runs it) with the Groq LLM zero-shot
on the hand-checked test set. Writes docs/evaluation.md.

  python ml/evaluate.py                       # both models (needs GROQ_API_KEY)
  python ml/evaluate.py --skip-llm            # BERT only

LLM answers are cached in data/llm_zero_shot_cache.jsonl, so a re-run does not call Groq again.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

from seqeval.metrics import classification_report, f1_score

from datautils import ROOT, align_entities, read_jsonl, write_jsonl

sys.path.insert(0, str(ROOT / "backend"))
from resume_screener.llm import GroqClient, LLMError  # noqa: E402
from resume_screener.ner import OnnxTagger  # noqa: E402
from resume_screener.text import LABELS, spans_to_bio  # noqa: E402

ZERO_SHOT_SYSTEM = """Extract named entities from a piece of a resume or job description.
Labels: SKILL (skills, tools, languages, methods), DEGREE (degrees and qualifications),
COLLEGE (universities, colleges, institutes, schools), COMPANY (organisations the person worked at),
YEARS_EXP (phrases giving an amount of work experience, e.g. "3 years").
Copy each entity text exactly from the input, list every mention in order, repeats included.
Reply with JSON only: {"entities": [{"text": "...", "label": "..."}]}"""


def score(true: list[list[str]], pred: list[list[str]]) -> dict:
    tokens = [(t, p) for ts, ps in zip(true, pred) for t, p in zip(ts, ps)]
    per_label = classification_report(true, pred, output_dict=True, zero_division=0)
    return {
        "token_accuracy": sum(t == p for t, p in tokens) / len(tokens) if tokens else 0.0,
        "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
        "micro_f1": f1_score(true, pred, average="micro", zero_division=0),
        "per_label": {lab: per_label.get(lab, {}).get("f1-score", 0.0) for lab in LABELS},
    }


def run_bert(rows, model_dir):
    tagger = OnnxTagger(model_dir)
    tagger.tag(["warm", "up"])
    preds, times = [], []
    for r in rows:
        t0 = time.perf_counter()
        preds.append(tagger.tag(r["words"]))
        times.append(time.perf_counter() - t0)
    return preds, times


def run_llm(rows, model, sleep):
    cache_path = ROOT / "data" / "llm_zero_shot_cache.jsonl"
    cache = {(c["id"], c["model"]): c for c in (read_jsonl(cache_path) if cache_path.exists() else [])}
    client = GroqClient(os.environ["GROQ_API_KEY"], model)
    preds, times = [], []
    for i, r in enumerate(rows):
        key = (r["id"], model)
        if key not in cache:
            msgs = [{"role": "system", "content": ZERO_SHOT_SYSTEM}, {"role": "user", "content": r["text"]}]
            for attempt in range(4):
                try:
                    t0 = time.perf_counter()
                    raw = client.chat_json(msgs)
                    seconds = time.perf_counter() - t0
                    break
                except LLMError as e:
                    print(f"{r['id']}: {e}", file=sys.stderr)
                    time.sleep(20 * (attempt + 1))
            else:
                raw, seconds = {"entities": []}, None
            cache[key] = {"id": r["id"], "model": model, "entities": raw.get("entities", []), "seconds": seconds}
            write_jsonl(cache_path, list(cache.values()))
            print(f"llm {i + 1}/{len(rows)}")
            time.sleep(sleep)
        c = cache[key]
        ents = c["entities"] if isinstance(c["entities"], list) else []
        _, tags = spans_to_bio(r["text"], align_entities(r["text"], [e for e in ents if isinstance(e, dict)]))
        preds.append(tags)
        if c["seconds"] is not None:
            times.append(c["seconds"])
    return preds, times


def fmt_ms(times):
    if not times:
        return "n/a"
    ms = statistics.median(times) * 1000
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:,.0f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--model-dir", default="models/resume-ner")
    ap.add_argument("--llm-model", default=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    rows = read_jsonl(ROOT / args.test)
    true = [r["tags"] for r in rows]
    checked = sum(bool(r.get("checked")) for r in rows)
    results = {}

    preds, times = run_bert(rows, ROOT / args.model_dir)
    results["Fine-tuned DistilBERT (ONNX, int8)"] = (score(true, preds), fmt_ms(times))
    print(classification_report(true, preds, digits=3, zero_division=0))

    if not args.skip_llm:
        preds, times = run_llm(rows, args.llm_model, args.sleep)
        results[f"Groq {args.llm_model}, zero-shot"] = (score(true, preds), fmt_ms(times))
        print(classification_report(true, preds, digits=3, zero_division=0))

    lines = [
        "# Evaluation: BERT vs LLM on the hand-checked test set",
        "",
        f"Test set: `{args.test}`, {len(rows)} examples ({checked} hand-checked), never used in training.",
        "F1 is entity-level (seqeval): an entity counts only if its span and label are both exactly right.",
        "Token accuracy counts every word, including the many `O` words, so it always looks high.",
        "Time is the median for one text: BERT on this machine's CPU, the LLM including the network round trip.",
        "",
        "| Model | Token accuracy | Macro F1 | Micro F1 | Time per text |",
        "|---|---|---|---|---|",
    ]
    for name, (s, ms) in results.items():
        lines.append(f"| {name} | {s['token_accuracy']:.3f} | {s['macro_f1']:.3f} | {s['micro_f1']:.3f} | {ms} |")
    lines += ["", "## F1 per label", "", "| Model | " + " | ".join(LABELS) + " |",
              "|---|" + "---|" * len(LABELS)]
    for name, (s, _) in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{s['per_label'][l]:.3f}" for l in LABELS) + " |")
    lines += ["", "## Decision", "",
              "_Write one sentence here: which model you would put in production, and why._", ""]
    out = ROOT / "docs" / "evaluation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: v[0] | {"time": v[1]} for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
