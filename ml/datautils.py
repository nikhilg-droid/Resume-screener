"""Helpers shared by the data scripts: turning LLM entity lists into character spans,
and a hand-editable markup format for checking the test set: [Python](SKILL)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from resume_screener.text import LABELS, Span, spans_to_bio  # noqa: E402

_MARKUP_RE = re.compile(r"\[([^\[\]\n]+)\]\((" + "|".join(LABELS) + r")\)")


def _find(text: str, needle: str, start: int) -> int:
    """Find needle at a word boundary, case-insensitively, from position start."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    m = re.compile(pattern, re.IGNORECASE).search(text, start)
    return m.start() if m else -1


def align_entities(text: str, entities: list[dict]) -> list[Span]:
    """The LLM returns entities as {"text", "label"} in reading order. Find each one in the
    text, moving forward so repeats map to later mentions. Unknown labels and entities
    not found in the text are dropped."""
    spans: list[Span] = []
    taken: list[tuple[int, int]] = []
    cursor = 0
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        label, needle = ent.get("label"), str(ent.get("text", "")).strip()
        if label not in LABELS or not needle:
            continue
        pos, search_from = -1, cursor
        while True:
            pos = _find(text, needle, search_from)
            if pos == -1 and search_from == cursor and cursor > 0:
                search_from = 0  # out of order: look again from the top
                continue
            if pos == -1:
                break
            end = pos + len(needle)
            if not any(pos < b and end > a for a, b in taken):
                break
            search_from = pos + 1
        if pos == -1:
            continue
        end = pos + len(needle)
        spans.append(Span(pos, end, label))
        taken.append((pos, end))
        cursor = end
    return sorted(spans, key=lambda s: s.start)


def to_markup(text: str, spans: list[Span]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        out.append(text[pos:s.start])
        out.append(f"[{text[s.start:s.end]}]({s.label})")
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


def from_markup(markup: str) -> tuple[str, list[Span]]:
    text_parts, spans, pos, length = [], [], 0, 0
    for m in _MARKUP_RE.finditer(markup):
        before = markup[pos:m.start()]
        text_parts.append(before)
        length += len(before)
        spans.append(Span(length, length + len(m.group(1)), m.group(2)))
        text_parts.append(m.group(1))
        length += len(m.group(1))
        pos = m.end()
    text_parts.append(markup[pos:])
    return "".join(text_parts), spans


def make_record(rid: str, text: str, spans: list[Span]) -> dict:
    words, tags = spans_to_bio(text, spans)
    return {
        "id": rid,
        "text": text,
        "spans": [{"start": s.start, "end": s.end, "label": s.label} for s in spans],
        "words": [w.text for w in words],
        "tags": tags,
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_spans(record: dict) -> list[Span]:
    return [Span(s["start"], s["end"], s["label"]) for s in record["spans"]]
