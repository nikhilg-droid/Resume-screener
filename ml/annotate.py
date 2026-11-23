"""Hand-check the test set in any text editor.

1. export: writes data/test_review.txt. Each example is a block like
       ### s999-3-0
       [MBA](DEGREE) from [TAPMI](COLLEGE), [3 years](YEARS_EXP) at [Infosys](COMPANY)
   Fix wrong labels by editing the brackets: add [text](LABEL), delete a wrong one by
   removing the brackets, change LABEL. Delete a whole block to drop that example.
   Labels: SKILL DEGREE COLLEGE COMPANY YEARS_EXP. Mark a block checked by adding ' ✓' to its ### line.

2. import: reads the edited file back into data/test.jsonl (only blocks marked ✓ unless --all).

  python ml/annotate.py export --src data/test_raw.jsonl --limit 200
  python ml/annotate.py import
"""
from __future__ import annotations

import argparse
import re

from datautils import ROOT, from_markup, make_record, read_jsonl, record_spans, to_markup, write_jsonl

REVIEW = ROOT / "data" / "test_review.txt"
TEST = ROOT / "data" / "test.jsonl"
CHECK = "✓"


def export(src: str, limit: int) -> None:
    rows = read_jsonl(ROOT / src)[:limit]
    blocks = [f"### {r['id']}\n{to_markup(r['text'], record_spans(r))}\n" for r in rows]
    REVIEW.write_text("\n".join(blocks), encoding="utf-8")
    print(f"wrote {len(rows)} examples to {REVIEW}")


def parse_review(content: str, include_unchecked: bool) -> list[dict]:
    rows = []
    for block in re.split(r"^### ", content, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        checked = CHECK in header
        rid = header.replace(CHECK, "").strip()
        if not (checked or include_unchecked):
            continue
        text, spans = from_markup(body.rstrip("\n"))
        if text.strip():
            rows.append(make_record(rid, text, spans) | {"checked": checked})
    return rows


def do_import(include_unchecked: bool) -> None:
    rows = parse_review(REVIEW.read_text(encoding="utf-8"), include_unchecked)
    write_jsonl(TEST, rows)
    checked = sum(r["checked"] for r in rows)
    print(f"wrote {len(rows)} examples ({checked} hand-checked) to {TEST}")
    if checked < 150:
        print("note: the brief asks for 150-200 hand-checked examples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import"])
    ap.add_argument("--src", default="data/test_raw.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--all", action="store_true", help="import unchecked blocks too")
    a = ap.parse_args()
    export(a.src, a.limit) if a.cmd == "export" else do_import(a.all)
