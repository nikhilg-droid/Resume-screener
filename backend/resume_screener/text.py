"""Turning resume text into words, BIO tags, entity spans and BERT-sized chunks.

The same word tokenizer is used to build training data (ml/) and at inference time,
so the model sees words split the same way in both places.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = ("SKILL", "DEGREE", "COLLEGE", "COMPANY", "YEARS_EXP")
TAGS = ("O",) + tuple(f"{p}-{label}" for label in LABELS for p in ("B", "I"))

# A word starts and ends with a letter/digit, and may contain + # . / & ' - inside,
# so "C++", "C#", "Node.js", "B.Tech" and "scikit-learn" stay whole.
# Any other non-space character is a word of its own (punctuation).
_WORD_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9+#./&'\-]*[A-Za-z0-9+#])?|[^\sA-Za-z0-9]")


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

    def text_in(self, text: str) -> str:
        return text[self.start:self.end]


@dataclass(frozen=True)
class Chunk:
    text: str
    offset: int


def tokenize_words(text: str) -> list[Word]:
    return [Word(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def spans_to_bio(text: str, spans: list[Span]) -> tuple[list[Word], list[str]]:
    """Tag every word: B- for the first word touching a span, I- for the rest, O outside.
    A word counts as inside if it overlaps the span at all, which forgives off-by-one offsets."""
    words = tokenize_words(text)
    tags = ["O"] * len(words)
    for span in spans:
        inside = [i for i, w in enumerate(words) if w.start < span.end and w.end > span.start]
        for n, i in enumerate(inside):
            tags[i] = f"{'B' if n == 0 else 'I'}-{span.label}"
    return words, tags


def bio_to_spans(text: str, words: list[Word], tags: list[str]) -> list[Span]:
    """Join tagged words back into entity spans. A stray I- tag starts a new entity."""
    spans: list[Span] = []
    current: list | None = None  # [start, end, label]
    for word, tag in zip(words, tags):
        if tag == "O" or "-" not in tag:
            if current:
                spans.append(Span(*current))
            current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and current and current[2] == label:
            current[1] = word.end
        else:
            if current:
                spans.append(Span(*current))
            current = [word.start, word.end, label]
    if current:
        spans.append(Span(*current))
    return spans


def chunk_text(text: str, max_words: int = 200) -> list[Chunk]:
    """Split text into chunks of at most max_words words, keeping whole lines together
    where possible. Each chunk is an exact slice of the original text."""
    segments: list[tuple[int, int, int]] = []  # (start, end, word_count)
    for line in re.finditer(r"[^\n]+", text):
        words = list(re.finditer(r"\S+", line.group()))
        base = line.start()
        for i in range(0, len(words), max_words):
            piece = words[i:i + max_words]
            segments.append((base + piece[0].start(), base + piece[-1].end(), len(piece)))

    chunks: list[Chunk] = []
    group: list[tuple[int, int, int]] = []
    for seg in segments:
        if group and sum(s[2] for s in group) + seg[2] > max_words:
            chunks.append(Chunk(text[group[0][0]:group[-1][1]], group[0][0]))
            group = []
        group.append(seg)
    if group:
        chunks.append(Chunk(text[group[0][0]:group[-1][1]], group[0][0]))
    return chunks
