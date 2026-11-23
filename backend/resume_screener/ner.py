"""Named-entity recognition with the fine-tuned DistilBERT, exported to ONNX.

Only onnxruntime + tokenizers are used here (no PyTorch), so the back end fits on Vercel.
The model folder holds three files written by ml/train_ner.ipynb:
  model.onnx      quantised 8-bit token-classification model
  tokenizer.json  the fast tokenizer
  labels.json     list of tags, index = class id
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .text import Span, bio_to_spans, chunk_text, tokenize_words

MODEL_FILES = ("model.onnx", "tokenizer.json", "labels.json")
MAX_TOKENS = 512


class WordTagger(Protocol):
    def tag(self, words: list[str]) -> list[str]: ...


def first_subtoken_tags(word_ids, logits, id2label: dict[int, str], n_words: int) -> list[str]:
    """Pick one tag per word from the logits of its first sub-token.
    Words cut off by truncation get 'O'."""
    tags = ["O"] * n_words
    seen: set[int] = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen or wid >= n_words:
            continue
        seen.add(wid)
        tags[wid] = id2label[int(np.argmax(logits[pos]))]
    return tags


def extract_entities(text: str, tagger: WordTagger, max_words: int = 200) -> list[Span]:
    """Split the resume into chunks, tag each chunk, and return entity spans
    with offsets in the original text."""
    spans: list[Span] = []
    for chunk in chunk_text(text, max_words=max_words):
        words = tokenize_words(chunk.text)
        if not words:
            continue
        tags = tagger.tag([w.text for w in words])
        for s in bio_to_spans(chunk.text, words, tags):
            spans.append(Span(s.start + chunk.offset, s.end + chunk.offset, s.label))
    return spans


class OnnxTagger:
    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.no_padding()
        labels = json.loads((model_dir / "labels.json").read_text())
        self.id2label = dict(enumerate(labels))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def tag(self, words: list[str]) -> list[str]:
        enc = self.tokenizer.encode(words, is_pretokenized=True)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        logits = self.session.run(None, feeds)[0][0]
        return first_subtoken_tags(enc.word_ids, logits, self.id2label, len(words))


def download_model(repo: str, dest: str | Path) -> Path:
    """Fetch the model files from the Hugging Face Hub once (cold start) into dest."""
    import httpx

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    for name in MODEL_FILES:
        target = dest / name
        if target.exists():
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        tmp = target.with_suffix(".part")
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_bytes(1 << 20):
                    f.write(block)
        tmp.rename(target)
    return dest


def load_tagger(model_dir: str | None, model_repo: str | None) -> OnnxTagger:
    """Use a local model folder if it has the files, else download from the Hub to /tmp."""
    if model_dir and all((Path(model_dir) / f).exists() for f in MODEL_FILES):
        return OnnxTagger(model_dir)
    if not model_repo:
        raise RuntimeError("No NER model: set MODEL_DIR to a folder with model.onnx, or MODEL_REPO.")
    started = time.perf_counter()
    path = download_model(model_repo, "/tmp/resume-ner")
    print(f"downloaded NER model in {time.perf_counter() - started:.1f}s")
    return OnnxTagger(path)
