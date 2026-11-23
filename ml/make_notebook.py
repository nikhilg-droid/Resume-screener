"""Builds ml/train_ner.ipynb from the cells below (so the notebook stays reviewable in git).
  python ml/make_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("md", """# Resume NER: fine-tune DistilBERT, export to ONNX, quantise

Runs on a free Colab / Kaggle T4 GPU in about 10 minutes.

1. Load `data/train.jsonl` (made by `ml/generate_data.py`), split by document
2. Fine-tune `distilbert-base-cased` for token classification (11 BIO tags)
3. Score on the validation split with `seqeval`
4. Export to ONNX, quantise to 8-bit, check it matches PyTorch
5. Upload `model.onnx`, `tokenizer.json`, `labels.json` to the Hugging Face Hub

The hand-checked test set is **not** used here. It is only used by `ml/evaluate.py`."""),
("code", """!pip -q install "transformers==4.47.1" "datasets==3.2.0" "seqeval==1.2.2" "onnx==1.17.0" "onnxruntime==1.20.1" "accelerate>=1.2" huggingface_hub"""),
("code", """# --- settings ---
REPO_URL = "https://github.com/nikhilg-droid/Resume-screener.git"   # or upload data/train.jsonl by hand
TRAIN_FILE = "resume-screener/data/train.jsonl"
MODEL_NAME = "distilbert-base-cased"   # cased: capital letters help spot names of colleges and companies
HF_REPO = "1233knfx/resume-ner-distilbert-onnx"
EPOCHS = 5
LR = 5e-5
OUT = "resume-ner"
SEED = 42"""),
("code", """import os
if not os.path.exists(TRAIN_FILE):
    !git clone -q {REPO_URL}
print(os.path.exists(TRAIN_FILE))"""),
("md", "## 1. Data"),
("code", """import json, random
from collections import Counter

# Same tag order as backend/resume_screener/text.py (TAGS)
LABELS = ["SKILL", "DEGREE", "COLLEGE", "COMPANY", "YEARS_EXP"]
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
tag2id = {t: i for i, t in enumerate(TAGS)}

rows = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8") if l.strip()]
# Split by document, so chunks of one resume never sit in both train and validation
docs = sorted({r["doc_id"] for r in rows})
random.Random(SEED).shuffle(docs)
val_docs = set(docs[: max(1, len(docs) // 10)])
train_rows = [r for r in rows if r["doc_id"] not in val_docs]
val_rows = [r for r in rows if r["doc_id"] in val_docs]
print(len(rows), "chunks |", len(train_rows), "train |", len(val_rows), "validation")
print(Counter(t[2:] for r in rows for t in r["tags"] if t.startswith("B-")))"""),
("md", "## 2. Tokenise and align tags to sub-words\nOnly the first sub-word of each word gets the tag; the rest get -100 and are ignored by the loss. The app decodes the same way (`first_subtoken_tags`)."),
("code", """from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(batch):
    enc = tokenizer(batch["words"], is_split_into_words=True, truncation=True, max_length=512)
    all_labels = []
    for i, tags in enumerate(batch["tags"]):
        prev, labels = None, []
        for wid in enc.word_ids(i):
            labels.append(-100 if wid is None or wid == prev else tag2id[tags[wid]])
            prev = wid
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc

cols = ["words", "tags"]
train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in train_rows]).map(encode, batched=True, remove_columns=cols)
val_ds = Dataset.from_list([{k: r[k] for k in cols} for r in val_rows]).map(encode, batched=True, remove_columns=cols)
train_ds"""),
("md", "## 3. Fine-tune"),
("code", """import numpy as np, torch
from seqeval.metrics import classification_report, f1_score
from transformers import (AutoModelForTokenClassification, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

set_seed(SEED)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id=tag2id)

def to_tags(preds, labels):
    true, pred = [], []
    for p_row, l_row in zip(preds, labels):
        keep = [(TAGS[p], TAGS[l]) for p, l in zip(p_row, l_row) if l != -100]
        pred.append([p for p, _ in keep]); true.append([l for _, l in keep])
    return true, pred

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    true, pred = to_tags(np.argmax(logits, -1), labels)
    return {"macro_f1": f1_score(true, pred, average="macro"), "micro_f1": f1_score(true, pred)}

args = TrainingArguments(
    output_dir="checkpoints", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=16, per_device_eval_batch_size=32, weight_decay=0.01,
    warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", save_total_limit=1, fp16=torch.cuda.is_available(),
    report_to="none", logging_steps=20, seed=SEED)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorForTokenClassification(tokenizer),
                  compute_metrics=compute_metrics)
trainer.train()"""),
("code", """out = trainer.predict(val_ds)
true, pred = to_tags(np.argmax(out.predictions, -1), out.label_ids)
print(classification_report(true, pred, digits=3))
trainer.save_model(OUT + "-pt"); tokenizer.save_pretrained(OUT + "-pt")"""),
("md", "## 4. Export to ONNX and quantise to 8-bit\nONNX runs with `onnxruntime` alone, so the Vercel back end needs no PyTorch."),
("code", """import os, shutil
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic

os.makedirs(OUT, exist_ok=True)
model = trainer.model.to("cpu").eval()
sample = tokenizer(["Python", "at", "Infosys"], is_split_into_words=True, return_tensors="pt")
torch.onnx.export(
    model, (sample["input_ids"], sample["attention_mask"]), f"{OUT}/model-fp32.onnx",
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "logits": {0: "batch", 1: "seq"}},
    opset_version=17)
quantize_dynamic(f"{OUT}/model-fp32.onnx", f"{OUT}/model.onnx", weight_type=QuantType.QInt8)
os.remove(f"{OUT}/model-fp32.onnx")

# The app loads the fast tokenizer from tokenizer.json and the tag list from labels.json
tokenizer.backend_tokenizer.save(f"{OUT}/tokenizer.json")
json.dump(TAGS, open(f"{OUT}/labels.json", "w"))
for f in sorted(Path(OUT).iterdir()):
    print(f"{f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")"""),
("code", """# Check: the quantised ONNX model agrees with PyTorch on the validation set
import onnxruntime as ort
sess = ort.InferenceSession(f"{OUT}/model.onnx", providers=["CPUExecutionProvider"])
agree = total = 0
for r in val_rows[:100]:
    enc = tokenizer(r["words"], is_split_into_words=True, truncation=True, max_length=512, return_tensors="np")
    onnx_pred = sess.run(None, {"input_ids": enc["input_ids"].astype("int64"),
                                "attention_mask": enc["attention_mask"].astype("int64")})[0].argmax(-1)
    with torch.no_grad():
        pt_pred = model(input_ids=torch.tensor(enc["input_ids"]),
                        attention_mask=torch.tensor(enc["attention_mask"])).logits.argmax(-1).numpy()
    agree += (onnx_pred == pt_pred).sum(); total += onnx_pred.size
print(f"ONNX vs PyTorch token agreement: {agree / total:.2%}")"""),
("md", "## 5. Upload to the Hugging Face Hub\nThe back end downloads these three files on a cold start (set `MODEL_REPO` on Vercel)."),
("code", """from huggingface_hub import HfApi, login
login()  # paste a write token from huggingface.co/settings/tokens
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(folder_path=OUT, repo_id=HF_REPO, allow_patterns=["model.onnx", "tokenizer.json", "labels.json"])
print(f"https://huggingface.co/{HF_REPO}")"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }


if __name__ == "__main__":
    path = Path(__file__).with_name("train_ner.ipynb")
    path.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {path}")
