import numpy as np

from fakes import DictTagger
from resume_screener.ner import extract_entities, first_subtoken_tags
from resume_screener.text import Span

ID2LABEL = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}


def test_first_subtoken_tags_uses_first_piece_of_each_word_and_skips_specials():
    # [CLS] Py ##thon SQL [SEP]  -> words 0, 0, 1
    word_ids = [None, 0, 0, 1, None]
    logits = np.array([
        [9, 0, 0],   # CLS (ignored)
        [0, 9, 0],   # Py      -> B-SKILL
        [9, 0, 0],   # ##thon  (ignored, not first piece)
        [0, 9, 0],   # SQL     -> B-SKILL
        [9, 0, 0],   # SEP
    ])
    assert first_subtoken_tags(word_ids, logits, ID2LABEL, n_words=2) == ["B-SKILL", "B-SKILL"]


def test_first_subtoken_tags_marks_truncated_words_O():
    word_ids = [None, 0, None]
    logits = np.array([[9, 0, 0], [0, 9, 0], [9, 0, 0]])
    assert first_subtoken_tags(word_ids, logits, ID2LABEL, n_words=3) == ["B-SKILL", "O", "O"]


def test_extract_entities_returns_spans_in_original_text_coordinates():
    text = "Skills: Python, SQL\nWorked at Infosys"
    tagger = DictTagger({"Python": "SKILL", "SQL": "SKILL", "Infosys": "COMPANY"})
    spans = extract_entities(text, tagger)
    assert [(s.text_in(text), s.label) for s in spans] == [
        ("Python", "SKILL"), ("SQL", "SKILL"), ("Infosys", "COMPANY")
    ]


def test_extract_entities_chunks_long_resumes():
    text = "\n".join(["filler words here"] * 5 + ["Python"])
    tagger = DictTagger({"Python": "SKILL"})
    spans = extract_entities(text, tagger, max_words=6)
    assert len(tagger.calls) == 3
    assert spans == [Span(text.index("Python"), len(text), "SKILL")]


def test_extract_entities_on_empty_text_does_not_call_model():
    tagger = DictTagger({})
    assert extract_entities("  ", tagger) == []
    assert tagger.calls == []
