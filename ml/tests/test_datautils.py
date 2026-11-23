from datautils import align_entities, from_markup, make_record, to_markup
from resume_screener.text import Span


def test_align_finds_entities_in_order_and_handles_repeats():
    text = "Python at Infosys. Later Python at TCS."
    ents = [{"text": "Python", "label": "SKILL"}, {"text": "Infosys", "label": "COMPANY"},
            {"text": "Python", "label": "SKILL"}, {"text": "TCS", "label": "COMPANY"}]
    spans = align_entities(text, ents)
    assert [(s.text_in(text), s.start) for s in spans] == [
        ("Python", 0), ("Infosys", 10), ("Python", 25), ("TCS", 35)]


def test_align_is_case_insensitive_and_drops_unknown_or_missing():
    text = "skills: python"
    ents = [{"text": "Python", "label": "SKILL"}, {"text": "Java", "label": "SKILL"},
            {"text": "skills", "label": "NAME"}]
    assert align_entities(text, ents) == [Span(8, 14, "SKILL")]


def test_align_does_not_match_inside_other_words():
    text = "JavaScript and Java"
    assert align_entities(text, [{"text": "Java", "label": "SKILL"}]) == [Span(15, 19, "SKILL")]


def test_markup_round_trip():
    text = "MBA from TAPMI, 3 years"
    spans = [Span(0, 3, "DEGREE"), Span(9, 14, "COLLEGE"), Span(16, 23, "YEARS_EXP")]
    markup = to_markup(text, spans)
    assert markup == "[MBA](DEGREE) from [TAPMI](COLLEGE), [3 years](YEARS_EXP)"
    assert from_markup(markup) == (text, spans)


def test_from_markup_ignores_brackets_that_are_not_labels():
    assert from_markup("[x](FOO) [y](SKILL)") == ("[x](FOO) y", [Span(9, 10, "SKILL")])


def test_make_record_adds_words_and_tags():
    r = make_record("r1", "Python dev", [Span(0, 6, "SKILL")])
    assert r["words"] == ["Python", "dev"]
    assert r["tags"] == ["B-SKILL", "O"]
    assert r["spans"] == [{"start": 0, "end": 6, "label": "SKILL"}]


def test_review_file_keeps_only_checked_blocks_by_default():
    from annotate import parse_review
    content = "### a ✓\n[Python](SKILL) dev\n\n### b\n[SQL](SKILL)\n"
    rows = parse_review(content, include_unchecked=False)
    assert [(r["id"], r["text"], r["tags"]) for r in rows] == [("a", "Python dev", ["B-SKILL", "O"])]
    assert len(parse_review(content, include_unchecked=True)) == 2


def test_score_reports_entity_level_macro_f1_and_token_accuracy():
    from evaluate import score
    true = [["B-SKILL", "O", "B-COMPANY", "I-COMPANY"]]
    pred = [["B-SKILL", "O", "B-COMPANY", "O"]]  # company span cut short: entity is wrong
    s = score(true, pred)
    assert s["token_accuracy"] == 0.75
    assert s["per_label"]["SKILL"] == 1.0
    assert s["per_label"]["COMPANY"] == 0.0
    assert s["macro_f1"] == 0.5


def test_align_skips_malformed_entities_from_the_llm():
    text = "Python and SQL"
    ents = ["Python", None, {"text": "SQL"}, {"label": "SKILL"}, {"text": 5, "label": "SKILL"},
            {"text": "SQL", "label": "SKILL"}]
    assert align_entities(text, ents) == [Span(11, 14, "SKILL")]
