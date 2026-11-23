from resume_screener.text import Span, bio_to_spans, chunk_text, spans_to_bio, tokenize_words


def words_only(text):
    return [w.text for w in tokenize_words(text)]


def test_tokenize_keeps_tech_words_whole():
    assert words_only("Skills: C++, C#, Node.js and scikit-learn.") == [
        "Skills", ":", "C++", ",", "C#", ",", "Node.js", "and", "scikit-learn", "."
    ]


def test_tokenize_records_character_offsets():
    text = "B.Tech, IIT"
    for w in tokenize_words(text):
        assert text[w.start:w.end] == w.text


def test_spans_to_bio_tags_first_word_B_and_rest_I():
    text = "Worked at Tata Consultancy Services using Python"
    spans = [Span(10, 35, "COMPANY"), Span(42, 48, "SKILL")]
    words, tags = spans_to_bio(text, spans)
    assert [w.text for w in words] == ["Worked", "at", "Tata", "Consultancy", "Services", "using", "Python"]
    assert tags == ["O", "O", "B-COMPANY", "I-COMPANY", "I-COMPANY", "O", "B-SKILL"]


def test_spans_to_bio_tags_words_that_only_partly_overlap_a_span():
    # LLM-made labels are sometimes off by a character; the word is still tagged.
    text = "Python developer"
    words, tags = spans_to_bio(text, [Span(1, 5, "SKILL")])
    assert tags == ["B-SKILL", "O"]


def test_bio_to_spans_round_trips():
    text = "MBA from TAPMI Manipal, 3 years of experience"
    spans = [Span(0, 3, "DEGREE"), Span(9, 22, "COLLEGE"), Span(24, 31, "YEARS_EXP")]
    words, tags = spans_to_bio(text, spans)
    assert bio_to_spans(text, words, tags) == spans


def test_bio_to_spans_treats_stray_I_as_start_of_entity():
    text = "Python SQL"
    words = tokenize_words(text)
    assert bio_to_spans(text, words, ["I-SKILL", "B-SKILL"]) == [Span(0, 6, "SKILL"), Span(7, 10, "SKILL")]


def test_bio_to_spans_splits_on_label_change():
    text = "Python Infosys"
    words = tokenize_words(text)
    assert bio_to_spans(text, words, ["B-SKILL", "I-COMPANY"]) == [Span(0, 6, "SKILL"), Span(7, 14, "COMPANY")]


def test_span_text_helper():
    assert Span(0, 6, "SKILL").text_in("Python rocks") == "Python"


def test_chunk_text_keeps_lines_together_and_offsets_valid():
    text = "line one here\nline two here\nline three here"
    chunks = chunk_text(text, max_words=6)
    assert [c.text for c in chunks] == ["line one here\nline two here", "line three here"]
    for c in chunks:
        assert text[c.offset:c.offset + len(c.text)] == c.text


def test_chunk_text_splits_a_single_long_line():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, max_words=10)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]
    for c in chunks:
        assert text[c.offset:c.offset + len(c.text)] == c.text


def test_chunk_text_ignores_blank_text():
    assert chunk_text("   \n\n ") == []
