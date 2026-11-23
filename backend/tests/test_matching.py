from resume_screener.matching import match_skills, normalize_skill, parse_years


def test_normalize_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_skill("  Python. ") == "python"
    assert normalize_skill("Machine   Learning") == "machine learning"


def test_normalize_maps_common_aliases():
    assert normalize_skill("JS") == normalize_skill("JavaScript")
    assert normalize_skill("ML") == normalize_skill("machine learning")
    assert normalize_skill("Postgres") == normalize_skill("PostgreSQL")
    assert normalize_skill("ReactJS") == normalize_skill("React")


def test_normalize_keeps_symbols_that_change_meaning():
    assert normalize_skill("C++") != normalize_skill("C")
    assert normalize_skill("C#") != normalize_skill("C")


def test_match_skills_scores_share_of_required_skills_found():
    m = match_skills(["Python", "SQL", "Docker", "AWS"], ["python", "Docker", "Excel"])
    assert m.score == 50
    assert m.matched == ["Python", "Docker"]
    assert m.missing == ["SQL", "AWS"]


def test_match_skills_counts_duplicate_required_skills_once():
    m = match_skills(["Python", "python", "SQL"], ["Python"])
    assert m.score == 50
    assert m.matched == ["Python"]


def test_match_skills_with_no_required_skills_is_zero():
    m = match_skills([], ["Python"])
    assert (m.score, m.matched, m.missing) == (0, [], [])


def test_parse_years_reads_numbers_in_experience_phrases():
    assert parse_years("3 years") == 3
    assert parse_years("5+ years of experience") == 5
    assert parse_years("2.5 yrs") == 2.5
    assert parse_years("eighteen months") is None
