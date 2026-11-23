import json

import httpx
import pytest

from resume_screener.llm import GroqClient, LLMError, build_fit_messages, fit_input_hash, parse_fit_report
from resume_screener.matching import SkillMatch


def make_messages(**kw):
    args = dict(
        job_title="Data Analyst",
        job_description="Need SQL, Python and AWS.",
        resume_text="Python developer at Infosys.",
        entities={"SKILL": ["Python"], "COMPANY": ["Infosys"]},
        match=SkillMatch(33, ["Python"], ["SQL", "AWS"]),
    )
    args.update(kw)
    return build_fit_messages(**args)


def test_fit_prompt_contains_job_resume_and_missing_skills():
    user = make_messages()[-1]["content"]
    assert "Need SQL, Python and AWS." in user
    assert "Python developer at Infosys." in user
    assert "SQL, AWS" in user
    assert "five" in make_messages()[0]["content"].lower()


def test_fit_prompt_truncates_very_long_resumes():
    user = make_messages(resume_text="x" * 50_000)[-1]["content"]
    assert len(user) < 12_000


def test_parse_fit_report_accepts_valid_reply():
    r = parse_fit_report({"summary": " Good fit. ", "questions": ["a?", "b?", "c?", "d?", "e?"]})
    assert r == {"summary": "Good fit.", "questions": ["a?", "b?", "c?", "d?", "e?"]}


def test_parse_fit_report_keeps_only_first_five_questions():
    r = parse_fit_report({"summary": "ok", "questions": [f"q{i}?" for i in range(8)]})
    assert len(r["questions"]) == 5


@pytest.mark.parametrize("bad", [
    {"questions": ["a?"] * 5},
    {"summary": "ok", "questions": ["a?"] * 3},
    {"summary": "ok", "questions": "a?"},
    {"summary": "", "questions": ["a?"] * 5},
])
def test_parse_fit_report_rejects_bad_replies(bad):
    with pytest.raises(LLMError):
        parse_fit_report(bad)


def test_fit_input_hash_changes_when_inputs_change():
    a = fit_input_hash("model-a", make_messages())
    assert a == fit_input_hash("model-a", make_messages())
    assert a != fit_input_hash("model-b", make_messages())
    assert a != fit_input_hash("model-a", make_messages(resume_text="other"))


def test_groq_client_sends_key_in_header_and_parses_json_content():
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        content = json.dumps({"summary": "ok", "questions": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = GroqClient("secret-key", "small-model", transport=httpx.MockTransport(handler))
    assert client.chat_json([{"role": "user", "content": "hi"}]) == {"summary": "ok", "questions": []}
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "small-model"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert "secret-key" not in json.dumps(seen["body"])


def test_groq_client_turns_rate_limit_into_llm_error():
    client = GroqClient("k", "m", transport=httpx.MockTransport(lambda r: httpx.Response(429, json={})))
    with pytest.raises(LLMError, match="rate limit"):
        client.chat_json([{"role": "user", "content": "hi"}])


def test_groq_client_rejects_non_json_content():
    reply = {"choices": [{"message": {"content": "not json"}}]}
    client = GroqClient("k", "m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    with pytest.raises(LLMError):
        client.chat_json([{"role": "user", "content": "hi"}])
