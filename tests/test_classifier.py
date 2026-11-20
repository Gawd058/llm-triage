from conftest import EASY, HARD, chat

from triage.classifier import extract_text, profile_request


def test_trivial_question_is_easy():
    profile = profile_request(chat(EASY))
    assert profile.difficulty < 0.25
    assert profile.signals == ()


def test_code_plus_reasoning_is_hard():
    profile = profile_request(chat(HARD))
    assert profile.difficulty >= 0.55
    assert "code" in profile.signals
    assert any(s.startswith("reasoning:") for s in profile.signals)


def test_math_signal():
    profile = profile_request(chat("Prove that the derivative of x^2 is 2x."))
    assert "math" in profile.signals


def test_long_prompt_raises_difficulty():
    short = profile_request(chat("summarize this: " + "word " * 10))
    long = profile_request(chat("summarize this: " + "word " * 1500))
    assert long.difficulty > short.difficulty
    assert any(s.startswith("length:") for s in long.signals)


def test_content_parts_and_completions_prompt_are_read():
    parts = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
    }
    assert extract_text(parts) == ("hello", 1)
    assert extract_text({"prompt": ["a", "b"]}) == ("a\nb", 1)


def test_max_tokens_caps_expected_output():
    profile = profile_request(chat(HARD, max_tokens=20))
    assert profile.expected_output_tokens == 20
    unbounded = profile_request(chat(HARD))
    assert unbounded.expected_output_tokens > 20


def test_priority_header_is_case_insensitive_and_validated():
    assert profile_request(chat(EASY), {"X-Priority": "HIGH"}).priority == "high"
    assert profile_request(chat(EASY), {"x-priority": "urgent!!"}).priority == "normal"
    assert profile_request(chat(EASY)).priority == "normal"


def test_pinned_model_and_min_tier_header():
    profile = profile_request(
        {"model": "large", "messages": []}, {"x-triage-min-tier": "2"}, {"small", "large"}
    )
    assert profile.pinned == "large"
    assert profile.min_tier == 2
    assert profile_request({"model": "auto", "messages": []}, {}, {"large"}).pinned is None
    assert profile_request(chat(EASY), {"x-triage-min-tier": "abc"}).min_tier is None
