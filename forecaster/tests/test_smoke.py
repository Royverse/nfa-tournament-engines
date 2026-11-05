import json

from nfa_forecaster.config import load_config
from nfa_forecaster.smoke import (
    Check,
    check_gemini,
    check_groq,
    check_metaculus,
    check_openrouter,
    check_openrouter_free,
    render_checks,
    smoke_passed,
)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_metaculus_ok_names_the_bot_account():
    check = check_metaculus("tok", get=lambda url, **kw: FakeResponse(200, {"id": 7, "username": "nfa-bot"}))
    assert check.status == "ok" and "nfa-bot" in check.detail


def test_metaculus_missing_or_rejected_token_fails():
    assert check_metaculus(None).status == "fail"
    assert check_metaculus("bad", get=lambda url, **kw: FakeResponse(401)).detail == "HTTP 401 from /api/users/me"


def test_openrouter_reports_remaining_credit_and_fails_when_empty():
    ok = check_openrouter("k", get=lambda url, **kw: FakeResponse(200, {"data": {"usage": 3.5, "limit": 50, "limit_remaining": 46.5}}))
    assert ok.status == "ok" and "$46.50 of $50.00" in ok.detail
    empty = check_openrouter("k", get=lambda url, **kw: FakeResponse(200, {"data": {"usage": 5, "limit": 5, "limit_remaining": 0}}))
    assert empty.status == "fail" and "out of credits" in empty.detail
    assert check_openrouter(None).status == "skip"


def test_gemini_probe_uses_the_bare_model_name_and_a_header_key():
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResponse(200)

    check = check_gemini("key", "gemini/gemini-3.6-flash", post=post)
    assert check.status == "ok" and check.name == "Gemini · gemini-3.6-flash"
    assert seen["url"].endswith("/models/gemini-3.6-flash:generateContent")
    # A requests error quotes the URL, so the key must never be in it.
    assert "key" not in seen.get("params", {}) and seen["headers"]["x-goog-api-key"] == "key"


def test_gemini_daily_quota_and_busy_are_warnings_but_a_retired_model_fails():
    daily = FakeResponse(429, {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    assert check_gemini("k", "gemini/m", post=lambda url, **kw: daily).detail.startswith("m: today's free quota")
    assert check_gemini("k", "gemini/m", post=lambda url, **kw: FakeResponse(503)).status == "warn"
    assert check_gemini("k", "gemini/m", post=lambda url, **kw: FakeResponse(404)).status == "fail"
    assert check_gemini(None, "gemini/m").status == "skip"


def test_groq_flags_models_missing_from_its_catalogue_and_probes_the_rest():
    listing = FakeResponse(200, {"data": [{"id": "openai/gpt-oss-120b"}]})
    checks = check_groq(
        "k",
        ["groq/openai/gpt-oss-120b", "groq/retired/model"],
        get=lambda url, **kw: listing,
        post=lambda url, **kw: FakeResponse(200),
    )
    assert [(c.name, c.status) for c in checks] == [
        ("Groq · openai/gpt-oss-120b", "ok"),
        ("Groq · retired/model", "fail"),
    ]
    assert check_groq(None, ["groq/x"])[0].status == "skip"
    assert check_groq("k", []) == []


def test_openrouter_free_reports_the_daily_allowance_and_probes_one_free_model():
    seen = {}

    def post(url, **kw):
        seen["model"] = kw["json"]["model"]
        return FakeResponse(200)

    checks = check_openrouter_free(
        "k",
        ["openrouter/a/one:free", "openrouter/b/two:free"],
        get=lambda url, **kw: FakeResponse(200, {"data": {"is_free_tier": True}}),
        post=post,
    )
    assert checks[0].status == "ok" and "50/day" in checks[0].detail
    assert checks[1].name == "OpenRouter · a/one:free" and seen["model"] == "a/one:free"


def test_no_check_ever_prints_a_key():
    key = "AIzaSy-this-is-a-secret"
    checks = [
        check_gemini(key, "gemini/gemini-3.6-flash", post=lambda url, **kw: FakeResponse(429)),
        check_openrouter(key, get=lambda url, **kw: FakeResponse(401)),
        check_metaculus(key, get=lambda url, **kw: FakeResponse(403)),
        *check_groq(key, ["groq/m"], get=lambda url, **kw: FakeResponse(401)),
        *check_openrouter_free(key, ["openrouter/m:free"], get=lambda url, **kw: FakeResponse(401)),
    ]
    assert key not in render_checks(checks)


def test_smoke_passes_with_metaculus_and_one_answering_model_even_if_others_are_busy():
    config = load_config({"METACULUS_TOKEN": "t", "GEMINI_API_KEY": "g"})
    good = [
        Check("Metaculus bot token", "ok", ""),
        Check("OpenRouter (funded ensemble)", "skip", ""),
        Check("Gemini · gemini-3.7-flash", "warn", "daily quota"),
        Check("Gemini · gemini-3.6-flash", "ok", ""),
    ]
    assert smoke_passed(config, good)
    only_a_key_check = [Check("Metaculus bot token", "ok", ""), Check("OpenRouter free models (reserve)", "ok", "")]
    assert not smoke_passed(config, only_a_key_check)
    broken = good + [Check("Groq · retired", "fail", "")]
    assert not smoke_passed(config, broken)
    no_metaculus = [Check("Metaculus bot token", "fail", ""), Check("Gemini · gemini-3.6-flash", "ok", "")]
    assert not smoke_passed(config, no_metaculus)


def test_precedent_check_runs_the_real_source_on_a_resolved_question():
    from nfa_forecaster.smoke import check_precedents

    def post(post_id, resolution):
        return {"id": post_id, "title": f"Q{post_id}", "question": {"title": f"Q{post_id}", "type": "binary", "resolution": resolution}}

    def get(url, params=None, **kw):
        if "similar_to_post_id" in params:
            return FakeResponse(200, {"results": [post(2, "yes"), post(3, "no"), post(4, "yes")]})
        return FakeResponse(200, {"results": [post(1, None), post(9, "no")]})

    on = load_config({"METACULUS_TOKEN": "t", "FORECASTER_PRECEDENTS": "on"})
    check = check_precedents(on, "t", get=get)
    assert check.status == "ok" and check.detail == "similar questions to post 9: 2 of 3 resolved Yes."
    off = load_config({"METACULUS_TOKEN": "t"})
    assert check_precedents(off, "t", get=get).detail.endswith("set FORECASTER_PRECEDENTS=on to use it")
    assert check_precedents(off, None).status == "skip"


def test_precedent_check_reports_hidden_outcomes_without_failing():
    from nfa_forecaster.smoke import check_precedents

    hidden = {"results": [{"id": 1, "title": "Q", "question": {"title": "Q", "status": "resolved", "resolution": None}}]}
    off = load_config({"METACULUS_TOKEN": "t"})
    check = check_precedents(off, "t", get=lambda url, **kw: FakeResponse(200, hidden))
    assert check.status == "skip" and "no outcomes" in check.detail


def test_a_slow_model_is_a_named_warning_not_a_failure():
    import requests

    def slow(url, **kw):
        raise requests.ReadTimeout("read timed out")

    check = check_gemini("k", "gemini/gemma-4-31b-it", post=slow)
    assert (check.name, check.status) == ("Gemini · gemma-4-31b-it", "warn")
