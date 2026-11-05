import pytest
import requests

from nfa_forecaster.web_search import WebSearchFailed, grounded_search, parse_grounded_answer

# Shape of a real grounded answer (gemini-2.5-flash, 2026-09-26), trimmed.
GROUNDED = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "planning the search", "thought": True},
                    {"text": "On September 16, 2026, the FOMC raised the target range by 0.25 points."},
                ]
            },
            "groundingMetadata": {
                "webSearchQueries": ["FOMC September 2026 decision"],
                "groundingChunks": [
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x", "title": "federalreserve.gov"}},
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/y", "title": "reuters.com"}},
                    {"web": {"uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/z", "title": "reuters.com"}},
                ],
            },
        }
    ]
}


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_parse_drops_thinking_and_deduplicates_sources():
    text, sources = parse_grounded_answer(GROUNDED)
    assert text == "On September 16, 2026, the FOMC raised the target range by 0.25 points."
    assert sources == ["federalreserve.gov", "reuters.com"]


def test_search_sends_the_google_search_tool_and_the_key_in_a_header():
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResponse(200, GROUNDED)

    answer = grounded_search("what did the Fed do?", "SECRET", post=post, models=["gemini-2.5-flash"])
    assert seen["json"]["tools"] == [{"google_search": {}}]
    assert seen["headers"] == {"x-goog-api-key": "SECRET"} and "SECRET" not in seen["url"]
    assert "Sources: federalreserve.gov, reuters.com" in answer
    assert answer.endswith("(Searched with Google by gemini-2.5-flash.)")


def test_search_falls_back_to_the_next_model_and_names_every_failure():
    responses = iter([FakeResponse(429), FakeResponse(200, {"candidates": [{"content": {"parts": []}}]})])
    with pytest.raises(WebSearchFailed, match="a: HTTP 429; b: empty answer"):
        grounded_search("q", "k", post=lambda url, **kw: next(responses), models=["a", "b"])

    def flaky(url, **kw):
        if "/a:" in url:
            raise requests.ConnectionError("url with ?key=SECRET")
        return FakeResponse(200, GROUNDED)

    assert "FOMC" in grounded_search("q", "SECRET", post=flaky, models=["a", "b"])


def test_a_failure_message_never_carries_the_key():
    def boom(url, **kw):
        raise requests.ConnectionError("HTTPSConnectionPool: Max retries exceeded with url: /x?key=SECRET")

    with pytest.raises(WebSearchFailed) as err:
        grounded_search("q", "SECRET", post=boom, models=["a"])
    assert "SECRET" not in str(err.value)


def _answer(text, sources=()):
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "groundingMetadata": {"groundingChunks": [{"web": {"title": s}} for s in sources]},
            }
        ]
    }


class ScriptedGemini:
    """Answers by what the request asks for: planner (no tools), search, or source reading."""

    def __init__(self, plan="rate cut odds October 2026\nCPI release date", fail=()):
        self.plan = plan
        self.fail = dict(fail)
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        model = url.split("/models/")[1].split(":")[0]
        tools = [next(iter(t)) for t in (json or {}).get("tools", [])]
        prompt = json["contents"][0]["parts"][0]["text"]
        self.calls.append((model, tuple(tools)))
        if model in self.fail:
            return self.fail[model]
        if not tools:
            return FakeResponse(200, _answer(self.plan))
        if "url_context" in tools:
            return FakeResponse(200, _answer("FRED shows 4.1% for August 2026; next release 2 Oct."))
        return FakeResponse(200, _answer(f"found: {prompt[:40]}", ["reuters.com"]))


def test_deep_search_asks_what_is_missing_then_searches_for_it():
    from nfa_forecaster.web_search import GeminiResearch

    gemini = ScriptedGemini()
    text = GeminiResearch("k", post=gemini).deep_search("first pass", "Will the Fed cut?", "Resolves Yes if...", "2026-09-26")
    assert text.startswith("found: first pass")
    assert "### Follow-up searches (what the first pass left open)" in text
    assert "**rate cut odds October 2026**" in text and "**CPI release date**" in text
    # One search, one planner call, two follow-up searches.
    assert [tools for _, tools in gemini.calls].count(("google_search",)) == 3
    assert [tools for _, tools in gemini.calls].count(()) == 1


def test_a_planner_that_sees_nothing_missing_costs_no_extra_searches():
    from nfa_forecaster.web_search import GeminiResearch

    gemini = ScriptedGemini(plan="NONE")
    text = GeminiResearch("k", post=gemini).deep_search("first pass", "t", None, "2026-09-26")
    assert "Follow-up" not in text and len(gemini.calls) == 2


def test_resolution_sources_are_read_with_the_url_tool():
    from nfa_forecaster.web_search import GeminiResearch

    gemini = ScriptedGemini()
    text = GeminiResearch("k", post=gemini).read_sources(["https://fred.stlouisfed.org/series/UNRATE"], "t", None, "2026-09-26")
    assert "4.1% for August 2026" in text and text.endswith("(Read directly by gemini-2.5-flash.)")
    assert gemini.calls[0][1] == ("url_context", "google_search")


def test_a_model_out_for_the_day_is_benched_and_one_rejecting_a_tool_only_for_that_tool():
    from nfa_forecaster.web_search import GeminiResearch

    class Body(FakeResponse):
        def __init__(self, code, text):
            super().__init__(code)
            self.text = text

    gemini = ScriptedGemini(fail={"gemini-2.5-flash": Body(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")})
    research = GeminiResearch("k", post=gemini)
    research.search("q")
    research.search("q")
    assert [m for m, _ in gemini.calls].count("gemini-2.5-flash") == 1
    assert research.benched["gemini-2.5-flash"] == "daily limit reached"

    gemini = ScriptedGemini(fail={"gemini-2.5-flash": Body(400, "Url Context as tool is not enabled for this model")})
    research = GeminiResearch("k", post=gemini)
    research.read_sources(["https://x.org"], "t", None, "d")
    research.search("q")  # the same model still searches
    assert ("gemini-2.5-flash", ("google_search",)) in gemini.calls


def test_source_urls_skip_metaculus_and_trailing_punctuation():
    from nfa_forecaster.web_search import source_urls

    urls = source_urls(
        "Per https://fred.stlouisfed.org/series/UNRATE. See https://www.metaculus.com/questions/1/",
        "(source: https://www.bls.gov/cpi/)",
        None,
    )
    assert urls == ["https://fred.stlouisfed.org/series/UNRATE", "https://www.bls.gov/cpi/"]


def test_planner_output_is_cleaned_and_capped():
    from nfa_forecaster.web_search import parse_queries

    assert parse_queries("1. first search\n- second search\n* third search") == ["first search", "second search"]
    assert parse_queries("NONE") == []


def test_exa_camel_case_published_date_and_temporal_auditor_timezone_safety(monkeypatch):
    from nfa_forecaster.research_harness import _extract_json_dict
    from nfa_forecaster.research_models import TemporalBadge
    from nfa_forecaster.search_engine import ResilientSearchCascade
    from nfa_forecaster.temporal import TemporalAuditor

    # 1. Verify _extract_json_dict handles markdown fences with trailing text and uppercase JSON
    raw_llm = 'Here is the plan:\n```JSON\n{"status_queries": ["q1"], "has_critical_gaps": true}\n```\nHope this helps!'
    parsed = _extract_json_dict(raw_llm)
    assert parsed["status_queries"] == ["q1"]
    assert parsed["has_critical_gaps"] is True

    # 2. Verify TemporalAuditor handles timezone-naive ISO dates without raising TypeError
    assert TemporalAuditor.classify_date("2024-05-10", current_year=2026) == TemporalBadge.HISTORICAL
    assert TemporalAuditor.classify_date("2026-01-01", current_year=2026) == TemporalBadge.CURRENT
    assert TemporalAuditor.classify_date(None, current_year=2026) == TemporalBadge.UNDATED

    # 3. Verify ResilientSearchCascade._search_exa reads camelCase publishedDate and passes topic="news"
    cascade = ResilientSearchCascade({"EXA_API_KEY": "test-key", "TAVILY_API_KEY": "test-tavily"})
    captured_payloads = []

    def fake_post(url, headers, payload):
        captured_payloads.append((url, payload))
        return {
            "results": [
                {
                    "url": "https://reuters.com/article",
                    "title": "Rate Decision",
                    "text": "Fed holds rates steady.",
                    "publishedDate": "2026-10-20T12:00:00Z",
                }
            ]
        }

    monkeypatch.setattr(cascade, "_http_post_json", fake_post)
    results = cascade.search_sync("Fed rate decision", max_results=2, topic="news")
    assert len(results) == 1
    assert results[0].published_date == "2026-10-20T12:00:00Z"
    assert captured_payloads[0][1].get("category") == "news"
