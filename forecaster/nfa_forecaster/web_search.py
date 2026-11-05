"""Live web research on the free Gemini key: search, spot the gaps, search again, and
read the question's own resolution sources.

Research quality is the bottleneck the 2026 analyses name most often, and the
evidence on how to do it is consistent:

- Iterative search ("generate queries → search → identify gaps → search
  again") beats a single pass (Metaculus, Bot Advice Fall 2025). A top-15
  Summer 2026 bot runs two gap-fill rounds (nostreambot's design log).
- Reading the pages a question resolves from is the other half: the same bot
  fetches every resolution source the question names.

Both run on Gemini's built-in tools, free on this key (measured 2026-09-26):
`google_search` (gemini-2.5-flash and gemma-4-26b reported the Fed's 16 Sep
2026 decision, with sources) and `url_context` (gemini-2.5-flash read the FRED
and BLS pages and returned August's unemployment rate and the next release
date; Gemma models reject that tool).

Free models allow ~20 requests a day each, so every call walks a short list of
models and a model out for the day is benched for the rest of the run, as in
models.py. The key travels in a header, never in the URL: a requests error
message quotes the URL, and error text must never carry a key.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import requests

GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Each list is tried in order; each call spends one of that model's free daily requests.
SEARCH_MODELS: tuple[str, ...] = ("gemini-2.5-flash", "gemma-4-26b-a4b-it", "gemma-4-31b-it", "gemini-3-flash-preview")
READER_MODELS: tuple[str, ...] = ("gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash")
PLANNER_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash")
TIMEOUT_SECONDS = 90
MAX_SOURCES = 10
MAX_FOLLOW_UPS = 2
MAX_SOURCE_URLS = 4
FINDINGS_FOR_PLANNER = 4000

_URL = re.compile(r"https?://[^\s<>\"'\])]+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

Http = Callable[..., requests.Response]


class WebSearchFailed(RuntimeError):
    """No model produced an answer."""


def parse_grounded_answer(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """(answer text without thinking parts, distinct source titles)."""
    candidate = (payload.get("candidates") or [{}])[0] or {}
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    sources: list[str] = []
    for chunk in (candidate.get("groundingMetadata") or {}).get("groundingChunks") or []:
        title = ((chunk or {}).get("web") or {}).get("title", "").strip()
        if title and title not in sources:
            sources.append(title)
    return text.strip(), sources[:MAX_SOURCES]


def source_urls(*texts: str | None) -> list[str]:
    """Web pages the question itself names, Metaculus's own pages excluded."""
    found: list[str] = []
    for text in texts:
        for url in _URL.findall(text or ""):
            url = url.rstrip(".,;:")
            if "metaculus.com" not in url and url not in found:
                found.append(url)
    return found[:MAX_SOURCE_URLS]


def parse_queries(text: str, limit: int = MAX_FOLLOW_UPS) -> list[str]:
    """The planner's one-search-per-line answer, cleaned; [] when it says NONE."""
    queries = []
    for line in text.splitlines():
        query = _LIST_MARKER.sub("", line).strip().strip('"')
        if not query or query.upper().startswith("NONE"):
            continue
        queries.append(query)
    return queries[:limit]


class GeminiResearch:
    """Gemini calls for research, with a per-run bench shared by every question."""

    def __init__(self, key: str, *, post: Http = requests.post) -> None:
        self._key = key
        self._post = post
        # model -> reason; (model, tools) -> reason for a tool the model rejects.
        self.benched: dict[object, str] = {}
        self._cache: dict[str, str] = {}

    def _ask(self, prompt: str, models: Sequence[str], tools: Sequence[str]) -> tuple[str, list[str], str]:
        """(text, source titles, model that answered)."""
        tool_key = tuple(tools)
        failures: list[str] = []
        for model in models:
            if model in self.benched or (model, tool_key) in self.benched:
                continue
            body: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
            if tools:
                body["tools"] = [{tool: {}} for tool in tools]
            try:
                response = self._post(
                    GENERATE_URL.format(model=model),
                    headers={"x-goog-api-key": self._key},
                    json=body,
                    timeout=TIMEOUT_SECONDS,
                )
            except requests.RequestException as e:
                failures.append(f"{model}: {type(e).__name__}")
                continue
            if response.status_code != 200:
                failures.append(f"{model}: HTTP {response.status_code}")
                text = getattr(response, "text", "") or ""
                if response.status_code == 429 and ("PerDay" in text or "per day" in text.lower()):
                    self.benched[model] = "daily limit reached"
                elif response.status_code in (400, 403, 404):
                    self.benched[(model, tool_key)] = f"HTTP {response.status_code}"
                continue
            text, sources = parse_grounded_answer(response.json())
            if not text:
                failures.append(f"{model}: empty answer")
                continue
            return text, sources, model
        raise WebSearchFailed("; ".join(failures) or "every model is benched")

    def search(self, prompt: str, models: Sequence[str] = SEARCH_MODELS) -> str:
        cache_key = prompt.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        text, sources, model = self._ask(prompt, models, ("google_search",))
        lines = [text, ""]
        if sources:
            lines.append("Sources: " + ", ".join(sources))
        lines.append(f"(Searched with Google by {model}.)")
        res = "\n".join(lines)
        self._cache[cache_key] = res
        return res

    def plan_follow_ups(self, title: str, resolution: str | None, today: str, findings: str) -> list[str]:
        prompt = f"""You plan research for a forecaster. Today is {today}.

Question: {title}
Resolution criteria: {resolution or "None given."}

Research so far:
{findings[:FINDINGS_FOR_PLANNER]}

List the {MAX_FOLLOW_UPS} web searches that would most change this forecast: a missing recent fact, the latest value of the data the question resolves on, a base rate for similar events, or a scheduled date before the deadline. One search per line, no numbering, no commentary. If nothing important is missing, write NONE."""
        try:
            text, _, _ = self._ask(prompt, PLANNER_MODELS, ())
        except WebSearchFailed:
            return []
        return parse_queries(text)

    def deep_search(self, prompt: str, title: str, resolution: str | None, today: str) -> str:
        """Search, ask what is still missing, then run those searches in parallel."""
        first = self.search(prompt)
        queries = self.plan_follow_ups(title, resolution, today, first)
        if not queries:
            return first

        def follow_up(query: str) -> str:
            try:
                return self.search(
                    f"Today is {today}. Search the web: {query}\n"
                    f'Report the specific, dated facts you find that bear on this forecasting question: "{title}". '
                    "At most 8 bullet points, naming sources. Do not forecast."
                )
            except WebSearchFailed as e:
                return f"Unavailable ({e})."

        with ThreadPoolExecutor(max_workers=MAX_FOLLOW_UPS) as pool:
            answers = list(pool.map(follow_up, queries))
        parts = [first, "", "### Follow-up searches (what the first pass left open)"]
        for query, answer in zip(queries, answers):
            parts += [f"**{query}**", answer, ""]
        return "\n".join(parts).strip()

    def read_sources(self, urls: Sequence[str], title: str, resolution: str | None, today: str) -> str:
        """What the question's own resolution sources show right now."""
        listing = "\n".join(f"- {url}" for url in urls)
        prompt = f"""Today is {today}. Read these pages, which the forecasting question below names:
{listing}

Question: {title}
Resolution criteria: {resolution or "None given."}

For each page: the current value, status or latest entry that bears on how the question resolves, with its date, and any scheduled update before the deadline. Say plainly if a page could not be read. Do not forecast."""
        text, _, model = self._ask(prompt, READER_MODELS, ("url_context", "google_search"))
        return f"{text}\n\n(Read directly by {model}.)"


def grounded_search(
    prompt: str,
    key: str,
    *,
    post: Http = requests.post,
    models: Sequence[str] = SEARCH_MODELS,
) -> str:
    """One Google-grounded answer, for callers without a GeminiResearch (the smoke test)."""
    return GeminiResearch(key, post=post).search(prompt, models)
