"""Key and service check — proves every configured provider and model answers.

Prints one line per check: OK / WARN / FAIL / SKIP and a plain reason. It
never prints a key, a key's length, or a key's prefix, and it sends keys in
headers, never in URLs (a requests error message quotes the URL). Run it
after adding or rotating a secret ("Forecaster — check" → smoke).

  OK    works now
  WARN  exists but is busy or out of today's quota; the pool falls back past it
  FAIL  broken: a rejected key or a retired model id — fix the config
  SKIP  its key is not set

Cost: one tiny request per configured free model (a free-tier request), one
Google-grounded search, and 1 of AskNews's monthly searches. Metaculus and
OpenRouter key checks are free reads. Alpha Vantage is deliberately not
called — its 25/day quota is shared with the trading app.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import requests

from nfa_forecaster.config import ForecasterConfig, is_openrouter_free
from nfa_forecaster.precedents import POSTS_URL, precedent_research, summarise_precedents
from nfa_forecaster.web_search import WebSearchFailed, grounded_search

METACULUS_ME_URL = "https://www.metaculus.com/api/users/me"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_SECONDS = 30
PING = "Reply with the single word OK."

Http = Callable[..., requests.Response]


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail" | "skip"
    detail: str


def _model_check(name: str, response: requests.Response, label: str) -> Check:
    """Classify one model probe: busy or rate-limited is a WARN, broken is a FAIL."""
    code = response.status_code
    if code == 200:
        return Check(name, "ok", f"{label} answered")
    if code == 429:
        body = response.text or ""
        when = "today's free quota is used up (resets 07:00 UTC)" if "PerDay" in body or "per day" in body.lower() else "rate-limited right now"
        return Check(name, "warn", f"{label}: {when} — the pool falls back past it")
    if code >= 500:
        return Check(name, "warn", f"{label}: provider busy (HTTP {code}) — the pool falls back past it")
    if code == 404:
        return Check(name, "fail", f"{label}: model not found (HTTP 404) — retired id, update config.py")
    return Check(name, "fail", f"{label}: HTTP {code}")


def check_metaculus(token: str | None, get: Http = requests.get) -> Check:
    if not token:
        return Check("Metaculus bot token", "fail", "METACULUS_TOKEN is not set")
    response = get(METACULUS_ME_URL, headers={"Authorization": f"Token {token}"}, timeout=TIMEOUT_SECONDS)
    if response.status_code != 200:
        return Check("Metaculus bot token", "fail", f"HTTP {response.status_code} from /api/users/me")
    me = response.json()
    return Check(
        "Metaculus bot token", "ok", f"authenticated as '{me.get('username')}' (user id {me.get('id')})"
    )


def check_openrouter(key: str | None, get: Http = requests.get) -> Check:
    if not key:
        return Check("OpenRouter (funded ensemble)", "skip", "No OpenRouter key set — running zero-cost multi-provider ensemble")
    response = get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_SECONDS)
    if response.status_code != 200:
        return Check("OpenRouter (funded ensemble)", "fail", f"HTTP {response.status_code} from /api/v1/key")
    data: dict[str, Any] = (response.json() or {}).get("data") or {}
    usage = float(data.get("usage") or 0)
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    if limit is None:
        detail = f"key valid, no credit limit set, ${usage:.2f} used so far"
    else:
        detail = f"key valid, ${float(remaining or 0):.2f} of ${float(limit):.2f} credit remaining"
    if remaining is not None and float(remaining) <= 0:
        return Check("OpenRouter (funded ensemble)", "fail", detail + " — out of credits")
    return Check("OpenRouter (funded ensemble)", "ok", detail)


def check_gemini(key: str | None, model: str, post: Http = requests.post) -> Check:
    bare = model.removeprefix("gemini/")
    name = f"Gemini · {bare}"
    if not key:
        return Check(name, "skip", "GEMINI_API_KEY not set")
    try:
        response = post(
            GEMINI_GENERATE_URL.format(model=bare),
            headers={"x-goog-api-key": key},
            json={"contents": [{"parts": [{"text": PING}]}]},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return Check(name, "warn", f"{bare}: no answer within {TIMEOUT_SECONDS}s — the pool falls back past it")
    return _model_check(name, response, bare)


def check_groq(key: str | None, models: list[str], get: Http = requests.get, post: Http = requests.post) -> list[Check]:
    if not models:
        return []
    if not key:
        return [Check("Groq (free reserve)", "skip", "GROQ_API_KEY not set")]
    headers = {"Authorization": f"Bearer {key}"}
    listing = get(GROQ_MODELS_URL, headers=headers, timeout=TIMEOUT_SECONDS)
    if listing.status_code != 200:
        return [Check("Groq (free reserve)", "fail", f"HTTP {listing.status_code} from /models — key rejected?")]
    available = {m.get("id") for m in (listing.json() or {}).get("data") or [] if isinstance(m, dict)}
    checks = []
    for model in models:
        bare = model.removeprefix("groq/")
        name = f"Groq · {bare}"
        if bare not in available:
            offered = ", ".join(sorted(m for m in available if m))
            checks.append(Check(name, "fail", f"{bare} is not in Groq's model list — update config.py. Groq offers: {offered}"))
            continue
        response = post(
            GROQ_CHAT_URL,
            headers=headers,
            json={"model": bare, "messages": [{"role": "user", "content": PING}], "max_tokens": 16},
            timeout=TIMEOUT_SECONDS,
        )
        checks.append(_model_check(name, response, bare))
    return checks


def check_openrouter_free(key: str | None, models: list[str], get: Http = requests.get, post: Http = requests.post) -> list[Check]:
    if not models:
        return []
    if not key:
        return [Check("OpenRouter free models (reserve)", "skip", "OPENROUTER_FREE_KEY not set")]
    headers = {"Authorization": f"Bearer {key}"}
    info = get(OPENROUTER_KEY_URL, headers=headers, timeout=TIMEOUT_SECONDS)
    if info.status_code != 200:
        return [Check("OpenRouter free models (reserve)", "fail", f"HTTP {info.status_code} from /api/v1/key")]
    data: dict[str, Any] = (info.json() or {}).get("data") or {}
    # OpenRouter: 50 free-model requests a day until an account has bought
    # $10 of credits in total, 1,000 a day after.
    cap = "50/day (account never bought credits)" if data.get("is_free_tier") else "1,000/day"
    checks = [Check("OpenRouter free models (reserve)", "ok", f"key valid · free-model allowance {cap} · can never bill")]
    bare = models[0].removeprefix("openrouter/")
    response = post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json={"model": bare, "messages": [{"role": "user", "content": PING}], "max_tokens": 16},
        timeout=TIMEOUT_SECONDS,
    )
    checks.append(_model_check(f"OpenRouter · {bare}", response, bare))
    return checks



def check_exa_api(env: Mapping[str, str]) -> Check:
    key = env.get("EXA_API_KEY")
    if not key:
        return Check("Exa.ai neural search", "skip", "EXA_API_KEY not set (optional — get free key at exa.ai)")
    try:
        import urllib.request, json
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=json.dumps({"query": "forecast benchmark", "num_results": 1}).encode("utf-8"),
            headers={"x-api-key": key, "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            count = len(data.get("results", []))
            return Check("Exa.ai neural search", "ok", f"answered successfully ({count} result, neural search active)")
    except Exception as e:
        return Check("Exa.ai neural search", "fail", f"Exa API failed: {type(e).__name__} ({e})")


def check_tavily_api(env: Mapping[str, str]) -> Check:
    key = env.get("TAVILY_API_KEY")
    if not key:
        return Check("Tavily news search", "skip", "TAVILY_API_KEY not set (optional — get free key at tavily.com)")
    try:
        import urllib.request, json
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps({"api_key": key, "query": "forecast benchmark", "max_results": 1, "topic": "news"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            count = len(data.get("results", []))
            return Check("Tavily news search", "ok", f"answered successfully ({count} result, news search active)")
    except Exception as e:
        return Check("Tavily news search", "fail", f"Tavily API failed: {type(e).__name__} ({e})")

def check_web_search(config: ForecasterConfig, key: str | None) -> Check:
    name = "Google-grounded web search"
    if not config.use_web_search or not key:
        return Check(name, "skip", "needs GEMINI_API_KEY (and FORECASTER_WEB_SEARCH not off)")
    try:
        answer = grounded_search("Search the web: what is today's date, and one top world news headline from today?", key)
    except WebSearchFailed as e:
        return Check(name, "warn", f"no grounding model answered ({e}) — forecasts go ahead without it")
    sources = "with sources" if "\nSources: " in answer else "without source links"
    return Check(name, "ok", f"Google search answered, {sources}")


def check_precedents(config: ForecasterConfig, token: str | None, get: Http = requests.get) -> Check:
    """Proves the precedent source end to end on a question Metaculus has already resolved.

    Runs even while precedents are off, so the day Metaculus starts showing
    bot accounts the outcomes, this row says so.
    """
    name = "Metaculus precedents"
    if not token:
        return Check(name, "skip", "needs METACULUS_TOKEN")
    response = get(
        POSTS_URL,
        params={"statuses": "resolved", "limit": 10},
        headers={"Authorization": f"Token {token}"},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        return Check(name, "fail", f"HTTP {response.status_code} listing resolved questions")
    posts = (response.json() or {}).get("results") or []
    sample = next((p for p in posts if summarise_precedents([p], exclude_post_id=-1)), None)
    if sample is None:
        questions = [p.get("question") or {} for p in posts if isinstance(p, dict)]
        statuses = sorted({str(q.get("status")) for q in questions})
        resolutions = sorted({repr(q.get("resolution")) for q in questions})[:5]
        return Check(
            name,
            "warn" if config.use_precedents else "skip",
            f"Metaculus shows this bot {len(posts)} resolved question(s) (statuses {statuses}) but no outcomes"
            f" (resolutions {resolutions}), so precedents stay off",
        )
    text = precedent_research(sample["id"], str(sample.get("title") or ""), token, get=get)
    rate = next((line for line in text.splitlines() if line.startswith("Yes/no base rate")), None)
    if rate:
        detail = f"similar questions to post {sample['id']}: {rate.removeprefix('Yes/no base rate among these: ')}"
        if not config.use_precedents:
            detail += " — this works now: set FORECASTER_PRECEDENTS=on to use it"
        return Check(name, "ok", detail)
    return Check(name, "warn", f"post {sample['id']}: {text.splitlines()[0]}")


async def _asknews_search(env: Mapping[str, str]) -> int:
    from asknews_sdk import AsyncAskNewsSDK

    async with AsyncAskNewsSDK(
        client_id=env.get("ASKNEWS_CLIENT_ID") or None,
        client_secret=env.get("ASKNEWS_SECRET") or None,
        api_key=env.get("ASKNEWS_API_KEY") or None,
        scopes={"news"},
    ) as ask:
        response = await ask.news.search_news(query="central bank interest rate decision", n_articles=1, return_type="dicts")
        return len(response.as_dicts or [])


def check_asknews(config: ForecasterConfig, env: Mapping[str, str]) -> Check:
    if not config.use_asknews:
        return Check("AskNews (news research)", "skip", "ASKNEWS_API_KEY not set (optional — get it at my.asknews.app)")
    try:
        found = asyncio.run(_asknews_search(env))
    except Exception as e:
        return Check("AskNews (news research)", "fail", f"search failed ({type(e).__name__})")
    return Check("AskNews (news research)", "ok", f"search answered ({found} article)")


def run_checks(config: ForecasterConfig, env: Mapping[str, str]) -> list[Check]:
    models = [*dict.fromkeys([*config.forecast_models, *config.reserve_models, config.parser_model or ""])]
    gemini = [m for m in models if m.startswith("gemini/")]
    groq = [m for m in models if m.startswith("groq/")]
    free_openrouter = [m for m in models if is_openrouter_free(m)]

    checks: list[Callable[[], Check | list[Check]]] = [
        lambda: check_metaculus(env.get("METACULUS_TOKEN")),
        lambda: check_openrouter(env.get("OPENROUTER_API_KEY")),
        *[(lambda m=m: check_gemini(env.get("GEMINI_API_KEY"), m)) for m in gemini],
        lambda: check_groq(env.get("GROQ_API_KEY"), groq),
        lambda: check_openrouter_free(env.get("OPENROUTER_FREE_KEY"), free_openrouter),
        lambda: check_exa_api(env),
        lambda: check_tavily_api(env),
        lambda: check_web_search(config, env.get("GEMINI_API_KEY")),
        lambda: check_precedents(config, env.get("METACULUS_TOKEN")),
        lambda: check_asknews(config, env),
    ]
    results: list[Check] = []
    for run in checks:
        try:
            outcome = run()
        except requests.RequestException as e:
            outcome = Check("network", "fail", f"request failed ({type(e).__name__})")
        results += outcome if isinstance(outcome, list) else [outcome]
    return results


def smoke_passed(config: ForecasterConfig, checks: list[Check]) -> bool:
    """Pass when Metaculus works, a model answers, and nothing is broken (WARN is fine)."""
    by_name = {c.name: c for c in checks}
    metaculus_ok = by_name.get("Metaculus bot token", Check("", "fail", "")).status == "ok"
    # A model probe is named "<provider> · <model>"; the funded key is checked
    # by reading its credit, since a probe would spend it.
    provider_ok = any(
        c.status == "ok" for c in checks if " · " in c.name or c.name == "OpenRouter (funded ensemble)"
    )
    no_failures = all(c.status != "fail" for c in checks)
    return metaculus_ok and provider_ok and no_failures and config.is_runnable


def render_checks(checks: list[Check]) -> str:
    icon = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {icon.get(c.status, c.status)} {c.status.upper()} | {c.detail} |")
    return "\n".join(lines)
