"""Crowd odds from public prediction markets, as one research input.

Polymarket (Gamma API) and Manifold are both readable without a key. A
market is only an anchor when it asks the same thing, by the same deadline,
under the same resolution rules — so this module never decides relevance. It
finds candidates, filters out thin markets, and hands them to the forecaster
with that warning attached.

Rules note: the FutureEval rules permit using publicly available forecasts
from other platforms (they forbid creating questions elsewhere to feed a bot).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import requests

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"

# Thin markets are noise, not crowd wisdom.
MIN_POLYMARKET_VOLUME_USD = 1_000.0
MIN_MANIFOLD_TRADERS = 10
MAX_QUOTES_PER_SOURCE = 4
HTTP_TIMEOUT_SECONDS = 10

GetJson = Callable[[str, dict[str, Any]], Any]

_STOPWORDS = {
    "will", "the", "a", "an", "of", "in", "on", "by", "to", "be", "is", "are", "and", "or",
    "for", "at", "as", "than", "more", "less", "before", "after", "between", "what", "which",
    "who", "how", "many", "much", "does", "do", "did", "its", "it", "from", "with", "this",
    "that", "any", "least", "most", "end", "next", "have", "has", "been",
}
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)


@dataclass(frozen=True)
class MarketQuote:
    source: str
    question: str
    probability: float
    activity: str
    closes: str | None
    url: str


def default_get_json(url: str, params: dict[str, Any]) -> Any:
    response = requests.get(
        url,
        params=params,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": "nfa-forecaster/1.0 (+https://notfinancialadvice.site)"},
    )
    response.raise_for_status()
    return response.json()


def build_search_query(title: str, max_words: int = 8) -> str:
    """Turn a question title into a short keyword query.

    Search engines on both venues match keywords, and a full question with
    dates and filler words returns nothing. Dates and filler are dropped;
    numbers and names are kept.
    """
    text = re.sub(r"\([^)]*\)", " ", title)
    # "January 1, 2027" is a deadline, not a keyword; "October 2026" is kept
    # because it often names the period the question is about.
    text = re.sub(rf"\b({_MONTHS})\.?\s+\d{{1,2}}(st|nd|rd|th)?\b,?\s*(\d{{4}}\b)?", " ", text, flags=re.I)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    text = re.sub(r"[^\w\s$%.\-]", " ", text)
    words = [w.strip(".-") for w in text.split()]
    kept = [w for w in words if w and w.lower() not in _STOPWORDS]
    return " ".join(kept[:max_words])


def _iso_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date().isoformat()
        return str(value)[:10]
    except (ValueError, OSError, OverflowError):
        return None


def _as_list(value: Any) -> list[Any]:
    # Gamma encodes outcomes and outcomePrices as JSON strings inside the JSON.
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_polymarket(payload: Any) -> list[MarketQuote]:
    ranked: list[tuple[float, MarketQuote]] = []
    for event in (payload or {}).get("events") or []:
        event_slug = event.get("slug") or ""
        for market in event.get("markets") or []:
            if market.get("closed") or not market.get("active", True):
                continue
            outcomes = [str(o).lower() for o in _as_list(market.get("outcomes"))]
            prices = _as_list(market.get("outcomePrices"))
            if "yes" not in outcomes or len(prices) != len(outcomes):
                continue
            try:
                probability = float(prices[outcomes.index("yes")])
                volume = float(market.get("volumeNum") or market.get("volume") or 0)
            except (TypeError, ValueError):
                continue
            if volume < MIN_POLYMARKET_VOLUME_USD or not 0 <= probability <= 1:
                continue
            quote = MarketQuote(
                source="Polymarket",
                question=str(market.get("question") or event.get("title") or "").strip(),
                probability=probability,
                activity=f"${volume:,.0f} traded",
                closes=_iso_date(market.get("endDate") or event.get("endDate")),
                url=f"https://polymarket.com/event/{event_slug}" if event_slug else "https://polymarket.com",
            )
            ranked.append((volume, quote))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [quote for _, quote in ranked[:MAX_QUOTES_PER_SOURCE]]


def parse_manifold(payload: Any) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []
    for market in payload or []:
        if market.get("isResolved") or market.get("outcomeType") != "BINARY":
            continue
        traders = int(market.get("uniqueBettorCount") or 0)
        probability = market.get("probability")
        if traders < MIN_MANIFOLD_TRADERS or not isinstance(probability, (int, float)):
            continue
        quotes.append(
            MarketQuote(
                source="Manifold",
                question=str(market.get("question") or "").strip(),
                probability=float(probability),
                activity=f"{traders} traders (play money)",
                closes=_iso_date(market.get("closeTime")),
                url=str(market.get("url") or "https://manifold.markets"),
            )
        )
    return quotes[:MAX_QUOTES_PER_SOURCE]


def search_markets(title: str, get_json: GetJson = default_get_json) -> tuple[list[MarketQuote], list[str]]:
    """Search both venues. Returns quotes plus a note for every venue that failed."""
    query = build_search_query(title)
    if not query:
        return [], ["no searchable keywords in the question title"]

    quotes: list[MarketQuote] = []
    notes: list[str] = []
    try:
        quotes += parse_polymarket(
            get_json(POLYMARKET_SEARCH_URL, {"q": query, "limit_per_type": 6, "events_status": "active"})
        )
    except Exception as e:  # one venue failing must not cost the other
        notes.append(f"Polymarket search failed ({type(e).__name__})")
    try:
        quotes += parse_manifold(
            get_json(MANIFOLD_SEARCH_URL, {"term": query, "limit": 8, "filter": "open", "contractType": "BINARY"})
        )
    except Exception as e:
        notes.append(f"Manifold search failed ({type(e).__name__})")
    return quotes, notes


def format_markets_block(title: str, quotes: list[MarketQuote], notes: list[str]) -> str:
    lines = [f'Search terms: "{build_search_query(title)}"']
    if quotes:
        lines.append(
            "These markets are RELATED, not necessarily the same question. Use one as a strong anchor "
            "only if its event, deadline and resolution rules match; otherwise say how it differs and "
            "adjust, or ignore it."
        )
        for q in quotes:
            closes = f" · closes {q.closes}" if q.closes else ""
            lines.append(f'- {q.source}: "{q.question}" — {q.probability:.0%} Yes · {q.activity}{closes} · {q.url}')
    else:
        lines.append("No liquid related markets found on Polymarket or Manifold.")
    lines += [f"- note: {n}" for n in notes]
    return "\n".join(lines)


def market_research(title: str, get_json: GetJson = default_get_json) -> str:
    quotes, notes = search_markets(title, get_json)
    return format_markets_block(title, quotes, notes)
