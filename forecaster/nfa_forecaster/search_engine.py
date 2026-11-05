"""Resilient multi-provider search engine cascade with rate-limit pacing and automatic fallback.

Supports:
1. Exa.ai API (Semantic neural search, exact date filtering, text highlights)
2. Tavily API (News filter, recency windows, structured markdown)
3. Gemini Google Search Grounding (Fallback tool)
4. Jina AI Reader (r.jina.ai keyless URL-to-Markdown extractor)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from nfa_forecaster.research_models import SearchResultItem, TemporalBadge

logger = logging.getLogger(__name__)


class SearchProviderError(RuntimeError):
    """Raised when a specific provider fails."""


class ResilientSearchCascade:
    """Cascade: Exa.ai -> Tavily -> Gemini Grounding -> Jina Reader."""

    def __init__(self, env: Mapping[str, str], timeout_seconds: float = 15.0) -> None:
        self.env = env
        self.timeout = timeout_seconds
        self._cache: dict[str, List[SearchResultItem]] = {}

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    def _http_post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **headers}
        req = urllib.request.Request(url, data=data, headers=req_headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    def _http_get(self, url: str, headers: Optional[dict[str, str]] = None) -> str:
        req_headers = {"User-Agent": "Mozilla/5.0 (compatible; NfaResearchBot/2.0)", **(headers or {})}
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def search_sync(
        self, query: str, max_results: int = 4, topic: str = "general", start_published_date: Optional[str] = None
    ) -> List[SearchResultItem]:
        cache_key = f"{topic}:{start_published_date or ''}:{query.strip().lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        providers = [
            ("exa", self._search_exa),
            ("tavily", self._search_tavily),
        ]

        errors: list[str] = []
        for name, provider_func in providers:
            try:
                results = provider_func(
                    query, max_results, topic=topic, start_published_date=start_published_date
                )
                if results:
                    self._cache[cache_key] = results
                    return results
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}({e})")
                logger.debug("Search provider %s failed for '%s': %s", name, query, e)
                continue

        logger.warning("All primary search providers failed for query '%s': %s", query, "; ".join(errors))
        return []

    # --- Tier 1: Exa.ai API ---
    def _search_exa(
        self, query: str, count: int, topic: str = "general", start_published_date: Optional[str] = None
    ) -> List[SearchResultItem]:
        key = self.env.get("EXA_API_KEY")
        if not key:
            raise SearchProviderError("EXA_API_KEY missing")

        url = "https://api.exa.ai/search"
        headers = {"x-api-key": key}
        payload: dict[str, Any] = {
            "query": query,
            "num_results": count,
            "use_autoprompt": True,
            "type": "auto",
            "contents": {"text": {"max_characters": 1200}, "highlights": True},
        }
        if topic == "news":
            payload["category"] = "news"
        if start_published_date:
            payload["start_published_date"] = start_published_date

        data = self._http_post_json(url, headers, payload)
        items: List[SearchResultItem] = []
        for r in data.get("results", []):
            text_snippet = r.get("text") or (r.get("highlights") or [""])[0] or ""
            pub_date = r.get("publishedDate") or r.get("published_date")
            items.append(
                SearchResultItem(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=text_snippet.strip(),
                    published_date=pub_date,
                    source_domain=self._extract_domain(r.get("url", "")),
                    engine="exa",
                )
            )
        return items

    # --- Tier 2: Tavily Search API ---
    def _search_tavily(
        self, query: str, count: int, topic: str = "general", start_published_date: Optional[str] = None
    ) -> List[SearchResultItem]:
        key = self.env.get("TAVILY_API_KEY")
        if not key:
            raise SearchProviderError("TAVILY_API_KEY missing")

        url = "https://api.tavily.com/search"
        tavily_topic = "news" if topic == "news" else "general"
        payload: dict[str, Any] = {
            "api_key": key,
            "query": query,
            "max_results": count,
            "topic": tavily_topic,
        }
        if tavily_topic == "news":
            payload["days"] = 14

        data = self._http_post_json(url, {}, payload)
        items: List[SearchResultItem] = []
        for r in data.get("results", []):
            items.append(
                SearchResultItem(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", "").strip(),
                    published_date=r.get("published_date"),
                    source_domain=self._extract_domain(r.get("url", "")),
                    engine="tavily",
                )
            )
        return items

    # --- Tier 3: Universal Keyless Document Extractor via Jina Reader ---
    def read_url_markdown(self, url: str) -> str:
        if not url.startswith("http"):
            return ""
        jina_url = f"https://r.jina.ai/{url}"
        try:
            raw = self._http_get(jina_url)
            return raw[:3500].strip()
        except Exception as e:
            logger.debug("Jina reader failed for %s: %s", url, e)
            return ""
