"""Master Agentic Research Harness implementing a Claude Code-grade superforecasting retrieval loop:
1. Cognitive Decomposition (Orthogonal Query Vectors)
2. Fast Multi-Provider Parallel Retrieval & Document Scraping (Exa, Tavily, Jina Reader)
3. Gap Reflection Loop (Detecting Missing 2026 Facts / Resolution Ambiguities)
4. Structured Research Dossier Synthesis with Deterministic Temporal Badging
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Mapping, Sequence

from nfa_forecaster.prompts import QuestionText
from nfa_forecaster.research_models import (
    GapAnalysisResult,
    QueryVectorPlan,
    SearchResultItem,
    TemporalBadge,
)
from nfa_forecaster.search_engine import ResilientSearchCascade
from nfa_forecaster.temporal import TemporalAuditor
from nfa_forecaster.web_search import source_urls

logger = logging.getLogger(__name__)
# Research content for a live question; muted when run logs are public (see bot.forecast_log).
forecast_log = logging.getLogger("nfa_forecaster.forecasts")


def _extract_json_dict(raw_text: str) -> dict[str, Any]:
    """Robustly extracts a JSON object from LLM text, handling markdown fences and preamble."""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return json.loads(fence_match.group(1))
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))
    return json.loads(cleaned)


class AgenticResearchHarness:
    """Institutional-grade research pipeline inspired by Claude Code and Perplexity Deep Research."""

    def __init__(self, llm_pool: Any, env: Mapping[str, str], max_concurrency: int = 4) -> None:
        self.llm_pool = llm_pool
        self.env = env
        self.cascade = ResilientSearchCascade(env)
        self.auditor = TemporalAuditor()
        self._sem = asyncio.Semaphore(max_concurrency)

    async def execute_research(self, q: QuestionText, today_str: str) -> str:
        """Executes the full 4-layer research loop and returns a structured markdown dossier."""
        # 1. Cognitive Decomposition
        plan = await self._decompose_question(q, today_str)

        # Deterministic URL merge from resolution_criteria, fine_print, and background
        regex_urls = source_urls(q.resolution_criteria, q.fine_print, q.background)
        plan.direct_urls = list(dict.fromkeys([*plan.direct_urls, *regex_urls]))[:4]

        # 2. Fast Parallel Multi-Query & Document Retrieval
        items, urls = await self._retrieve_evidence(plan, q)

        # 3. Reflection & Gap Analysis Loop
        gap = await self._check_gaps(q, items, today_str)
        if gap.has_critical_gaps and gap.follow_up_queries:
            logger.info("Critical gaps identified; dispatching %d follow-up queries", len(gap.follow_up_queries[:2]))
            forecast_log.info("Gaps: %s. Follow-ups: %s", gap.reasoning, gap.follow_up_queries[:2])
            follow_ups = await self._dispatch_follow_ups(gap.follow_up_queries[:2], q)
            items.extend(follow_ups)

        # 4. Structured Research Dossier Synthesis
        return await self._synthesize_dossier(q, items, urls, today_str)

    # ---------------------------------------------------- Layer 1: Decomposition
    async def _decompose_question(self, q: QuestionText, today: str) -> QueryVectorPlan:
        prompt = f"""You are the Lead Research Strategist for an elite superforecasting engine. Today is {today}.
Decompose this forecasting question into targeted, orthogonal search queries.

Question: {q.title}
Resolution criteria: {q.resolution_criteria or 'None'}
Fine print: {q.fine_print or 'None'}

Return ONLY a JSON object with this exact structure:
{{
  "status_queries": ["1-2 queries for the latest 2026 status or current metric value"],
  "base_rate_queries": ["1 query for historical frequency / reference class base rates"],
  "official_calendar_queries": ["1 query for official agency release calendar or meeting dates"],
  "red_team_queries": ["1 query for obstacles, failure modes, or opposing consensus"],
  "direct_urls": ["Any official URLs explicitly named in the resolution criteria or fine print"]
}}"""
        try:
            _, resp = await self.llm_pool.ask(
                prompt,
                preferred=[
                    "gemini/gemini-3.5-flash-lite",
                    "gemini/gemini-3.1-flash-lite",
                    "gemini/gemma-4-31b-it",
                    "groq/qwen/qwen3.8-27b",
                ],
            )
            data = _extract_json_dict(resp)
            return QueryVectorPlan(
                status_queries=data.get("status_queries", []),
                base_rate_queries=data.get("base_rate_queries", []),
                official_calendar_queries=data.get("official_calendar_queries", []),
                red_team_queries=data.get("red_team_queries", []),
                direct_urls=data.get("direct_urls", []),
            )
        except Exception:
            return QueryVectorPlan(
                status_queries=[f'"{q.title}" current status 2026'],
                base_rate_queries=[f'"{q.title}" historical base rate statistics'],
                official_calendar_queries=[f'"{q.title}" official release date schedule 2026'],
                red_team_queries=[f'"{q.title}" delays obstacles risks 2026'],
                direct_urls=[],
            )

    # ------------------------------------------------------- Layer 2: Retrieval
    async def _bounded_search(self, query: str, max_results: int = 3, topic: str = "general") -> List[SearchResultItem]:
        async with self._sem:
            return await asyncio.to_thread(self.cascade.search_sync, query=query, max_results=max_results, topic=topic)

    async def _bounded_read_url(self, url: str) -> str:
        async with self._sem:
            return await asyncio.to_thread(self.cascade.read_url_markdown, url=url)

    async def _retrieve_evidence(self, plan: QueryVectorPlan, q: QuestionText) -> tuple[List[SearchResultItem], List[str]]:
        tagged_queries: list[tuple[str, str]] = [
            *((q_str, "news") for q_str in plan.status_queries),
            *((q_str, "general") for q_str in plan.base_rate_queries),
            *((q_str, "general") for q_str in plan.official_calendar_queries),
            *((q_str, "news") for q_str in plan.red_team_queries),
        ]

        # Execute search queries with bounded concurrency to avoid rate-limit bursts
        search_tasks = [
            self._bounded_search(query=q_str, max_results=3, topic=topic)
            for q_str, topic in tagged_queries
        ]
        url_tasks = [
            self._bounded_read_url(url=u)
            for u in plan.direct_urls
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        scraped_docs = await asyncio.gather(*url_tasks, return_exceptions=True)

        items: List[SearchResultItem] = []
        urls: List[str] = []

        for r in search_results:
            if isinstance(r, list):
                for item in r:
                    badge = self.auditor.classify_date(item.published_date)
                    items.append(
                        SearchResultItem(
                            url=item.url,
                            title=item.title,
                            snippet=self.auditor.audit_snippet(item.snippet, badge),
                            published_date=item.published_date,
                            source_domain=item.source_domain,
                            engine=item.engine,
                            temporal_badge=badge,
                        )
                    )
                    urls.append(item.url)

        for u, doc_text in zip(plan.direct_urls, scraped_docs):
            if isinstance(doc_text, str) and doc_text.strip():
                items.append(
                    SearchResultItem(
                        url=u,
                        title=f"[Direct Resolution Source] {u}",
                        snippet=self.auditor.audit_snippet(doc_text[:800], TemporalBadge.CURRENT),
                        published_date=None,
                        source_domain=u,
                        engine="jina_reader",
                        temporal_badge=TemporalBadge.CURRENT,
                    )
                )
                urls.append(u)

        return items, list(dict.fromkeys(urls))

    # ---------------------------------------------------- Layer 3: Gap Analysis
    async def _check_gaps(self, q: QuestionText, items: List[SearchResultItem], today: str) -> GapAnalysisResult:
        snippets = "\n".join(f"- [{it.engine}] {it.title}: {it.snippet[:250]}" for it in items[:10])
        prompt = f"""Today is {today}. Audit these research findings for the forecasting question: "{q.title}".
Resolution criteria: {q.resolution_criteria or 'None'}

Findings:
{snippets}

Determine if critical information is missing to resolve this forecast:
1. Is the current (2026) status missing or only supported by stale (<=2025) data?
2. Is the exact date of the next catalyst, release, or decision unknown?
3. If critical gaps exist, generate up to 2 sharp, surgical search queries to resolve them.

Respond in JSON ONLY:
{{
  "has_critical_gaps": true/false,
  "reasoning": "brief explanation",
  "follow_up_queries": ["query 1", "query 2"]
}}"""
        try:
            _, resp = await self.llm_pool.ask(
                prompt,
                preferred=[
                    "gemini/gemini-3.5-flash-lite",
                    "gemini/gemini-3.1-flash-lite",
                    "gemini/gemma-4-31b-it",
                    "groq/qwen/qwen3.8-27b",
                ],
            )
            data = _extract_json_dict(resp)
            return GapAnalysisResult(
                has_critical_gaps=bool(data.get("has_critical_gaps", False)),
                reasoning=data.get("reasoning", ""),
                follow_up_queries=data.get("follow_up_queries", []),
            )
        except Exception:
            return GapAnalysisResult(has_critical_gaps=False, reasoning="")

    async def _dispatch_follow_ups(self, queries: Sequence[str], q: QuestionText) -> List[SearchResultItem]:
        tasks = [
            self._bounded_search(query=fq, max_results=3, topic="news")
            for fq in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: List[SearchResultItem] = []
        for r in results:
            if isinstance(r, list):
                for item in r:
                    badge = self.auditor.classify_date(item.published_date)
                    items.append(
                        SearchResultItem(
                            url=item.url,
                            title=f"[Follow-up] {item.title}",
                            snippet=self.auditor.audit_snippet(item.snippet, badge),
                            published_date=item.published_date,
                            source_domain=item.source_domain,
                            engine=item.engine,
                            temporal_badge=badge,
                        )
                    )
        return items

    # ---------------------------------------------------- Layer 4: Synthesis
    async def _synthesize_dossier(self, q: QuestionText, items: List[SearchResultItem], urls: List[str], today: str) -> str:
        evidence = "\n\n".join(
            f"Engine: {it.engine} | Source: {it.url}\nDate: {it.published_date or 'Undated'}\nSnippet: {it.snippet}"
            for it in items[:15]
        )
        prompt = f"""You are the Lead Intelligence Officer for a superforecasting tournament team. Today is {today}.
Synthesize the verified evidence into a Structured Research Dossier for this question:
Question: {q.title}
Resolution criteria: {q.resolution_criteria or 'None'}
Fine print: {q.fine_print or 'None'}

Evidence:
{evidence}

Format in Markdown with these exact section headers:
### 1. Current Ground Truth & Latest Reading (2026)
(State confirmed current status/metric with date and source. If no 2026 update exists, explicitly warn "NO RECENT 2026 CONFIRMATION".)

### 2. Base Rate & Historical Reference Classes
(Quantified historical frequencies or comparable historical precedents.)

### 3. Official Source Calendar & Upcoming Catalysts
(Dated calendar of scheduled releases, meetings, or votes before resolution.)

### 4. Disconfirming Scenarios & Fine-Print Traps
(Counter-arguments, edge-case clauses, or measurement quirks.)

Do not give a probability forecast. Keep it dense, objective, and factual."""

        try:
            _, dossier_md = await self.llm_pool.ask(
                prompt,
                preferred=[
                    "gemini/gemini-2.5-flash",
                    "gemini/gemma-4-31b-it",
                    "groq/qwen/qwen3.8-27b",
                    "openrouter/qwen/qwen3.8-27b:free",
                    "gemini/gemini-3.5-flash-lite",
                ],
            )
            return dossier_md
        except Exception as e:
            return f"Dossier synthesis fallback ({type(e).__name__}):\n\n" + evidence[:2000]
