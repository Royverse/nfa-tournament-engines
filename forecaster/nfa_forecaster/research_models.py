"""Data contracts for the agentic research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TemporalBadge(str, Enum):
    FRESH = "FRESH (<72h)"
    CURRENT = "CURRENT (2026)"
    HISTORICAL = "HISTORICAL (<=2025)"
    UNDATED = "UNDATED"


@dataclass(frozen=True)
class SearchResultItem:
    url: str
    title: str
    snippet: str
    published_date: Optional[str] = None
    source_domain: str = ""
    engine: str = "unknown"
    temporal_badge: TemporalBadge = TemporalBadge.UNDATED


@dataclass
class QueryVectorPlan:
    status_queries: List[str]
    base_rate_queries: List[str]
    official_calendar_queries: List[str]
    red_team_queries: List[str]
    direct_urls: List[str]


@dataclass
class GapAnalysisResult:
    has_critical_gaps: bool
    reasoning: str
    follow_up_queries: List[str] = field(default_factory=list)


@dataclass
class ResearchDossier:
    question_title: str
    created_at: str
    raw_markdown_summary: str
    sources_consulted: List[str]
