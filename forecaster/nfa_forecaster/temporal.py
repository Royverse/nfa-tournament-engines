"""Temporal verification engine ensuring ground truth validity."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from nfa_forecaster.research_models import TemporalBadge


class TemporalAuditor:
    """Verifies publication dates and applies age badges against the current calendar."""

    @staticmethod
    def classify_date(date_str: Optional[str], current_year: Optional[int] = None) -> TemporalBadge:
        if not date_str:
            return TemporalBadge.UNDATED

        now = datetime.now(timezone.utc)
        target_year = current_year if current_year is not None else now.year

        cleaned = str(date_str).strip()
        year_match = re.search(r"\b(202[0-9])\b", cleaned)
        if not year_match:
            return TemporalBadge.UNDATED

        year = int(year_match.group(1))
        if year < target_year:
            return TemporalBadge.HISTORICAL

        try:
            iso_clean = cleaned.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = now - dt
            if 0 <= delta.total_seconds() <= 72 * 3600:
                return TemporalBadge.FRESH
            return TemporalBadge.CURRENT
        except Exception:
            return TemporalBadge.CURRENT

    @staticmethod
    def audit_snippet(text: str, badge: TemporalBadge) -> str:
        """Injects explicit temporal badge prefix to prevent LLM cognitive confusion."""
        return f"[{badge.value}] {text.strip()}"
