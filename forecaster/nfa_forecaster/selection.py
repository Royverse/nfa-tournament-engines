"""Which open questions a run works on, and in what order.

Main-tournament questions stay open for only ~1.5 hours, while MiniBench can
release dozens at once. A run therefore takes the unforecast questions that
close soonest first, up to a per-run cap; anything left over is picked up by
the next run (already-forecast questions are always skipped).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Sequence, TypeVar

Q = TypeVar("Q")

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)
URGENT_CLOSE_MINUTES = 30


def _close_time(question: Any) -> datetime:
    value = getattr(question, "close_time", None)
    if value is None:
        return _FAR_FUTURE
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _identity(question: Any) -> Any:
    return getattr(question, "id_of_question", None) or id(question)


def select_questions(questions: Sequence[Q], max_questions: int, now: datetime) -> tuple[list[Q], int]:
    """Return (questions to forecast now, number deferred to a later run)."""
    seen: set[Any] = set()
    candidates: list[Q] = []
    for question in questions:
        key = _identity(question)
        if key in seen or getattr(question, "already_forecasted", False):
            continue
        seen.add(key)
        if _close_time(question) <= now:
            continue
        candidates.append(question)
    candidates.sort(key=_close_time)
    chosen = candidates[: max(0, max_questions)]
    return chosen, len(candidates) - len(chosen)


def split_urgent_questions(
    questions: Sequence[Q], now: datetime, urgent_minutes: int = URGENT_CLOSE_MINUTES
) -> tuple[list[Q], list[Q]]:
    """Split already-ordered questions into (closing within `urgent_minutes`, later).

    On 2026-10-20 (run #39373569596), Q46162 had ~3.5 minutes left when the run
    started, but ran concurrently with non-urgent Q46163, which shared rate-limit
    pacers and finished 18 seconds earlier while Q46162 missed close by 34s.
    Running urgent questions one-by-one ahead of non-urgent ones gives them
    uncontested model pacer slots and immediate publication.
    """
    cutoff = now + timedelta(minutes=urgent_minutes)
    urgent = [q for q in questions if _close_time(q) <= cutoff]
    later = [q for q in questions if _close_time(q) > cutoff]
    return urgent, later
