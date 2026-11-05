"""Per-model request pacing for rate-limited free tiers.

Google's free Gemini tier allows 5 requests per minute per model (measured in
production 2026-09-26: "Quota exceeded for metric
generate_content_free_tier_requests, limit: 5, model: gemini-3.6-flash").
The first dry run fired every ensemble member and every parse at once and
lost all three questions to 429s. A Pacer spaces calls to one model evenly,
so they queue instead of failing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable


class Pacer:
    """Hands out evenly spaced start times — one call per `interval` seconds."""

    def __init__(self, requests_per_minute: float, clock: Callable[[], float] = time.monotonic) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self._clock = clock
        self._next_slot = 0.0

    def reserve(self) -> float:
        """Claim the next slot; return how long to wait for it (0 if free now).

        No await between reading and advancing the slot, so this is atomic
        under asyncio even with many concurrent callers.
        """
        now = self._clock()
        slot = max(now, self._next_slot)
        self._next_slot = slot + self.interval
        return slot - now

    async def wait(self) -> None:
        delay = self.reserve()
        if delay > 0:
            await asyncio.sleep(delay)


_PACERS: dict[str, Pacer] = {}


def pacer_for(model: str, requests_per_minute: float) -> Pacer:
    """One shared pacer per model id, so every caller of a model shares its budget."""
    pacer = _PACERS.get(model)
    if pacer is None:
        pacer = _PACERS[model] = Pacer(requests_per_minute)
    return pacer
