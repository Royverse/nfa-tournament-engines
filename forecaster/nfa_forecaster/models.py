"""The model pool: every AI call goes down an ordered list of models until one answers.

Free tiers run out mid-day. On 2026-09-26 the first real run lost the
practice question because three of four Gemini models had spent their 20
requests for the day, while other models on the same key, and the Groq key,
sat unused. So no call depends on one model.

  - Each ensemble slot starts at its own model, then falls back through the
    other members, then the reserve pool (see config.py).
  - A model that says it is done for the day (daily quota, no credits, bad
    key, unknown model) is benched for the rest of the run, so later calls
    skip it without spending a request on it.
  - A model that times out is benched too. Each timeout already cost
    MEMBER_TIMEOUT_SECONDS x MEMBER_TRIES of billed runner time (bot.py), and
    a model that hung once tends to hang again: on 2026-11-05 one ":free"
    model stalled a run for 8 minutes before the pool moved on.
  - A model that fails for a moment (503 "high demand", per-minute limit) is
    skipped for this call only.

Every fallback and every benching is recorded, and the run summary reports
them, so a question answered by a reserve model is never passed off as the
planned ensemble.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)

# Errors that mean "this model will not answer again this run". Matched
# against the exception text, which is how litellm surfaces provider errors.
_SPENT_FOR_TODAY = re.compile(
    r"PerDay|per[ _-]?day|\bRPD\b|\bTPD\b|daily|free-models-per-day", re.IGNORECASE
)
# litellm's exception classes for 401 / 403 / 404.
_GONE_TYPES = frozenset({"AuthenticationError", "PermissionDeniedError", "NotFoundError"})
# 402 and a retired model id arrive as generic errors, so match their text.
_GONE_TEXT = re.compile(
    r"\b402\b|insufficient[ _]credits|payment required|no longer available", re.IGNORECASE
)


class Llm(Protocol):
    async def invoke(self, prompt: str) -> str: ...


def bench_reason(error: BaseException) -> str | None:
    """Why a failed model should sit out the rest of the run, or None to keep it."""
    text = f"{type(error).__name__}: {error}"
    rate_limited = "RateLimit" in type(error).__name__ or "429" in text or "RESOURCE_EXHAUSTED" in text
    if rate_limited and _SPENT_FOR_TODAY.search(text):
        return "daily limit reached"
    if type(error).__name__ in _GONE_TYPES or _GONE_TEXT.search(text):
        return f"unavailable ({type(error).__name__})"
    if "Timeout" in type(error).__name__ or isinstance(error, TimeoutError):
        return "timed out"
    return None


class AllModelsFailed(RuntimeError):
    """Every model in the chain failed or was benched."""


@dataclass
class ModelPool:
    """Named models, a shared bench, and a log of what answered instead of what."""

    llms: Mapping[str, Llm]
    benched: dict[str, str] = field(default_factory=dict)
    fallbacks: list[tuple[str, str]] = field(default_factory=list)

    def chain(self, preferred: Sequence[str]) -> list[str]:
        """`preferred` first, then every other pool model in pool order; benched ones dropped."""
        ordered = list(dict.fromkeys([*preferred, *self.llms]))
        return [m for m in ordered if m in self.llms and m not in self.benched]

    async def ask(self, prompt: str, preferred: Sequence[str]) -> tuple[str, str]:
        """(model that answered, answer). Raises AllModelsFailed if none did."""
        wanted = preferred[0] if preferred else None
        failures: list[str] = []
        for model in self.chain(preferred):
            if model in self.benched:
                continue
            try:
                answer = await self.llms[model].invoke(prompt)
            except Exception as e:  # provider errors come in many types; each is recorded
                reason = bench_reason(e)
                if reason:
                    self.benched.setdefault(model, reason)
                failures.append(f"{model}: {type(e).__name__}")
                logger.warning(f"{model} failed ({type(e).__name__}{'; ' + reason if reason else ''}) — trying the next model")
                continue
            if wanted and model != wanted:
                self.fallbacks.append((wanted, model))
            return model, answer
        raise AllModelsFailed("no model answered: " + ("; ".join(failures) or "every model is benched"))

    def report(self) -> list[str]:
        """Plain lines for the run summary."""
        lines = [f"benched for this run: `{m}` — {why}" for m, why in self.benched.items()]
        if self.fallbacks:
            counts: dict[tuple[str, str], int] = {}
            for pair in self.fallbacks:
                counts[pair] = counts.get(pair, 0) + 1
            lines += [f"fallback: `{used}` answered for `{wanted}` ×{n}" for (wanted, used), n in counts.items()]
        return lines
