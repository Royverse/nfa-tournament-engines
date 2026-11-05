"""Read the final answer out of a forecaster's text without an extra AI call.

Every prompt ends by asking for a fixed format ("Probability: ZZ%",
"Option: P", "Percentile N: X"). When the model followed it, reading the
numbers directly is free and exact. Only when it didn't does the bot fall
back to the AI parser. On the free Gemini tier (20 requests per model per
day, measured 2026-09-26) that halves the daily budget each question costs.

Each function returns None when it is not certain, never a guess.
"""

from __future__ import annotations

import re
from typing import Sequence

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def parse_binary(text: str) -> float | None:
    """The last "Probability: ZZ%" in the text, as a decimal."""
    matches = re.findall(r"Probability\s*[:=]\s*\**\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.I)
    if not matches:
        return None
    value = float(matches[-1])
    return value / 100 if 0 <= value <= 100 else None


def parse_multiple_choice(text: str, options: Sequence[str]) -> dict[str, float] | None:
    """One probability per option from "Option: P" lines; None unless every option is found.

    Accepts decimals (0.35) or percentages (35 / 35%). Values are normalised
    to sum to 1 only if they already sum to roughly 1 (or 100) — a list that
    doesn't add up is a misread, not something to rescale.
    """
    values: dict[str, float] = {}
    for option in options:
        pattern = rf"^[ \t*\-•]*(?:Option[ \t_]*)?{re.escape(option)}[ \t]*\**[ \t]*[:=][ \t]*\**[ \t]*(\d+(?:\.\d+)?)[ \t]*(%?)"
        found = re.findall(pattern, text, flags=re.I | re.M)
        if not found:
            return None
        number, percent_sign = found[-1]
        values[option] = float(number) / (100 if percent_sign else 1)
    total = sum(values.values())
    if 90 <= total <= 110:
        values = {k: v / 100 for k, v in values.items()}
        total = sum(values.values())
    if not 0.9 <= total <= 1.1:
        return None
    return {k: v / total for k, v in values.items()}


def parse_percentiles(text: str, expected: Sequence[int]) -> list[tuple[float, float]] | None:
    """[(0.05, v), (0.10, v), ...] from "Percentile N: value" lines.

    Only plain numbers are accepted (commas and a leading $ are fine). Units
    like "1.2 million" or "5k" are left to the AI parser, which knows how to
    convert them into the question's units. Values must strictly increase.
    """
    found: dict[int, float] = {}
    # [ \t] rather than \s: whitespace must not run onto the next line, or the
    # next line's "Percentile" would be read as this value's unit.
    for pct, raw, tail in re.findall(
        rf"Percentile[ \t]*(\d+)[ \t]*[:=][ \t]*\**[ \t]*\$?[ \t]*({_NUMBER})[ \t]*([A-Za-z%]*)", text, flags=re.I
    ):
        if tail and tail != "%":
            return None
        found[int(pct)] = _to_float(raw)
    if any(p not in found for p in expected):
        return None
    points = [(p / 100, found[p]) for p in expected]
    values = [v for _, v in points]
    if any(b <= a for a, b in zip(values, values[1:])):
        return None
    return points
