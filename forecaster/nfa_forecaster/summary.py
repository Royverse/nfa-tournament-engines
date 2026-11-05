"""The run report shown in the GitHub Actions summary and printed to the log.

Transparency is the point: every question the run touched is listed with its
final forecast, cost and time, and every failure with its reason — a question
that silently went unforecast is money silently lost.

With redact_forecasts, the forecast values and reasoning are left out. That is
for a live run whose logs are public: the app's rule is that the past is
public and live positions are not, and the bot's comment on Metaculus shows
the reasoning once the question closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

EXPLANATION_CHARS = 6000


@dataclass(frozen=True)
class QuestionOutcome:
    title: str
    url: str | None
    kind: str
    forecast: str | None
    cost_usd: float | None
    minutes: float | None
    error: str | None
    explanation: str | None = None


def _cell(text: str | None) -> str:
    if not text:
        return "—"
    return text.replace("|", "\\|").replace("\n", "; ").strip()


def _link(outcome: QuestionOutcome) -> str:
    title = _cell(outcome.title)[:90]
    return f"[{title}]({outcome.url})" if outcome.url else title


def render_summary(
    heading: str,
    config_lines: Sequence[str],
    outcomes: Sequence[QuestionOutcome],
    notes: Sequence[str] = (),
    free_tier: bool = False,
    redact_forecasts: bool = False,
) -> str:
    done = [o for o in outcomes if o.error is None]
    failed = [o for o in outcomes if o.error is not None]
    total_cost = sum(o.cost_usd or 0 for o in done)
    # litellm prices every call at list price; a free tier charges nothing.
    cost = f"${total_cost:.2f} at list price, $0 charged (free tiers)" if free_tier else f"${total_cost:.2f}"

    lines = [f"## {heading}", ""]
    lines.append(f"**{len(done)} forecast · {len(failed)} failed** · estimated AI cost {cost}")
    lines += [f"- {n}" for n in notes]
    if redact_forecasts:
        lines.append("- forecasts and reasoning withheld from this public log until each question closes")
    lines += ["", "<details><summary>Configuration</summary>", ""]
    lines += [f"- {c}" for c in config_lines]
    lines += ["", "</details>", ""]

    if outcomes:
        lines += ["| Question | Type | Forecast | Cost | Minutes |", "|---|---|---|---|---|"]
        for o in outcomes:
            if o.error:
                forecast = f"❌ {_cell(o.error)[:160]}"
            else:
                forecast = "posted (withheld)" if redact_forecasts else _cell(o.forecast)[:160]
            cost = f"${o.cost_usd:.3f}" if o.cost_usd is not None else "—"
            minutes = f"{o.minutes:.1f}" if o.minutes is not None else "—"
            lines.append(f"| {_link(o)} | {o.kind} | {forecast} | {cost} | {minutes} |")
        lines.append("")
    else:
        lines += ["No questions needed forecasting this run.", ""]

    for o in [] if redact_forecasts else done:
        if o.explanation:
            text = o.explanation[:EXPLANATION_CHARS]
            if len(o.explanation) > EXPLANATION_CHARS:
                text += "\n\n…(truncated — the full reasoning is in the bot's comment on Metaculus)"
            lines += [f"<details><summary>Reasoning — {_cell(o.title)[:90]}</summary>", "", text, "", "</details>", ""]
    return "\n".join(lines)
