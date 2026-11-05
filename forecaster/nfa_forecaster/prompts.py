"""Forecasting prompts.

Built on Metaculus's template prompts, changed where the 2025–2026 evidence
says it matters:

- Outside view first. Looking up base rates and similar resolved questions
  was done by 34% of Fall 2025 winners and 0% of non-winners.
- Crowd check. Public forecasts from other platforms are allowed, and a
  liquid market on the same question is the strongest single anchor.
- Status quo weight, kept from the template: most short-window questions
  resolve toward "nothing changed".
- No "update like a Bayesian" framing — prompts built on it underperformed.
- Wider numeric tails (P5/P95 asked for explicitly): numeric errors came from
  overconfident intervals.

Prompt text is plain data so it can be read and diffed in tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionText:
    title: str
    background: str | None
    resolution_criteria: str | None
    fine_print: str | None
    close_time: str | None
    resolution_time: str | None
    open_time: str | None = None
    forecasters: int | None = None


def _status(q: QuestionText) -> str:
    """The question's own facts on Metaculus. Some questions are about Metaculus
    itself: the first practice question asked for "new forecasters per day",
    and the models, not given these numbers, said they could not see them."""
    facts = []
    if q.open_time:
        facts.append(f"it opened {q.open_time}")
    if q.forecasters is not None:
        facts.append(f"{q.forecasters} forecasters have forecast it so far")
    return f"\nOn Metaculus, {' and '.join(facts)}." if facts else ""


def _context(q: QuestionText, research: str, today: str) -> str:
    return f"""Question:
{q.title}

Background:
{q.background or "None given."}

Resolution criteria (these have NOT been met yet):
{q.resolution_criteria or "None given."}

Fine print:
{q.fine_print or "None given."}

Today is {today}. Forecasts close {q.close_time or "unknown"}; the question resolves {q.resolution_time or "unknown"}.{_status(q)}

Research from your assistants (web search with follow-ups, the question's own resolution sources, prediction markets, statistical baselines):
{research or "No research was available this run."}"""


_SHARED_STEPS = """1. Time left: how long until the outcome is known.
2. Status quo: what happens if nothing changes? The world usually changes slowly, and most questions resolve toward the status quo (~70% of short-horizon binary questions resolve No).
3. Outside view: name a reference class of similar situations and estimate how often each outcome happens over a comparable window. Give numbers. If the research has a statistical baseline for this data series, it IS your outside view; if it lists similar resolved Metaculus questions that truly match, their outcomes are your reference class.
4. Crowd check: if a prediction market in the research asks essentially the same question (same event, deadline and resolution rules), state its number and treat it as a strong anchor. If it differs in deadline or definition, say how and adjust, or ignore it.
5. Inside view: the specific evidence for and against, and how far each piece moves you from the outside view. Note the date of each key fact and prefer the newest. An announcement, plan or target date is not the event: announced things often slip past their deadline."""

# Five orthogonal cognitive lenses rotated across the 5 ensemble slots.
# Sending identical prompts to every slot causes ensemble collapse; assigning
# complementary analytical priorities (Outside-View Base Rate, Inside-View
# Catalyst, Red-Team Fine-Print Auditor, Market & Econometric Synthesizer,
# Superforecaster Integrator) produces genuinely decorrelated reasoning paths.
SLOT_PERSPECTIVES: tuple[str, ...] = (
    "Perspective for this forecast — OUTSIDE-VIEW & BASE-RATE ANCHOR: Weight historical base rates, reference classes, and status-quo persistence heavily. In short windows (<60 days), dramatic departures from the status quo happen far less often than headlines suggest (~70% of short-window binary event questions resolve No). Move away from the historical base rate only when concrete, dated, already-scheduled mechanisms exist.",
    "Perspective for this forecast — INSIDE-VIEW & CATALYST ANALYST: Map the concrete causal chain, decision-makers' incentives, and exact calendar of upcoming catalysts between today and the close/resolution date. Distinguish strictly between an announcement/intention and the completed resolution event.",
    "Perspective for this forecast — RED-TEAM & FINE-PRINT AUDITOR: Scrutinise the exact resolution criteria, data source definitions, reporting lags, units, and edge-case clauses. Identify the #1 way a naive forecaster could misread the rules, units, or current status of the resolution metric, and stress-test the consensus against that trap.",
    "Perspective for this forecast — MARKET & ECONOMETRIC SYNTHESIZER: Prioritise quantitative anchors—liquid prediction market odds (Polymarket, Kalshi, Manifold), statistical baselines from the data desk (FRED, Yahoo Finance empirical quantiles), and latest numerical readings from the resolution source—adjusting strictly for any difference in deadline or threshold.",
    "Perspective for this forecast —CALIBRATED SUPERFORECASTER INTEGRATOR: Synthesise the outside view, inside view, market odds, and tail risks into a well-calibrated distribution. Guard actively against overconfidence and narrative availability bias.",
)


def perspective_for_slot(slot: int | None) -> str:
    """Return the slot-specific perspective block, or empty string when slot is None."""
    if slot is None:
        return ""
    lens = SLOT_PERSPECTIVES[slot % len(SLOT_PERSPECTIVES)]
    return f"\n{lens}\n"


def binary_prompt(q: QuestionText, research: str, today: str, slot: int | None = None) -> str:
    return f"""You are a professional forecaster. Your forecasts are scored with a log score, which punishes confident mistakes very heavily.
{perspective_for_slot(slot)}
{_context(q, research, today)}

Work through these steps, writing each one out:
{_SHARED_STEPS}
6. Scenarios: one brief, plausible path to Yes and one to No.
7. Calibration: forecasts below 5% or above 95% need overwhelming evidence; leave room for the unknown.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100"""


def multiple_choice_prompt(
    q: QuestionText, options: list[str], research: str, today: str, slot: int | None = None
) -> str:
    return f"""You are a professional forecaster. Your forecasts are scored with a log score, which punishes a near-zero probability on the option that happens very heavily.
{perspective_for_slot(slot)}
{_context(q, research, today)}

The options are: {options}

Work through these steps, writing each one out:
{_SHARED_STEPS}
6. Surprises: describe one scenario that produces an unexpected option.
7. Calibration: leave some probability on every option; nothing gets less than 1% without overwhelming evidence.

The last thing you write is your final probabilities for the {len(options)} options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N"""


NUMERIC_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def _numeric_answer_format(placeholder: str) -> str:
    rows = [f"Percentile {p}: {placeholder}" for p in NUMERIC_PERCENTILES]
    rows[0] += " (lowest)"
    rows[-1] += " (highest)"
    body = "\n".join(rows)
    return f'The last thing you write is your final answer as:\n"\n{body}\n"'


def numeric_prompt(
    q: QuestionText,
    research: str,
    today: str,
    units: str | None,
    lower_bound_message: str,
    upper_bound_message: str,
    slot: int | None = None,
) -> str:
    return f"""You are a professional forecaster. Your distribution is scored with a log score: if the outcome lands where you put almost no probability, the penalty is severe, so your tails must be honest.
{perspective_for_slot(slot)}
{_context(q, research, today)}

Units for the answer: {units or "Not stated (infer them from the question)"}
{lower_bound_message}
{upper_bound_message}

Formatting instructions:
- Give values in the EXACT requested units and scale shown in the bounds above (e.g. if the bounds are 110000000 and 150000000, write 114700000, NEVER 114.7).
- Never use scientific notation.
- Values must strictly increase from percentile 5 to percentile 95, and stay inside the stated closed bounds.

Work through these steps, writing each one out:
{_SHARED_STEPS}
6. Trend: the outcome if the current trend continues.
7. Tails: one unexpected scenario for a low outcome and one for a high outcome. Your 5–95 range should be wide enough that a surprise still lands inside it (raw intuition is usually ~30% too narrow).

{_numeric_answer_format("XX")}"""


def date_prompt(
    q: QuestionText,
    research: str,
    today: str,
    lower_bound_message: str,
    upper_bound_message: str,
    slot: int | None = None,
) -> str:
    return f"""You are a professional forecaster. Your distribution over dates is scored with a log score, so your tails must be honest.
{perspective_for_slot(slot)}
{_context(q, research, today)}

{lower_bound_message}
{upper_bound_message}

Formatting instructions:
- Write every date as YYYY-MM-DD. If the hour matters, use YYYY-MM-DDTHH:MM:SSZ (UTC).
- Dates must be in chronological order from percentile 5 (earliest) to percentile 95 (latest).

Work through these steps, writing each one out:
{_SHARED_STEPS}
6. Trend: when it happens if the current pace continues.
7. Tails: one scenario for a much earlier date and one for a much later date.

{_numeric_answer_format("YYYY-MM-DD")}"""


def web_research_prompt(q: QuestionText, today: str) -> str:
    return f"""You are an assistant to a superforecaster. They will forecast the question below. Today is {today}.
Search for, then produce a concise but detailed rundown of, the most relevant, most recent news and facts, with dates and sources. Say whether the question would resolve Yes or No on current information, and list any scheduled events before the resolution date that could change it. Do not produce a forecast yourself.

Question:
{q.title}

Resolution criteria:
{q.resolution_criteria or "None given."}

Fine print:
{q.fine_print or "None given."}"""
