"""Prompts, research composition, question selection and the run summary."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nfa_forecaster import prompts
from nfa_forecaster.research import compose_research, news_query
from nfa_forecaster.selection import select_questions, split_urgent_questions
from nfa_forecaster.summary import QuestionOutcome, render_summary

QT = prompts.QuestionText(
    title="Will X happen?",
    background="Some background.",
    resolution_criteria="Resolves Yes if X.",
    fine_print=None,
    close_time="2026-10-01T12:00:00+00:00",
    resolution_time="2026-12-31T00:00:00+00:00",
)


def test_binary_prompt_puts_the_outside_view_before_the_inside_view_and_ends_with_the_answer_format():
    text = prompts.binary_prompt(QT, "RESEARCH BLOCK", "2026-09-26")
    assert text.index("Outside view") < text.index("Inside view")
    assert "Crowd check" in text and "RESEARCH BLOCK" in text
    assert text.rstrip().endswith('"Probability: ZZ%", 0-100')
    assert "Bayes" not in text  # prompts framed around Bayesian updating underperformed


def test_numeric_prompt_asks_for_explicit_tails():
    text = prompts.numeric_prompt(QT, "", "2026-09-26", "USD", "The outcome can not be lower than 0 USD.", "")
    for p in prompts.NUMERIC_PERCENTILES:
        assert f"Percentile {p}:" in text
    assert "Percentile 5: XX (lowest)" in text and "Percentile 95: XX (highest)" in text


def test_multiple_choice_prompt_lists_options_in_order():
    text = prompts.multiple_choice_prompt(QT, ["A", "B", "C"], "", "2026-09-26")
    assert "['A', 'B', 'C']" in text and "3 options" in text


def test_missing_fields_are_said_out_loud_not_left_blank():
    text = prompts.binary_prompt(QT, "", "2026-09-26")
    assert "Fine print:\nNone given." in text
    assert "No research was available this run." in text


def test_each_ensemble_slot_gets_a_distinct_cognitive_perspective():
    slot_prompts = [prompts.binary_prompt(QT, "", "2026-09-26", slot=i) for i in range(5)]
    assert len(set(slot_prompts)) == 5
    assert "OUTSIDE-VIEW & BASE-RATE ANCHOR" in slot_prompts[0]
    assert "RED-TEAM & FINE-PRINT AUDITOR" in slot_prompts[2]


def test_news_query_is_bounded():
    assert news_query("T", "C" * 5000).startswith("T\nC")
    assert len(news_query("T", "C" * 5000)) == 1500


def test_compose_research_accounts_for_every_source():
    text = compose_research(
        [("News", "headline"), ("Web", ValueError("secret detail in message")), ("Markets", "   "), ("Data", "")]
    )
    assert "## News\nheadline" in text
    assert "## Web\nUnavailable this run (ValueError)." in text
    assert "secret detail" not in text
    assert "## Markets" not in text


def test_compose_research_with_nothing():
    assert compose_research([]) == "No research source returned anything this run."


def _q(qid, close_in_minutes, forecasted=False):
    now = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    close = None if close_in_minutes is None else now + timedelta(minutes=close_in_minutes)
    return SimpleNamespace(id_of_question=qid, close_time=close, already_forecasted=forecasted)


def test_select_questions_orders_by_soonest_close_and_caps_the_run():
    now = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    qs = [_q(1, 600), _q(2, 60), _q(3, None), _q(4, 30, forecasted=True), _q(5, -5), _q(2, 60)]
    chosen, deferred = select_questions(qs, 2, now)
    assert [q.id_of_question for q in chosen] == [2, 1]
    assert deferred == 1  # question 3 (no close time) waits; 4 is done, 5 is closed, the duplicate 2 is dropped


def test_split_urgent_questions_isolates_closing_soon_candidates():
    now = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
    qs = [_q(1, 4), _q(2, 25), _q(3, 90), _q(4, None)]
    urgent, later = split_urgent_questions(qs, now, urgent_minutes=30)
    assert [q.id_of_question for q in urgent] == [1, 2]
    assert [q.id_of_question for q in later] == [3, 4]


def test_summary_lists_failures_with_reasons_and_totals_cost():
    outcomes = [
        QuestionOutcome("Q one", "https://www.metaculus.com/questions/1", "binary", "12.0%", 0.25, 2.0, None, "# SUMMARY\nwhy"),
        QuestionOutcome("Q | two", None, "numeric", None, None, None, "RuntimeError: parser failed"),
    ]
    text = render_summary("Forecaster run", ["funding: free"], outcomes, ["tournament `minibench`: 2 open"])
    assert "**1 forecast · 1 failed** · estimated AI cost $0.25" in text
    assert "[Q one](https://www.metaculus.com/questions/1)" in text
    assert "❌ RuntimeError: parser failed" in text
    assert "Q \\| two" in text  # pipes escaped so the table survives
    assert "<summary>Reasoning — Q one</summary>" in text


def test_summary_when_nothing_to_do():
    assert "No questions needed forecasting this run." in render_summary("Forecaster run", [], [])


def test_free_tier_runs_do_not_claim_a_cost_they_never_paid():
    outcomes = [QuestionOutcome("Q", None, "binary", "40%", 0.53, 1.0, None)]
    text = render_summary("Forecaster run", [], outcomes, free_tier=True)
    assert "$0.53 at list price, $0 charged (free tiers)" in text


def test_a_redacted_summary_keeps_the_accounting_but_not_the_forecast():
    outcomes = [
        QuestionOutcome("Will X happen?", "https://www.metaculus.com/questions/1", "binary", "37.0%", 0.01, 2.5, None, "secret reasoning"),
        QuestionOutcome("Will Y happen?", None, "binary", None, None, None, "AllModelsFailed"),
    ]
    text = render_summary("Forecaster run", [], outcomes, redact_forecasts=True)
    assert "37.0%" not in text and "secret reasoning" not in text
    assert "posted (withheld)" in text and "AllModelsFailed" in text
    assert "1 forecast · 1 failed" in text
    assert "withheld from this public log" in text

