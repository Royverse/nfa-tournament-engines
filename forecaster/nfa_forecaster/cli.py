"""Command-line entry point.

    python -m nfa_forecaster.cli run     forecast open questions in the tournaments and publish
    python -m nfa_forecaster.cli test    dry run on Metaculus's bot-testing-area (publishes nothing by default)
    python -m nfa_forecaster.cli smoke   check every key and service, forecast nothing

Exit codes: 0 = done (or not configured yet — a skip is not a failure);
1 = something failed and a human should look. A red run is what the
Netlify watchdog and GitHub's own notifications key off.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Sequence

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optional dotenv support for local runs
try:
    from dotenv import load_dotenv

    load_dotenv()
    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_here, "..", ".env"))
    load_dotenv(os.path.join(_here, "..", "..", ".env"))
    load_dotenv(os.path.join(_here, "..", "..", ".env.local"))
except ImportError:
    pass

from nfa_forecaster.config import (
    ConfigError,
    ForecasterConfig,
    apply_repository_variables,
    load_config,
    select_variant_token,
)
from nfa_forecaster.selection import select_questions, split_urgent_questions
from nfa_forecaster.summary import QuestionOutcome, render_summary

logger = logging.getLogger("nfa_forecaster")

BOT_TESTING_AREA = "bot-testing-area"
# Set to "1" where run logs are public (the public engines repo): a live run
# then keeps forecast values and reasoning out of the log and the summary.
REDACT_ENV = "FORECASTER_REDACT_LIVE"
DEFAULT_MAX_QUESTIONS_PER_RUN = 20


def _write_step_summary(markdown: str) -> None:
    print(markdown)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")


def _kind(question: Any) -> str:
    name = type(question).__name__.replace("Question", "").lower()
    return name or "question"


def _outcome_from_report(report: Any) -> QuestionOutcome:
    question = report.question
    try:
        forecast = type(report).make_readable_prediction(report.prediction)
    except Exception:
        forecast = str(report.prediction)[:200]
    return QuestionOutcome(
        title=question.question_text,
        url=question.page_url,
        kind=_kind(question),
        forecast=forecast,
        cost_usd=report.price_estimate,
        minutes=report.minutes_taken,
        error=None,
        explanation=report.explanation,
    )


def _outcome_from_error(error: BaseException, titles_by_url: dict[str, str]) -> QuestionOutcome:
    message = str(error)
    match = re.search(r"https://www\.metaculus\.com/questions/\d+", message)
    url = match.group(0) if match else None
    return QuestionOutcome(
        title=titles_by_url.get(url or "", url or "unknown question"),
        url=url,
        kind="—",
        forecast=None,
        cost_usd=None,
        minutes=None,
        error=f"{type(error).__name__}: {message[:300]}",
    )


def _fetch_open_questions(tournaments: Sequence[str]) -> tuple[list[Any], list[str]]:
    from forecasting_tools import MetaculusClient

    client = MetaculusClient()
    questions: list[Any] = []
    notes: list[str] = []
    for tournament in tournaments:
        try:
            found = client.get_all_open_questions_from_tournament(tournament)
            questions += found
            unforecast = sum(1 for q in found if not q.already_forecasted)
            notes.append(f"tournament `{tournament}`: {len(found)} open, {unforecast} not yet forecast")
        except Exception as e:
            notes.append(f"❌ tournament `{tournament}`: could not list questions ({type(e).__name__})")
    return questions, notes


def _forecast(
    config: ForecasterConfig,
    questions: list[Any],
    *,
    publish: bool,
    skip_forecasted: bool,
    now: datetime | None = None,
) -> tuple[list[Any], list[str]]:
    """(one report or exception per question, the model pool's fallback report)."""
    from nfa_forecaster.bot import NfaForecaster

    bot = NfaForecaster(config, publish=publish, skip_previously_forecasted=skip_forecasted, env=os.environ)
    urgent, later = split_urgent_questions(questions, now or datetime.now(timezone.utc))

    async def _run_batches() -> list[Any]:
        if not urgent or len(questions) <= 1:
            return list(await bot.forecast_questions(questions, return_exceptions=True))
        out: list[Any] = []
        for q in urgent:
            out.extend(await bot.forecast_questions([q], return_exceptions=True))
        if later:
            out.extend(await bot.forecast_questions(later, return_exceptions=True))
        return out

    reports = asyncio.run(_run_batches())
    return reports, bot.model_report()


def _outcomes(reports: list[Any], questions: list[Any]) -> list[QuestionOutcome]:
    titles = {q.page_url: q.question_text for q in questions}
    return [
        _outcome_from_error(r, titles) if isinstance(r, BaseException) else _outcome_from_report(r)
        for r in reports
    ]


def command_run(config: ForecasterConfig, max_questions: int) -> int:
    redact = os.environ.get(REDACT_ENV) == "1"
    if redact:
        logging.getLogger("nfa_forecaster.forecasts").setLevel(logging.WARNING)
    questions, notes = _fetch_open_questions(config.tournaments)
    now = datetime.now(timezone.utc)
    selected, deferred = select_questions(questions, max_questions, now)
    if deferred:
        notes.append(f"{deferred} more unforecast question(s) deferred to the next run (per-run cap {max_questions})")
    reports, model_notes = (
        _forecast(config, selected, publish=True, skip_forecasted=True, now=now) if selected else ([], [])
    )
    outcomes = _outcomes(reports, selected)
    _write_step_summary(
        render_summary(
            "Forecaster run",
            config.describe(),
            outcomes,
            notes + model_notes,
            config.funding == "free",
            redact_forecasts=redact,
        )
    )
    listing_failed = any(n.startswith("❌") for n in notes)
    return 1 if listing_failed or any(o.error for o in outcomes) else 0


def command_test(config: ForecasterConfig, limit: int, publish: bool) -> int:
    questions, notes = _fetch_open_questions([BOT_TESTING_AREA])
    chosen = questions[: max(0, limit)]
    notes.append(
        f"dry run on {len(chosen)} bot-testing-area question(s) — "
        + ("PUBLISHED to the testing area" if publish else "nothing was published")
    )
    reports, model_notes = _forecast(config, chosen, publish=publish, skip_forecasted=False) if chosen else ([], [])
    outcomes = _outcomes(reports, chosen)
    _write_step_summary(
        render_summary("Forecaster test run", config.describe(), outcomes, notes + model_notes, config.funding == "free")
    )
    return 1 if any(o.error for o in outcomes) or any(n.startswith("❌") for n in notes) else 0


def command_smoke(config: ForecasterConfig) -> int:
    from nfa_forecaster.smoke import render_checks, run_checks, smoke_passed

    checks = run_checks(config, os.environ)
    passed = smoke_passed(config, checks)
    verdict = "✅ Ready to forecast." if passed else "❌ Not ready — fix the failing rows above."
    body = "\n".join(
        ["## Forecaster smoke test", "", render_checks(checks), "", verdict, "", "<details><summary>Configuration</summary>", ""]
        + [f"- {line}" for line in config.describe()]
        + ["", "</details>"]
    )
    _write_step_summary(body)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nfa_forecaster", description="nfa. Metaculus forecasting bot")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="forecast open, unforecast tournament questions and publish")
    run.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=f"per-run cap (default: FORECASTER_MAX_QUESTIONS_PER_RUN or {DEFAULT_MAX_QUESTIONS_PER_RUN})",
    )
    test = sub.add_parser("test", help="dry run on the bot-testing-area tournament")
    test.add_argument("--limit", type=int, default=3)
    test.add_argument("--publish", action="store_true", help="also publish to the testing area")
    sub.add_parser("smoke", help="check keys and services without forecasting")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # litellm logs every call at INFO, twice; the pool's own lines say which model answered.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    try:
        # The library reads METACULUS_TOKEN and model settings straight from
        # os.environ, so the variant and repository variables are applied to
        # the process environment before anything else is constructed.
        applied = apply_repository_variables(os.environ, os.environ.get("FORECASTER_VARS_JSON"))
        if applied:
            logger.info(f"Repository variables applied: {', '.join(sorted(applied))}")
        select_variant_token(os.environ)
        config = load_config(os.environ)
    except ConfigError as e:
        _write_step_summary(f"## Forecaster\n\n❌ Configuration error: {e}")
        return 1

    if args.command == "smoke":
        return command_smoke(config)

    reason = config.not_runnable_reason()
    if reason:
        _write_step_summary(f"## Forecaster\n\n⏭️ Not configured yet — skipped. {reason}")
        return 0
    if args.command == "run":
        default_cap = 5 if config.funding == "free" else DEFAULT_MAX_QUESTIONS_PER_RUN
        max_questions = args.max_questions or int(
            os.environ.get("FORECASTER_MAX_QUESTIONS_PER_RUN") or default_cap
        )
        return command_run(config, max_questions)
    return command_test(config, args.limit, args.publish)


if __name__ == "__main__":
    sys.exit(main())
