"""The forecasting bot: forecasting-tools' ForecastBot with a multi-model ensemble.

The library drives the per-question flow; this class fills in each step:

  research   AskNews, Google-grounded web search, similar resolved Metaculus
             questions, prediction markets and the data desk, in parallel
  forecast   one prediction per ensemble slot; each slot starts at its own
             model and falls back through the pool if that model is out
  aggregate  the library's median (binary), mean (multiple choice), median CDF (numeric)
  publish    the forecast plus a private comment carrying every member's reasoning

Every member's forecast is capped (binary) or floored (multiple choice) before
aggregation, so the aggregate is too — see calibration.py for why.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Mapping, MutableMapping

from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    structure_output,
)

from nfa_forecaster import calibration, prompts
from nfa_forecaster.config import ForecasterConfig, free_tier_rpm, is_openrouter_free
from nfa_forecaster.data_desk import data_desk_research
from nfa_forecaster.markets import market_research
from nfa_forecaster.models import AllModelsFailed, ModelPool, bench_reason
from nfa_forecaster.pacing import pacer_for
from nfa_forecaster.parsing import parse_binary, parse_multiple_choice, parse_percentiles
from nfa_forecaster.precedents import precedent_research
from nfa_forecaster.research import compose_research, news_query
from nfa_forecaster.web_search import GeminiResearch, WebSearchFailed, source_urls
from nfa_forecaster.research_harness import AgenticResearchHarness

logger = logging.getLogger(__name__)
# Per-model forecast values. A separate logger, so a run whose logs are public
# can mute it (cli.REDACT_ENV) without losing the pool's warnings.
forecast_log = logging.getLogger("nfa_forecaster.forecasts")

# (ensemble slot, question key) of the current prediction task. Each
# prediction runs in its own asyncio task, so a ContextVar is task-local.
_SLOT: ContextVar[tuple[int, int]] = ContextVar("nfa_slot", default=(0, 0))

# Every model in the pool is a reasoning model, so none gets a temperature:
# each provider's default applies. A low temperature hurts all three families
# we use. Gemini 3 warns that below 1.0 it "can cause infinite loops, degraded
# reasoning performance" (it printed that on every call of the 2026-09-26
# runs). Anthropic rejects any temperature but 1 once extended thinking is on,
# which reasoning_effort turns on. OpenAI reasoning models accept only their
# default.
MEMBER_TEMPERATURE = None
MEMBER_TIMEOUT_SECONDS = 180
# Tries per model before the pool moves on. Low on purpose: the next model is
# a better bet than a third wait on a model answering "503 high demand".
MEMBER_TRIES = 2
MAX_PARSER_ATTEMPTS = 3

# GeneralLlm.to_dict() dumps every litellm kwarg, and for Metaculus-proxy
# models ("metaculus/...") those include the bot token as api_key and in an
# Authorization header. The library puts that dict in run configs and, with
# extra metadata on, in the Metaculus comment — which goes public when the
# question closes. So: extra metadata stays off, and the dict is redacted.
_SECRET_LLM_FIELDS = frozenset({"api_key", "extra_headers", "base_url", "api_base"})

_MC_PARSE_INSTRUCTIONS = """Make sure that all option names are one of the following:
{options}

The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability."""

_NUMERIC_PARSE_INSTRUCTIONS = """The text given to you is trying to give a forecast distribution for a numeric question.
- This text is trying to answer the numeric question: "{title}".
- When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
- The units for the forecast are: {units}
- As an example, someone else guessed that the answer will be between {lower} {units} and {upper} {units}, so the numbers parsed from an answer like this would be verbatim "{lower}" and "{upper}".
- If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
- If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
- Turn any values that are in scientific notation into regular numbers."""

_DATE_PARSE_INSTRUCTIONS = """The text given to you is trying to give a forecast distribution for a date question.
- This text is trying to answer the question: "{title}".
- As an example, someone else guessed that the answer will be between {lower} and {upper}, so the numbers parsed from an answer like this would be verbatim "{lower}" and "{upper}".
- The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
- If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text."""


class PacedLlm(GeneralLlm):
    """A GeneralLlm whose every API call waits for its model's pacing slot.

    `_mockable_direct_call_to_model` is the one method every real request
    passes through — plain invokes, structured parsing, and each retry — so
    pacing it covers them all. Sharing `benched` lets concurrent tasks and
    internal retries bail out before reserving a pacer slot on a model that
    just exhausted its daily quota.
    """

    def __init__(
        self,
        model: str,
        *,
        requests_per_minute: float,
        benched: MutableMapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._pacer = pacer_for(model, requests_per_minute)
        self._benched = benched

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        if self._benched is not None and self.model in self._benched:
            raise RuntimeError(f"{self.model} is benched ({self._benched[self.model]})")
        await self._pacer.wait()
        if self._benched is not None and self.model in self._benched:
            raise RuntimeError(f"{self.model} is benched ({self._benched[self.model]})")
        try:
            return await super()._mockable_direct_call_to_model(prompt)
        except Exception as e:
            if self._benched is not None:
                reason = bench_reason(e)
                if reason:
                    self._benched.setdefault(self.model, reason)
            raise


def make_llm(
    model: str,
    config: ForecasterConfig,
    env: Mapping[str, str],
    *,
    benched: MutableMapping[str, str] | None = None,
    **kwargs: Any,
) -> GeneralLlm:
    """Free-tier models are paced; paid models get the configured reasoning effort.

    OpenRouter ":free" models run on OPENROUTER_FREE_KEY (the trading app's
    OpenRouter key). A ":free" model never bills, and that key is handed to
    ":free" models only, so the forecaster cannot spend the trading app's money.
    """
    if is_openrouter_free(model):
        kwargs["api_key"] = env.get("OPENROUTER_FREE_KEY") or None
    rpm = free_tier_rpm(model, config.gemini_rpm)
    if rpm is not None:
        return PacedLlm(model, requests_per_minute=rpm, benched=benched, **kwargs)
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    return GeneralLlm(model=model, **kwargs)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def question_text(question: MetaculusQuestion) -> prompts.QuestionText:
    return prompts.QuestionText(
        title=question.question_text,
        background=question.background_info,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
        close_time=_iso(question.close_time),
        resolution_time=_iso(question.scheduled_resolution_time),
        open_time=_iso(question.open_time),
        forecasters=question.num_forecasters,
    )


def _target_date(question: MetaculusQuestion) -> date | None:
    when = question.scheduled_resolution_time or question.close_time
    return when.date() if when else None


def _signed(model: str, wanted: str, reasoning: str) -> str:
    stand_in = f" (standing in for `{wanted}`)" if model != wanted else ""
    return f"*Ensemble member: `{model}`{stand_in}*\n\n{reasoning}"


def bound_messages(question: NumericQuestion | DateQuestion) -> tuple[str, str]:
    """(lower, upper) bound sentences, as in Metaculus's template."""
    if isinstance(question, DateQuestion):
        upper: Any = question.upper_bound.date().isoformat()
        lower: Any = question.lower_bound.date().isoformat()
        unit = ""
    else:
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        unit = question.unit_of_measure or ""
    if question.open_upper_bound:
        upper_message = f"The question creator thinks the number is likely not higher than {upper} {unit}."
    else:
        upper_message = f"The outcome can not be higher than {upper} {unit}."
    if question.open_lower_bound:
        lower_message = f"The question creator thinks the number is likely not lower than {lower} {unit}."
    else:
        lower_message = f"The outcome can not be lower than {lower} {unit}."
    return lower_message, upper_message


class NfaForecaster(ForecastBot):
    def __init__(
        self,
        config: ForecasterConfig,
        *,
        publish: bool,
        skip_previously_forecasted: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> None:
        reason = config.not_runnable_reason()
        if reason:
            raise ValueError(reason)
        self.config = config
        self._env: Mapping[str, str] = env or {}
        gemini_key = self._env.get("GEMINI_API_KEY")
        # One per bot, so a search model spent on one question is skipped for the rest.
        self._gemini_research = GeminiResearch(gemini_key) if config.use_web_search and gemini_key else None
        self._agentic_harness = None  # initialized after self._pool
        models = dict.fromkeys([*config.forecast_models, *config.reserve_models])
        shared_benched: dict[str, str] = {}
        self._pool = ModelPool(
            {
                m: make_llm(
                    m,
                    config,
                    self._env,
                    benched=shared_benched,
                    temperature=MEMBER_TEMPERATURE,
                    timeout=MEMBER_TIMEOUT_SECONDS,
                    allowed_tries=MEMBER_TRIES,
                )
                for m in models
            },
            benched=shared_benched,
        )
        # Parsing is a fallback (answers are parsed directly first), and any
        # pool model can do it: the parser first, then the reserve's weakest
        # models first, so strong models' quota is kept for forecasting.
        parser_chain = dict.fromkeys(
            [config.parser_model or "", *reversed(config.reserve_models), *reversed(config.forecast_models)]
        )
        self._parsers = {
            m: make_llm(
                m,
                config,
                self._env,
                benched=shared_benched,
                temperature=MEMBER_TEMPERATURE,
                allowed_tries=MEMBER_TRIES,
            )
            for m in parser_chain
            if m
        }
        if config.use_web_search:
            self._agentic_harness = AgenticResearchHarness(self._pool, self._env)
        self._slots: dict[int, int] = {}
        self._used: dict[int, set[str]] = {}  # per question: models that answered
        self._limiter: asyncio.Semaphore | None = None
        self._limiter_loop: asyncio.AbstractEventLoop | None = None
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=len(config.forecast_models),
            use_research_summary_to_forecast=False,
            enable_summarize_research=False,
            publish_reports_to_metaculus=publish,
            skip_previously_forecasted_questions=skip_previously_forecasted,
            extra_metadata_in_explanation=False,
            required_successful_predictions=0.5,
            llms={
                "default": self._pool.llms[config.forecast_models[0]],
                "parser": next(iter(self._parsers.values())),
                "researcher": (
                    GeneralLlm(model=config.research_model, temperature=MEMBER_TEMPERATURE, allowed_tries=2)
                    if config.research_model
                    else None
                ),
                "summarizer": None,
            },
        )

    def make_llm_dict(self) -> dict[str, Any]:
        """The library's model summary, with credentials redacted (see _SECRET_LLM_FIELDS)."""
        redacted: dict[str, Any] = {}
        for purpose, value in super().make_llm_dict().items():
            if isinstance(value, dict):
                value = {k: ("[redacted]" if k in _SECRET_LLM_FIELDS else v) for k, v in value.items()}
            redacted[purpose] = value
        return redacted

    # ------------------------------------------------------------ concurrency

    def _question_limiter(self) -> asyncio.Semaphore:
        # A semaphore binds to the event loop it is first used in; the CLI may
        # run more than one loop, so make one per loop.
        loop = asyncio.get_running_loop()
        if self._limiter is None or self._limiter_loop is not loop:
            self._limiter_loop = loop
            self._limiter = asyncio.Semaphore(self.config.max_concurrent_questions)
        return self._limiter

    async def _run_individual_question(self, question: MetaculusQuestion):  # type: ignore[override]
        async with self._question_limiter():
            return await super()._run_individual_question(question)

    # ------------------------------------------------------- ensemble routing

    def preferred_for_slot(self, slot: int) -> list[str]:
        """The slot's own member first, then the other members from there on."""
        members = list(self.config.forecast_models)
        start = slot % len(members)
        return members[start:] + members[:start]

    async def _make_prediction(self, question: MetaculusQuestion, research: str):  # type: ignore[override]
        # The library asks for `predictions_per_research_report` predictions per
        # question, concurrently. Numbering them per question (per notepad) and
        # rotating through the ensemble gives each member exactly one of them.
        # The increment has no await between read and write, so it is atomic
        # under asyncio.
        notepad = await self._get_notepad(question)
        key = id(notepad)
        slot = self._slots.get(key, 0)
        self._slots[key] = slot + 1
        token = _SLOT.set((slot, key))
        try:
            return await super()._make_prediction(question, research)
        finally:
            _SLOT.reset(token)

    def fallback_order(self, own: str, chain: list[str], used: set[str]) -> list[str]:
        """The slot's own model, then members before the reserve (quality) and,
        within each, models this question has not heard from yet (diversity)."""
        members = set(self.config.forecast_models)
        rest = [m for m in chain if m != own]
        return [own, *sorted(rest, key=lambda m: (m not in members, m in used))]

    async def _ask(self, prompt: str) -> tuple[str, str, str]:
        """(model that answered, the model this slot wanted, answer)."""
        slot, question_key = _SLOT.get()
        preferred = self.preferred_for_slot(slot)
        used = self._used.setdefault(question_key, set())
        order = self.fallback_order(preferred[0], self._pool.chain(preferred), used)
        model, answer = await self._pool.ask(prompt, order)
        used.add(model)
        return model, preferred[0], answer

    async def _structure(self, text: str, output_type: Any, **kwargs: Any) -> Any:
        """forecasting-tools' structure_output, down the parser chain.

        Shares the pool's bench: a model out of quota for forecasting is out
        for parsing too, since both spend the same quota. Stops after
        MAX_PARSER_ATTEMPTS: when three parsers fail, the text itself usually
        has no parseable answer, and more tries would only burn quota.
        """
        failures: list[str] = []
        for model, llm in self._parsers.items():
            if model in self._pool.benched:
                continue
            if len(failures) >= MAX_PARSER_ATTEMPTS:
                break
            try:
                return await structure_output(
                    text_to_structure=text,
                    output_type=output_type,
                    model=llm,
                    num_validation_samples=self.config.parser_samples,
                    **kwargs,
                )
            except Exception as e:  # recorded, then the next parser is tried
                reason = bench_reason(e)
                if reason:
                    self._pool.benched.setdefault(model, reason)
                failures.append(f"{model}: {type(e).__name__}")
        raise AllModelsFailed("no parser answered: " + ("; ".join(failures) or "every parser is benched"))

    def model_report(self) -> list[str]:
        return self._pool.report()

    # --------------------------------------------------------------- research

    async def run_research(self, question: MetaculusQuestion) -> str:
        qt = question_text(question)
        jobs: dict[str, Awaitable[str]] = {}
        if self.config.use_asknews:
            jobs["News (AskNews)"] = AskNewsSearcher().get_formatted_news_async(
                news_query(question.question_text, question.resolution_criteria)
            )
        if self._agentic_harness is not None and (self.config.use_exa or self.config.use_tavily):
            today = _today()
            jobs["Deep Research Dossier (Exa / Tavily / Grounded)"] = self._agentic_harness.execute_research(qt, today)
        if self._gemini_research is not None:
            today = _today()
            if not (self.config.use_exa or self.config.use_tavily):
                jobs["Web search (Google, with follow-ups)"] = asyncio.to_thread(
                    self._gemini_research.deep_search,
                    prompts.web_research_prompt(qt, today),
                    question.question_text,
                    question.resolution_criteria,
                    today,
                )
            urls = source_urls(question.resolution_criteria, question.fine_print, question.background_info)
            if urls:
                jobs["Resolution sources (read directly)"] = asyncio.to_thread(
                    self._gemini_research.read_sources, urls, question.question_text, question.resolution_criteria, today
                )
        if self.config.research_model:
            jobs["Web research"] = self.get_llm("researcher", "llm").invoke(prompts.web_research_prompt(qt, _today()))
        if self.config.use_precedents and self._env.get("METACULUS_TOKEN") and question.id_of_post:
            jobs["Similar resolved questions (Metaculus)"] = asyncio.to_thread(
                precedent_research, question.id_of_post, question.question_text, self._env["METACULUS_TOKEN"]
            )
        if self.config.use_markets:
            jobs["Prediction markets"] = asyncio.to_thread(market_research, question.question_text)
        if self.config.use_data_desk:
            jobs["Statistical baseline (data desk)"] = asyncio.to_thread(
                data_desk_research,
                [question.question_text, question.resolution_criteria, question.fine_print, question.background_info],
                _target_date(question),
                alpha_vantage_key=self._env.get("ALPHA_VANTAGE_API_KEY") or None,
            )
        results = await asyncio.gather(*jobs.values(), return_exceptions=True)
        for title, result in zip(jobs, results):
            if isinstance(result, BaseException):
                # WebSearchFailed text is built from model names and HTTP codes only, so it is
                # safe to log; any other error is logged by type, since its text may carry URLs.
                detail = f": {result}" if isinstance(result, WebSearchFailed) else ""
                logger.warning(
                    f"Research source '{title}' failed for {question.page_url}: {type(result).__name__}{detail}"
                )
        return compose_research(zip(jobs, results))

    # ---------------------------------------------------------------- binary

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        slot, _ = _SLOT.get()
        model, wanted, reasoning = await self._ask(
            prompts.binary_prompt(question_text(question), research, _today(), slot=slot)
        )
        raw = parse_binary(reasoning)
        if raw is None:
            parsed: BinaryPrediction = await self._structure(reasoning, BinaryPrediction)
            raw = parsed.prediction_in_decimal
        probability = calibration.cap_probability(raw, self.config.binary_floor, self.config.binary_ceiling)
        forecast_log.info(f"{question.page_url} · {model} → {probability:.3f}")
        return ReasonedPrediction(prediction_value=probability, reasoning=_signed(model, wanted, reasoning))

    # ------------------------------------------------------- multiple choice

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        slot, _ = _SLOT.get()
        options = list(question.options)
        model, wanted, reasoning = await self._ask(
            prompts.multiple_choice_prompt(question_text(question), options, research, _today(), slot=slot)
        )
        direct = parse_multiple_choice(reasoning, options)
        if direct is None:
            parsed: PredictedOptionList = await self._structure(
                reasoning,
                PredictedOptionList,
                additional_instructions=_MC_PARSE_INSTRUCTIONS.format(options=options),
            )
            direct = {o.option_name: o.probability for o in parsed.predicted_options}
        aligned = calibration.align_options(direct, options)
        floor = min(self.config.mc_floor, 0.5 / len(options))
        floored = calibration.floor_and_normalise(aligned, floor)
        prediction = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=name, probability=p) for name, p in floored.items()]
        )
        forecast_log.info(f"{question.page_url} · {model} → {floored}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=_signed(model, wanted, reasoning))

    # --------------------------------------------------------------- numeric

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        slot, _ = _SLOT.get()
        lower_message, upper_message = bound_messages(question)
        model, wanted, reasoning = await self._ask(
            prompts.numeric_prompt(
                question_text(question),
                research,
                _today(),
                question.unit_of_measure,
                lower_message,
                upper_message,
                slot=slot,
            )
        )
        direct = parse_percentiles(reasoning, prompts.NUMERIC_PERCENTILES)
        if direct is not None:
            raw_pairs = [(p, v) for p, v in direct]
        else:
            parsed_pcts = await self._structure(
                reasoning,
                list[Percentile],
                additional_instructions=_NUMERIC_PARSE_INSTRUCTIONS.format(
                    title=question.question_text,
                    units=question.unit_of_measure,
                    lower=question.lower_bound,
                    upper=question.upper_bound,
                ),
            )
            raw_pairs = [(p.percentile, p.value) for p in parsed_pcts]

        calibrated_pairs = calibration.calibrate_numeric_percentiles(
            raw_pairs,
            float(question.lower_bound),
            float(question.upper_bound),
            open_lower_bound=bool(getattr(question, "open_lower_bound", True)),
            open_upper_bound=bool(getattr(question, "open_upper_bound", True)),
        )
        percentiles = [Percentile(percentile=p, value=v) for p, v in calibrated_pairs]
        prediction = NumericDistribution.from_question(percentiles, question)
        forecast_log.info(f"{question.page_url} · {model} → {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=_signed(model, wanted, reasoning))

    # ------------------------------------------------------------------ date

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        slot, _ = _SLOT.get()
        lower_message, upper_message = bound_messages(question)
        model, wanted, reasoning = await self._ask(
            prompts.date_prompt(
                question_text(question), research, _today(), lower_bound_message=lower_message, upper_bound_message=upper_message, slot=slot
            )
        )
        date_percentiles: list[DatePercentile] = await self._structure(
            reasoning,
            list[DatePercentile],
            additional_instructions=_DATE_PARSE_INSTRUCTIONS.format(
                title=question.question_text, lower=question.lower_bound, upper=question.upper_bound
            ),
        )
        raw_pairs = [(p.percentile, p.value.timestamp()) for p in date_percentiles]
        calibrated_pairs = calibration.calibrate_numeric_percentiles(
            raw_pairs,
            float(question.lower_bound.timestamp()),
            float(question.upper_bound.timestamp()),
            open_lower_bound=bool(getattr(question, "open_lower_bound", True)),
            open_upper_bound=bool(getattr(question, "open_upper_bound", True)),
        )
        percentiles = [Percentile(percentile=p, value=v) for p, v in calibrated_pairs]
        prediction = NumericDistribution.from_question(percentiles, question)
        forecast_log.info(f"{question.page_url} · {model} → {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=_signed(model, wanted, reasoning))
