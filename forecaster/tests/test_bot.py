"""End-to-end tests of the bot against the real forecasting-tools library.

No network: every model is a fake, and the parser is replaced with a
deterministic one. These run in CI (forecaster-ci.yml), where the library is
installed; locally they are skipped if it is not.
"""

import asyncio
import re

import pytest

pytest.importorskip("forecasting_tools")

from forecasting_tools import (  # noqa: E402
    BinaryPrediction,
    BinaryQuestion,
    MultipleChoiceQuestion,
    PredictedOption,
    PredictedOptionList,
)

from nfa_forecaster import bot as bot_module  # noqa: E402
from nfa_forecaster.bot import NfaForecaster  # noqa: E402
from nfa_forecaster.config import load_config  # noqa: E402
from nfa_forecaster.models import ModelPool  # noqa: E402

MODELS = "openrouter/a/one,openrouter/b/two,openrouter/c/three"


class FakeLlm:
    def __init__(self, model, answer):
        self.model = model
        self.answer = answer
        self.prompts = []

    async def invoke(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        return self.answer


class RateLimitError(Exception):
    """Named like litellm's."""


class SpentForTheDay(FakeLlm):
    async def invoke(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        raise RateLimitError('429 RESOURCE_EXHAUSTED {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}')


def _config(**extra):
    env = {
        "METACULUS_TOKEN": "t",
        "OPENROUTER_API_KEY": "o",
        "FORECASTER_MODELS": MODELS,
        "FORECASTER_RESERVE_MODELS": "off",
        "FORECASTER_RESEARCH_MODEL": "off",
        "FORECASTER_MARKETS": "off",
        "FORECASTER_DATA_DESK": "off",
        "FORECASTER_PRECEDENTS": "off",
    }
    env.update(extra)
    return load_config(env)


def _bot(answers, env=None, **extra):
    bot = NfaForecaster(_config(**extra), publish=False, skip_previously_forecasted=False, env=env or {})
    fakes = {model: FakeLlm(model, answers[model]) for model in bot._pool.llms}
    bot._pool = ModelPool(fakes)
    return bot, fakes


async def _fake_structure_output(text_to_structure, output_type, **kwargs):
    # Same parameter names as forecasting_tools.structure_output: the bot
    # passes them by keyword.
    if output_type is BinaryPrediction:
        percent = float(re.search(r"Probability:\s*([\d.]+)%", text_to_structure).group(1))
        return BinaryPrediction(prediction_in_decimal=percent / 100)
    if output_type is PredictedOptionList:
        pairs = re.findall(r"(\S+):\s*([\d.]+)", text_to_structure)
        return PredictedOptionList(predicted_options=[PredictedOption(option_name=n, probability=float(p)) for n, p in pairs])
    raise AssertionError(f"unexpected output type {output_type}")


@pytest.fixture(autouse=True)
def fake_parser(monkeypatch):
    monkeypatch.setattr(bot_module, "structure_output", _fake_structure_output)


def test_pinned_tournaments_match_the_library_current_season():
    # config.CURRENT_TOURNAMENTS is hard-pinned so the watcher and the engine
    # agree without importing the library. Bumping forecasting-tools to a new
    # season must fail here until the pin (and the Netlify watcher's copy) move.
    from forecasting_tools import MetaculusClient

    from nfa_forecaster.config import CURRENT_TOURNAMENTS

    assert CURRENT_TOURNAMENTS == (
        str(MetaculusClient.CURRENT_AI_COMPETITION_ID),
        str(MetaculusClient.CURRENT_MINIBENCH_ID),
    )


def _binary():
    return BinaryQuestion(
        question_text="Will the test pass?",
        id_of_post=101,
        id_of_question=201,
        page_url="https://www.metaculus.com/questions/101",
        resolution_criteria="Resolves Yes if it passes.",
    )


def test_each_ensemble_member_forecasts_exactly_once_and_extremes_are_capped():
    # Funded mode caps at 2%: 0.5% and 1% both become 2%, so the median of
    # (2%, 2%, 60%) is 2%.
    answers = {
        "openrouter/a/one": "Probability: 0.5%",
        "openrouter/b/two": "Probability: 1%",
        "openrouter/c/three": "Probability: 60%",
    }
    bot, fakes = _bot(answers)
    report = asyncio.run(bot.forecast_question(_binary()))
    assert [len(f.prompts) for f in fakes.values()] == [1, 1, 1]
    assert report.prediction == pytest.approx(0.02)
    # The private comment carries every member's reasoning, labelled by model.
    for model in fakes:
        assert f"`{model}`" in report.explanation
    assert "standing in for" not in report.explanation


def test_well_formatted_answers_skip_the_ai_parser(monkeypatch):
    # The free Gemini tier allows 20 calls per model per day; the parser
    # must not spend them when the answer already reads "Probability: N%".
    calls = []

    async def counting_parser(text_to_structure, output_type, **kwargs):
        calls.append(output_type)
        return await _fake_structure_output(text_to_structure, output_type, **kwargs)

    monkeypatch.setattr(bot_module, "structure_output", counting_parser)
    bot, _ = _bot({m: "reasoning\nProbability: 25%" for m in MODELS.split(",")})
    report = asyncio.run(bot.forecast_question(_binary()))
    assert report.prediction == pytest.approx(0.25)
    assert calls == []


def test_badly_formatted_answers_still_get_parsed_by_the_ai(monkeypatch):
    calls = []

    async def fallback_parser(text_to_structure, output_type, **kwargs):
        calls.append(output_type)
        return BinaryPrediction(prediction_in_decimal=0.6)

    monkeypatch.setattr(bot_module, "structure_output", fallback_parser)
    bot, _ = _bot({m: "I'd put it at roughly sixty percent." for m in MODELS.split(",")})
    report = asyncio.run(bot.forecast_question(_binary()))
    assert report.prediction == pytest.approx(0.6)
    assert len(calls) == 3


def test_a_parse_that_keeps_failing_stops_after_three_parsers(monkeypatch):
    calls = []

    async def always_fails(text_to_structure, output_type, **kwargs):
        calls.append(kwargs["model"].model)
        raise ValueError("no probability in the text")

    monkeypatch.setattr(bot_module, "structure_output", always_fails)
    bot, _ = _bot({m: "no number here" for m in MODELS.split(",")})
    with pytest.raises(Exception):
        asyncio.run(bot.forecast_question(_binary()))
    assert len(calls) <= 3 * 3  # at most three parsers per member


def test_repeated_models_in_the_ensemble_are_each_called_per_slot():
    config_models = "openrouter/a/one,openrouter/a/one,openrouter/c/three"
    answers = {"openrouter/a/one": "Probability: 40%", "openrouter/c/three": "Probability: 70%"}
    bot, fakes = _bot(answers, FORECASTER_MODELS=config_models)
    report = asyncio.run(bot.forecast_question(_binary()))
    assert len(fakes["openrouter/a/one"].prompts) == 2
    assert len(fakes["openrouter/c/three"].prompts) == 1
    assert report.prediction == pytest.approx(0.40)


def test_a_failing_member_is_replaced_and_labelled_in_the_comment():
    class Broken(FakeLlm):
        async def invoke(self, prompt, system_prompt=None):
            raise RuntimeError("provider down")

    answers = {m: "Probability: 30%" for m in MODELS.split(",")}
    bot, fakes = _bot(answers)
    bot._pool = ModelPool({**fakes, "openrouter/b/two": Broken("openrouter/b/two", "")})
    report = asyncio.run(bot.forecast_question(_binary()))
    assert report.prediction == pytest.approx(0.30)
    assert "(standing in for `openrouter/b/two`)" in report.explanation


def test_the_practice_question_failure_of_2026_09_26_now_succeeds():
    # Three of four Gemini members were out for the day; the question was lost.
    members = "gemini/a,gemini/b,gemini/c,gemini/d"
    bot = NfaForecaster(
        _config(OPENROUTER_API_KEY="", GEMINI_API_KEY="g", FORECASTER_MODELS=members, FORECASTER_RESERVE_MODELS="groq/r"),
        publish=False,
        skip_previously_forecasted=False,
    )
    spent = {m: SpentForTheDay(m, "") for m in ("gemini/a", "gemini/b", "gemini/c")}
    bot._pool = ModelPool({**spent, "gemini/d": FakeLlm("gemini/d", "Probability: 20%"), "groq/r": FakeLlm("groq/r", "Probability: 30%")})
    report = asyncio.run(bot.forecast_question(_binary()))
    assert 0.2 <= report.prediction <= 0.3
    # Each spent model was asked once, then benched for the rest of the run.
    assert all(len(f.prompts) == 1 for f in spent.values())
    assert set(bot._pool.benched) == set(spent)
    assert any("daily limit reached" in line for line in bot.model_report())


def test_multiple_choice_labels_are_aligned_and_every_option_keeps_a_floor():
    question = MultipleChoiceQuestion(
        question_text="Who wins?",
        id_of_post=102,
        id_of_question=202,
        page_url="https://www.metaculus.com/questions/102",
        options=["Alpha", "Beta", "Gamma"],
    )
    answer = "option_alpha: 0.9\nBeta: 0.1\nGamma: 0.0"
    bot, _ = _bot({m: answer for m in MODELS.split(",")})
    report = asyncio.run(bot.forecast_question(question))
    probabilities = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert set(probabilities) == {"Alpha", "Beta", "Gamma"}
    assert probabilities["Gamma"] >= 0.01 - 1e-9
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["Alpha"] > probabilities["Beta"] > probabilities["Gamma"]


def test_free_tiers_are_paced_through_the_one_real_call_path(monkeypatch):
    import time

    from forecasting_tools import GeneralLlm

    from nfa_forecaster.bot import PacedLlm, make_llm

    starts = []

    async def fake_direct_call(self, prompt):
        starts.append(time.monotonic())
        return "ok"

    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", fake_direct_call)
    config = _config(FORECASTER_GEMINI_RPM="1200")  # one call per 50 ms
    llm = make_llm("gemini/gemini-pacing-test", config, {}, temperature=0)
    assert isinstance(llm, PacedLlm)
    assert isinstance(make_llm("groq/openai/gpt-oss-120b", config, {}), PacedLlm)
    assert isinstance(make_llm("openrouter/x/y:free", config, {}), PacedLlm)
    assert not isinstance(make_llm("openrouter/a/one", config, {}), PacedLlm)

    async def burst():
        await asyncio.gather(*(llm._mockable_direct_call_to_model("p") for _ in range(3)))

    asyncio.run(burst())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) == 3 and all(g >= 0.04 for g in gaps), gaps


def test_the_trading_openrouter_key_only_ever_reaches_free_models():
    from nfa_forecaster.bot import make_llm

    config = _config()  # funded, reasoning effort high
    env = {"OPENROUTER_FREE_KEY": "trading-key"}
    free = make_llm("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", config, env)
    paid = make_llm("openrouter/openai/gpt-6-sol", config, env)
    assert free.litellm_kwargs["api_key"] == "trading-key"
    assert paid.litellm_kwargs.get("api_key") is None
    # High reasoning goes to paid frontier models only; free tiers keep their default.
    assert paid.litellm_kwargs["reasoning_effort"] == "high"
    assert "reasoning_effort" not in free.litellm_kwargs


def test_the_bot_token_never_reaches_a_comment_or_the_config(monkeypatch):
    # A Metaculus-proxy model carries the token inside its litellm kwargs.
    token = "metaculus-token-SHOULD-NEVER-BE-PUBLISHED"
    monkeypatch.setenv("METACULUS_TOKEN", token)
    models = "metaculus/anthropic/claude-sonnet-5,openrouter/b/two,openrouter/c/three"
    answers = {m: "Probability: 40%" for m in models.split(",")}
    bot, _ = _bot(answers, FORECASTER_MODELS=models)
    report = asyncio.run(bot.forecast_question(_binary()))
    assert token not in report.explanation
    assert token not in str(bot.make_llm_dict())


def test_research_failure_is_recorded_and_forecasting_continues(monkeypatch):
    def broken_markets(title):
        raise TimeoutError("polymarket slow")

    monkeypatch.setattr(bot_module, "market_research", broken_markets)
    answers = {m: "Probability: 50%" for m in MODELS.split(",")}
    bot, fakes = _bot(answers, FORECASTER_MARKETS="on")
    report = asyncio.run(bot.forecast_question(_binary()))
    assert report.prediction == pytest.approx(0.5)
    first_prompt = next(iter(fakes.values())).prompts[0]
    assert "## Prediction markets\nUnavailable this run (TimeoutError)." in first_prompt


def test_web_research_sources_and_question_facts_reach_every_member(monkeypatch):
    from nfa_forecaster.web_search import GeminiResearch

    seen = {}

    def fake_deep_search(self, prompt, title, resolution, today):
        seen["searched"] = title
        return "Fed raised rates on 2026-09-16."

    def fake_read_sources(self, urls, title, resolution, today):
        seen["urls"] = list(urls)
        return "FRED shows 4.1% for August 2026."

    def fake_precedents(post_id, title, token):
        seen["precedents"] = (post_id, token)
        return "3 of 5 similar questions resolved Yes."

    monkeypatch.setattr(GeminiResearch, "deep_search", fake_deep_search)
    monkeypatch.setattr(GeminiResearch, "read_sources", fake_read_sources)
    monkeypatch.setattr(bot_module, "precedent_research", fake_precedents)
    answers = {m: "Probability: 50%" for m in MODELS.split(",")}
    bot, fakes = _bot(
        answers,
        env={"GEMINI_API_KEY": "g-key", "METACULUS_TOKEN": "m-token"},
        GEMINI_API_KEY="g-key",
        FORECASTER_PRECEDENTS="on",
    )
    question = BinaryQuestion(
        question_text="Will unemployment exceed 4.5%?",
        id_of_post=101,
        id_of_question=201,
        page_url="https://www.metaculus.com/questions/101",
        resolution_criteria="Resolves per https://fred.stlouisfed.org/series/UNRATE.",
        num_forecasters=37,
    )
    asyncio.run(bot.forecast_question(question))
    assert seen == {
        "searched": "Will unemployment exceed 4.5%?",
        "urls": ["https://fred.stlouisfed.org/series/UNRATE"],
        "precedents": (101, "m-token"),
    }
    for fake in fakes.values():
        prompt = fake.prompts[0]
        assert "## Web search (Google, with follow-ups)\nFed raised rates on 2026-09-16." in prompt
        assert "## Resolution sources (read directly)\nFRED shows 4.1% for August 2026." in prompt
        assert "## Similar resolved questions (Metaculus)\n3 of 5 similar questions resolved Yes." in prompt
        assert "37 forecasters have forecast it so far" in prompt


def test_fallbacks_prefer_members_then_models_the_question_has_not_heard_from():
    bot, _ = _bot({m: "Probability: 50%" for m in MODELS.split(",")})
    chain = ["openrouter/a/one", "openrouter/b/two", "openrouter/c/three", "reserve/x", "reserve/y"]
    order = bot.fallback_order("openrouter/a/one", chain, used={"openrouter/b/two", "reserve/x"})
    assert order == ["openrouter/a/one", "openrouter/c/three", "openrouter/b/two", "reserve/y", "reserve/x"]
