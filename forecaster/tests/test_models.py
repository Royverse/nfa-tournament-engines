import asyncio

import pytest

from nfa_forecaster.models import AllModelsFailed, ModelPool, bench_reason


class RateLimitError(Exception):
    """Named like litellm's, which is how bench_reason recognises a 429."""


class NotFoundError(Exception):
    pass


class ServiceUnavailableError(Exception):
    pass


# The exact text of the 429 that cost the practice question on 2026-09-26.
DAILY_429 = RateLimitError(
    'geminiException - {"error": {"code": 429, "message": "Quota exceeded for metric: '
    'generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.7-flash", '
    '"status": "RESOURCE_EXHAUSTED", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}'
)
MINUTE_429 = RateLimitError('{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "retryDelay": "26s"}')


class Fake:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    async def invoke(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def test_bench_reason_separates_spent_for_the_day_from_a_passing_blip():
    assert bench_reason(DAILY_429) == "daily limit reached"
    assert bench_reason(RateLimitError("Rate limit reached ... tokens per day (TPD): Limit 200000")) == "daily limit reached"
    assert bench_reason(NotFoundError("model gone")) == "unavailable (NotFoundError)"
    assert bench_reason(Exception("HTTP 402: insufficient credits")) == "unavailable (Exception)"
    # Transient: skip for this call, keep for the next.
    assert bench_reason(MINUTE_429) is None
    assert bench_reason(ServiceUnavailableError("503 This model is currently experiencing high demand")) is None


def test_the_practice_question_failure_now_falls_back_and_answers():
    # 2026-09-26: three of four Gemini members were out for the day.
    pool = ModelPool(
        {
            "gemini/a": Fake(error=DAILY_429),
            "gemini/b": Fake(error=DAILY_429),
            "gemini/c": Fake(answer="Probability: 40%"),
            "groq/d": Fake(answer="Probability: 45%"),
        }
    )
    model, answer = asyncio.run(pool.ask("q", ["gemini/a", "gemini/b", "gemini/c"]))
    assert (model, answer) == ("gemini/c", "Probability: 40%")
    assert pool.benched == {"gemini/a": "daily limit reached", "gemini/b": "daily limit reached"}
    assert pool.fallbacks == [("gemini/a", "gemini/c")]


def test_a_benched_model_is_never_asked_again_in_the_run():
    spent = Fake(error=DAILY_429)
    pool = ModelPool({"a": spent, "b": Fake(answer="ok")})
    for _ in range(3):
        asyncio.run(pool.ask("q", ["a"]))
    assert spent.calls == 1


def test_model_benched_mid_loop_is_skipped_without_calling_it():
    # Simulate a concurrent slot benching model 'b' while 'a' was being invoked
    b_called = False

    class BenchMidLoop(Fake):
        def __init__(self, pool_ref):
            super().__init__(error=ServiceUnavailableError("503"))
            self.pool_ref = pool_ref

        async def invoke(self, prompt):
            self.pool_ref[0].benched["b"] = "daily limit reached"
            return await super().invoke(prompt)

    class TrackB(Fake):
        async def invoke(self, prompt):
            nonlocal b_called
            b_called = True
            return "b answered"

    pool_holder = []
    pool = ModelPool(
        {
            "a": BenchMidLoop(pool_holder),
            "b": TrackB(),
            "c": Fake(answer="c answered"),
        }
    )
    pool_holder.append(pool)
    model, answer = asyncio.run(pool.ask("q", ["a", "b", "c"]))
    assert model == "c"
    assert answer == "c answered"
    assert b_called is False  # 'b' was skipped because it was benched mid-loop


def test_a_transient_failure_is_retried_on_the_next_call():
    busy = Fake(error=ServiceUnavailableError("503"))
    pool = ModelPool({"a": busy, "b": Fake(answer="ok")})
    asyncio.run(pool.ask("q", ["a"]))
    asyncio.run(pool.ask("q", ["a"]))
    assert busy.calls == 2 and pool.benched == {}


def test_chain_puts_preferred_first_then_the_rest_of_the_pool_in_order():
    pool = ModelPool({m: Fake(answer="x") for m in ("m1", "m2", "r1", "r2")})
    assert pool.chain(["m2", "m1"]) == ["m2", "m1", "r1", "r2"]
    pool.benched["r1"] = "daily limit reached"
    assert pool.chain(["m2"]) == ["m2", "m1", "r2"]


def test_all_models_failing_raises_with_every_reason():
    pool = ModelPool({"a": Fake(error=DAILY_429), "b": Fake(error=TimeoutError())})
    with pytest.raises(AllModelsFailed, match="a: RateLimitError; b: TimeoutError"):
        asyncio.run(pool.ask("q", ["a"]))


def test_report_lists_benchings_and_counts_fallbacks():
    pool = ModelPool({"a": Fake(error=DAILY_429), "b": Fake(answer="x")})
    asyncio.run(pool.ask("q", ["a"]))
    asyncio.run(pool.ask("q", ["a"]))
    assert pool.report() == [
        "benched for this run: `a` — daily limit reached",
        "fallback: `b` answered for `a` ×2",
    ]


def test_a_timeout_benches_the_model_for_the_rest_of_the_run():
    # A timeout has already burned MEMBER_TIMEOUT_SECONDS x MEMBER_TRIES of runner time.
    assert bench_reason(TimeoutError()) == "timed out"

    class Timeout(Exception):
        """Named like litellm's timeout error."""

    assert bench_reason(Timeout("Request timed out")) == "timed out"
