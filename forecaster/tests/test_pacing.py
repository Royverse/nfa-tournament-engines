import asyncio
import time

import pytest

from nfa_forecaster.pacing import Pacer, pacer_for


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_simultaneous_callers_get_evenly_spaced_slots():
    clock = FakeClock()
    pacer = Pacer(4, clock=clock)  # the free-tier budget: one call per 15 s
    assert [pacer.reserve() for _ in range(4)] == [0.0, 15.0, 30.0, 45.0]


def test_an_idle_pacer_lets_the_next_call_straight_through():
    clock = FakeClock()
    pacer = Pacer(4, clock=clock)
    pacer.reserve()
    clock.now += 60
    assert pacer.reserve() == 0.0


def test_pacer_rejects_nonsense():
    with pytest.raises(ValueError):
        Pacer(0)


def test_one_shared_pacer_per_model():
    assert pacer_for("gemini/test-shared", 4) is pacer_for("gemini/test-shared", 4)
    assert pacer_for("gemini/test-a", 4) is not pacer_for("gemini/test-b", 4)


def test_wait_really_spaces_concurrent_calls():
    pacer = Pacer(60 / 0.05)  # one call per 50 ms
    starts: list[float] = []

    async def call():
        await pacer.wait()
        starts.append(time.monotonic())

    async def main():
        await asyncio.gather(*(call() for _ in range(3)))

    asyncio.run(main())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.04 for g in gaps), gaps
