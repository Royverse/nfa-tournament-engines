import json
from pathlib import Path

import pytest

from nfa_forecaster.markets import (
    MANIFOLD_SEARCH_URL,
    POLYMARKET_SEARCH_URL,
    build_search_query,
    format_markets_block,
    market_research,
    parse_manifold,
    parse_polymarket,
    search_markets,
)

FIXTURES = Path(__file__).parent / "fixtures"
# Real responses recorded 2026-09-26 (trimmed) for the query "fed rate cut".
POLYMARKET = json.loads((FIXTURES / "polymarket_search.json").read_text())
MANIFOLD = json.loads((FIXTURES / "manifold_search.json").read_text())


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Will the Fed cut rates before January 1, 2027?", "Fed cut rates"),
        ("Will SpaceX launch Starship (flight 12) by 2026-12-31?", "SpaceX launch Starship"),
        ("What will be the unemployment rate in October 2026?", "unemployment rate October 2026"),
        ("How many named Atlantic hurricanes will form in 2026?", "named Atlantic hurricanes form 2026"),
    ],
)
def test_build_search_query_keeps_names_and_drops_filler(title, expected):
    assert build_search_query(title) == expected


def test_parse_polymarket_skips_closed_markets_and_orders_by_volume():
    quotes = parse_polymarket(POLYMARKET)
    assert [q.question for q in quotes] == [
        "Will no Fed rate cuts happen in 2026?",
        "Will 1 Fed rate cut happen in 2026?",
        "Will 2 Fed rate cuts happen in 2026?",
        "Will 3 Fed rate cuts happen in 2026?",
    ]
    assert quotes[0].probability == pytest.approx(0.9695)
    assert quotes[0].url.startswith("https://polymarket.com/event/")
    assert all(q.source == "Polymarket" for q in quotes)


def test_parse_polymarket_reads_json_encoded_outcomes_and_ignores_non_yes_no():
    payload = {
        "events": [
            {
                "slug": "e",
                "markets": [
                    {"question": "Q1", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.25", "0.75"]', "volumeNum": 5000, "active": True, "closed": False},
                    {"question": "Q2", "outcomes": '["Up", "Down"]', "outcomePrices": '["0.5", "0.5"]', "volumeNum": 5000, "active": True, "closed": False},
                    {"question": "Q3", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.5", "0.5"]', "volumeNum": 10, "active": True, "closed": False},
                ],
            }
        ]
    }
    quotes = parse_polymarket(payload)
    assert [(q.question, q.probability) for q in quotes] == [("Q1", 0.25)]


def test_parse_manifold_drops_thin_markets():
    quotes = parse_manifold(MANIFOLD)
    # The 8-trader Fed market is below the 10-trader floor.
    assert all("Oct. 28" not in q.question for q in quotes)
    assert {q.question for q in quotes} >= {
        "Will the U.S. Federal Reserve cut interest rates at least 3 times before December 31, 2026?"
    }
    assert all(q.activity.endswith("(play money)") for q in quotes)


def test_search_markets_survives_one_venue_failing():
    def fake_get(url, params):
        if url == POLYMARKET_SEARCH_URL:
            raise TimeoutError("slow")
        assert url == MANIFOLD_SEARCH_URL
        return MANIFOLD

    quotes, notes = search_markets("Will the Fed cut interest rates 3 times in 2026?", fake_get)
    assert quotes and all(q.source == "Manifold" for q in quotes)
    assert notes == ["Polymarket search failed (TimeoutError)"]


def test_block_warns_that_markets_are_related_not_identical():
    quotes = parse_polymarket(POLYMARKET)
    block = format_markets_block("Will the Fed cut rates in 2026?", quotes, [])
    assert "RELATED, not necessarily the same question" in block
    assert "97% Yes" in block


def test_block_says_so_when_nothing_liquid_was_found():
    text = market_research("Will a meteor land in my garden?", lambda url, params: {"events": []} if url == POLYMARKET_SEARCH_URL else [])
    assert "No liquid related markets found" in text
