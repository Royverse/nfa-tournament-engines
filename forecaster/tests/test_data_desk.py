import math
from datetime import date, datetime, timedelta, timezone

import pytest

from nfa_forecaster.data_desk import (
    ALPHA_VANTAGE_URL,
    FRED_CSV_URL,
    InsufficientData,
    Series,
    change_quantiles,
    data_desk_research,
    detect_series,
    next_observation_dates,
    random_walk_quantiles,
    weekdays_between,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Resolves per https://fred.stlouisfed.org/series/UNRATE", Series("fred", "UNRATE")),
        ("See fred.stlouisfed.org/series/cpiaucsl for data", Series("fred", "CPIAUCSL")),
        ("According to FRED series ID: DGS10 on the resolution date", Series("fred", "DGS10")),
        ("What will happen next with respect to the 30-year U.S. Treasury yield?", Series("fred", "DGS30")),
        ("Will the price of regular unleaded gas in the U.S. reach $5 a gallon?", Series("fred", "GASREGW")),
        ("Close price per https://finance.yahoo.com/quote/NVDA/history", Series("stock", "NVDA")),
        ("Apple Inc. (NASDAQ: AAPL) closing price", Series("stock", "AAPL")),
        ("the stock with ticker symbol TSLA", Series("stock", "TSLA")),
        ("Will the UN hold a vote on climate?", None),
    ],
)
def test_detect_series(text, expected):
    assert detect_series([text]) == expected


def test_fred_wins_over_a_ticker_in_the_same_question():
    assert detect_series(["NASDAQ: AAPL", "https://fred.stlouisfed.org/series/SP500"]) == Series("fred", "SP500")


def test_weekdays_between_skips_weekends():
    friday = date(2026, 9, 25)
    assert weekdays_between(friday, date(2026, 9, 28)) == 1  # Monday
    assert weekdays_between(friday, date(2026, 10, 2)) == 5
    assert weekdays_between(friday, friday) == 0


def test_random_walk_quantiles_are_centred_and_widen_with_horizon():
    closes = [100 * math.exp(0.01 * ((-1) ** i)) for i in range(80)]  # choppy, zero drift
    sigma, short = random_walk_quantiles(closes, 1)
    _, long = random_walk_quantiles(closes, 25)
    assert sigma > 0
    median_short = dict(short)[0.5]
    assert median_short == pytest.approx(closes[-1])
    spread = lambda qs: dict(qs)[0.95] - dict(qs)[0.05]
    assert spread(long) == pytest.approx(spread(short) * 5, rel=0.25)  # ~sqrt(25)
    assert [v for _, v in short] == sorted(v for _, v in short)


def test_random_walk_needs_enough_history():
    with pytest.raises(InsufficientData):
        random_walk_quantiles([100.0] * 10, 5)


def test_change_quantiles_log_for_positive_levels():
    values = [100 * 1.01**i for i in range(60)]  # +1% per step, every step
    cq = change_quantiles(values, 2)
    assert cq.uses_log
    for _, v in cq.quantiles:
        assert v == pytest.approx(values[-1] * 1.01**2)


def test_change_quantiles_absolute_when_series_crosses_zero():
    values = [(-1) ** i * 0.5 + i * 0.01 for i in range(60)]
    cq = change_quantiles(values, 1)
    assert not cq.uses_log
    assert [v for _, v in cq.quantiles] == sorted(v for _, v in cq.quantiles)


def test_next_observation_dates_for_monthly_and_quarterly():
    assert next_observation_dates(date(2026, 8, 1), 30.5, 3) == [date(2026, 9, 1), date(2026, 10, 1), date(2026, 11, 1)]
    assert next_observation_dates(date(2026, 4, 1), 91, 2) == [date(2026, 7, 1), date(2026, 10, 1)]


def _fred_csv(series_id: str, start: date, n: int, step_days: int, value=lambda i: 4.0 + 0.01 * i) -> str:
    rows = [f"observation_date,{series_id}"]
    for i in range(n):
        rows.append(f"{(start + timedelta(days=step_days * i)).isoformat()},{value(i):.4f}")
    rows.append(f"{(start + timedelta(days=step_days * n)).isoformat()},.")  # missing value marker
    return "\n".join(rows)


def test_fred_monthly_block_shows_three_rows_and_the_release_lag_warning():
    csv_text = _fred_csv("UNRATE", date(2020, 1, 1), 70, 30)

    def get_text(url, params):
        assert url == FRED_CSV_URL and params == {"id": "UNRATE"}
        return csv_text

    block = data_desk_research(
        ["per https://fred.stlouisfed.org/series/UNRATE"], date(2026, 11, 5), get_text=get_text
    )
    assert "FRED series **UNRATE**" in block
    assert "released with a lag" in block
    assert block.count("ahead) |") == 3


def test_stock_block_uses_yahoo_and_never_touches_alpha_vantage_when_yahoo_works():
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    timestamps = [int((start + timedelta(days=i)).timestamp()) for i in range(90)]
    closes = [100 + (i % 3) for i in range(90)]
    payload = {"chart": {"result": [{"timestamp": timestamps, "indicators": {"quote": [{"close": closes}]}}]}}
    calls = []

    def get_json(url, params):
        calls.append(url)
        return payload

    block = data_desk_research(["(NASDAQ: AAPL)"], date(2026, 8, 20), alpha_vantage_key="k", get_json=get_json)
    assert "Stock **AAPL**" in block and "Yahoo Finance" in block
    assert all(ALPHA_VANTAGE_URL not in c for c in calls)


def test_stock_block_falls_back_to_alpha_vantage():
    series = {
        (date(2026, 5, 1) + timedelta(days=i)).isoformat(): {"4. close": str(50 + i % 4)} for i in range(60)
    }

    def get_json(url, params):
        if url == ALPHA_VANTAGE_URL:
            return {"Time Series (Daily)": series}
        raise ConnectionError("yahoo down")

    block = data_desk_research(["ticker: MSFT"], date(2026, 7, 15), alpha_vantage_key="k", get_json=get_json)
    assert "Alpha Vantage" in block


def test_failures_report_the_type_only_never_the_message():
    # A requests error message can contain the request URL, which for Alpha
    # Vantage contains the API key.
    def get_json(url, params):
        raise ConnectionError(f"failed GET {url}?apikey=SECRET-KEY-VALUE")

    block = data_desk_research(["ticker: MSFT"], date(2026, 7, 15), alpha_vantage_key="SECRET-KEY-VALUE", get_json=get_json)
    assert "SECRET-KEY-VALUE" not in block
    assert "ConnectionError" in block


def test_no_series_or_no_target_returns_empty():
    assert data_desk_research(["Will it rain in Paris?"], date(2026, 10, 1)) == ""
    assert data_desk_research(["ticker: MSFT"], None) == ""
