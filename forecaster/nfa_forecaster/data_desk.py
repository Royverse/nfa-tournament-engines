"""Statistical baselines for questions about a data series.

MiniBench questions are generated from public data (FRED, stock prices,
Google Trends). For those, a model guessing a number is worse than a simple
statistical view of how much the series normally moves over the horizon. The
data desk detects a FRED series or a stock ticker in the question, pulls the
history from a keyless source, and writes that outside view into the research.

Stock history comes from Yahoo's keyless chart API. Alpha Vantage is only a
fallback, because its 25-calls/day key is shared with the trading scanner.
"""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence

import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
HTTP_TIMEOUT_SECONDS = 15

PERCENTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
STOCK_RETURN_WINDOW = 60  # trading days of returns used for volatility
MIN_OBSERVATIONS = 20
FRED_CHANGE_WINDOW = 240  # most recent observations used for change quantiles

_FRED_PATTERNS = (
    re.compile(r"fred\.stlouisfed\.org/(?:series|data)/([A-Za-z0-9_]{2,30})"),
    re.compile(r"fred\.stlouisfed\.org/graph/\?id=([A-Za-z0-9_]{2,30})"),
    re.compile(r"\bFRED\b[^.\n]{0,40}?\bseries(?:\s+ID)?\s*[:\-]?\s*\(?([A-Z][A-Z0-9_]{2,29})\b"),
)
_STOCK_PATTERNS = (
    re.compile(r"finance\.yahoo\.com/quote/([A-Za-z0-9.\-^=]{1,12})"),
    re.compile(r"\b(?:NASDAQ|NYSE|NYSEARCA|NYSE American|AMEX|BATS)\s*:\s*([A-Z][A-Z.\-]{0,6})\b"),
    re.compile(r"\bticker(?:\s+symbol)?\s*[:(]?\s*\$?([A-Z][A-Z.\-]{0,6})\b"),
)
# Natural-language macro phrases that resolve on canonical FRED series even when
# the question text does not paste a literal fred.stlouisfed.org URL (e.g. Q45969
# on the 30-year U.S. Treasury yield or Q46057 on regular unleaded gas).
_FRED_NATURAL_LANGUAGE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b30[\s\-]year\s+(?:u\.?s\.?\s+)?treasury\s+yield\b", re.I), "DGS30"),
    (re.compile(r"\b10[\s\-]year\s+(?:u\.?s\.?\s+)?treasury\s+yield\b", re.I), "DGS10"),
    (re.compile(r"\b2[\s\-]year\s+(?:u\.?s\.?\s+)?treasury\s+yield\b", re.I), "DGS2"),
    (re.compile(r"\b(?:u\.?s\.?\s+)?unemployment\s+rate\b", re.I), "UNRATE"),
    (re.compile(r"\bregular\s+(?:unleaded\s+)?gas(?:oline)?\s+in\s+the\s+u\.?s\.?", re.I), "GASREGW"),
    (re.compile(r"\beffective\s+federal\s+funds\s+rate\b", re.I), "EFFR"),
)

GetText = Callable[[str, dict[str, Any]], str]
GetJson = Callable[[str, dict[str, Any]], Any]


class InsufficientData(ValueError):
    """Too little history to say anything honest."""


@dataclass(frozen=True)
class Series:
    kind: str  # "fred" | "stock"
    identifier: str


def _http_get(url: str, params: dict[str, Any]) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0 (nfa-forecaster data desk)"},
    )
    response.raise_for_status()
    return response


def default_get_text(url: str, params: dict[str, Any]) -> str:
    return _http_get(url, params).text


def default_get_json(url: str, params: dict[str, Any]) -> Any:
    return _http_get(url, params).json()


def detect_series(texts: Sequence[str | None]) -> Series | None:
    """Find the first FRED series or stock ticker referenced by the question."""
    blob = "\n".join(t for t in texts if t)
    for pattern in _FRED_PATTERNS:
        match = pattern.search(blob)
        if match:
            return Series("fred", match.group(1).upper())
    for pattern in _STOCK_PATTERNS:
        match = pattern.search(blob)
        if match:
            return Series("stock", match.group(1).upper())
    for pattern, series_id in _FRED_NATURAL_LANGUAGE_PATTERNS:
        if pattern.search(blob):
            return Series("fred", series_id)
    return None


def _quantile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation empirical quantile of an ascending sequence."""
    if not sorted_values:
        raise InsufficientData("no values")
    position = p * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def weekdays_between(start: date, end: date) -> int:
    """Weekdays after `start` up to and including `end` (market holidays ignored)."""
    if end <= start:
        return 0
    full_weeks, extra = divmod((end - start).days, 7)
    count = full_weeks * 5
    for offset in range(1, extra + 1):
        if (start + timedelta(days=offset)).weekday() < 5:
            count += 1
    return count


def random_walk_quantiles(
    closes: Sequence[float], horizon_days: int, percentiles: Sequence[float] = PERCENTILES
) -> tuple[float, list[tuple[float, float]]]:
    """Zero-drift lognormal quantiles of the price `horizon_days` trading days out.

    Returns (daily volatility, [(percentile, price)]). Volatility is the stdev
    of the last STOCK_RETURN_WINDOW daily log returns.
    """
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    returns = returns[-STOCK_RETURN_WINDOW:]
    if len(returns) < MIN_OBSERVATIONS:
        raise InsufficientData(f"only {len(returns)} daily returns")
    sigma = statistics.stdev(returns)
    spread = sigma * math.sqrt(max(horizon_days, 1))
    last = closes[-1]
    normal = statistics.NormalDist()
    return sigma, [(p, last * math.exp(normal.inv_cdf(p) * spread)) for p in percentiles]


@dataclass(frozen=True)
class ChangeQuantiles:
    uses_log: bool
    sample_size: int
    quantiles: list[tuple[float, float]]

    @property
    def kind(self) -> str:
        return "log" if self.uses_log else "absolute"


def change_quantiles(
    values: Sequence[float], steps: int, percentiles: Sequence[float] = PERCENTILES
) -> ChangeQuantiles:
    """Quantiles of the value `steps` observations ahead, from historical k-step changes.

    Uses log changes when the recent series is strictly positive (levels and
    indices), absolute changes otherwise (rates that cross zero).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    recent = list(values[-(FRED_CHANGE_WINDOW + steps):])
    use_log = all(v > 0 for v in recent)
    changes = sorted(
        (math.log(b / a) if use_log else b - a) for a, b in zip(recent, recent[steps:])
    )
    if len(changes) < MIN_OBSERVATIONS:
        raise InsufficientData(f"only {len(changes)} historical {steps}-step changes")
    last = values[-1]
    out = []
    for p in percentiles:
        q = _quantile(changes, p)
        out.append((p, last * math.exp(q) if use_log else last + q))
    return ChangeQuantiles(uses_log=use_log, sample_size=len(changes), quantiles=out)


def _median_gap_days(dates: Sequence[date]) -> float:
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
    if not gaps:
        raise InsufficientData("cannot infer the series frequency")
    return statistics.median(gaps)


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year, month = d.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(d.day, 28))


def next_observation_dates(last: date, gap_days: float, count: int) -> list[date]:
    if 27 <= gap_days <= 32:
        return [_add_months(last, k) for k in range(1, count + 1)]
    if 85 <= gap_days <= 95:
        return [_add_months(last, 3 * k) for k in range(1, count + 1)]
    if 360 <= gap_days <= 370:
        return [_add_months(last, 12 * k) for k in range(1, count + 1)]
    return [last + timedelta(days=round(gap_days * k)) for k in range(1, count + 1)]


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:.4g}"


def _percentile_header() -> str:
    return "| | " + " | ".join(f"P{round(p * 100)}" for p in PERCENTILES) + " |\n|---|" + "---|" * len(PERCENTILES)


def fetch_fred(series_id: str, get_text: GetText) -> list[tuple[date, float]]:
    text = get_text(FRED_CSV_URL, {"id": series_id})
    rows: list[tuple[date, float]] = []
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or len(header) < 2:
        raise InsufficientData("FRED returned no CSV header")
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        try:
            rows.append((date.fromisoformat(row[0]), float(row[1])))
        except ValueError:
            continue
    if len(rows) < MIN_OBSERVATIONS:
        raise InsufficientData(f"FRED series {series_id} has {len(rows)} observations")
    return rows


def fred_block(series_id: str, target: date, get_text: GetText) -> str:
    rows = fetch_fred(series_id, get_text)
    dates = [d for d, _ in rows]
    values = [v for _, v in rows]
    gap = _median_gap_days(dates)
    last_date, last_value = rows[-1]

    lines = [
        f"FRED series **{series_id}** — last observation {last_date.isoformat()} = {_fmt(last_value)} "
        f"(one observation every ~{gap:.0f} days). Source: https://fred.stlouisfed.org/series/{series_id}",
    ]
    if gap <= 7:
        # Daily/weekly series: one row at the question's resolution date.
        if gap <= 1.5:
            steps = weekdays_between(last_date, target)
        else:
            steps = round((target - last_date).days / gap)
        steps = max(1, steps)
        cq = change_quantiles(values, steps)
        lines += [
            f"Baseline for the value around {target.isoformat()} ({steps} observations ahead), from "
            f"{cq.kind} {steps}-step changes over the last {cq.sample_size} observations:",
            _percentile_header(),
            f"| {target.isoformat()} | " + " | ".join(_fmt(v) for _, v in cq.quantiles) + " |",
        ]
    else:
        # Monthly/quarterly/annual: data is published with a lag, so show the
        # next three observations and let the forecaster pick the one the
        # question actually asks about.
        lines += [
            "Official data is released with a lag — pick the row for the observation period the question asks about.",
            _percentile_header(),
        ]
        cq = None
        for k, obs_date in enumerate(next_observation_dates(last_date, gap, 3), start=1):
            cq = change_quantiles(values, k)
            lines.append(f"| {obs_date.isoformat()} ({k} ahead) | " + " | ".join(_fmt(v) for _, v in cq.quantiles) + " |")
        if cq is not None:
            lines.append(f"Method: empirical {cq.kind} k-step changes over the last ~{cq.sample_size} observations.")
    lines.append(
        "This is the outside view: how much this series normally moves. Adjust it only for specific, "
        "dated evidence (a scheduled release, a known shock)."
    )
    return "\n".join(lines)


def _yahoo_closes(symbol: str, get_json: GetJson) -> list[tuple[date, float]]:
    payload = get_json(YAHOO_CHART_URL.format(symbol=symbol), {"range": "6mo", "interval": "1d"})
    result = ((payload or {}).get("chart") or {}).get("result") or []
    if not result:
        raise InsufficientData(f"Yahoo has no chart for {symbol}")
    timestamps = result[0].get("timestamp") or []
    closes = (((result[0].get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []
    rows = [
        (datetime.fromtimestamp(t, tz=timezone.utc).date(), float(c))
        for t, c in zip(timestamps, closes)
        if c is not None
    ]
    if len(rows) < MIN_OBSERVATIONS:
        raise InsufficientData(f"Yahoo returned {len(rows)} closes for {symbol}")
    return rows


def _alpha_vantage_closes(symbol: str, api_key: str, get_json: GetJson) -> list[tuple[date, float]]:
    payload = get_json(
        ALPHA_VANTAGE_URL,
        {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "compact", "apikey": api_key},
    )
    series = (payload or {}).get("Time Series (Daily)")
    if not series:
        raise InsufficientData("Alpha Vantage returned no daily series (rate limit or unknown symbol)")
    rows = sorted((date.fromisoformat(d), float(bar["4. close"])) for d, bar in series.items())
    if len(rows) < MIN_OBSERVATIONS:
        raise InsufficientData(f"Alpha Vantage returned {len(rows)} closes for {symbol}")
    return rows


def stock_block(
    symbol: str, target: date, get_json: GetJson, alpha_vantage_key: str | None
) -> str:
    source = "Yahoo Finance"
    try:
        rows = _yahoo_closes(symbol, get_json)
    except Exception:
        if not alpha_vantage_key:
            raise
        rows = _alpha_vantage_closes(symbol, alpha_vantage_key, get_json)
        source = "Alpha Vantage"
    last_date, last_close = rows[-1]
    horizon = max(1, weekdays_between(last_date, target))
    sigma, qs = random_walk_quantiles([c for _, c in rows], horizon)
    return "\n".join(
        [
            f"Stock **{symbol}** — last close {last_date.isoformat()} = {_fmt(last_close)} ({source}). "
            f"Daily volatility {sigma:.2%} (≈{sigma * math.sqrt(252):.0%} a year) over the last "
            f"{min(STOCK_RETURN_WINDOW, len(rows) - 1)} sessions.",
            f"Random-walk baseline for the close around {target.isoformat()} ({horizon} trading days ahead, zero drift):",
            _percentile_header(),
            f"| {target.isoformat()} | " + " | ".join(_fmt(v) for _, v in qs) + " |",
            "Prices are close to a random walk over short horizons; beat this only with specific news "
            "(earnings date, guidance, a pending deal).",
        ]
    )


def data_desk_research(
    texts: Sequence[str | None],
    target: date | None,
    *,
    alpha_vantage_key: str | None = None,
    get_text: GetText = default_get_text,
    get_json: GetJson = default_get_json,
) -> str:
    """Return a baseline section, or "" when the question is not about a data series."""
    series = detect_series(texts)
    if series is None or target is None:
        return ""
    try:
        if series.kind == "fred":
            return fred_block(series.identifier, target, get_text)
        return stock_block(series.identifier, target, get_json, alpha_vantage_key)
    except Exception as e:
        # Never str(e): a requests error message can carry the request URL,
        # and the Alpha Vantage URL carries the key.
        return f"Detected {series.kind} series {series.identifier}, but no baseline could be built ({type(e).__name__})."
