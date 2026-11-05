"""Post-processing that keeps a forecast away from the extremes.

Log scoring punishes a confident miss without limit: a 1% forecast on
something that happens costs ln(0.01) = -4.6, while being right at 99%
instead of 97% gains only ln(0.99/0.97) = 0.02. Among Fall 2025 prize-winning
bots, capping forecasts was the single strongest differentiator
(r = +0.48, p = 0.005), and LLM ensembles are measurably overconfident.
These functions are the cap. They are pure so they can be tested exhaustively.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Iterable, Mapping, Sequence


def cap_probability(p: float, floor: float, ceiling: float) -> float:
    """Clamp a binary probability into [floor, ceiling]."""
    if not 0.0 <= floor < ceiling <= 1.0:
        raise ValueError(f"invalid caps [{floor}, {ceiling}]")
    if p != p:  # NaN
        raise ValueError("probability is NaN")
    return min(ceiling, max(floor, p))


def aggregate_binary_probabilities(
    probabilities: Sequence[float],
    floor: float = 0.02,
    ceiling: float = 0.98,
    logit_weight: float = 0.5,
) -> float:
    """Combine binary ensemble forecasts via a blend of median and trimmed log-odds mean.

    Why both:
    - The median is immune to a single broken stand-in model.
    - The trimmed log-odds (logit) mean respects log-score geometry when models
      have strong evidence near the tails (e.g. two models at 5% and two at 15%
      pool to ~8.7% in log-odds vs 10% linearly).
    - When 4+ forecasts exist, the highest and lowest are trimmed before taking
      the log-odds mean so one hallucinated extreme cannot hijack the logit average.
    """
    if not probabilities:
        raise ValueError("no probabilities to aggregate")
    capped = sorted(cap_probability(float(p), floor, ceiling) for p in probabilities)
    median_p = float(statistics.median(capped))
    if len(capped) <= 2 or logit_weight <= 0.0:
        return cap_probability(median_p, floor, ceiling)

    trimmed = capped[1:-1] if len(capped) >= 4 else capped
    logits = [math.log(p / (1.0 - p)) for p in trimmed]
    mean_logit = sum(logits) / len(logits)
    logit_p = 1.0 / (1.0 + math.exp(-mean_logit))
    blended = (1.0 - logit_weight) * median_p + logit_weight * logit_p
    return cap_probability(blended, floor, ceiling)


def _normalise_label(label: str) -> str:
    # Applied to both sides of the match, so stripping a leading "Option_"
    # cannot turn one real option name into another.
    text = re.sub(r"^option[\s_:\-]+", "", label.strip().lower())
    return re.sub(r"\s+", " ", text).strip(" .:\"'")


def align_options(parsed: Mapping[str, float], options: Iterable[str]) -> dict[str, float]:
    """Map a parser's option labels onto the question's exact option names.

    The parser sometimes returns "Option_A", different casing or trailing
    punctuation. Metaculus needs the exact labels, and the library's aggregator
    rejects any mismatch, so every canonical option gets a value: its matched
    probability, or 0.0 when the parser dropped it (the floor lifts it later).
    """
    canonical = list(options)
    by_exact = {name: float(value) for name, value in parsed.items()}
    by_normalised: dict[str, float] = {}
    for name, value in parsed.items():
        by_normalised.setdefault(_normalise_label(name), float(value))

    aligned: dict[str, float] = {}
    for option in canonical:
        if option in by_exact:
            aligned[option] = by_exact[option]
        else:
            aligned[option] = by_normalised.get(_normalise_label(option), 0.0)
    return aligned


def floor_and_normalise(probabilities: Mapping[str, float], floor: float) -> dict[str, float]:
    """Give every option at least `floor` and make the total exactly 1.

    p_i' = floor + (1 - n * floor) * p_i / sum(p)

    This keeps the ranking of options, guarantees each option >= floor, and
    sums to 1 by construction. With all-zero input it returns a uniform split.
    """
    n = len(probabilities)
    if n == 0:
        raise ValueError("no options to normalise")
    if floor < 0 or n * floor >= 1:
        raise ValueError(f"floor {floor} is impossible for {n} options")
    values = {k: max(0.0, float(v)) for k, v in probabilities.items()}
    total = sum(values.values())
    if total <= 0:
        return {k: 1.0 / n for k in values}
    spare = 1.0 - n * floor
    return {k: floor + spare * v / total for k, v in values.items()}


def aggregate_multiple_choice(
    member_distributions: Sequence[Mapping[str, float]],
    options: Iterable[str],
    floor: float,
) -> dict[str, float]:
    """Combine multiple-choice forecasts using per-option median + floor normalisation.

    Per-option median resists a single fallback model putting 80% on the wrong
    option far better than the arithmetic mean.
    """
    canonical = list(options)
    if not canonical:
        raise ValueError("no options to aggregate")
    if not member_distributions:
        raise ValueError("no member distributions to aggregate")
    raw_medians: dict[str, float] = {}
    for opt in canonical:
        vals = [float(dist.get(opt, 0.0)) for dist in member_distributions]
        raw_medians[opt] = float(statistics.median(vals))
    effective_floor = min(floor, 0.5 / len(canonical))
    return floor_and_normalise(raw_medians, effective_floor)


_SCALE_FACTORS: tuple[float, ...] = (1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9)


def calibrate_numeric_percentiles(
    percentiles: Sequence[tuple[float, float]],
    lower_bound: float,
    upper_bound: float,
    *,
    open_lower_bound: bool = True,
    open_upper_bound: bool = True,
    tail_widen_factor: float = 1.30,
) -> list[tuple[float, float]]:
    """Repair unit-scale mismatches, widen overconfident LLM tails, and prevent boundary pile-ups.

    1. Auto-unit repair: if all values lie outside [lower_bound, upper_bound] and
       multiplying by 10^3, 10^6, or 10^9 (or dividing) moves the median squarely
       inside [lower_bound, upper_bound], rescale the values. (Fixes e.g. a model
       writing '114.7' instead of '114,700,000' when bounds are in raw dollars.)
    2. Tail widening: raw LLM 5th-95th percentile intervals are empirically ~30%
       too narrow on FutureEval numeric questions; widen distances from the median
       by `tail_widen_factor`.
    3. Boundary de-clamping & strict monotonicity: when closed bounds clamp multiple
       low percentiles onto `lower_bound` (as happened on live Q45961 where P10 and
       P20 both clamped to 110,000,000.0), space them strictly inside the range so
       `NumericDistribution.from_question` never builds a degenerate step cliff.
    """
    if not percentiles:
        raise ValueError("no percentiles provided")
    if upper_bound <= lower_bound:
        raise ValueError(f"invalid bounds [{lower_bound}, {upper_bound}]")

    sorted_pairs = sorted((float(p), float(v)) for p, v in percentiles)
    ps = [p for p, _ in sorted_pairs]
    vs = [v for _, v in sorted_pairs]

    span = upper_bound - lower_bound
    mid_idx = len(vs) // 2
    median_val = vs[mid_idx]

    # 1. Unit-scale repair when the median is wildly off by a power of 1,000
    if lower_bound > 0 and median_val > 0 and (median_val < lower_bound * 0.05 or median_val > upper_bound * 20.0):
        best_scale = 1.0
        target_center = 0.5 * (lower_bound + upper_bound)
        best_dist = abs(math.log10(max(median_val, 1e-12) / target_center))
        for factor in _SCALE_FACTORS:
            candidate = median_val * factor
            if lower_bound <= candidate <= upper_bound:
                dist = abs(math.log10(candidate / target_center))
                if dist < best_dist:
                    best_dist = dist
                    best_scale = factor
        if best_scale != 1.0:
            vs = [v * best_scale for v in vs]
            median_val = vs[mid_idx]

    # 2. Widen tails around the median to correct LLM overconfidence
    if tail_widen_factor > 1.0 and len(vs) >= 3:
        vs = [median_val + tail_widen_factor * (v - median_val) for v in vs]

    # 3. Clamp to soft interior margins so closed/open bounds never collapse multiple
    # percentiles onto the exact same boundary value (which causes Q45961-style cliffs).
    n = len(vs)
    min_step = max(span * 1e-4, 1e-9)
    low_margin = lower_bound - 0.15 * span if open_lower_bound else lower_bound + min_step
    high_margin = upper_bound + 0.15 * span if open_upper_bound else upper_bound - min_step

    clamped = [min(high_margin, max(low_margin, v)) for v in vs]

    # Forward pass: enforce strict minimum spacing
    for i in range(1, n):
        if clamped[i] <= clamped[i - 1] + min_step:
            clamped[i] = clamped[i - 1] + min_step

    # If the forward pass pushed the top above high_margin, pull back from the top
    if clamped[-1] > high_margin:
        clamped[-1] = high_margin
        for i in range(n - 2, -1, -1):
            if clamped[i] >= clamped[i + 1] - min_step:
                clamped[i] = clamped[i + 1] - min_step

    return list(zip(ps, clamped))
