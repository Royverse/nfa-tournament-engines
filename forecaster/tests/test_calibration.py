import math

import pytest

from nfa_forecaster.calibration import (
    aggregate_binary_probabilities,
    aggregate_multiple_choice,
    align_options,
    calibrate_numeric_percentiles,
    cap_probability,
    floor_and_normalise,
)


@pytest.mark.parametrize(
    "p, expected",
    [(0.0, 0.03), (0.01, 0.03), (0.03, 0.03), (0.5, 0.5), (0.97, 0.97), (0.999, 0.97), (1.0, 0.97)],
)
def test_cap_probability(p, expected):
    assert cap_probability(p, 0.03, 0.97) == expected


def test_cap_rejects_nan_and_bad_caps():
    with pytest.raises(ValueError):
        cap_probability(float("nan"), 0.03, 0.97)
    with pytest.raises(ValueError):
        cap_probability(0.5, 0.9, 0.1)


def test_capping_is_cheap_when_right_and_saves_a_lot_when_wrong():
    # The rationale in calibration.py, as numbers: right at 97% instead of 99%
    # costs ~0.02 nats; wrong at 3% instead of 1% saves ~1.1 nats.
    cost_when_right = math.log(0.99) - math.log(0.97)
    saving_when_wrong = math.log(0.03) - math.log(0.01)
    assert cost_when_right < 0.021
    assert saving_when_wrong > 1.09


def test_align_options_maps_parser_labels_to_exact_names():
    options = ["Joe Biden", "Donald Trump", "Other"]
    parsed = {"joe biden": 0.2, "Option_Donald Trump": 0.7, "other.": 0.1}
    assert align_options(parsed, options) == {"Joe Biden": 0.2, "Donald Trump": 0.7, "Other": 0.1}


def test_align_options_fills_dropped_options_with_zero():
    assert align_options({"A": 0.9}, ["A", "B"]) == {"A": 0.9, "B": 0.0}


def test_align_prefers_exact_match():
    assert align_options({"1": 0.3, "Option 1": 0.7}, ["Option 1"]) == {"Option 1": 0.7}


def test_floor_and_normalise_sums_to_one_and_respects_floor():
    result = floor_and_normalise({"A": 0.98, "B": 0.02, "C": 0.0}, 0.01)
    assert math.isclose(sum(result.values()), 1.0, abs_tol=1e-12)
    assert all(v >= 0.01 - 1e-12 for v in result.values())
    assert result["A"] > result["B"] > result["C"]


def test_floor_and_normalise_all_zero_is_uniform():
    assert floor_and_normalise({"A": 0, "B": 0}, 0.01) == {"A": 0.5, "B": 0.5}


def test_floor_and_normalise_rejects_impossible_floor():
    with pytest.raises(ValueError):
        floor_and_normalise({"A": 1, "B": 1}, 0.5)
    with pytest.raises(ValueError):
        floor_and_normalise({}, 0.01)


def test_aggregate_binary_probabilities_trims_outliers_and_blends_logit_with_median():
    # 5 forecasts: one wild outlier (95%), four clustered around 10%-20%
    p = aggregate_binary_probabilities([0.05, 0.10, 0.12, 0.20, 0.95], floor=0.02, ceiling=0.98)
    assert 0.10 <= p <= 0.14


def test_aggregate_multiple_choice_uses_per_option_median_to_resist_outlier():
    dists = [
        {"A": 0.70, "B": 0.20, "C": 0.10},
        {"A": 0.75, "B": 0.15, "C": 0.10},
        {"A": 0.05, "B": 0.05, "C": 0.90},  # single rogue stand-in
    ]
    agg = aggregate_multiple_choice(dists, ["A", "B", "C"], floor=0.01)
    assert math.isclose(sum(agg.values()), 1.0, abs_tol=1e-12)
    assert agg["A"] > 0.65 and agg["C"] < 0.15


def test_calibrate_numeric_percentiles_rescales_million_unit_mismatch_and_prevents_q45961_boundary_pileup():
    # Live Q45961 scenario: bounds are [110_000_000, 150_000_000], model outputs in millions (111..130)
    raw = [(0.05, 108.0), (0.10, 110.0), (0.25, 113.0), (0.50, 118.0), (0.75, 123.0), (0.90, 128.0), (0.95, 132.0)]
    calibrated = calibrate_numeric_percentiles(
        raw, 110_000_000.0, 150_000_000.0, open_lower_bound=False, open_upper_bound=False
    )
    vals = [v for _, v in calibrated]
    # Rescaled into raw dollars, strictly increasing, and never piled up at 110_000_000.0
    assert vals[3] == pytest.approx(118_000_000.0)
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[0] > 110_000_000.0 and vals[-1] < 150_000_000.0

