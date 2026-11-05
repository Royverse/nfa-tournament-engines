import pytest

from nfa_forecaster.parsing import parse_binary, parse_multiple_choice, parse_percentiles

PCTS = (5, 10, 25, 50, 75, 90, 95)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("reasoning...\nProbability: 12%", 0.12),
        ("Probability: 40%\n...changed my mind...\nProbability: 35.5 %", 0.355),
        ("**Probability: 7%**", 0.07),
        ("probability = 3%", 0.03),
    ],
)
def test_parse_binary(text, expected):
    assert parse_binary(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["I think it is unlikely.", "Probability: 150%", "Probability: about a third"])
def test_parse_binary_refuses_to_guess(text):
    assert parse_binary(text) is None


def test_parse_multiple_choice_decimals_and_percentages():
    options = ["Democratic Party", "Republican Party", "Other"]
    assert parse_multiple_choice(
        "Democratic Party: 0.55\nRepublican Party: 0.43\nOther: 0.02", options
    ) == pytest.approx({"Democratic Party": 0.55, "Republican Party": 0.43, "Other": 0.02})
    assert parse_multiple_choice(
        "- Option_Democratic Party: 55%\n- Republican Party: 43%\n- Other: 2%", options
    ) == pytest.approx({"Democratic Party": 0.55, "Republican Party": 0.43, "Other": 0.02})


def test_parse_multiple_choice_needs_every_option_and_a_sane_total():
    options = ["A", "B", "C"]
    assert parse_multiple_choice("A: 0.5\nB: 0.5", options) is None
    assert parse_multiple_choice("A: 0.5\nB: 0.5\nC: 0.5", options) is None


def test_parse_percentiles_plain_numbers_with_thousands_separators():
    lines = [f"Percentile {p}: {v:,}" for p, v in zip(PCTS, [10, 20, 35, 50, 80, 1200, 1500])]
    assert parse_percentiles("\n".join(lines), PCTS) == [
        (0.05, 10), (0.10, 20), (0.25, 35), (0.50, 50), (0.75, 80), (0.90, 1200), (0.95, 1500)
    ]


def test_parse_percentiles_leaves_units_and_bad_orders_to_the_ai_parser():
    with_units = "\n".join(f"Percentile {p}: {p} million" for p in PCTS)
    assert parse_percentiles(with_units, PCTS) is None
    not_increasing = "\n".join(f"Percentile {p}: 5" for p in PCTS)
    assert parse_percentiles(not_increasing, PCTS) is None
    missing = "\n".join(f"Percentile {p}: {p}" for p in PCTS[:-1])
    assert parse_percentiles(missing, PCTS) is None
