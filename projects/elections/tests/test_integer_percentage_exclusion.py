"""Anchor 1's null with the paper's over-99 exclusion, and the honest-data bias beside it.

Kobak, Shpilkin and Pshenichnikov (2016), section 3.1, drop every station whose turnout or
result is over 99 per cent before counting integer percentages. Without that rule the
binomial-redraw null over-predicts hits at 100 per cent (a station at 100 is a hit in every
draw, and a station at 97-99.9 adds the chance its redraw lands on its denominator), so on
honest data with near-complete turnout the test reports a negative "excess". These tests
build honest synthetic stations, where there is nothing to find.

The rule removes most of that bias, not all of it: a kept station observed at 98.5 per cent
can still be redrawn onto 100. That residual is what :func:`integer_percentage_honest_bias`
exists to make visible, and the last tests here check that it does.

Everything here is synthetic; no test reads the real precinct tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from elections.analysis import integer_percentage as ip

ELECTION = "2011-duma"


def _honest_tidy(
    n: int,
    *,
    turnout: tuple[float, float],
    seed: int,
    registered: tuple[int, int] = (101, 250),
    share: tuple[float, float] = (0.45, 0.65),
    region: str = "R",
) -> pd.DataFrame:
    """Stations whose every count is a binomial draw at a continuous true share."""
    rng = np.random.default_rng(seed)
    reg = rng.integers(registered[0], registered[1] + 1, n)
    boxes = rng.binomial(reg, rng.uniform(*turnout, n))
    invalid = np.zeros(n, dtype=np.int64)
    winner = rng.binomial(boxes, rng.uniform(*share, n))
    return pd.DataFrame(
        {
            "election": ELECTION,
            "region": region,
            "registered": reg,
            "in_mobile_boxes": np.zeros(n, dtype=np.int64),
            "in_stationary_boxes": boxes,
            "valid": boxes,
            "invalid": invalid,
            "winner_votes": winner,
        }
    )


def _stations(rows: list[tuple[int, int, int, str]]) -> pd.DataFrame:
    """Hand-built stations: (registered, ballots in boxes, winner's votes, region)."""
    frame = pd.DataFrame(
        rows, columns=["registered", "in_stationary_boxes", "winner_votes", "region"]
    )
    frame["election"] = ELECTION
    frame["in_mobile_boxes"] = 0
    frame["valid"] = frame["in_stationary_boxes"]
    frame["invalid"] = 0
    return frame


def _mixture(seed: int, n: int = 3000) -> pd.DataFrame:
    """Small honest stations, 15% of them with true turnout of 97-99.9 per cent.

    In the real 101-250 band, 6 to 10 per cent of stations report 99 per cent or more.
    """
    near = round(0.15 * n)
    return pd.concat(
        [
            _honest_tidy(n - near, turnout=(0.4, 0.9), seed=seed),
            _honest_tidy(near, turnout=(0.97, 0.999), seed=seed + 1000),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def near_complete() -> pd.DataFrame:
    """Honest stations of 101-250 voters, all with true turnout between 97 and 99.9 per cent."""
    return _honest_tidy(3000, turnout=(0.97, 0.999), seed=11)


@pytest.fixture(scope="module")
def ordinary() -> pd.DataFrame:
    """Honest stations of 101-250 voters at ordinary turnout, where the null holds."""
    return _honest_tidy(3000, turnout=(0.4, 0.8), seed=12)


SEEDS = range(20, 28)


def test_12_honest_small_stations_read_as_no_excess_by_default():
    """The published procedure read honest data of this shape as a deficit (mean z near -3)."""
    z = [
        ip.integer_percentage_excess(
            _mixture(seed), election=ELECTION, n_mc=200, seed=5
        ).test.statistic
        for seed in SEEDS
    ]
    assert abs(np.mean(z)) < 1.25, f"mean z over {len(z)} honest samples is {np.mean(z):.2f}"


def test_12_without_the_rule_the_same_honest_data_read_as_a_deficit():
    """The miscalibration the rule removes is real: this is what was published."""
    z = [
        ip.integer_percentage_excess(
            _mixture(seed), election=ELECTION, exclude_over_pct=None, n_mc=200, seed=5
        ).test.statistic
        for seed in SEEDS
    ]
    assert np.mean(z) < -2.0


def test_12_the_rule_removes_every_observed_hit_at_100_and_most_of_the_deficit(near_complete):
    on = ip.integer_percentage_excess(near_complete, election=ELECTION, n_mc=200, seed=5)
    off = ip.integer_percentage_excess(
        near_complete, election=ELECTION, exclude_over_pct=None, n_mc=200, seed=5
    )
    assert on.test.details["observed_per_integer"][100] == 0
    assert off.test.details["observed_per_integer"][100] > 0
    assert on.settings["n_excluded_high"] > 0
    assert off.settings["n_excluded_high"] == 0
    assert abs(on.excess) < 0.5 * abs(off.excess)


def test_12_the_rule_drops_on_turnout_or_winner_share_whichever_quantity_is_tested():
    frame = _stations(
        [
            (200, 200, 100, "A"),  # turnout 100%
            (200, 199, 100, "A"),  # turnout 99.5%
            (200, 198, 100, "A"),  # turnout 99.0%: not over 99, kept
            (200, 120, 119, "A"),  # winner's share 99.2%
            (200, 120, 60, "A"),  # ordinary
            (200, 130, 70, "A"),  # ordinary
        ]
    )
    for quantity in ip.PERCENTAGE_QUANTITIES:
        result = ip.integer_percentage_excess(
            frame, election=ELECTION, quantity=quantity, n_mc=20, seed=1
        )
        assert result.settings["n_excluded_high"] == 3
        assert result.settings["n_stations"] == 6
        assert result.settings["exclude_over_pct"] == ip.PAPER_EXCLUSION_PCT
        assert result.test.n == 3


def test_12_the_rule_is_the_papers_99_per_cent():
    assert ip.PAPER_EXCLUSION_PCT == 99.0


def test_12_size_bands_apply_the_rule_and_count_it(near_complete):
    table = ip.integer_percentage_excess_by_size_band(
        near_complete, election=ELECTION, n_mc=200, seed=5
    )
    assert "n_excluded_high" in table.columns
    band = table.set_index("size_band").loc["101-250"]
    whole = ip.integer_percentage_excess(near_complete, election=ELECTION, n_mc=20, seed=5)
    assert band["n_excluded_high"] == whole.settings["n_excluded_high"] > 0
    assert (
        band["n_tested"] + band["n_excluded_high"] + band["n_excluded_small"] == band["n_stations"]
    )


def test_12_a_band_the_rule_empties_is_reported_rather_than_raising():
    frame = _stations([(3000, 3000, 2000, "A"), (3000, 2990, 1500, "A"), (200, 120, 60, "A")])
    table = ip.integer_percentage_excess_by_size_band(frame, election=ELECTION, n_mc=20, seed=1)
    band = table.set_index("size_band").loc["2001-5000"]
    assert band["n_excluded_high"] == 2
    assert band["n_tested"] == 0
    assert np.isnan(band["z"])


def test_12_regions_apply_the_rule_and_a_region_it_empties_stays_in_the_table():
    ordinary = _honest_tidy(300, turnout=(0.5, 0.7), seed=3, region="ordinary")
    complete = _stations([(200, 200, 199, "complete")] * 20)
    frame = pd.concat([ordinary, complete], ignore_index=True)
    table = ip.integer_percentage_excess_by_region(frame, election=ELECTION, n_mc=50, seed=1)
    by_region = table.set_index("region")
    assert by_region.loc["complete", "n_excluded_high"] == 20
    assert by_region.loc["complete", "n"] == 0
    assert by_region.loc["complete", "n_stations"] == 20
    assert np.isnan(by_region.loc["complete", "excess_per_unit"])
    assert by_region.loc["ordinary", "n_excluded_high"] == 0
    assert by_region.loc["ordinary", "n"] == 300
    assert list(table["rank"]) == list(range(1, len(table) + 1))


def test_12_honest_bias_is_near_zero_where_the_null_holds(ordinary):
    table = ip.integer_percentage_honest_bias(ordinary, election=ELECTION, n_rep=6, n_mc=50, seed=2)
    assert table["stratum"].iloc[0] == "all"
    assert "101-250" in set(table["stratum"])
    overall = table.set_index("stratum").loc["all"]
    assert overall["n_rep"] == 6
    assert abs(overall["honest_bias"]) < 4 * overall["honest_bias_se"]
    assert abs(overall["honest_z_mean"]) < 1.0


def test_12_honest_bias_makes_the_residual_visible_and_the_unruled_bias_larger(near_complete):
    """All stations near 100 per cent: the rule leaves a residual, and the bias column shows it."""
    on = ip.integer_percentage_honest_bias(
        near_complete, election=ELECTION, n_rep=6, n_mc=50, seed=2
    )
    off = ip.integer_percentage_honest_bias(
        near_complete, election=ELECTION, exclude_over_pct=None, n_rep=6, n_mc=50, seed=2
    )
    on_all = on.set_index("stratum").loc["all"]
    off_all = off.set_index("stratum").loc["all"]
    assert on_all["honest_bias"] < -3 * on_all["honest_bias_se"]
    assert off_all["honest_bias"] < -3 * off_all["honest_bias_se"]
    assert off_all["honest_bias"] < 2 * on_all["honest_bias"]


def test_12_an_invalid_ceiling_is_refused(near_complete):
    with pytest.raises(ValueError, match="exclude_over_pct"):
        ip.integer_percentage_excess(near_complete, election=ELECTION, exclude_over_pct=150.0)
