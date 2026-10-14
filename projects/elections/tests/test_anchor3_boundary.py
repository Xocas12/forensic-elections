"""Anchor 3 at the edge of the support: turnout cannot exceed 100, and the density must know it.

Issue #14. The location density used to be convolved with zero padding past 0 and 100, and
the peak finder counted only interior sign changes. A distribution whose density rises all
the way to complete turnout was then reported as an interior mode a few points short of 100,
and the report described that mode with a sentence typed by hand rather than read from the
table. Everything here is synthetic; no test reads the real precinct files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from elections.analysis import turnout_bimodality as tb

ROOT = Path(__file__).resolve().parents[3]


def _rising_into_100(seed: int = 0) -> np.ndarray:
    """The issue's synthetic check: a principal mode at 55 and a density rising into 100."""
    rng = np.random.default_rng(seed)
    principal = rng.normal(55.0, 10.0, 90_000)
    edge = 100.0 * rng.beta(8.0, 1.0, 20_000)
    return np.clip(np.concatenate([principal, edge]), 0.0, 100.0)


def _tidy_from_turnout(turnout: np.ndarray, share: np.ndarray) -> pd.DataFrame:
    """A tidy-shaped frame whose ballots-in-boxes turnout and winner's share are the inputs."""
    registered = 10_000
    ballots = np.rint(turnout / 100.0 * registered).astype(int)
    winner = np.rint(share / 100.0 * ballots).astype(int)
    return pd.DataFrame(
        {
            "election": "2018-presidential",
            "region": "Testonia",
            "uik": np.arange(turnout.size),
            "registered": registered,
            "in_mobile_boxes": 0,
            "in_stationary_boxes": ballots,
            "valid": ballots,
            "invalid": 0,
            "winner_votes": winner,
        }
    )


def _load_run_analysis():
    spec = importlib.util.spec_from_file_location(
        "run_analysis_under_test", ROOT / "scripts" / "run_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_density_keeps_its_mass_at_the_edges_of_the_support():
    """Zero padding throws away the half of each kernel that falls past 0 or 100."""
    values = _rising_into_100()
    density = tb._density(values, tb.MODE_BANDWIDTH_PP)
    assert density.sum() == pytest.approx(values.size, rel=1e-9)


def test_a_density_rising_into_100_peaks_at_100_not_a_few_points_short():
    density = tb._density(_rising_into_100(), tb.MODE_BANDWIDTH_PP)
    peaks = [float(tb._GRID[i]) for i in tb._peak_indices(density)]
    assert peaks[-1] == 100.0
    assert not any(90.0 < p < 100.0 for p in peaks), f"a boundary-pulled false peak: {peaks}"
    # slope at the edge is flat or rising, never the sag zero padding produces
    assert density[-1] >= density[-30]


def test_mode_locations_report_a_maximum_at_complete_turnout_and_flag_it():
    turnout = _rising_into_100()
    share = np.where(turnout > 90.0, 88.0, 70.0)
    table = tb.turnout_mode_locations(
        _tidy_from_turnout(turnout, share), election="2018-presidential"
    )
    for weighting in ("stations", "registered"):
        sub = table[table["weight_by"] == weighting]
        top = sub.loc[sub["mode_turnout_pct"].idxmax()]
        assert top["mode_turnout_pct"] == 100.0
        assert bool(top["at_boundary"])
        # the one-sided window at 100 holds only the near-complete-turnout stations
        assert top["winner_share_at_mode_pct"] == pytest.approx(88.0, abs=0.1)
        interior = sub[~sub["at_boundary"]]
        assert interior["mode_turnout_pct"].between(50.0, 60.0).all()


def test_an_interior_mode_well_short_of_100_is_still_reported_where_it_is():
    """Reflection must not drag a genuine interior mode to the boundary."""
    rng = np.random.default_rng(1)
    values = np.clip(
        np.concatenate([rng.normal(55.0, 8.0, 60_000), rng.normal(85.0, 2.0, 15_000)]), 0, 100
    )
    peaks = [float(tb._GRID[i]) for i in tb._peak_indices(tb._density(values, 2.0))]
    assert any(abs(p - 85.0) < 1.0 for p in peaks)
    assert 100.0 not in peaks


def test_the_bootstrap_reflects_draws_back_into_the_support_instead_of_piling_them_on_100():
    draws = np.array([-3.0, 0.0, 42.0, 100.0, 104.0])
    assert tb._reflect_into_support(draws).tolist() == [3.0, 0.0, 42.0, 100.0, 96.0]


def test_the_critical_bandwidth_counts_a_maximum_at_100_as_a_mode():
    values = _rising_into_100()
    h = tb.critical_bandwidth(values)
    below = tb._density(values, 0.95 * h)
    assert tb._n_modes(below) >= 2
    assert tb._GRID[tb._peak_indices(below)[-1]] == 100.0


def test_high_mode_location_interval_is_a_percentile_interval_around_the_estimate():
    turnout = _rising_into_100()
    tidy = _tidy_from_turnout(turnout, np.full(turnout.size, 70.0))
    result = tb.high_mode_location_interval(tidy, election="2018-presidential", n_boot=19, seed=3)
    assert result["mode_turnout_pct"] == 100.0
    assert result["ci_low"] <= result["mode_turnout_pct"] <= result["ci_high"]
    assert 0.0 <= result["share_at_boundary"] <= 1.0
    assert result["n_boot"] == 19
    assert "not evidence of falsification" in result["caveat"]


# --------------------------------------------------------------------------- the report


def _mode_row(election, weight_by, turnout, share, mass, at_boundary=False):
    return {
        "election": election,
        "weight_by": weight_by,
        "mode_turnout_pct": turnout,
        "winner_share_at_mode_pct": share,
        "mass_within_2_5pp": mass,
        "at_boundary": at_boundary,
    }


def _location_row(election, weight_by, estimate, low, high, at_100):
    return {
        "election": election,
        "weight_by": weight_by,
        "mode_turnout_pct": estimate,
        "ci_low": low,
        "ci_high": high,
        "share_at_boundary": at_100,
    }


def test_the_location_sentence_is_read_off_the_mode_table_and_does_not_overclaim():
    """The committed 2018 rows had second modes at 93.6 and 94.8 under a 'within two points'
    sentence. A table like that must produce a sentence that says so."""
    run_analysis = _load_run_analysis()
    modes = pd.DataFrame(
        [
            _mode_row("2011-duma", "registered", 51.5, 34.5, 0.185),
            _mode_row("2011-duma", "registered", 98.6, 83.7, 0.043),
            _mode_row("2011-duma", "stations", 52.8, 37.9, 0.142),
            _mode_row("2011-duma", "stations", 98.8, 82.7, 0.077),
            _mode_row("2018-presidential", "registered", 23.2, 78.4, 0.0002),
            _mode_row("2018-presidential", "registered", 61.1, 72.9, 0.235),
            _mode_row("2018-presidential", "registered", 93.6, 88.1, 0.046),
            _mode_row("2018-presidential", "stations", 61.7, 73.5, 0.180),
            _mode_row("2018-presidential", "stations", 94.8, 87.8, 0.074),
        ]
    )
    location = pd.DataFrame(
        [
            _location_row("2011-duma", "registered", 98.6, 98.0, 99.0, 0.0),
            _location_row("2011-duma", "stations", 98.8, 98.3, 99.1, 0.0),
            _location_row("2018-presidential", "registered", 93.6, 92.9, 94.4, 0.0),
            _location_row("2018-presidential", "stations", 94.8, 94.0, 95.5, 0.0),
        ]
    )
    text = "\n".join(run_analysis.mode_location_lines(modes, location, 199))

    assert "in 2 of them the highest-turnout mode lies within 2 points" in text
    assert "every one of them" not in text
    assert "from 93.6 to 98.8" in text
    # each case's sentence names its own modes and its own interval
    assert "highest-turnout mode at 93.6%" in text
    assert "[92.9, 94.4]" in text
    # the dropped tail mode is named, not silently omitted
    assert "not shown" in text and "2018 presidential, registered, at 23.2" in text
    assert "| 23.2 |" not in text


def test_the_location_sentence_names_a_boundary_maximum_and_agrees_when_all_cases_do():
    run_analysis = _load_run_analysis()
    modes = pd.DataFrame(
        [
            _mode_row(e, w, 55.0, 40.0, 0.2)
            for e in ("2011-duma", "2018-presidential")
            for w in ("registered", "stations")
        ]
        + [
            _mode_row(e, w, 100.0, 85.0, 0.05, at_boundary=True)
            for e in ("2011-duma", "2018-presidential")
            for w in ("registered", "stations")
        ]
    ).sort_values(["election", "weight_by", "mode_turnout_pct"])
    location = pd.DataFrame(
        [
            _location_row(e, w, 100.0, 100.0, 100.0, 1.0)
            for e in ("2011-duma", "2018-presidential")
            for w in ("registered", "stations")
        ]
    )
    text = "\n".join(run_analysis.mode_location_lines(modes, location, 199))
    assert "There is a second mode in all 4 election-and-weighting cases" in text
    assert "in every one of them it lies within 2 points of complete turnout" in text
    assert "100.0 (boundary)" in text
    assert "a maximum at the boundary" in text
    assert "all at 100.0" in text
    assert "+45.0 to +45.0 points" in text
    assert "not shown" not in text


def test_the_unimodality_sentence_follows_the_p_values():
    run_analysis = _load_run_analysis()
    mixed = pd.DataFrame({"election": ["2011-duma", "2018-presidential"], "pvalue": [0.001, 0.4]})
    text = " ".join(run_analysis.unimodality_verdict_lines(mixed, 999))
    assert "rejected at 5% in the 2011 Duma and not in the 2018 presidential" in text
    assert "at that floor" not in text
