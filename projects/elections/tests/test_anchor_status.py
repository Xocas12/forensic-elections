"""The status the documents give the anchors must be the status the anchors' own rules allow.

``docs/validation_anchors.md`` fixed what counts as reproduced for anchor 1 before the run:
all three items, or a partial failure. Item 3, the Poland and Spain control, has not run, so
anchor 1 is a partial failure, and the top-level documents must say so rather than "reproduced".
Anchor 3's work order put its content in the confound, the ``heterogeneous_mixture``
calibration, which has not run either. And both were computed before gate G1 signed off a
pre-registration, which ``RESULTS.md`` must state.

The national anchor-1 table must also carry what its work order asked for: Monte Carlo
p-values next to the normal approximation, and the ``min_denominator`` sweep.

``RESULTS.md`` is generated, so the generator is checked as well as its output: a re-run must
not quietly drop any of this.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from elections.analysis import integer_percentage

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "projects" / "elections" / "results"
GENERATOR = ROOT / "scripts" / "run_analysis.py"

#: Status claims the anchors' own rules do not allow.
OVERCLAIMS = (
    "Two of the three validation anchors are reproduced",
    "anchors 1 and 3 reproduced",
    "Anchors 1 and\n3 are reproduced",
    "reproduced with their controls",
    "the control that matters",
)


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Line breaks folded to spaces, so a phrase wrapped across lines is still found."""
    return " ".join(text.split())


@pytest.mark.parametrize("name", ["README.md", "ROADMAP.md", "RESULTS.md"])
def test_anchor_1_is_described_as_a_partial_failure(name):
    assert "partial failure" in _flat(_read(name)), (
        f"{name} must describe anchor 1 as a partial failure until the Poland/Spain control runs"
    )


@pytest.mark.parametrize("name", ["README.md", "ROADMAP.md", "RESULTS.md"])
def test_no_document_claims_more_than_the_anchor_rules_allow(name):
    text = _read(name)
    flat = _flat(text)
    for claim in OVERCLAIMS:
        assert claim not in text and _flat(claim) not in flat, f"{name} still says {claim!r}"


@pytest.mark.parametrize("name", ["README.md", "ROADMAP.md", "RESULTS.md"])
def test_anchor_3_status_names_the_unrun_heterogeneous_mixture_calibration(name):
    assert "heterogeneous_mixture" in _read(name)


def test_results_states_it_was_computed_before_g1():
    assert "computed before G1" in _flat(_read("RESULTS.md"))


def test_roadmap_does_not_mark_tasks_8_and_10_done():
    for line in _read("ROADMAP.md").splitlines():
        if line.startswith(("| 8 |", "| 10 |")):
            assert "**done**" not in line, line


def test_the_national_table_carries_monte_carlo_pvalues():
    with (RESULTS_DIR / "anchor1_national.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    for row in rows:
        mc = float(row["mc_pvalue"])
        assert 0.0 < mc <= 1.0
    assert "| MC p |" in _read("RESULTS.md")


def test_the_national_table_carries_the_min_denominator_sweep():
    with (RESULTS_DIR / "anchor1_min_denominator_sweep.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    floors = tuple(int(f) for f in integer_percentage.MIN_DENOMINATOR_SWEEP)
    assert floors == (100, 250, 500, 1000)
    cells = {(r["election"], r["quantity"], int(r["min_denominator"])) for r in rows}
    for election in ("2011-duma", "2018-presidential"):
        for quantity in integer_percentage.PERCENTAGE_QUANTITIES:
            for floor in floors:
                assert (election, quantity, floor) in cells
    text = _read("RESULTS.md")
    assert "### The denominator floor" in text
    assert "| floor 100 | floor 250 | floor 500 | floor 1,000 |" in text


def test_the_generator_writes_all_of_it():
    source = _flat(GENERATOR.read_text(encoding="utf-8"))
    for needle in (
        '"mc_pvalue": res.test.details["mc_pvalue"]',
        "integer_percentage_excess_by_min_denominator(",
        "anchor1_min_denominator_sweep.csv",
        "These results were computed before G1.",
        "Anchor 1 is a partial failure by its own rule.",
        "heterogeneous_mixture",
    ):
        assert needle in source, f"scripts/run_analysis.py no longer writes {needle!r}"


def test_min_denominator_sweep_gives_one_row_per_floor(tidy):
    table = integer_percentage.integer_percentage_excess_by_min_denominator(
        tidy, election="2011-duma", n_mc=20, seed=1, floors=(100, 1000, 2000)
    )
    assert list(table["min_denominator"]) == [100, 1000, 2000]
    assert {"n_tested", "n_excluded_small", "excess", "z", "pvalue", "mc_pvalue"} <= set(
        table.columns
    )
    assert (table["n_tested"] + table["n_excluded_small"] == table["n_stations"]).all()
    # a stricter floor can only exclude more stations
    assert table["n_excluded_small"].is_monotonic_increasing
    assert table["mc_pvalue"].between(1 / 21, 1.0).all()


def test_the_sweep_at_the_default_floor_is_the_headline_number(tidy):
    headline = integer_percentage.integer_percentage_excess(
        tidy, election="2018-presidential", quantity="winner_share", n_mc=20, seed=3
    )
    table = integer_percentage.integer_percentage_excess_by_min_denominator(
        tidy, election="2018-presidential", quantity="winner_share", n_mc=20, seed=3
    )
    row = table.loc[table["min_denominator"] == 100].iloc[0]
    assert row["excess"] == pytest.approx(headline.excess)
    assert row["mc_pvalue"] == pytest.approx(headline.test.details["mc_pvalue"])
