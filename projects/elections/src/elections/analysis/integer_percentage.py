"""Anchor 1: excess mass at integer percentages (the sawtooth).

Method
------
Kobak, D., S. Shpilkin and M. S. Pshenichnikov, 2016. "Integer percentages as electoral
falsification fingerprints", *Annals of Applied Statistics* 10(1), 54-73. Turnout and vote
share reported as percentages pile up on whole numbers far more often than binomial noise
allows, and the excess concentrates in particular regions.

The estimator is :func:`forensics_core.digits.integer_pct.integer_excess`, whose null redraws
each station's numerator as ``Binomial(registered, observed share)``; that is the paper's own
null, so the functions here choose inputs and strata and do not reimplement statistics.

What must be reproduced (``docs/validation_anchors.md``, Tier 1 item 1)
----------------------------------------------------------------------
1. A significant positive excess nationally in 2011 and in 2018.
2. The excess concentrated in the regions the paper names.
3. The excess vanishing on the Poland 2010 and Spain 2011 control tables in the same
   supplement. This last one is what shows the test is not detecting arithmetic, and it is
   the number the ``gosplan`` project will inherit, since it cannot compute a false-positive
   rate for itself.

Items 1 and 2 are computed here from the Russian tables. Item 3 needs the Poland and Spain
control tables, which are not in ``data/raw``; :mod:`elections.analysis` does not fabricate
them, and ``RESULTS.md`` records that control as outstanding rather than passed.

The trap this must not fall into (``docs/known_traps.md``, trap 1)
------------------------------------------------------------------
Small stations land on integer percentages honestly, and about 18 per cent of the 2011 sample
has 250 or fewer registered voters. A pooled national statistic is therefore not a result
until it is shown to survive size conditioning: hence
:func:`integer_percentage_excess_by_size_band`, and hence the requirement that any headline
number be reported next to the count of stations the denominator floor removed.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pandas as pd
from forensics_core.digits import IntegerExcessResult
from forensics_core.digits.integer_pct import integer_excess, integer_excess_by_group

from elections.clean.derive import ballots_counted, turnout_ballots_in_boxes, winner_share
from elections.clean.schema import ELECTIONS
from elections.features.size import SIZE_BAND_LABELS, iter_size_strata

__all__ = [
    "PERCENTAGE_QUANTITIES",
    "integer_percentage_excess",
    "integer_percentage_excess_by_region",
    "integer_percentage_excess_by_size_band",
]

#: The two percentages the paper analyses, as columns of the derived tidy frame.
PERCENTAGE_QUANTITIES = ("turnout_boxes", "winner_share")

Quantity = Literal["turnout_boxes", "winner_share"]

#: Which protocol line is the binomial denominator of each quantity. Recorded in every
#: result's ``settings`` so that a number can never be read without knowing what it is a
#: share of.
DENOMINATOR_COLUMN: dict[str, str] = {
    "turnout_boxes": "registered",
    "winner_share": "ballots_counted",
}


def _select(tidy: pd.DataFrame, election: str) -> pd.DataFrame:
    """The rows of one election, or a loud failure naming the valid keys."""
    if election not in ELECTIONS:
        raise ValueError(f"unknown election {election!r}; expected one of {list(ELECTIONS)}")
    if "election" not in tidy.columns:
        raise ValueError(
            "tidy frame has no 'election' column; pass elections.clean.build_tidy output"
        )
    rows = tidy.loc[tidy["election"] == election]
    if rows.empty:
        raise ValueError(f"no rows for election {election!r} in the frame given")
    return rows


def _inputs(rows: pd.DataFrame, quantity: str) -> tuple[pd.Series, pd.Series]:
    """The percentage and its binomial denominator, both computed here.

    The percentage is recomputed from the canonical protocol columns rather than read off the
    frame, so that the denominator in the null is the same quantity that formed the ratio.
    """
    if quantity not in PERCENTAGE_QUANTITIES:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {list(PERCENTAGE_QUANTITIES)}"
        )
    if quantity == "turnout_boxes":
        return turnout_ballots_in_boxes(rows, as_percent=True), rows["registered"].astype(float)
    return winner_share(rows, as_percent=True), ballots_counted(rows)


def integer_percentage_excess(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    n_mc: int = 200,
    seed: int | None = None,
) -> IntegerExcessResult:
    """Excess mass at integer percentages for one election, nationally.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame from :func:`elections.clean.build_tidy`. Percentages are computed here
        from the canonical columns rather than taken from the frame, so that the denominator
        used is unambiguous and recorded in the result's ``settings``.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    quantity : {"turnout_boxes", "winner_share"}, default "turnout_boxes"
        Which percentage to test. ``turnout_boxes`` is ballots in boxes over registered
        voters, with the registered voters as the binomial denominator; ``winner_share`` is
        the winner's votes over valid plus invalid ballots, with that total as the
        denominator.
    tolerance : float, default 0.05
        Half-width, in percentage points, of the window that counts as "on an integer".
    min_denominator : int, default 100
        Stations with a smaller denominator are excluded and counted.
    n_mc : int, default 200
        Monte Carlo replications of the binomial null.
    seed : int, optional
        Seeds the generator.

    Returns
    -------
    forensics_core.digits.IntegerExcessResult
        As returned by :func:`forensics_core.digits.integer_pct.integer_excess`, with the
        election, the quantity and the denominator column added to ``settings``.
    """
    rows = _select(tidy, election)
    pct, denominators = _inputs(rows, quantity)
    result = integer_excess(
        pct.to_numpy(),
        denominators.to_numpy(),
        tolerance=tolerance,
        min_denominator=min_denominator,
        n_mc=n_mc,
        seed=seed,
    )
    return dataclasses.replace(
        result,
        settings={
            **result.settings,
            "election": election,
            "quantity": quantity,
            "denominator_column": DENOMINATOR_COLUMN[quantity],
            "n_stations": len(rows),
        },
    )


def integer_percentage_excess_by_region(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    n_mc: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    """The same excess computed within each federal subject, for the regional ranking.

    Anchor 1 item 2 is a claim about *which* regions carry the excess, so the comparison must
    be a ranking rather than a count of regions that reject: with roughly 85 regions, several
    will reject by chance (``docs/known_traps.md``, trap 6).

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    quantity : {"turnout_boxes", "winner_share"}, default "turnout_boxes"
        As in :func:`integer_percentage_excess`.
    tolerance, min_denominator, n_mc, seed
        As in :func:`integer_percentage_excess`.

    Returns
    -------
    pandas.DataFrame
        One row per region, as returned by
        :func:`forensics_core.digits.integer_pct.integer_excess_by_group`, sorted by excess
        per unit descending. Carries ``election`` and ``quantity`` columns so that tables for
        the two elections can be concatenated without losing which is which, and a
        ``rank`` column so the ranking is explicit rather than positional.
    """
    rows = _select(tidy, election)
    pct, denominators = _inputs(rows, quantity)
    table = integer_excess_by_group(
        pct.to_numpy(),
        denominators.to_numpy(),
        rows["region"].to_numpy(),
        tolerance=tolerance,
        min_denominator=min_denominator,
        n_mc=n_mc,
        seed=seed,
    )
    table = table.rename(columns={"group": "region"})
    table = table.sort_values("excess_per_unit", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    table.insert(0, "election", election)
    table.insert(1, "quantity", quantity)
    table.insert(2, "rank", range(1, len(table) + 1))
    return table


def integer_percentage_excess_by_size_band(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    n_mc: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    """The excess within each precinct-size band: the trap-1 control.

    The headline national number is not interpretable on its own. This table is what shows
    whether the excess is present among large stations, where integer percentages are not
    arithmetically favoured, or only among small ones, where they are.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    quantity : {"turnout_boxes", "winner_share"}, default "turnout_boxes"
        As in :func:`integer_percentage_excess`.
    tolerance, min_denominator, n_mc, seed
        As in :func:`integer_percentage_excess`.

    Returns
    -------
    pandas.DataFrame
        One row per band of :data:`elections.features.size.SIZE_BAND_LABELS`, with the fields
        of :class:`forensics_core.digits.IntegerExcessResult` plus the band's station count
        and the number of stations the denominator floor removed within it. Bands are in
        ascending size order, not sorted by the statistic, because the question this table
        answers is whether the excess grows or dies with station size.
    """
    rows = _select(tidy, election)
    records: list[dict[str, object]] = []
    for label, band in iter_size_strata(rows):
        pct, denominators = _inputs(band, quantity)
        result = integer_excess(
            pct.to_numpy(),
            denominators.to_numpy(),
            tolerance=tolerance,
            min_denominator=min_denominator,
            n_mc=n_mc,
            seed=seed,
        )
        n_tested = int(len(band) - result.n_excluded_small)
        records.append(
            {
                "election": election,
                "quantity": quantity,
                "size_band": label,
                "n_stations": len(band),
                "n_tested": n_tested,
                "n_excluded_small": int(result.n_excluded_small),
                "observed": float(result.observed),
                "expected_mean": float(result.expected_mean),
                "expected_sd": float(result.expected_sd),
                "excess": float(result.excess),
                "excess_per_unit": float(result.excess / n_tested) if n_tested else float("nan"),
                "z": float(result.test.statistic),
                "pvalue": float(result.test.pvalue),
            }
        )
    table = pd.DataFrame.from_records(records)
    if not table.empty:
        order = {label: index for index, label in enumerate(SIZE_BAND_LABELS)}
        table = table.sort_values("size_band", key=lambda s: s.map(order)).reset_index(drop=True)
    return table
