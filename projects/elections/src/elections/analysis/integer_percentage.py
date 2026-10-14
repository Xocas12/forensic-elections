"""Anchor 1: excess mass at integer percentages (the sawtooth).

Method
------
Kobak, D., S. Shpilkin and M. S. Pshenichnikov, 2016. "Integer percentages as electoral
falsification fingerprints", *Annals of Applied Statistics* 10(1), 54-73. Turnout and vote
share reported as percentages pile up on whole numbers far more often than binomial noise
allows, and the excess concentrates in particular regions.

The estimator is :func:`forensics_core.digits.integer_pct.integer_excess`, whose null redraws
each station's numerator as ``Binomial(denominator, observed share)``. The functions here
choose inputs and strata and do not reimplement statistics.

That redraw is the paper's null only together with the rule the paper applies before counting
(section 3.1): every station whose turnout or winner's share is over 99 per cent is excluded,
because 100 per cent is always an integer. Without it the null over-predicts hits at 100: a
station observed at 100 per cent is a hit in every draw, and every station at 97-99.9 per cent
adds the chance that its redraw lands exactly on its denominator, so the expected count at
100 sits above what honest data produce and the excess is biased low, most of all among small
stations and in regions with near-complete turnout. Every function here therefore applies
that exclusion by default (:data:`PAPER_EXCLUSION_PCT`) and reports how many stations it
removed, next to the count the denominator floor removed.

The rule removes most of that bias but not all of it: a kept station observed at, say, 98.5
per cent can still be redrawn onto 100 in the null, where no kept station can be in the data.
Whatever calibration error is left is measured, not assumed away:
:func:`integer_percentage_honest_bias` simulates honest data at the observed shares and real
denominators, runs it through the same procedure, and reports the mean excess it produces.

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

import numpy as np
import pandas as pd
from forensics_core.digits import IntegerExcessResult
from forensics_core.digits.integer_pct import integer_excess, integer_excess_by_group

from elections.clean.derive import ballots_counted, turnout_ballots_in_boxes, winner_share
from elections.clean.schema import ELECTIONS
from elections.features.size import SIZE_BAND_LABELS, iter_size_strata

__all__ = [
    "PAPER_EXCLUSION_PCT",
    "PERCENTAGE_QUANTITIES",
    "integer_percentage_excess",
    "integer_percentage_excess_by_region",
    "integer_percentage_excess_by_size_band",
    "integer_percentage_honest_bias",
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

#: Kobak, Shpilkin and Pshenichnikov (2016), section 3.1: "Prior to this counting, we excluded
#: all polling stations with turnout or result being over 99%". A station is dropped when
#: either percentage is above this, whichever of the two is being tested.
PAPER_EXCLUSION_PCT = 99.0


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


def _check_ceiling(exclude_over_pct: float | None) -> None:
    if exclude_over_pct is None:
        return
    if not np.isfinite(exclude_over_pct) or not 0.0 < exclude_over_pct <= 100.0:
        raise ValueError(
            f"exclude_over_pct must lie in (0, 100] or be None; got {exclude_over_pct}"
        )


def _over_ceiling(rows: pd.DataFrame, exclude_over_pct: float | None) -> np.ndarray:
    """Stations the paper's rule drops: turnout or winner's share above ``exclude_over_pct``.

    Both percentages are checked whichever one is tested, as in the paper. A station whose
    percentage is undefined is not dropped here; ``integer_excess`` drops and counts it.
    """
    if exclude_over_pct is None:
        return np.zeros(len(rows), dtype=bool)
    turnout = turnout_ballots_in_boxes(rows, as_percent=True).to_numpy(dtype=float)
    share = winner_share(rows, as_percent=True).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return (turnout > exclude_over_pct) | (share > exclude_over_pct)


def integer_percentage_excess(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    exclude_over_pct: float | None = PAPER_EXCLUSION_PCT,
    n_mc: int = 200,
    seed: int | np.random.Generator | None = None,
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
    exclude_over_pct : float or None, default :data:`PAPER_EXCLUSION_PCT`
        Stations whose turnout or winner's share is above this are excluded before counting,
        as in the paper, and counted in ``settings["n_excluded_high"]``. ``None`` turns the
        rule off, which leaves the null over-predicting hits at 100 per cent.
    n_mc : int, default 200
        Monte Carlo replications of the binomial null.
    seed : int or numpy.random.Generator, optional
        Seeds the generator.

    Returns
    -------
    forensics_core.digits.IntegerExcessResult
        As returned by :func:`forensics_core.digits.integer_pct.integer_excess` on the
        stations that survive the exclusion, with the election, the quantity, the
        denominator column, the station count before any exclusion (``n_stations``), the
        ceiling used (``exclude_over_pct``) and the stations it removed
        (``n_excluded_high``) added to ``settings``.
    """
    _check_ceiling(exclude_over_pct)
    rows = _select(tidy, election)
    high = _over_ceiling(rows, exclude_over_pct)
    pct, denominators = _inputs(rows.loc[~high], quantity)
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
            "exclude_over_pct": exclude_over_pct,
            "n_excluded_high": int(high.sum()),
        },
    )


def integer_percentage_excess_by_region(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    exclude_over_pct: float | None = PAPER_EXCLUSION_PCT,
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
    tolerance, min_denominator, exclude_over_pct, n_mc, seed
        As in :func:`integer_percentage_excess`.

    Returns
    -------
    pandas.DataFrame
        One row per region, as returned by
        :func:`forensics_core.digits.integer_pct.integer_excess_by_group`, sorted by excess
        per unit descending. Carries ``election`` and ``quantity`` columns so that tables for
        the two elections can be concatenated without losing which is which, and a
        ``rank`` column so the ranking is explicit rather than positional. ``n_stations`` is
        the region's station count before any exclusion and ``n_excluded_high`` the number
        the over-99 rule removed; a region the rule empties is kept, with ``n = 0`` and null
        statistics, so it is visible rather than silently missing from the ranking.
    """
    _check_ceiling(exclude_over_pct)
    rows = _select(tidy, election)
    high = _over_ceiling(rows, exclude_over_pct)
    kept = rows.loc[~high]
    pct, denominators = _inputs(kept, quantity)
    table = integer_excess_by_group(
        pct.to_numpy(),
        denominators.to_numpy(),
        kept["region"].to_numpy(),
        tolerance=tolerance,
        min_denominator=min_denominator,
        n_mc=n_mc,
        seed=seed,
    )
    table = table.rename(columns={"group": "region"})
    counts = (
        pd.DataFrame({"region": rows["region"].to_numpy(), "high": high})
        .groupby("region", sort=False)
        .agg(n_stations=("high", "size"), n_excluded_high=("high", "sum"))
        .reset_index()
    )
    if table.empty:
        table = pd.DataFrame(columns=["region", "n", "n_excluded_small"])
    table = counts.merge(table, on="region", how="left")
    table["n"] = table["n"].fillna(0).astype(int)
    table["n_excluded_small"] = table["n_excluded_small"].fillna(0).astype(int)
    table["n_excluded_high"] = table["n_excluded_high"].astype(int)
    table = table.sort_values("excess_per_unit", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    table.insert(0, "election", election)
    table.insert(1, "quantity", quantity)
    table.insert(2, "rank", range(1, len(table) + 1))
    return table


def _band_record(
    band: pd.DataFrame,
    quantity: str,
    *,
    exclude_over_pct: float | None,
    tolerance: float,
    min_denominator: int,
    n_mc: int,
    seed: int | np.random.Generator | None,
) -> dict[str, object]:
    """The excess within one size band, with both exclusions counted.

    A band in which nothing survives the exclusions is reported with null statistics rather
    than raising, so one empty band does not sink the table.
    """
    high = _over_ceiling(band, exclude_over_pct)
    pct, denominators = _inputs(band.loc[~high], quantity)
    record: dict[str, object] = {
        "n_stations": len(band),
        "n_excluded_high": int(high.sum()),
    }
    values = pct.to_numpy(dtype=float)
    dens = denominators.to_numpy(dtype=float)
    try:
        result = integer_excess(
            values,
            dens,
            tolerance=tolerance,
            min_denominator=min_denominator,
            n_mc=n_mc,
            seed=seed,
        )
    except ValueError as exc:
        if "no unit survives exclusion" not in str(exc):
            raise
        nan = float("nan")
        record.update(
            n_tested=0,
            n_excluded_small=int(
                np.sum(np.isfinite(dens) & np.isfinite(values) & (dens < min_denominator))
            ),
            observed=nan,
            expected_mean=nan,
            expected_sd=nan,
            excess=nan,
            excess_per_unit=nan,
            z=nan,
            pvalue=nan,
        )
        return record
    n_tested = int(result.test.n)
    record.update(
        n_tested=n_tested,
        n_excluded_small=int(result.n_excluded_small),
        observed=float(result.observed),
        expected_mean=float(result.expected_mean),
        expected_sd=float(result.expected_sd),
        excess=float(result.excess),
        excess_per_unit=float(result.excess / n_tested) if n_tested else float("nan"),
        z=float(result.test.statistic),
        pvalue=float(result.test.pvalue),
    )
    return record


def integer_percentage_excess_by_size_band(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    exclude_over_pct: float | None = PAPER_EXCLUSION_PCT,
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
    tolerance, min_denominator, exclude_over_pct, n_mc, seed
        As in :func:`integer_percentage_excess`.

    Returns
    -------
    pandas.DataFrame
        One row per band of :data:`elections.features.size.SIZE_BAND_LABELS`, with the fields
        of :class:`forensics_core.digits.IntegerExcessResult` plus the band's station count,
        the number of stations the over-99 rule removed (``n_excluded_high``) and the number
        the denominator floor removed (``n_excluded_small``) within it. Bands are in
        ascending size order, not sorted by the statistic, because the question this table
        answers is whether the excess grows or dies with station size.
    """
    _check_ceiling(exclude_over_pct)
    rows = _select(tidy, election)
    records: list[dict[str, object]] = []
    for label, band in iter_size_strata(rows):
        record = _band_record(
            band,
            quantity,
            exclude_over_pct=exclude_over_pct,
            tolerance=tolerance,
            min_denominator=min_denominator,
            n_mc=n_mc,
            seed=seed,
        )
        records.append({"election": election, "quantity": quantity, "size_band": label, **record})
    table = pd.DataFrame.from_records(records)
    if not table.empty:
        order = {label: index for index, label in enumerate(SIZE_BAND_LABELS)}
        table = table.sort_values("size_band", key=lambda s: s.map(order)).reset_index(drop=True)
    return table


def _honest_frame(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One honest replicate of ``rows``: every numerator redrawn at its observed share.

    Ballots in boxes are drawn as ``Binomial(registered, observed turnout)`` and the winner's
    votes as ``Binomial(valid + invalid, observed winner's share)``, so both percentages are
    honest binomial noise around the shares the data report, on the real denominators. The
    other columns the two percentages read are kept as they are.
    """
    out = rows[["election", "region", "registered", "valid", "invalid"]].copy()
    for target, share, n in (
        (
            "in_stationary_boxes",
            turnout_ballots_in_boxes(rows).to_numpy(dtype=float),
            rows["registered"].to_numpy(dtype="float64", na_value=np.nan),
        ),
        (
            "winner_votes",
            winner_share(rows).to_numpy(dtype=float),
            ballots_counted(rows).to_numpy(dtype=float),
        ),
    ):
        drawn = np.full(len(rows), np.nan)
        ok = np.isfinite(share) & np.isfinite(n) & (n > 0)
        drawn[ok] = rng.binomial(np.round(n[ok]).astype(np.int64), np.clip(share[ok], 0.0, 1.0))
        out[target] = drawn
    out["in_mobile_boxes"] = 0.0
    return out


def integer_percentage_honest_bias(
    tidy: pd.DataFrame,
    *,
    election: str,
    quantity: Quantity = "turnout_boxes",
    tolerance: float = 0.05,
    min_denominator: int = 100,
    exclude_over_pct: float | None = PAPER_EXCLUSION_PCT,
    n_rep: int = 20,
    n_mc: int = 50,
    seed: int | None = None,
) -> pd.DataFrame:
    """The excess the procedure reports on honest data: its calibration error, measured.

    Each replicate redraws every station's numerators as binomial noise around the shares
    the data report, on the real denominators (so there is, by construction, no excess to
    find), applies the same exclusions and runs the same test nationally and within each
    size band. The mean excess over replicates is the bias of the procedure at these shares
    and sizes. It is printed beside each real excess so that a calibration error cannot pass
    for a finding, in either direction.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    quantity : {"turnout_boxes", "winner_share"}, default "turnout_boxes"
        As in :func:`integer_percentage_excess`.
    tolerance, min_denominator, exclude_over_pct
        As in :func:`integer_percentage_excess`. The over-99 rule is applied to the
        simulated percentages, as it would be to real ones.
    n_rep : int, default 20
        Honest replicates.
    n_mc : int, default 50
        Monte Carlo draws of the null within each replicate. Its noise averages out over the
        replicates, so it can be smaller than the ``n_mc`` of the real test.
    seed : int, optional
        Seeds the simulation and every test run inside it.

    Returns
    -------
    pandas.DataFrame
        One row per stratum: ``"all"`` (the national test) first, then each size band in
        ascending order. Columns ``election``, ``quantity``, ``stratum``, ``n_rep``,
        ``honest_bias`` (mean excess over replicates), ``honest_bias_se`` (its standard
        error), ``honest_bias_per_unit`` (mean of excess per tested station) and
        ``honest_z_mean`` (mean z, which should sit near zero for a calibrated test).
    """
    _check_ceiling(exclude_over_pct)
    if int(n_rep) < 2:
        raise ValueError(f"n_rep must be at least 2 for a standard error; got {n_rep}")
    rows = _select(tidy, election)
    if quantity not in PERCENTAGE_QUANTITIES:
        raise ValueError(
            f"unknown quantity {quantity!r}; expected one of {list(PERCENTAGE_QUANTITIES)}"
        )
    rng = np.random.default_rng(seed)
    settings = {
        "tolerance": tolerance,
        "min_denominator": min_denominator,
        "exclude_over_pct": exclude_over_pct,
        "n_mc": n_mc,
    }
    draws: dict[str, list[tuple[float, float, float]]] = {}
    for _ in range(int(n_rep)):
        honest = _honest_frame(rows, rng)
        strata = [("all", honest), *iter_size_strata(honest)]
        for label, stratum in strata:
            record = _band_record(stratum, quantity, seed=rng, **settings)
            n_tested = record["n_tested"]
            per_unit = record["excess"] / n_tested if n_tested else float("nan")
            draws.setdefault(label, []).append((record["excess"], per_unit, record["z"]))
    records = []
    for label in ("all", *SIZE_BAND_LABELS):
        if label not in draws:
            continue
        values = np.asarray(draws[label], dtype=float)
        excess = values[:, 0]
        finite = excess[np.isfinite(excess)]
        records.append(
            {
                "election": election,
                "quantity": quantity,
                "stratum": label,
                "n_rep": int(finite.size),
                "honest_bias": float(finite.mean()) if finite.size else float("nan"),
                "honest_bias_se": (
                    float(finite.std(ddof=1) / np.sqrt(finite.size))
                    if finite.size > 1
                    else float("nan")
                ),
                "honest_bias_per_unit": float(np.nanmean(values[:, 1]))
                if np.isfinite(values[:, 1]).any()
                else float("nan"),
                "honest_z_mean": float(np.nanmean(values[:, 2]))
                if np.isfinite(values[:, 2]).any()
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)
