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

Items 1 and 2 are computed here from the Russian tables. For item 2 "the regions the paper
names" means Table S4 of the paper's supplement, held in :data:`PAPER_TOP_REGIONS`, and the
comparison is :func:`paper_region_overlap`, fixed in advance in the anchors document. Item 3 needs the Poland and Spain
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
import math
from typing import Literal

import pandas as pd
from forensics_core.digits import IntegerExcessResult
from forensics_core.digits.integer_pct import integer_excess, integer_excess_by_group

from elections.clean.derive import ballots_counted, turnout_ballots_in_boxes, winner_share
from elections.clean.schema import ELECTIONS
from elections.features.size import SIZE_BAND_LABELS, iter_size_strata

__all__ = [
    "PAPER_OVERLAP_ALPHA",
    "PAPER_TOP_REGIONS",
    "PAPER_TOP_TEN",
    "PERCENTAGE_QUANTITIES",
    "integer_percentage_excess",
    "integer_percentage_excess_by_region",
    "integer_percentage_excess_by_size_band",
    "paper_region_overlap",
    "paper_region_ranks",
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


@dataclasses.dataclass(frozen=True)
class PaperRegion:
    """One row of the paper's Table S4, and the spellings it has in the precinct tables."""

    iso: str
    paper_name: str
    peak_thousands: int
    table_names: tuple[str, ...]


#: The item-2 benchmark: Table S4 of the supplement to Kobak, Shpilkin and Pshenichnikov,
#: arXiv:1410.6059v4, p. 22 ("Top 15 regions contributing to the integer anomaly"), in the
#: paper's order. ``peak_thousands`` is the height of the region's largest integer peak, in
#: thousands, relative to the Monte Carlo mean, maximised over all elections 2000-2012, over
#: turnout and leader's result, and over integers from 70 to 99 per cent. That is a different
#: statistic from the excess per station ranked here, and the paper's data end in 2012.
#: ``table_names`` are the region strings of the dkobak precinct tables; Moscow is spelled
#: with a capital in 2011 and a small letter in 2018. Recorded, with its source, in
#: ``docs/validation_anchors.md``; a test keeps the two in step.
PAPER_TOP_REGIONS: tuple[PaperRegion, ...] = (
    PaperRegion("DA", "Dagestan, Respublika", 87, ("Республика Дагестан",)),
    PaperRegion("BA", "Bashkortostan, Respublika", 64, ("Республика Башкортостан",)),
    PaperRegion("KEM", "Kemerovskaya Oblast", 52, ("Кемеровская область",)),
    PaperRegion("KDA", "Krasnodarskiy Krai", 51, ("Краснодарский край",)),
    PaperRegion("KO", "Komi, Respublika", 38, ("Республика Коми",)),
    PaperRegion("MOS", "Moskovskaya Oblast", 35, ("Московская область",)),
    PaperRegion("MOW", "Moscow", 34, ("Город Москва", "город Москва")),
    PaperRegion("KB", "Kabardino-Balkarskaya Respublika", 33, ("Кабардино-Балкарская Республика",)),
    PaperRegion("TA", "Tatarstan, Respublika", 33, ("Республика Татарстан (Татарстан)",)),
    PaperRegion("ROS", "Rostovskaya Oblast", 33, ("Ростовская область",)),
    PaperRegion("IN", "Ingushetiya, Respublika", 28, ("Республика Ингушетия",)),
    PaperRegion(
        "SE",
        "Severanaya Osetiya-Alaniya, Respublika",
        24,
        ("Республика Северная Осетия - Алания",),
    ),
    PaperRegion("MO", "Mordoviya, Respublika", 23, ("Республика Мордовия",)),
    PaperRegion(
        "KC",
        "Karachayevo-Cherkesskaya Respublika",
        21,
        ("Карачаево-Черкесская Республика",),
    ),
    PaperRegion("CE", "Chechenskaya Respublika", 18, ("Чеченская Республика",)),
)

#: The cut-off of the item-2 comparison, fixed in ``docs/validation_anchors.md``: how many of
#: the Table S4 regions fall in the top ten of the regional ranking.
PAPER_TOP_TEN = 10

#: The level at which the item-2 overlap counts as more than chance in an election, one-sided,
#: also fixed in ``docs/validation_anchors.md``.
PAPER_OVERLAP_ALPHA = 0.05


def _hypergeometric_sf(k: int, population: int, successes: int, draws: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(population, successes, draws), computed exactly."""
    total = math.comb(population, draws)
    tail = sum(
        math.comb(successes, i) * math.comb(population - successes, draws - i)
        for i in range(max(k, 0), min(successes, draws) + 1)
    )
    return tail / total


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


def paper_region_ranks(regions: pd.DataFrame) -> pd.DataFrame:
    """Where each region of the paper's Table S4 sits in the regional ranking.

    Anchor 1 item 2 asks for the excess to be concentrated in the regions the paper names.
    The paper's list is :data:`PAPER_TOP_REGIONS`; this looks every one of them up in the
    ranking, so the regions that rank low are reported next to those that rank high
    (``docs/known_traps.md``, trap 6: a ranking compared with the published one, not a count
    of regions that reject).

    Parameters
    ----------
    regions : pandas.DataFrame
        One or more elections' output of :func:`integer_percentage_excess_by_region`, with at
        least the columns ``election``, ``rank``, ``region``, ``n``, ``excess_per_unit``,
        ``z`` and ``pvalue``.

    Returns
    -------
    pandas.DataFrame
        One row per election and Table S4 region, in the paper's order within each election,
        with the region's ISO code, the paper's name for it, the region string it matched in
        the table, its ``rank`` out of ``n_regions``, its station count ``n``, its
        ``excess_per_unit``, ``z`` and ``pvalue``, and ``in_top`` (rank within
        :data:`PAPER_TOP_TEN`).

    Raises
    ------
    ValueError
        When a Table S4 region is missing from an election's table, or matches more than one
        row. A silent miss would shrink the list and change the comparison without saying so.
    """
    required = {"election", "rank", "region", "n", "excess_per_unit", "z", "pvalue"}
    missing_columns = required - set(regions.columns)
    if missing_columns:
        raise ValueError(f"regional table lacks columns {sorted(missing_columns)}")
    records: list[dict[str, object]] = []
    for election, table in regions.groupby("election", sort=False):
        n_regions = len(table)
        for paper in PAPER_TOP_REGIONS:
            match = table.loc[table["region"].isin(paper.table_names)]
            if len(match) != 1:
                raise ValueError(
                    f"{election}: Table S4 region {paper.iso} ({paper.paper_name}) matched "
                    f"{len(match)} rows of the regional table, expected exactly one; "
                    f"looked for {list(paper.table_names)}"
                )
            row = match.iloc[0]
            records.append(
                {
                    "election": election,
                    "iso": paper.iso,
                    "paper_name": paper.paper_name,
                    "region": row["region"],
                    "rank": int(row["rank"]),
                    "n_regions": n_regions,
                    "n": int(row["n"]),
                    "excess_per_unit": float(row["excess_per_unit"]),
                    "z": float(row["z"]),
                    "pvalue": float(row["pvalue"]),
                    "in_top": int(row["rank"]) <= PAPER_TOP_TEN,
                }
            )
    return pd.DataFrame.from_records(records)


def paper_region_overlap(regions: pd.DataFrame) -> pd.DataFrame:
    """The item-2 statistic: how many Table S4 regions are in the top ten, against chance.

    The comparison fixed in ``docs/validation_anchors.md``: count the regions of
    :data:`PAPER_TOP_REGIONS` ranked within the top :data:`PAPER_TOP_TEN` by excess per
    station, and compare that count with the hypergeometric distribution of the overlap
    between a random top ten and a fixed list of the same size. The median rank of the list
    is reported beside it. With roughly 85 regions several reject by chance, which is why the
    statistic is an overlap of rankings and not a count of rejections
    (``docs/known_traps.md``, trap 6).

    Parameters
    ----------
    regions : pandas.DataFrame
        As in :func:`paper_region_ranks`.

    Returns
    -------
    pandas.DataFrame
        One row per election, with ``n_regions`` ranked, ``n_paper`` (the Table S4 regions
        found, always 15), ``top`` (the cut-off), ``overlap`` (Table S4 regions within it),
        ``expected_overlap`` under chance, the exact one-sided hypergeometric ``pvalue``
        P(overlap >= observed), and the ``median_rank`` of the Table S4 regions.
    """
    ranks = paper_region_ranks(regions)
    records: list[dict[str, object]] = []
    for election, table in ranks.groupby("election", sort=False):
        n_regions = int(table["n_regions"].iloc[0])
        n_paper = len(table)
        overlap = int(table["in_top"].sum())
        records.append(
            {
                "election": election,
                "n_regions": n_regions,
                "n_paper": n_paper,
                "top": PAPER_TOP_TEN,
                "overlap": overlap,
                "expected_overlap": n_paper * PAPER_TOP_TEN / n_regions,
                "pvalue": _hypergeometric_sf(overlap, n_regions, n_paper, PAPER_TOP_TEN),
                "median_rank": float(table["rank"].median()),
            }
        )
    return pd.DataFrame.from_records(records)
