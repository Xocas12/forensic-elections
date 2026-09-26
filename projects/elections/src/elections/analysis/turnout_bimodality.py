"""Anchor 3: a second mode in the turnout distribution near complete turnout.

Method
------
Klimek, P., Y. Yegorov, R. Hanel and S. Thurner, 2012. "Statistical detection of systematic
election irregularities", *PNAS* 109(41), 16469-16473. The distribution of station turnout
shows a second mode close to 100 per cent that honest elections do not produce, and the
authors fit a parametric model of incremental and extreme ballot stuffing to the joint
turnout / vote-share distribution.

Two details from the paper are already confirmed and are reflected in the signatures below:
units with an electorate smaller than 100 are excluded, to keep very small communities from
producing extreme turnout and vote rates as artefacts; and the authors report that their
results do not depend much on the level of aggregation. Both were read from the PubMed Central
full text (registry entry ``pmc_klimek_2012``).

The unimodality statistic, and why this one
-------------------------------------------
Klimek et al. assert a second mode; they do not supply a hypothesis test for it, so one has to
be chosen and named rather than inherited. This module uses **Silverman's critical-bandwidth
test** (Silverman, B. W., 1981. "Using kernel density estimates to investigate multimodality",
*JRSS B* 43(1), 97-99):

* the statistic is the smallest Gaussian-kernel bandwidth ``h_crit`` at which the kernel
  density estimate of turnout still has exactly one local maximum. A distribution with a
  strong second mode needs a large bandwidth to smooth it away, so large ``h_crit`` is
  evidence against unimodality;
* the null distribution is obtained by the smoothed bootstrap the paper prescribes, drawing
  resamples from the density estimated at ``h_crit`` itself (the most nearly unimodal density
  consistent with the data) and recording how often a resample needs a bandwidth at least as
  large. That fraction is the p-value.

This is a test of the turnout margin only. It is reported together with
:func:`turnout_mode_locations`, never on its own, for the reason in trap 2 below.

What must be reproduced (``docs/validation_anchors.md``, Tier 1 item 3)
----------------------------------------------------------------------
A second mode near complete turnout in 2011 and 2018, located where the paper locates it.
Whether the paper's supplementary material holds its underlying data or only figures was
settled after that document was written: it holds neither data nor code, only an index page
and one PDF, so the replication runs on the precinct files acquired here and no Klimek data is
needed. The supplementary PDF itself is behind a bot block and remains a human task
(``data/ACCESS_NOTES.md``).

The caveat that belongs inside the result (``docs/known_traps.md``, trap 2)
--------------------------------------------------------------------------
Bimodality alone is not a signature. Genuine heterogeneity - urban against rural, republics
with different mobilisation patterns - produces a second mode in an honest election too. The
claim is about the joint distribution of turnout and vote share and about where the second
mode sits, which is why :func:`turnout_mode_locations` returns the location and not only a
test of unimodality, and why a bimodality result that is not accompanied by the joint
distribution is not evidence of anything. Every :class:`~forensics_core.TestResult` this
module returns carries that sentence in ``details["caveat"]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from forensics_core import TestResult

from elections.clean.derive import turnout_ballots_in_boxes, turnout_ballots_issued, winner_share
from elections.clean.schema import ELECTIONS

__all__ = [
    "KLIMEK_MIN_ELECTORATE",
    "TRAP_2_CAVEAT",
    "critical_bandwidth",
    "turnout_bimodality_test",
    "turnout_mode_locations",
]

#: The exclusion rule stated in the paper's Data and Methods section: units with an electorate
#: smaller than 100 are excluded.
KLIMEK_MIN_ELECTORATE = 100

#: Carried in the ``details`` of every result this module returns, so the number cannot travel
#: without it.
TRAP_2_CAVEAT = (
    "Bimodality alone is not evidence of falsification: urban/rural heterogeneity and "
    "differing mobilisation across republics produce a second mode in honest elections too. "
    "Read this only together with the joint turnout / vote-share mode locations."
)

TurnoutBasis = Literal["boxes", "issued"]

#: Grid the densities are evaluated on, in percentage points. Turnout is a percentage, so the
#: support is known and fixed rather than data-dependent.
_GRID = np.linspace(0.0, 100.0, 1001)

#: Bandwidth, in percentage points, used to *locate* modes (the test chooses its own).
#:
#: It must exceed one percentage point. Anchor 1 establishes that these same tables carry
#: excess mass on every whole number, so a narrower kernel resolves the integer sawtooth and
#: reports its teeth as a row of modes one point apart. That is a real feature of the data and
#: it is anchor 1's finding, not anchor 3's: at two points the sawtooth is smoothed away and
#: what is left are the broad modes Klimek et al. are describing.
MODE_BANDWIDTH_PP = 2.0


def _turnout(frame: pd.DataFrame, basis: str) -> pd.Series:
    if basis == "boxes":
        return turnout_ballots_in_boxes(frame, as_percent=True)
    if basis == "issued":
        return turnout_ballots_issued(frame, as_percent=True)
    raise ValueError(f"unknown turnout_basis {basis!r}; expected 'boxes' or 'issued'")


def _select(tidy: pd.DataFrame, election: str, min_electorate: int) -> tuple[pd.DataFrame, int]:
    """Election rows above the electorate floor, and the number the floor removed."""
    if election not in ELECTIONS:
        raise ValueError(f"unknown election {election!r}; expected one of {list(ELECTIONS)}")
    rows = tidy.loc[tidy["election"] == election]
    if rows.empty:
        raise ValueError(f"no rows for election {election!r} in the frame given")
    keep = rows["registered"].to_numpy(dtype=float) >= float(min_electorate)
    return rows.loc[keep], int((~keep).sum())


def _density(values: np.ndarray, bandwidth: float, weights: np.ndarray | None = None) -> np.ndarray:
    """Gaussian KDE on :data:`_GRID`, evaluated by binning first.

    Binning makes the bandwidth search affordable at 10^5 stations without changing the
    estimate materially: the bin width is two orders of magnitude below any bandwidth the
    search visits.
    """
    counts, edges = np.histogram(values, bins=_GRID.size, range=(0.0, 100.0), weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    step = centres[1] - centres[0]
    half = max(1, int(np.ceil(4.0 * bandwidth / step)))
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / bandwidth) ** 2)
    kernel /= kernel.sum()
    return np.convolve(counts.astype(float), kernel, mode="same")


def _n_modes(density: np.ndarray) -> int:
    """Local maxima of a density on a regular grid, counting plateaux once."""
    d = np.diff(density)
    sign = np.sign(d)
    nonzero = sign[sign != 0]
    if nonzero.size == 0:
        return 1
    return int(np.sum((nonzero[:-1] > 0) & (nonzero[1:] < 0)))


def critical_bandwidth(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    lo: float = 0.05,
    hi: float = 40.0,
    tolerance: float = 1e-3,
) -> float:
    """Smallest Gaussian bandwidth whose KDE of ``values`` has a single mode.

    Found by bisection, which is valid because the mode count of a Gaussian KDE is monotone
    non-increasing in the bandwidth (Silverman 1981).

    Parameters
    ----------
    values : numpy.ndarray
        Percentages in ``[0, 100]``.
    weights : numpy.ndarray, optional
        Per-observation weights.
    lo, hi : float
        Bracket for the search, in percentage points.
    tolerance : float
        Absolute width at which the bisection stops.

    Returns
    -------
    float
        The critical bandwidth.
    """
    if _n_modes(_density(values, lo, weights)) <= 1:
        return lo
    if _n_modes(_density(values, hi, weights)) > 1:
        return hi
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if _n_modes(_density(values, mid, weights)) > 1:
            lo = mid
        else:
            hi = mid
    return hi


def turnout_bimodality_test(
    tidy: pd.DataFrame,
    *,
    election: str,
    turnout_basis: TurnoutBasis = "boxes",
    min_electorate: int = KLIMEK_MIN_ELECTORATE,
    bin_width: float = 0.5,
    n_boot: int = 999,
    seed: int | None = None,
) -> TestResult:
    """Test the station turnout distribution for a second mode, for one election.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame from :func:`elections.clean.build_tidy`.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    turnout_basis : {"boxes", "issued"}, default "boxes"
        Which documented turnout definition to use.
    min_electorate : int, default 100
        Stations with fewer registered voters are excluded, following the paper.
    bin_width : float, default 0.5
        Width, in percentage points, of the turnout histogram bins. Recorded in ``details``;
        the density itself is evaluated on the fixed 0.1-point grid of this module.
    n_boot : int, default 999
        Smoothed-bootstrap resamples used for the null distribution of the statistic.
    seed : int, optional
        Seeds the generator.

    Returns
    -------
    forensics_core.TestResult
        ``statistic`` is the critical bandwidth; ``pvalue`` the smoothed-bootstrap p-value
        against unimodality. ``details`` records the excluded-station count, the bin width,
        the statistic's definition and citation, and the trap-2 caveat, so that the result
        cannot be quoted without the condition under which it means anything.
    """
    rows, n_excluded = _select(tidy, election, min_electorate)
    values = _turnout(rows, turnout_basis).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(f"{election}: fewer than two usable stations after the electorate floor")

    h_crit = critical_bandwidth(values)
    rng = np.random.default_rng(seed)
    # Silverman's smoothed bootstrap: resample the data, add N(0, h_crit) noise, and rescale
    # about the mean so the resample keeps the original variance.
    sd = values.std(ddof=1)
    scale = 1.0 / np.sqrt(1.0 + (h_crit**2) / (sd**2)) if sd > 0 else 1.0
    mean = values.mean()
    n_at_least = 0
    for _ in range(n_boot):
        draw = rng.choice(values, size=values.size, replace=True)
        draw = mean + scale * (draw - mean + h_crit * rng.standard_normal(values.size))
        if critical_bandwidth(np.clip(draw, 0.0, 100.0)) >= h_crit:
            n_at_least += 1
    pvalue = (n_at_least + 1) / (n_boot + 1)

    return TestResult(
        method="silverman_critical_bandwidth",
        statistic=float(h_crit),
        pvalue=float(pvalue),
        n=int(values.size),
        details={
            "statistic_name": "Silverman critical bandwidth (percentage points)",
            "definition": (
                "smallest Gaussian-kernel bandwidth at which the KDE of station turnout has "
                "exactly one local maximum; large values are evidence against unimodality"
            ),
            "citation": (
                "Silverman, B. W. (1981), Using kernel density estimates to investigate "
                "multimodality, JRSS B 43(1), 97-99; applied to the turnout distribution of "
                "Klimek et al. (2012), PNAS 109(41), 16469-16473"
            ),
            "null": "smoothed bootstrap from the density at the critical bandwidth",
            "alternative": "the turnout distribution has more than one mode",
            "election": election,
            "turnout_basis": turnout_basis,
            "bin_width": float(bin_width),
            "min_electorate": int(min_electorate),
            "n_stations": int(values.size),
            "n_excluded_small": n_excluded,
            "n_boot": int(n_boot),
            "caveat": TRAP_2_CAVEAT,
        },
    )


def turnout_mode_locations(
    tidy: pd.DataFrame,
    *,
    election: str,
    turnout_basis: TurnoutBasis = "boxes",
    min_electorate: int = KLIMEK_MIN_ELECTORATE,
    bin_width: float = 0.5,
    weight_by: Literal["stations", "registered"] = "registered",
    seed: int | None = None,
) -> pd.DataFrame:
    """Where the modes are, in turnout and in the joint turnout / vote-share plane.

    The anchor is about the *location* of the second mode, so the location is the output,
    not a by-product of a test. Reported both unweighted, one row per station, and weighted by
    registered voters, because a mode carried by a few large stations and a mode carried by
    many small ones are different claims.

    Parameters
    ----------
    tidy : pandas.DataFrame
        The tidy frame.
    election : str
        One of :data:`elections.clean.schema.ELECTIONS`.
    turnout_basis : {"boxes", "issued"}, default "boxes"
        Which documented turnout definition to use.
    min_electorate : int, default 100
        As in :func:`turnout_bimodality_test`.
    bin_width : float, default 0.5
        Histogram bin width in percentage points.
    weight_by : {"stations", "registered"}, default "registered"
        Weight each station equally, or by its electorate.
    seed : int, optional
        Unused; kept because the bandwidth here is fixed by rule rather than resampled, and
        the signature is the project's contract.

    Returns
    -------
    pandas.DataFrame
        One row per detected mode, with its turnout, the winner's vote share at it, the mass
        it carries and the weighting used. Modes are ordered by turnout ascending, and the
        frame carries both weightings when ``weight_by`` is the default, so that a mode
        present under one and absent under the other is visible rather than hidden.
    """
    if weight_by not in ("stations", "registered"):
        raise ValueError(f"unknown weight_by {weight_by!r}; expected 'stations' or 'registered'")
    rows, n_excluded = _select(tidy, election, min_electorate)
    turnout = _turnout(rows, turnout_basis).to_numpy(dtype=float)
    share = winner_share(rows, as_percent=True).to_numpy(dtype=float)
    electorate = rows["registered"].to_numpy(dtype=float)
    usable = np.isfinite(turnout) & np.isfinite(share) & np.isfinite(electorate)
    turnout, share, electorate = turnout[usable], share[usable], electorate[usable]

    weightings = ("stations", "registered") if weight_by == "registered" else ("stations",)
    records: list[dict[str, object]] = []
    for weighting in weightings:
        w = None if weighting == "stations" else electorate
        density = _density(turnout, MODE_BANDWIDTH_PP, w)
        total = density.sum()
        d = np.diff(density)
        sign = np.sign(d)
        peaks = [
            index + 1 for index in range(len(sign) - 1) if sign[index] > 0 and sign[index + 1] < 0
        ]
        for index in peaks:
            centre = _GRID[index]
            near = np.abs(turnout - centre) <= 2.5
            if not near.any():
                continue
            mass_w = w[near].sum() if w is not None else float(near.sum())
            mass_total = w.sum() if w is not None else float(turnout.size)
            records.append(
                {
                    "election": election,
                    "turnout_basis": turnout_basis,
                    "weight_by": weighting,
                    "mode_turnout_pct": float(centre),
                    "winner_share_at_mode_pct": float(
                        np.average(share[near], weights=w[near] if w is not None else None)
                    ),
                    "density_height": float(density[index] / total),
                    "mass_within_2_5pp": float(mass_w / mass_total),
                    "n_stations_within_2_5pp": int(near.sum()),
                    "n_excluded_small": n_excluded,
                    "caveat": TRAP_2_CAVEAT,
                }
            )
    table = pd.DataFrame.from_records(records)
    if not table.empty:
        table = table.sort_values(["weight_by", "mode_turnout_pct"]).reset_index(drop=True)
    return table
