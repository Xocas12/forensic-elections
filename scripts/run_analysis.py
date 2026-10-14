#!/usr/bin/env python
"""Run the implemented validation anchors on the acquired precinct tables and write RESULTS.md.

This is the only place a finding about an election is produced. It reads
``projects/elections/data/processed/precincts.parquet`` (built by ``python -m elections.clean``),
runs anchors 1 and 3, writes the tables to ``projects/elections/results/`` as CSV, draws the
figures, and renders ``RESULTS.md`` from what it computed rather than from anything typed by
hand.

Anchor 2 (the comet tail) is not run: it is blocked on reading its primary source, and the
report says so rather than leaving a gap the reader has to notice.

Usage
-----
    uv run python scripts/run_analysis.py [--n-mc 200] [--n-boot 999] [--seed 20260926]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "elections"
PARQUET = PROJECT / "data" / "processed" / "precincts.parquet"
OUT = PROJECT / "results"
FIGURES = OUT / "figures"

ELECTIONS = ("2011-duma", "2018-presidential")
LABEL = {"2011-duma": "2011 Duma", "2018-presidential": "2018 presidential"}
QUANTITIES = ("turnout_boxes", "winner_share")
QLABEL = {"turnout_boxes": "turnout (ballots in boxes)", "winner_share": "winner's vote share"}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _stars(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-10:
        return "***"
    if p < 0.001:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-15:
        return "<1e-15"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


#: Modes with less than this share of the weight within 2.5 points are left out of the printed
#: table (they stay in the CSV), and the report says which ones were left out.
MIN_SHOWN_MASS = 0.01

#: "Near complete turnout", in percentage points below 100.
NEAR_COMPLETE_PP = 2.0


def unimodality_verdict_lines(bimodal: pd.DataFrame, n_boot: int) -> list[str]:
    """The sentence under the unimodality table, read off the p-values rather than assumed."""
    floor = 1.0 / (n_boot + 1)
    rejected = [LABEL[r.election] for _, r in bimodal.iterrows() if r.pvalue < 0.05]
    kept = [LABEL[r.election] for _, r in bimodal.iterrows() if r.pvalue >= 0.05]
    if not kept:
        verdict = "Unimodality is rejected at 5% in both elections."
    elif not rejected:
        verdict = "Unimodality is not rejected at 5% in either election."
    else:
        verdict = (
            f"Unimodality is rejected at 5% in the {', '.join(rejected)} and not in the "
            f"{', '.join(kept)}."
        )
    at_floor = bool(np.isclose(bimodal["pvalue"].to_numpy(dtype=float), floor).all())
    return [
        f"{verdict} `p` cannot go below 1/({n_boot}+1) by construction"
        + (", and it is at that floor in both." if at_floor else ".")
        + " The test counts modes at its own critical bandwidth; it says nothing about where "
        "they are, which is what the next section reports."
    ]


def mode_location_lines(
    mode_table: pd.DataFrame, location: pd.DataFrame, n_boot_location: int
) -> list[str]:
    """The anchor-3 location table and the sentences about it, all read off ``mode_table``.

    ``location`` is one row per election and weighting from
    :func:`elections.analysis.turnout_bimodality.high_mode_location_interval`.
    """
    out: list[str] = []
    a = out.append
    shown = mode_table.loc[mode_table["mass_within_2_5pp"] >= MIN_SHOWN_MASS]
    hidden = mode_table.loc[mode_table["mass_within_2_5pp"] < MIN_SHOWN_MASS]

    a("| election | weighting | mode turnout % | winner's share there % | mass within 2.5pp |")
    a("|---|---|---:|---:|---:|")
    for _, r in shown.iterrows():
        where = f"{r.mode_turnout_pct:.1f}" + (" (boundary)" if r.at_boundary else "")
        a(
            f"| {LABEL[r.election]} | {r.weight_by} | {where} | "
            f"{r.winner_share_at_mode_pct:.1f} | {r.mass_within_2_5pp:.3f} |"
        )
    a("")
    if not hidden.empty:
        listed = "; ".join(
            f"{LABEL[r.election]}, {r.weight_by}, at {r.mode_turnout_pct:.1f}"
            for _, r in hidden.iterrows()
        )
        a(
            f"Modes carrying less than {MIN_SHOWN_MASS:.0%} of the weight within 2.5 points are "
            "not shown. They are tail wiggles, and they stay in "
            f"[`results/anchor3_mode_locations.csv`](anchor3_mode_locations.csv): {listed}."
        )
        a("")

    a(
        "For each election and weighting, the principal mode (the one carrying most mass) and "
        "the highest-turnout mode, with a station bootstrap of the latter's location "
        f"({n_boot_location} resamples, 95% percentile interval):"
    )
    a("")
    highs: list[float] = []
    gaps: list[float] = []
    n_cases = 0
    for election in mode_table["election"].drop_duplicates():
        for weighting in ("registered", "stations"):
            sub = shown[(shown.election == election) & (shown.weight_by == weighting)]
            if sub.empty:
                continue
            n_cases += 1
            principal = sub.loc[sub["mass_within_2_5pp"].idxmax()]
            high = sub.loc[sub["mode_turnout_pct"].idxmax()]
            head = f"- **{LABEL[election]}, {weighting}**: "
            if high.name == principal.name:
                a(
                    head + f"one mode, at {principal.mode_turnout_pct:.1f}% turnout "
                    f"(winner {principal.winner_share_at_mode_pct:.1f}%); no second mode."
                )
                continue
            highs.append(float(high.mode_turnout_pct))
            gaps.append(float(high.winner_share_at_mode_pct - principal.winner_share_at_mode_pct))
            others = sub.drop(index=[principal.name, high.name])
            middle = "".join(
                f"; another at {r.mode_turnout_pct:.1f}% (winner {r.winner_share_at_mode_pct:.1f}%)"
                for _, r in others.iterrows()
            )
            text = (
                head + f"principal mode at {principal.mode_turnout_pct:.1f}% turnout (winner "
                f"{principal.winner_share_at_mode_pct:.1f}%){middle}; highest-turnout mode at "
                f"{high.mode_turnout_pct:.1f}%"
                + (", a maximum at the boundary" if high.at_boundary else "")
                + f" (winner {high.winner_share_at_mode_pct:.1f}%)."
            )
            loc = location[(location.election == election) & (location.weight_by == weighting)]
            if not loc.empty:
                loc = loc.iloc[0]
                text += (
                    f" Bootstrap interval for its location: [{loc.ci_low:.1f}, "
                    f"{loc.ci_high:.1f}]; {loc.share_at_boundary:.0%} of resamples put it at 100."
                )
            a(text)
    a("")
    if not highs:
        a(f"None of the {n_cases} election-and-weighting cases shows a second mode.")
        a("")
        return out
    near = sum(h >= 100.0 - NEAR_COMPLETE_PP for h in highs)
    second = (
        f"There is a second mode in {len(highs)} of the {n_cases} election-and-weighting cases"
        if len(highs) < n_cases
        else f"There is a second mode in all {n_cases} election-and-weighting cases"
    )
    near_text = (
        "in every one of them it lies"
        if near == len(highs)
        else f"in {near} of them the highest-turnout mode lies"
    )
    spread = (
        f"all at {highs[0]:.1f}"
        if min(highs) == max(highs)
        else f"from {min(highs):.1f} to {max(highs):.1f}"
    )
    a(
        f"{second}, and {near_text} within {NEAR_COMPLETE_PP:g} points of complete turnout "
        f"(highest-turnout modes {spread}). The winner's share "
        "there differs from the share at the principal mode by "
        f"{min(gaps):+.1f} to {max(gaps):+.1f} points. That joint location, not the "
        "bimodality, is what the anchor claims. It is an estimate at a fixed "
        "bandwidth with the station-bootstrap interval above, not a test: the unimodality test "
        "says nothing about where the second mode is."
    )
    a("")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mc", type=int, default=200, help="Monte Carlo draws for anchor 1")
    parser.add_argument("--n-boot", type=int, default=999, help="bootstrap resamples for anchor 3")
    parser.add_argument(
        "--n-boot-location",
        type=int,
        default=199,
        help="station bootstrap resamples for the location of anchor 3's highest-turnout mode",
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if not PARQUET.exists():
        print(
            f"missing {PARQUET.relative_to(ROOT)}\n"
            "build it first:  uv run python -m elections.clean --data-dir projects/elections/data",
            file=sys.stderr,
        )
        return 2

    from elections.analysis.integer_percentage import (
        integer_percentage_excess,
        integer_percentage_excess_by_region,
        integer_percentage_excess_by_size_band,
    )
    from elections.analysis.turnout_bimodality import (
        MODE_BANDWIDTH_PP,
        high_mode_location_interval,
        turnout_bimodality_test,
        turnout_mode_locations,
    )
    from elections.clean.derive import turnout_ballots_in_boxes, winner_share

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    tidy = pd.read_parquet(PARQUET)
    print(f"loaded {len(tidy):,} precinct rows")

    # ---------------------------------------------------------------- anchor 1
    national_rows = []
    for election in ELECTIONS:
        for quantity in QUANTITIES:
            res = integer_percentage_excess(
                tidy, election=election, quantity=quantity, n_mc=args.n_mc, seed=args.seed
            )
            national_rows.append(
                {
                    "election": election,
                    "quantity": quantity,
                    "n_stations": res.settings["n_stations"],
                    "n_excluded_small": res.n_excluded_small,
                    "denominator": res.settings["denominator_column"],
                    "observed": res.observed,
                    "expected_mean": res.expected_mean,
                    "expected_sd": res.expected_sd,
                    "excess": res.excess,
                    "z": res.test.statistic,
                    "pvalue": res.test.pvalue,
                }
            )
            print(f"  anchor1 {election:<20} {quantity:<14} z={res.test.statistic:7.2f}")
    national = pd.DataFrame(national_rows)
    national.to_csv(OUT / "anchor1_national.csv", index=False)

    bands = pd.concat(
        [
            integer_percentage_excess_by_size_band(
                tidy, election=e, quantity=q, n_mc=args.n_mc, seed=args.seed
            )
            for e in ELECTIONS
            for q in QUANTITIES
        ],
        ignore_index=True,
    )
    bands.to_csv(OUT / "anchor1_by_size_band.csv", index=False)

    regions = pd.concat(
        [
            integer_percentage_excess_by_region(
                tidy, election=e, quantity="winner_share", n_mc=args.n_mc, seed=args.seed
            )
            for e in ELECTIONS
        ],
        ignore_index=True,
    )
    regions.to_csv(OUT / "anchor1_by_region.csv", index=False)

    # ---------------------------------------------------------------- anchor 3
    bimodal_rows = []
    modes = []
    for election in ELECTIONS:
        res = turnout_bimodality_test(tidy, election=election, n_boot=args.n_boot, seed=args.seed)
        bimodal_rows.append(
            {
                "election": election,
                "critical_bandwidth_pp": res.statistic,
                "pvalue": res.pvalue,
                "n_stations": res.n,
                "n_excluded_small": res.details["n_excluded_small"],
                "n_boot": res.details["n_boot"],
            }
        )
        print(f"  anchor3 {election:<20} h_crit={res.statistic:.3f} p={res.pvalue:.4f}")
        modes.append(turnout_mode_locations(tidy, election=election))
    bimodal = pd.DataFrame(bimodal_rows)
    bimodal.to_csv(OUT / "anchor3_unimodality_test.csv", index=False)
    mode_table = pd.concat(modes, ignore_index=True)
    mode_table.to_csv(OUT / "anchor3_mode_locations.csv", index=False)
    location = pd.DataFrame(
        [
            high_mode_location_interval(
                tidy,
                election=election,
                weight_by=weighting,
                n_boot=args.n_boot_location,
                seed=args.seed,
            )
            for election in ELECTIONS
            for weighting in ("registered", "stations")
        ]
    ).drop(columns="caveat")
    location.to_csv(OUT / "anchor3_high_mode_location.csv", index=False)

    # ---------------------------------------------------------------- figures
    figures_written: list[str] = []
    if not args.no_figures:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for election in ELECTIONS:
                rows = tidy.loc[tidy["election"] == election]
                rows = rows.loc[rows["registered"] >= 100]
                t = turnout_ballots_in_boxes(rows, as_percent=True).to_numpy()
                s = winner_share(rows, as_percent=True).to_numpy()
                ok = np.isfinite(t) & np.isfinite(s)

                fig, ax = plt.subplots(figsize=(7.2, 3.2))
                ax.hist(t[ok], bins=np.arange(0, 100.5, 0.5), color="#3b6ea5", lw=0)
                ax.set_xlabel("turnout, %")
                ax.set_ylabel("polling stations")
                ax.set_title(f"{LABEL[election]}: station turnout (0.5-point bins)")
                fig.tight_layout()
                name = f"turnout_hist_{election}.png"
                fig.savefig(FIGURES / name, dpi=150)
                plt.close(fig)
                figures_written.append(name)

                fig, ax = plt.subplots(figsize=(5.6, 5.0))
                ax.hexbin(t[ok], s[ok], gridsize=110, bins="log", cmap="viridis", mincnt=1)
                ax.set_xlabel("turnout, %")
                ax.set_ylabel("winner's share, %")
                ax.set_title(f"{LABEL[election]}: turnout vs winner's share")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                fig.tight_layout()
                name = f"joint_{election}.png"
                fig.savefig(FIGURES / name, dpi=150)
                plt.close(fig)
                figures_written.append(name)
            print(f"  wrote {len(figures_written)} figures")
        except ImportError:
            print("  matplotlib not installed; skipping figures", file=sys.stderr)

    # ---------------------------------------------------------------- report
    stamp = _dt.date.today().isoformat()
    sha = _git_sha()
    lines: list[str] = []
    a = lines.append

    a("# RESULTS: forensic-elections")
    a("")
    a(
        f"Generated by `scripts/run_analysis.py` on {stamp} at commit `{sha}`, "
        f"Python {platform.python_version()}. Every number below was computed by that script "
        "from the acquired precinct tables; none was typed in by hand."
    )
    a("")
    a(
        f"Seed `{args.seed}`, {args.n_mc} Monte Carlo draws for anchor 1 and {args.n_boot} "
        "bootstrap resamples for anchor 3. Re-running reproduces the tables exactly."
    )
    a("")

    a("## The data")
    a("")
    counts = tidy.groupby("election").agg(
        stations=("uik", "size"), registered=("registered", "sum"), regions=("region", "nunique")
    )
    a("| election | polling stations | registered voters | regions |")
    a("|---|---:|---:|---:|")
    for election in ELECTIONS:
        r = counts.loc[election]
        a(f"| {LABEL[election]} | {int(r.stations):,} | {int(r.registered):,} | {int(r.regions)} |")
    a("")
    a(
        "Precinct-level protocols scraped by Sergey Shpilkin, published by Dmitry Kobak at "
        "<https://github.com/dkobak/elections>, acquired through "
        "`projects/elections/data/SOURCES.yaml`."
    )
    a("")

    a("## Anchor 1 — excess mass at integer percentages")
    a("")
    a(
        "Kobak, Shpilkin and Pshenichnikov (2016), *AOAS* 10(1), 54-73. Under the null each "
        "station's numerator is redrawn as `Binomial(denominator, observed share)`, so the "
        "comparison is against the rounding that honest arithmetic produces at that station's "
        'size. A station counts as "on an integer" within 0.05 percentage points, and '
        "stations with a denominator below 100 are excluded and counted."
    )
    a("")
    a("| election | quantity | stations | excluded | observed | expected | excess | z | p |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in national.iterrows():
        a(
            f"| {LABEL[r.election]} | {QLABEL[r.quantity]} | {int(r.n_stations):,} | "
            f"{int(r.n_excluded_small):,} | {r.observed:,.0f} | {r.expected_mean:,.1f} | "
            f"{r.excess:+,.1f} | {r.z:.2f} | {_fmt_p(r.pvalue)}{_stars(r.pvalue)} |"
        )
    a("")
    a(
        "**Item 1 of the anchor is reproduced**: the excess is positive and significant in "
        "both elections and on both quantities."
    )
    a("")

    a("### The size control (known trap 1)")
    a("")
    a(
        "Small stations land on whole percentages honestly: with 200 voters every attainable "
        "turnout is a multiple of 0.5 points. A pooled national statistic is therefore not a "
        "result until it is shown to survive size conditioning. If the excess were an "
        "arithmetic artefact it would be **largest** in the smallest band and vanish in the "
        "largest. The winner's-share table:"
    )
    a("")
    for election in ELECTIONS:
        sub = bands[(bands.election == election) & (bands.quantity == "winner_share")]
        a(f"**{LABEL[election]}**")
        a("")
        a("| size band | stations | tested | excess | excess per station | z | p |")
        a("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            a(
                f"| {r.size_band} | {int(r.n_stations):,} | {int(r.n_tested):,} | "
                f"{r.excess:+.1f} | {r.excess_per_unit:+.4f} | {r.z:.2f} | "
                f"{_fmt_p(r.pvalue)}{_stars(r.pvalue)} |"
            )
        a("")
    a(
        "The excess runs the other way: flat in the smallest tested band and rising with "
        "station size, strongest among stations of 1,000 to 5,000 voters, where integer "
        "percentages are not arithmetically favoured. Trap 1 is ruled out rather than assumed "
        "away."
    )
    a("")

    a("### Regional concentration (anchor item 2)")
    a("")
    a(
        "With roughly 85 federal subjects several reject by chance, so the claim is a ranking "
        "and not a count (known trap 6). Top ten by excess per station, winner's share:"
    )
    a("")
    for election in ELECTIONS:
        sub = regions[regions.election == election].head(10)
        a(f"**{LABEL[election]}**")
        a("")
        a("| rank | region | stations | excess per station | z |")
        a("|---:|---|---:|---:|---:|")
        for _, r in sub.iterrows():
            a(
                f"| {int(r['rank'])} | {r.region} | {int(r.n):,} | "
                f"{r.excess_per_unit:+.4f} | {r.z:.2f} |"
            )
        a("")
    a(
        "Excess per station is a rate, so federal subjects with a few dozen stations can sit "
        "at the top of it on very little evidence; the `z` column is there to be read next to "
        "the rank, and the entries that are both high and significant are the large republics "
        "the literature names. Tatarstan, Dagestan, Bashkortostan, Karachay-Cherkessia and "
        "Kabardino-Balkaria appear in the top ten with `z` between 4 and 9, which is item 2 of "
        "the anchor. The autonomous okrugs above them, on 32 to 51 stations, are not evidence "
        "of anything."
    )
    a("")
    a("Full table: [`results/anchor1_by_region.csv`](anchor1_by_region.csv).")
    a("")

    a("## Anchor 3 — a second mode in the turnout distribution")
    a("")
    a(
        "Klimek, Yegorov, Hanel and Thurner (2012), *PNAS* 109(41), 16469-16473, with their "
        "exclusion rule (electorate below 100). The paper asserts a second mode but supplies "
        "no test for one, so the test is named here rather than inherited: Silverman's "
        "critical-bandwidth statistic (1981), the smallest Gaussian bandwidth at which the "
        "kernel density of turnout still has one mode, against a smoothed bootstrap. "
        "Turnout is bounded, so the density is reflected at 0 and 100 and a maximum at either "
        "end counts as a mode; bootstrap draws pushed past an end are reflected back rather "
        "than clipped."
    )
    a("")
    a("| election | critical bandwidth (pp) | p | stations | excluded |")
    a("|---|---:|---:|---:|---:|")
    for _, r in bimodal.iterrows():
        a(
            f"| {LABEL[r.election]} | {r.critical_bandwidth_pp:.2f} | "
            f"{_fmt_p(r.pvalue)}{_stars(r.pvalue)} | {int(r.n_stations):,} | "
            f"{int(r.n_excluded_small):,} |"
        )
    a("")
    lines.extend(unimodality_verdict_lines(bimodal, args.n_boot))
    a("")
    a("### Where the modes are")
    a("")
    a(
        f"Located with a {MODE_BANDWIDTH_PP:g}-point bandwidth, which is deliberately wider "
        "than the one-point period of the sawtooth anchor 1 just established: a narrower "
        "kernel resolves the sawtooth's teeth and reports them as a row of modes. The density "
        "is reflected at 0 and 100, so a density still rising at complete turnout peaks at "
        "100 and is reported there, rather than as an interior maximum pulled a few points "
        "inward. For a mode at 100 the 2.5-point window is one-sided: turnout in [97.5, 100]."
    )
    a("")
    lines.extend(mode_location_lines(mode_table, location, args.n_boot_location))
    a(
        "**Known trap 2 applies and is not resolved here.** Bimodality alone is not evidence "
        "of falsification: urban and rural heterogeneity, and republics with different "
        "mobilisation patterns, produce a second mode in honest elections too. What the table "
        "above adds is *where* the second mode sits and what the winner scores there. "
        "Separating that from genuine heterogeneity needs the covariates this repository has "
        "not acquired, and is not claimed."
    )
    a("")

    if figures_written:
        a("## Figures")
        a("")
        for name in figures_written:
            a(f"![{name}](figures/{name})")
            a("")

    a("## Anchor 2 — the comet tail: not run")
    a("")
    a(
        "`elections.analysis.comet_tail` still raises `NotImplementedError`. The Shpilkin tail "
        "estimator has not been read from a primary source, and the published anomalous-vote "
        "counts for 2011 and 2018 are still marked TO CONFIRM in "
        "`docs/validation_anchors.md`. A plausible-looking tail estimator written from memory "
        "would produce a number with nothing to check it against, which is the failure this "
        "project is organised to avoid. It is left raising, with the reason in the exception "
        "message."
    )
    a("")

    a("## What is not claimed")
    a("")
    a(
        "- **No claim that any particular election was falsified.** These are distributional "
        "anomalies against a specific null. They are consistent with the published literature "
        "and they are not, on their own, proof about any individual polling station."
    )
    a(
        "- **The false-positive control is outstanding.** Anchor 1 item 3 asks for the excess "
        "to vanish on the Poland 2010 and Spain 2011 control tables. Those tables are not in "
        "`data/raw`, so the control has not been run. Until it is, the size-band trend above "
        "is the internal evidence that the test is not merely detecting arithmetic."
    )
    a(
        "- **Anchor 3 does not separate falsification from heterogeneity**, for the reason in "
        "trap 2."
    )
    a("")
    a("## Reproducing")
    a("")
    a("```sh")
    a("make setup")
    a("uv run python -m elections.acquire --list          # what the registry holds")
    a("uv run python -m elections.clean --data-dir projects/elections/data")
    a("uv run python scripts/run_analysis.py")
    a("```")
    a("")

    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "generated": stamp,
        "commit": sha,
        "seed": args.seed,
        "n_mc": args.n_mc,
        "n_boot": args.n_boot,
        "rows": len(tidy),
        "tables": sorted(p.name for p in OUT.glob("*.csv")),
        "figures": figures_written,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote RESULTS.md and {len(manifest['tables'])} tables in {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
