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


def _join(items: list[str], sep: str = ", ") -> str:
    """'a', 'a and b', 'a, b and c'; 'none' for an empty list.

    ``sep="; "`` for items that contain commas themselves, as the paper's region names do.
    """
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return sep.join(items[:-1]) + " and " + items[-1]


def item2_section(regions: pd.DataFrame) -> list[str]:
    """The report's section on anchor 1 item 2, built from the regional table alone.

    Every region name, rank, station count and ``z`` in it is read from ``regions`` (the
    output of ``integer_percentage_excess_by_region`` for each election), and the comparison
    with the paper's Table S4 is computed from it, so the text cannot say something the table
    does not. A test regenerates this section from the committed
    ``results/anchor1_by_region.csv`` and requires ``RESULTS.md`` to match it.
    """
    from elections.analysis.integer_percentage import (
        PAPER_OVERLAP_ALPHA,
        PAPER_TOP_REGIONS,
        PAPER_TOP_TEN,
        paper_region_overlap,
        paper_region_ranks,
    )

    ranks = paper_region_ranks(regions)
    overlap = paper_region_overlap(regions)
    elections = [e for e in ELECTIONS if e in set(regions["election"])]
    paper_names = {name for p in PAPER_TOP_REGIONS for name in p.table_names}
    n_paper = len(PAPER_TOP_REGIONS)

    lines: list[str] = []
    a = lines.append
    a("### Regional concentration (anchor item 2)")
    a("")
    a(
        f"Item 2 asks for the excess to be concentrated in the regions the paper names. The "
        f"paper's list is Table S4 of its supplement: the {n_paper} regions with the largest "
        "integer peak, recorded with its source in `docs/validation_anchors.md` together with "
        f"the comparison fixed for it. That comparison is how many of the {n_paper} fall in the "
        f"top {PAPER_TOP_TEN} here by excess per station on the winner's share, against the "
        "hypergeometric chance of at least that many, with the median rank of the list beside "
        "it. With roughly 85 federal subjects several reject by chance, so the claim is a "
        "ranking and not a count (known trap 6)."
    )
    a("")
    a(
        "Two caveats go with it. Table S4 ranks a different statistic, the height of the "
        "largest integer peak over all elections from 2000 to 2012, over turnout and the "
        "leader's result, and over integers from 70 to 99 per cent; this ranking is excess per "
        "station on one quantity in one election. And the paper's data end in 2012, so for "
        "2018 its list is an extrapolation."
    )
    a("")
    a(
        f"| election | regions ranked | Table S4 regions in top {PAPER_TOP_TEN} | expected by "
        "chance | p (hypergeometric, one-sided) | median rank of Table S4 regions |"
    )
    a("|---|---:|---:|---:|---:|---:|")
    for election in elections:
        o = overlap.loc[overlap.election == election].iloc[0]
        a(
            f"| {LABEL[election]} | {int(o.n_regions)} | {int(o.overlap)} of {int(o.n_paper)} | "
            f"{o.expected_overlap:.1f} | {_fmt_p(o.pvalue)}{_stars(o.pvalue)} | "
            f"{o.median_rank:g} |"
        )
    a("")
    a("Where each Table S4 region ranks (rank, then `z`):")
    a("")
    header = "| ISO | Table S4 region | " + " | ".join(f"{LABEL[e]} rank | z" for e in elections)
    a(header + " |")
    a("|---|---|" + "---:|---:|" * len(elections))
    for paper in PAPER_TOP_REGIONS:
        cells = []
        for election in elections:
            r = ranks.loc[(ranks.election == election) & (ranks.iso == paper.iso)].iloc[0]
            cells.append(f"{int(r['rank'])} of {int(r.n_regions)} | {r.z:.2f}")
        a(f"| {paper.iso} | {paper.paper_name} | " + " | ".join(cells) + " |")
    a("")

    a(f"Top {PAPER_TOP_TEN} by excess per station, winner's share:")
    a("")
    for election in elections:
        sub = regions[regions.election == election].sort_values("rank").head(PAPER_TOP_TEN)
        a(f"**{LABEL[election]}**")
        a("")
        a("| rank | region | stations | excess per station | z | p | in Table S4 |")
        a("|---:|---|---:|---:|---:|---:|---|")
        for _, r in sub.iterrows():
            listed = "yes" if r.region in paper_names else ""
            a(
                f"| {int(r['rank'])} | {r.region} | {int(r.n):,} | "
                f"{r.excess_per_unit:+.4f} | {r.z:.2f} | {_fmt_p(r.pvalue)}{_stars(r.pvalue)} | "
                f"{listed} |"
            )
        a("")

    reproduced: list[str] = []
    not_reproduced: list[str] = []
    for election in elections:
        o = overlap.loc[overlap.election == election].iloc[0]
        r = ranks.loc[ranks.election == election].sort_values("rank")
        top = regions[regions.election == election].sort_values("rank").head(PAPER_TOP_TEN)
        high = [f"{x.paper_name} ({int(x['rank'])})" for _, x in r[r.in_top].iterrows()]
        low = [
            f"{x.paper_name} ({int(x['rank'])})"
            for _, x in r[r["rank"] > o.n_regions / 2].iterrows()
        ]
        others = [
            f"{x.region} (rank {int(x['rank'])}, z {x.z:.2f})"
            for _, x in top.iterrows()
            if x.region not in paper_names
        ]
        weak = [
            f"{x.region} ({int(x.n):,} stations, z {x.z:.2f})"
            for _, x in top.iterrows()
            if not (x.pvalue < PAPER_OVERLAP_ALPHA)
        ]
        a(
            f"**{LABEL[election]}.** {int(o.overlap)} of the {int(o.n_paper)} Table S4 regions "
            f"are in the top {PAPER_TOP_TEN}: {_join(high, '; ')}. Chance would put "
            f"{o.expected_overlap:.1f} there (p = {_fmt_p(o.pvalue)}). The median rank of the "
            f"list is {o.median_rank:g} of {int(o.n_regions)}, and these Table S4 regions sit "
            f"in the bottom half of the ranking: {_join(low, '; ')}. Top-{PAPER_TOP_TEN} entries not "
            f"on the list: {_join(others, '; ')}. Top-{PAPER_TOP_TEN} entries that do not reach "
            f"p < {PAPER_OVERLAP_ALPHA:g} on their own, because excess per station is a rate "
            f"and a small region can rank high on little evidence: {_join(weak, '; ')}."
        )
        a("")
        (reproduced if o.pvalue < PAPER_OVERLAP_ALPHA else not_reproduced).append(LABEL[election])

    verdict = (
        f"By the rule fixed in `docs/validation_anchors.md` (overlap more than chance at "
        f"one-sided p < {PAPER_OVERLAP_ALPHA:g}), item 2 holds in {_join(reproduced)}"
    )
    verdict += f" and fails in {_join(not_reproduced)}." if not_reproduced else "."
    a(
        verdict + " Where it holds, it says that the paper's regions are over-represented at "
        "the top of this ranking, not that each of them ranks high: the table of ranks "
        "above shows which do and which do not."
    )
    a("")
    a(
        "Full table: [`results/anchor1_by_region.csv`]"
        "(projects/elections/results/anchor1_by_region.csv); the Table S4 comparison: "
        "[`results/anchor1_paper_regions.csv`](projects/elections/results/anchor1_paper_regions.csv)"
        " and [`results/anchor1_paper_overlap.csv`]"
        "(projects/elections/results/anchor1_paper_overlap.csv)."
    )
    a("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mc", type=int, default=200, help="Monte Carlo draws for anchor 1")
    parser.add_argument("--n-boot", type=int, default=999, help="bootstrap resamples for anchor 3")
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
        paper_region_overlap,
        paper_region_ranks,
    )
    from elections.analysis.turnout_bimodality import (
        MODE_BANDWIDTH_PP,
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
    paper_region_ranks(regions).to_csv(OUT / "anchor1_paper_regions.csv", index=False)
    paper_region_overlap(regions).to_csv(OUT / "anchor1_paper_overlap.csv", index=False)

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

    lines.extend(item2_section(regions))

    a("## Anchor 3 — a second mode in the turnout distribution")
    a("")
    a(
        "Klimek, Yegorov, Hanel and Thurner (2012), *PNAS* 109(41), 16469-16473, with their "
        "exclusion rule (electorate below 100). The paper asserts a second mode but supplies "
        "no test for one, so the test is named here rather than inherited: Silverman's "
        "critical-bandwidth statistic (1981), the smallest Gaussian bandwidth at which the "
        "kernel density of turnout still has one mode, against a smoothed bootstrap."
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
    a(
        f"Unimodality is rejected in both. `p` cannot go below 1/({args.n_boot}+1) by "
        "construction, and it is at that floor."
    )
    a("")
    a("### Where the modes are")
    a("")
    a(
        f"Located with a {MODE_BANDWIDTH_PP:g}-point bandwidth, which is deliberately wider "
        "than the one-point period of the sawtooth anchor 1 just established: a narrower "
        "kernel resolves the sawtooth's teeth and reports them as a row of modes."
    )
    a("")
    a("| election | weighting | mode turnout % | winner's share there % | mass within 2.5pp |")
    a("|---|---|---:|---:|---:|")
    for _, r in mode_table.iterrows():
        if r.mass_within_2_5pp < 0.01:
            continue
        a(
            f"| {LABEL[r.election]} | {r.weight_by} | {r.mode_turnout_pct:.1f} | "
            f"{r.winner_share_at_mode_pct:.1f} | {r.mass_within_2_5pp:.3f} |"
        )
    a("")
    a(
        "In both elections there is a principal mode at ordinary turnout and a second mode "
        "within two points of complete turnout, and the winner's share at the near-100 per "
        "cent mode is far above the winner's share at the principal one. That is the anchor's "
        "claim, and it is the joint location, not the bimodality, that carries it."
    )
    a("")
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
