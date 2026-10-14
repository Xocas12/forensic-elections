"""Anchor 1 item 2: the paper's regional list, the comparison against it, and the report text.

Item 2 asks for the integer-percentage excess to be concentrated in "the regions the paper
names". These tests hold three things in place:

* the list is the paper's Table S4, and the copy in the anchors document and the copy in code
  are the same list;
* the comparison is computed (top-ten overlap against the hypergeometric chance of it), and
  is checked here against values worked out by hand on a synthetic ranking;
* the item-2 text of ``RESULTS.md`` is generated from the regional table, so it cannot name a
  region, a rank or a ``z`` the table does not hold. The last test regenerates that section
  from the committed ``results/anchor1_by_region.csv`` and requires ``RESULTS.md`` to match.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pandas as pd
import pytest
from elections.analysis import integer_percentage as ip

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects" / "elections"
ANCHORS = PROJECT / "docs" / "validation_anchors.md"
REGION_TABLE = PROJECT / "results" / "anchor1_by_region.csv"
RESULTS = ROOT / "RESULTS.md"


def _run_analysis():
    spec = importlib.util.spec_from_file_location(
        "run_analysis_under_test", ROOT / "scripts" / "run_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_regions(order: list[str], election: str = "2011-duma") -> pd.DataFrame:
    """A regional table in the shape of ``integer_percentage_excess_by_region``.

    ``order`` lists region strings from rank 1 downwards. ``z`` and the station count are
    made distinct per rank so that a test can tell which row a number in the text came from.
    """
    n = len(order)
    return pd.DataFrame(
        {
            "election": election,
            "quantity": "winner_share",
            "rank": range(1, n + 1),
            "region": order,
            "n": [1000 + 7 * i for i in range(n)],
            "excess_per_unit": [0.2 - 0.002 * i for i in range(n)],
            "z": [9.5 - 0.11 * i for i in range(n)],
            "pvalue": [0.001 if i % 3 else 0.2 for i in range(n)],
        }
    )


def _first_names() -> list[str]:
    return [p.table_names[0] for p in ip.PAPER_TOP_REGIONS]


def _fillers(k: int) -> list[str]:
    return [f"Синтетический регион {i}" for i in range(k)]


# ------------------------------------------------------------------------------ the list


def test_13_the_benchmark_is_the_fifteen_regions_of_table_s4_in_the_papers_order():
    assert [p.iso for p in ip.PAPER_TOP_REGIONS] == [
        "DA", "BA", "KEM", "KDA", "KO", "MOS", "MOW", "KB",
        "TA", "ROS", "IN", "SE", "MO", "KC", "CE",
    ]  # fmt: skip
    peaks = [p.peak_thousands for p in ip.PAPER_TOP_REGIONS]
    assert peaks == sorted(peaks, reverse=True), "Table S4 is sorted by peak height"
    assert peaks[0] == 87 and peaks[-1] == 18


def test_13_validation_anchors_records_the_same_list_with_its_source():
    text = ANCHORS.read_text(encoding="utf-8")
    assert "Table S4" in text
    assert "arXiv:1410.6059v4" in text
    assert "arxiv_kobak_1410_6059v4" in text, "the source must point at its registry entry"
    rows = re.findall(r"^\| ([A-Z]{2,3}) \| ([^|]+?) \| (\d+) \| ([^|]+?) \|$", text, re.M)
    assert [(iso, name, int(peak)) for iso, name, peak, _ in rows] == [
        (p.iso, p.paper_name, p.peak_thousands) for p in ip.PAPER_TOP_REGIONS
    ]
    for (_, _, _, table_cell), paper in zip(rows, ip.PAPER_TOP_REGIONS, strict=True):
        for name in paper.table_names:
            assert name in table_cell


def test_13_the_source_is_registered():
    import yaml

    registry = yaml.safe_load((PROJECT / "data" / "SOURCES.yaml").read_text(encoding="utf-8"))
    entry = {e["id"]: e for e in registry}["arxiv_kobak_1410_6059v4"]
    assert entry["status"] == "verified"
    assert entry["url"] == "https://arxiv.org/pdf/1410.6059v4"
    assert "Table S4" in entry["notes"]


# ------------------------------------------------------------------------ the comparison


def test_13_overlap_and_hypergeometric_p_match_a_hand_computation():
    names = _first_names()
    # 4 of the 15 in the top ten, the other 11 at ranks 41-51, out of 60 regions.
    order = names[:4] + _fillers(6) + _fillers(36)[6:] + names[4:] + _fillers(45)[36:]
    order = order + _fillers(60)[45 : 45 + 60 - len(order)]
    assert len(order) == 60 and len(set(order)) == 60
    table = _synthetic_regions(order)

    result = ip.paper_region_overlap(table).iloc[0]
    assert result.n_regions == 60
    assert result.n_paper == 15
    assert result.overlap == 4
    assert result.expected_overlap == pytest.approx(15 * 10 / 60)
    by_hand = sum(math.comb(15, i) * math.comb(45, 10 - i) for i in range(4, 11)) / math.comb(
        60, 10
    )
    assert result.pvalue == pytest.approx(by_hand, rel=1e-12)
    ranks = ip.paper_region_ranks(table)
    assert sorted(ranks["rank"]) == [1, 2, 3, 4, *range(41, 52)]
    assert result.median_rank == 44


def test_13_no_overlap_gives_p_of_one():
    order = _fillers(10) + _first_names() + _fillers(40)[10:]
    result = ip.paper_region_overlap(_synthetic_regions(order)).iloc[0]
    assert result.overlap == 0
    assert result.pvalue == pytest.approx(1.0)


def test_13_moscow_matches_either_spelling():
    names = _first_names()
    names[6] = "город Москва"
    ranks = ip.paper_region_ranks(_synthetic_regions(names + _fillers(20), "2018-presidential"))
    assert ranks.loc[ranks.iso == "MOW", "region"].item() == "город Москва"


def test_13_a_missing_table_s4_region_is_refused_not_dropped():
    names = _first_names()[:-1]  # Chechnya absent
    with pytest.raises(ValueError, match="CE"):
        ip.paper_region_overlap(_synthetic_regions(names + _fillers(20)))


# --------------------------------------------------------------------------- the report


def test_13_item_2_text_is_generated_from_the_table_it_describes():
    """Tatarstan and Karachay-Cherkessia at the bottom: the text must say so, from the table."""
    names = _first_names()
    tatarstan, kc = names[8], names[13]
    rest = [n for n in names if n not in (tatarstan, kc)]
    order = rest[:5] + _fillers(30) + rest[5:] + [kc, tatarstan]
    table = _synthetic_regions(order)
    text = "\n".join(_run_analysis().item2_section(table))

    n = len(order)
    assert f"{n - 1} of {n}" in text and f"{n} of {n}" in text
    for _, row in table.iterrows():
        if row["rank"] <= 10:
            assert f"| {row['rank']} | {row['region']} | {row['n']:,} |" in text
    ranks = ip.paper_region_ranks(table)
    for _, r in ranks.iterrows():
        assert f"| {r.iso} | {r.paper_name} | {r['rank']} of {n} | {r.z:.2f} |" in text
    assert f"Tatarstan, Respublika ({n})" in text, "a low-ranking S4 region must be named"
    overlap = ip.paper_region_overlap(table).iloc[0]
    assert f"{overlap.overlap} of the 15 Table S4 regions" in text
    # the old hand-typed sentence, whatever the table says
    assert "between 4 and 9" not in text
    assert "32 to 51 stations" not in text


def test_13_results_md_item_2_section_is_what_the_committed_table_generates():
    regions = pd.read_csv(REGION_TABLE)
    expected = "\n".join(_run_analysis().item2_section(regions))
    text = RESULTS.read_text(encoding="utf-8")
    start = text.index("### Regional concentration (anchor item 2)")
    end = text.index("\n## ", start)
    assert text[start:end].strip() == expected.strip()


def test_13_results_md_reports_the_comparison_for_both_elections():
    text = RESULTS.read_text(encoding="utf-8")
    regions = pd.read_csv(REGION_TABLE)
    overlap = ip.paper_region_overlap(regions)
    assert set(overlap["election"]) == {"2011-duma", "2018-presidential"}
    for _, o in overlap.iterrows():
        assert f"{o.overlap} of the 15 Table S4 regions" in text
