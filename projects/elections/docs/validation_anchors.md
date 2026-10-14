# Validation anchors: elections

What a working method must reproduce. A detector that fails these is broken, whatever else it
finds. Nothing here is an analysis result: these are published targets and the arithmetic
facts a loader must match before any test is run.

Every fact below was read from the source cited, in this scaffolding session. Items marked
**TO CONFIRM** were not read from a primary source and must be checked before use.

## Tier 0: the data must load correctly

Before any statistical claim, the loader must reproduce these totals. They are the cheapest
possible test that the scrape, the encoding and the column mapping are right.

| Election | Precinct rows | Registered voters (sum) | Check |
|---|---|---|---|
| 2011 State Duma, 4 Dec 2011 | 95,225 | 109,229,337 | United Russia 32,371,737 = 49.31 % |
| 2018 presidential, 18 Mar 2018 | 97,699 | 109,008,428 | Putin 56,430,712 = 76.69 % |

Sources: row counts and column counts were counted locally from
`dkobak/elections` `data/2011.csv.zip` (29 columns) and `data/2018.csv.zip` (24 columns).
The 2011 national totals were read from the archived Central Election Commission portal page
(Wayback snapshot 2021-12-29 of the `vybory.izbirkom.ru` results root for the 2011 Duma). The
2018 totals are those of Central Election Commission Resolution 152/1255-7 of 23 March 2018,
read from the archived English results page on `cikrf.ru`.

**A discrepancy to decide about, not to paper over.** The archived commission portal's own
2018 summary shows 56,426,399 votes for Putin and 109,001,306 registered voters, 4,313 votes
and 7,122 voters fewer than the resolution, reportedly after four polling stations were
cancelled. Pick one as the anchor, record which, and expect precinct sums to match that one.

## Tier 1: the three published signatures

These are the replication targets, in the order they should be attempted.

### 1. Integer-percentage excess mass (the sawtooth)

**Claim.** Turnout and vote-share percentages pile up at integer values far more often than
binomial noise allows, and the excess is concentrated in specific regions.

**Reference.** Kobak, Shpilkin & Pshenichnikov (2016), "Integer percentages as electoral
falsification fingerprints", *Annals of Applied Statistics* 10(1): 54–73. The supplementary
dataset is on Figshare (DOI 10.6084/m9.figshare.3126883.v2, CC BY 4.0) and covers Russia
2000–2012 plus Poland 2010 and Spain 2011 in a reduced ten-column format.

**Method.** `forensics_core.digits.integer_pct.integer_excess`, whose Monte-Carlo null redraws
each precinct's numerator as Binomial(registered voters, observed share). This is the paper's
own null model.

**What counts as reproduced.** A significant positive excess in 2011 and 2018 nationally; the
excess concentrated in the same regions the paper names; and the excess vanishing when the
test is applied to the Poland 2010 and Spain 2011 control data in the same supplement. That
last check is the one that shows the method is not detecting arithmetic.

**The regions the paper names (the item-2 benchmark).** Table S4 of the paper's supplement,
"Top 15 regions contributing to the integer anomaly", read from arXiv:1410.6059v4 (the
version of 29 June 2016), p. 22; registry entry `arxiv_kobak_1410_6059v4`. The paper defines
the column as "the height of the maximal integer peak, where maximum is taken over all
years, over both turnout and leader's result, and over all integer percentage values from
70% to 99%. Peak heights are measured relative to the mean Monte Carlo value." In the
paper's order:

| ISO code | Region name (as printed) | Maximal integer anomaly (10^3) | Region in the precinct tables |
|---|---|---:|---|
| DA | Dagestan, Respublika | 87 | Республика Дагестан |
| BA | Bashkortostan, Respublika | 64 | Республика Башкортостан |
| KEM | Kemerovskaya Oblast | 52 | Кемеровская область |
| KDA | Krasnodarskiy Krai | 51 | Краснодарский край |
| KO | Komi, Respublika | 38 | Республика Коми |
| MOS | Moskovskaya Oblast | 35 | Московская область |
| MOW | Moscow | 34 | Город Москва (2011), город Москва (2018) |
| KB | Kabardino-Balkarskaya Respublika | 33 | Кабардино-Балкарская Республика |
| TA | Tatarstan, Respublika | 33 | Республика Татарстан (Татарстан) |
| ROS | Rostovskaya Oblast | 33 | Ростовская область |
| IN | Ingushetiya, Respublika | 28 | Республика Ингушетия |
| SE | Severanaya Osetiya-Alaniya, Respublika | 24 | Республика Северная Осетия - Алания |
| MO | Mordoviya, Respublika | 23 | Республика Мордовия |
| KC | Karachayevo-Cherkesskaya Respublika | 21 | Карачаево-Черкесская Республика |
| CE | Chechenskaya Respublika | 18 | Чеченская Республика |

The last column maps the paper's names to the region strings of the dkobak tables; it is a
lookup, not something the paper prints. The same list is held in code as
`elections.analysis.integer_percentage.PAPER_TOP_REGIONS`, and a test fails if the two differ.

*The comparison.* Rank the federal subjects of each election by excess per station on the
winner's share (`integer_percentage_excess_by_region`). Count how many of the 15 regions
above fall in the top 10, and compare that count with the hypergeometric distribution of the
overlap between a random top 10 and a fixed list of 15 drawn from the same regions; report
the exact one-sided p, P(overlap >= observed), and the median rank of the 15 beside it.
Report every one of the 15 with its rank and `z`, those that rank low as well as those that
rank high. Item 2 counts as reproduced in an election when that p is below 0.05.

*What this does and does not establish.* It tests whether the paper's regions are
over-represented at the top of this ranking; it does not test that each of them ranks high.
Two differences limit it: Table S4 ranks a different statistic (maximal peak height across
2000-2012 and both quantities, not excess per station on one quantity in one election), and
the paper's data end in 2012, so for 2018 the list is an extrapolation. And it is not a blind
pre-registration: this comparison was fixed after the regional table of the first run had
been committed and inspected, when the earlier wording of item 2 turned out to have no
recorded benchmark at all.

### 2. The comet tail (turnout versus vote share)

**Claim.** In the two-dimensional histogram of precinct turnout against the winner's vote
share, honest precincts form a single cloud, while a tail extends toward the (100 %, 100 %)
corner. Shpilkin's estimator takes the vote total in the tail, above the level implied by the
distribution at ordinary turnout, as a count of anomalous votes.

**Reference.** Sergey Shpilkin's method, described in Kobak, Shpilkin & Pshenichnikov (2016)
and in their *Significance* article of the same year. **TO CONFIRM:** the exact estimator
definition and the published anomalous-vote counts for 2011 and 2018 were not read from a
primary source in this session; read the papers before treating any number as a target.

**Known fragility.** The estimate is sensitive to which regions are included. Report it with
and without the largest contributing regions, and at territorial-commission as well as
precinct level.

### 3. Turnout bimodality

**Claim.** The distribution of precinct turnout has a second mode near 100 % that honest
elections do not produce.

**Reference.** Klimek, Yegorov, Hanel & Thurner (2012), "Statistical detection of systematic
election irregularities", *PNAS* 109(41): 16469–16473. **TO CONFIRM:** whether the paper's
supplementary material contains the underlying election data or only code and figures was not
established; the follow-up hunt covering this was still running when this file was written.

**Caveat that belongs in the result.** Bimodality alone is not a signature; genuine urban and
rural heterogeneity produces it too. The claim is about the *joint* turnout–vote-share
distribution and the location of the second mode. See `known_traps.md`.

## Tier 2: the false-positive check

A detector that fires everywhere is useless. Before transferring any method to another
project, run it on a subsample where the published literature reports no anomaly, and on the
Poland and Spain control data in the Kobak et al. supplement. Record the false-positive rate
alongside every power number. This is what the `gosplan` project will inherit and cannot
compute for itself.

## Provenance of the data used

`dkobak/elections` states in its own README that the precinct data was scraped from the
official commission portal by Sergey Shpilkin (through 2021) and Ivan Shukshin (2024). The
sum of registered voters in the 2011 file matches the commission's own national total, which
is consistent with that provenance but is not an independent verification of it. The live
commission portal is unreachable from outside Russia, so re-scraping to check is not
available here; see `data/ACCESS_NOTES.md`.
