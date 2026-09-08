# ROADMAP

The order in which this programme gets built, and why that order.

This document is a map, not a specification. Where it summarises `CONTRACT.md`, the contract
wins. Every card below is a GitHub issue; the issue body is the card. Nothing here is a result,
and no number in any of these repositories may be cited as one.

## The programme

Statistical forensics of strategically reported data: detecting distortion in numbers produced
by agents with an incentive to distort them. Three repositories:

| Repository | Holds | Role |
|---|---|---|
| [`forensics-core`](https://github.com/Xocas12/forensics-core) | the shared method library | vendored into the other two as the submodule `packages/forensics_core` |
| [`forensic-elections`](https://github.com/Xocas12/forensic-elections) | `projects/elections` | the calibration project for digit and bunching methods |
| [`forensic-economy`](https://github.com/Xocas12/forensic-economy) | `projects/aaer`, `projects/china`, `projects/gosplan` | enforcement labels, provincial statistics, and the transfer target |

**Three projects have ground truth and one does not, and that asymmetry is the whole design.**
Methods are developed and scored where truth is knowable, then carried to the Soviet case where
it is not.

| Project | Labels | Role |
|---|---|---|
| `elections` | strong: three published, replicable signatures | calibrate digit and bunching methods |
| `aaer` | strong but selection-biased: enforcement actions | calibrate supervised and positive-unlabelled learning |
| `china` | partial: admitted episodes and a reform discontinuity | calibrate cross-source reconciliation |
| `gosplan` | almost none: **one** anchor event | **the target** |

**Unifying hypothesis.** Distortion concentrates at discontinuities in the incentive function.
Every project has a notch: the 100 per cent plan-fulfilment bonus, the round vote share, the
analyst consensus, the provincial growth target. WO-104 makes that a single experiment run
across four datasets instead of four separate stories.

**Second signal, weighted equally.** Fabricated series contain too little noise. A reported
yield series with less year-on-year variance than rainfall permits is impossible regardless of
its level.

## What each phase can establish

The "cannot establish" column is binding. An artefact from an early phase is not evidence for a
later claim, however suggestive it looks.

| Phase | Establishes | Cannot establish |
|---|---|---|
| **P0** acquisition | what is obtainable, at what cost, with what provenance | anything statistical whatsoever |
| **P1** calibration infrastructure | power as a function of sample size, effect size and aggregation; false-positive behaviour on controls | anything about whether a real dataset is honest |
| **P2** replication | that the implementation reproduces published results where labels are strong | anything about china or gosplan |
| **P3** reconciliation | whether the methods recover the admitted Chinese episodes | Soviet distortion |
| **P4** transfer | a bound under explicitly stated assumptions | a point estimate of aggregate Soviet distortion, or any causal claim |
| **P5** synthesis | what the programme learned, including what failed | |

## The gates

Each gate is a tracking issue in `forensics-core`, signed by the owner. Nothing downstream
starts until it is signed.

| Gate | Signs off that | Unblocks |
|---|---|---|
| **G0** data | the data is in hand or its absence is explained; the money decisions are made | P1 |
| **G1** methods frozen and pre-registered | the calibration infrastructure exists and the analysis plan is committed **before** any project-level result | P2, P3 |
| **G2** replication | elections and aaer hit or missed their published targets, documented either way | P5 |
| **G3** reconciliation | china is done, **and the gosplan anchor is unsealed** | P4 |
| **G4** transfer | the gosplan analysis has run once and been reported | P5 |
| **G5** final report | every claim carries its false-positive behaviour and its assumptions | |

## The held-out anchor

**This is the most important methodological rule in the programme.**

The Uzbek cotton affair, 1978 to 1983, is gosplan's only anchor. With one anchor, any peeking
destroys the claim, because a detector tuned while looking at its only test case is fit and
validated on the same event.

Therefore the cotton series, its physical correlates, and any breakdown that isolates it **must
not be plotted, tested, scored or summarised before G3 is signed.** Acquiring, transcribing and
validating that data is permitted. Looking at its distributional properties is not.

WO-500 enforces this in code rather than in prose, because the person who breaks the rule will
be someone who forgot it. The seal refuses to return held-out rows without a token that does not
exist until G3 is signed.

## The ideas this roadmap is built around

Five things worth knowing before reading the card list, because most of the cards exist to serve
one of them.

1. **The power atlas (WO-100, WO-101, WO-111).** elections has about 95,000 precincts; gosplan
   will have a few hundred sector-years. The most valuable thing the calibration projects can
   hand the target project is not a detector, it is a power curve. If integer-percentage excess
   has no power at n = 300, gosplan cannot use it, and knowing that before P4 saves a phase.
   The aggregation ladder answers the companion question: Soviet data is aggregate, so how much
   power survives aggregation? elections can answer that because it has a real hierarchy.

2. **The injection harness (WO-102).** gosplan has no labels, so make some: inject a distortion
   of known magnitude into a series believed clean and measure recovery. This is the only way to
   get an operating characteristic on the target's own data shape. The harness is deliberately
   built so an injector is not the estimator run backwards, because otherwise the power curve
   measures self-consistency.

3. **The false-positive budget (WO-103, WO-108, WO-204).** A detector that fires everywhere is
   useless. elections holds the programme's only genuine external controls, two non-Russian
   elections shipped in the same supplement as a replication target. Every claim ships with its
   behaviour on data where nothing should be found, and gosplan inherits that number because it
   cannot compute one of its own.

4. **The `gosplan-env` coupling (WO-506).** The owner's sibling repository simulates enterprises
   facing a plan, a bonus notch, an audit and a ratchet, and it knows the true quantity behind
   every reported one. That is a detector test bench for exactly the data shape gosplan faces.
   Its own plan scopes this coupling to **estimator robustness and nothing else**, and that
   scope is a feature: a detector that cannot find a distortion the simulator generated will not
   find one in the archives either, which is the strongest negative result available here. No
   number from a simulator is evidence about the historical USSR.

5. **A second, weaker label set (WO-505).** A case-level dataset of Soviet plan-fraud
   prosecutions, 1943 to 1962, recording reported against actual quantities. It is not a second
   anchor: it is prosecution-selected exactly as the enforcement releases are, and it covers a
   different period. But it may convert gosplan from one anchor to one anchor plus a
   positive-unlabelled label set, which would be a real upgrade to a project whose central
   weakness is having no labels. The card is written to be sceptical of it.


## The gates, as issues

| Gate | Issue |
|---|---|
| G0 | [GATE G0](https://github.com/Xocas12/forensics-core/issues/1) |
| G1 | [GATE G1](https://github.com/Xocas12/forensics-core/issues/2) |
| G2 | [GATE G2](https://github.com/Xocas12/forensics-core/issues/3) |
| G3 | [GATE G3](https://github.com/Xocas12/forensics-core/issues/4) |
| G4 | [GATE G4](https://github.com/Xocas12/forensics-core/issues/5) |
| G5 | [GATE G5](https://github.com/Xocas12/forensics-core/issues/6) |

## Cards in this repository (forensic-elections)


### elections

| Card | Phase | Gate | Diff | Depends on | What it buys |
|---|---|---|---|---|---|
| [WO-200](https://github.com/Xocas12/forensic-elections/issues/1) Run the acquisition for real and populate data/raw | P0 | G0 | 1 | - | (needs a human) Nothing has been downloaded into any tree. |
| [WO-201](https://github.com/Xocas12/forensic-elections/issues/2) Build the tidy artefact and run the acceptance checks against the real 2018 file | P0 | G0 | 2 | WO-200 | The 2011 file was validated during scaffolding; the 2018 file never was. |
| [WO-202](https://github.com/Xocas12/forensic-elections/issues/3) Decide the 2018 national anchor | P0 | G0 | 1 | - | (needs a human) The commission's own portal summary and its Resolution 152/1255-7 disagree by 4,313 votes and 7,122 registered voters, reportedly after four polling stations were cancelled. |
| [WO-203](https://github.com/Xocas12/forensic-elections/issues/4) Obtain the Klimek supplementary material and fill in two TO CONFIRM anchors | P0 | G0 | 1 | - | (blocked, needs a human) This single blocked source holds two of the three replication targets hostage. |
| [WO-204](https://github.com/Xocas12/forensic-elections/issues/5) Extract the Poland and Spain control tables from the acquired supplement | P1 | G1 | 2 | WO-200 | These two elections are the only genuine external controls the whole programme has: real elections, real data, from a supplement whose Russian half is a documented replication target. |
| [WO-205](https://github.com/Xocas12/forensic-elections/issues/6) Expose elections as the substrate for the power atlas and the aggregation ladder | P1 | G1 | 2 | WO-201 | The library's atlas cards need a real dataset with a large sample and a genuine hierarchy, and this is the only one in the programme that has both. |
| [WO-206](https://github.com/Xocas12/forensic-elections/issues/7) Populate the elections notch catalogue | P1 | G1 | 2 | WO-104 | Round vote shares and round turnout are this project's instance of the unifying hypothesis, and the catalogue schema requires each notch to record what someone actually gained by reporting on one side of the line. |
| [WO-207](https://github.com/Xocas12/forensic-elections/issues/8) Replicate the integer-percentage sawtooth, conditioned on precinct size | P2 | G2 | 3 | WO-201, WO-204, WO-110 | This is the cleanest published result in the programme and the first replication target. |
| [WO-208](https://github.com/Xocas12/forensic-elections/issues/9) Replicate the comet tail | P2 | G2 | 4 | WO-203, WO-207 | (blocked) The second published signature, and the one that yields an anomalous-vote count rather than a test statistic. |
| [WO-209](https://github.com/Xocas12/forensic-elections/issues/10) Replicate turnout bimodality | P2 | G2 | 3 | WO-203, WO-207 | (blocked) The third signature, and the weakest on its own: honest heterogeneity produces bimodality too. |
| [WO-210](https://github.com/Xocas12/forensic-elections/issues/11) Decompose detection power by method family, the project's actual research question | P2 | G2 | 4 | WO-207, WO-208, WO-209, WO-111 | The project's research question is not 'were these elections falsified', which the literature already answers, but 'how much detection power comes from each family of method'. |

## Cards in the other repositories

This roadmap is shared; the full card list lives in each repository's own copy.

- [forensics-core](https://github.com/Xocas12/forensics-core/issues) (16 cards)
- [forensic-economy](https://github.com/Xocas12/forensic-economy/issues) (27 cards)

## Working rules

One card, one branch, one pull request. Read only what the card's whitelist names, write only
what its "Write only" list names, run the completion command verbatim, and report in the card's
format.

When the card and its whitelist do not determine a choice, **file an ambiguity report and end
the session.** Choosing the reasonable default is a violation. Filing an ambiguity report is
correct behaviour, not failure.

The full rules are in [`CONTRACT.md`](CONTRACT.md): thirteen of them, each stating what a
violation looks like concretely and what catches it. The file is identical in all three
repositories, and `test_contract.py` freezes the rule numbers, because `gosplan/seal.py`
cites rule 5 and `eval/harness.py` cites rule 9 by number.

## Standing warning

Nothing in these repositories has been run. There are no results, no estimates and no findings.
Every module under any `analysis/` raises `NotImplementedError`, every `data/raw` is empty, and
all six gates are unsigned.
