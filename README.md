# forensic-elections

Statistical forensics of strategically reported data — the **elections** calibration project.
One of two project repositories that share the
[`forensics-core`](packages/forensics_core/) method library (a git submodule); the other is
`forensic-economy` (`aaer`, `china`, `gosplan`).

## The programme

Three of four projects have ground truth; one does not. Methods are developed and scored
where truth is knowable, then carried to the Soviet case where it is not.

| Project | Repository | Labels | Role |
|---|---|---|---|
| `elections` | **this repo** | Strong (published signatures, replicable prior findings) | Calibrate digit and bunching methods |
| `aaer` | forensic-economy | Strong (SEC enforcement actions as positives) | Calibrate supervised / PU learning, benchmark against published AUC |
| `china` | forensic-economy | Partial (self-admitted falsification, a statistical-reform discontinuity) | Calibrate cross-source reconciliation against physical proxies |
| `gosplan` | forensic-economy | Almost none (one anchor event) | **The target.** Methods validated above are transferred here |

**Unifying hypothesis.** Distortion concentrates at discontinuities in the incentive
function. Here the notch is the round-number vote share and turnout figure; the shared
library makes "find the notch, estimate excess mass around it" a first-class operation.

## This project

**Question.** Reproduce known falsification signatures in Russian federal elections (2011
State Duma, 2018 presidential) at precinct level, and quantify how much detection power comes
from each family of method. See [projects/elections/README.md](projects/elections/README.md)
for data status, current state and next actions, and
[projects/elections/docs/](projects/elections/docs/) for the research question, data
dictionary, validation anchors and known traps.

## Layout

```
forensic-elections/
├── config/forensics.toml        contact string (REQUIRED before any fetch) and per-host rate limits
├── packages/forensics_core/     git submodule → forensics-core
├── projects/elections/
│   ├── README.md
│   ├── data/SOURCES.yaml        the source registry — nothing enters a pipeline without an entry
│   ├── data/ACCESS_NOTES.md     what failed, what is gated, what needs a human
│   ├── data/{raw,interim,processed}/   gitignored; rebuilt by `make data`
│   ├── src/elections/{acquire,clean,features,analysis}/
│   ├── notebooks/00_data_audit.ipynb
│   ├── docs/{research_question,data_dictionary,validation_anchors,known_traps}.md
│   └── tests/
└── DATA_STATUS.md               source × access tier × status
```

## How to run

Requirements: [`uv`](https://docs.astral.sh/uv/), GNU make, git. Python 3.12 is pinned.

```sh
git clone --recurse-submodules <this repo>     # or: make submodule after a plain clone
make setup          # submodule init + uv sync --all-packages + pre-commit hooks
make test
make lint
# Put a real name and email in config/forensics.toml ([http].contact) first — every
# network script refuses to run while the placeholder is in place.
make data
make validate-sources
```

`make data` is idempotent: every successful fetch is checksummed and recorded in
`data/SOURCES.yaml`, every attempt is appended to `data/fetch_log.jsonl`, and a source is
never fetched twice unless forced.

## Rules the code enforces

1. No invented URLs: sources are verified by an actual fetch (HTTP status, bytes, SHA-256,
   timestamp) or recorded as `unverified` / `blocked` with a reason.
2. Every fetch attempt is logged.
3. No synthetic data under `data/`; fixtures live in `tests/fixtures/` with a `synthetic_` prefix.
4. Fail loudly and continue.
5. Raw data never enters git (`.gitignore` plus a pre-commit hook).
6. Access policies are respected: contact header, per-host rate limits, aggressive caching.
7. "Not found" and "not free" are different statuses.

## Status

Scaffold, shared library, source registry and acquisition pipeline built; free sources
attempted. **No analysis has been run and no findings exist in this tree.** See
[DATA_STATUS.md](DATA_STATUS.md).
