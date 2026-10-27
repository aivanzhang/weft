# Weft

Generate executable training environments from evidence. Test them against observed outcomes. Repair the environment when the evidence disagrees.

Weft generates and repairs reinforcement-learning environments using real-world evidence. It is a Loom-adjacent project with its own identity. This **separate hackathon repository** is adjacent to, not a fork of, the existing `../loom` data-ingestion product.

The original application used the name Loom. Historical application text and source snapshots retain that name; the project is now called **Weft**.

## Current state

A local iteration foundation now runs a real video through evidence extraction, a shared simulator, bounded repair, independent checks, and RL training. **The implemented profile is one isolated yellow ball moving horizontally, not arbitrary-video or 3D reconstruction.** The controls and parking objective are authored extensions, not recovered from the source.

The larger physical-first A–E scope remains the target. The rolling-ball clip is a reproducible engineering fixture, not a final decision about the product's first task family. No existing Loom code, data, credentials, or Git history was copied. Nothing is deployed, published, or submitted.

- [Scope and architecture](docs/architecture.md): target capabilities, component contracts, and current limits.
- [Iteration workflow](docs/iteration.md): change one hypothesis, run, inspect, and compare.
- [TODO and acceptance gates](TODO.md): verified state and the next independent work items.
- [Execution cost and bounds](docs/v1-cost-and-scope.md).

## Run locally

Tested on Python 3.14.7 / macOS arm64. Node.js is needed only for the Python/JavaScript physics parity test. Other Python versions/platforms are not yet verified against the lockfile.

For a fresh checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m weft fetch-sample
```

Reuse an existing valid environment. The sample is hash-pinned and already included locally; fetching verifies it without downloading again when present.

```sh
.venv/bin/python -m pytest
.venv/bin/python -m weft experiment experiments/rolling-ball-no-repair.json --out artifacts/baseline-01
.venv/bin/python -m weft experiment experiments/rolling-ball.json --out artifacts/candidate-01
.venv/bin/python -m weft compare artifacts/baseline-01 artifacts/candidate-01 --out artifacts/comparison-01.json
.venv/bin/python -m weft verify artifacts/candidate-01
```

Use a new output name for each run. Open `artifacts/candidate-01/index.html` directly for the local evidence viewer; no development server is needed. Alternatively, step the Python environment:

```sh
.venv/bin/python -m weft interact --world artifacts/candidate-01/world.json
```

Default runs use local CPU computation with **$0 external API/cloud charges**. Meta is not used by default. The optional Meta adapter is implemented but not live-verified; it needs configured access and an explicit inference budget. No credential presence alone authorizes a paid call.

## What a run preserves

Source attribution and hash; frame-level evidence and partitions; the object graph; initial/selected worlds; task definition; every repair decision; changed-action trials; training records and policy; evaluation trials; replay/video viewer; resolved experiment settings; implementation source snapshots; and a file-hash manifest. Failures preserve a `failure.json` instead of silently falling back.

Comparison requires matching evidence and scoring implementations for reconstruction. RL comparisons additionally require the same fitted world, task, implementation, and evaluation seeds. Results from different simulated worlds are descriptive—not proof of improved real-world performance.

## Hackathon reference archive

Intended entry: **Indie Developer / Push the Frontier** at the Meta Global AI Developer Hackathon. The original reference analyses below predate the one-video A–E scope; use [architecture](docs/architecture.md) and [TODO](TODO.md) for current implementation planning. Official snapshots remain unchanged.

1. [Hackathon reference hub](hackathon/README.md)
2. [What the judges want and how Loom fits](hackathon/analysis/fit-and-strategy.md)
3. [Requirements, dates, and unresolved conflicts](hackathon/analysis/requirements.md)
4. [Proposed demo and evaluation design](hackathon/analysis/demo-plan.md)
5. [Prior work versus hackathon work](hackathon/analysis/originality.md)
6. [Application context](hackathon/analysis/application-context.md)

The submission deadline is **November 4, 2026, at 6:00 p.m. PST**. Source snapshots retain the organizers' original timezone wording; planning documents use America/Los_Angeles.

## Repository boundaries

- Private GitHub repository: `kintsugi-software/weft`. No public release, deployment, or hackathon submission.
- `hackathon/sources/`: source snapshots and explicit unavailable-content records.
- `hackathon/analysis/`: interpretation and proposed project direction, not official rules.
- `src/weft/`: local A–E profile, experiment runner, and comparison tools.
- `experiments/`: versioned run configurations; input paths resolve relative to each config.
- `tests/`: contract, physics, provenance, repair, and comparison checks.
- `data/`: attributed source fixture; see [source provenance](data/README.md).
- `artifacts/`: ignored local run directories. Preserve runs; never edit scores to improve a result.
- `scripts/`: reference archival tooling.
- `.local/`: ignored raw HTTP responses and supporting artifacts; never publish blindly.
- Source documents retain their original ownership and terms. A public-code license is not selected yet; do not apply a future project license to third-party reference material.

The event requires a public repository at submission. This repository has **not** been made public. Publication and submission remain separate future actions.
