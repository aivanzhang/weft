# Iterating on Weft

For the new interactive exercise/tabletop workflow, start with the [local lab guide](local-lab.md). Commands below remain the original rolling-profile experiment/comparison contract. `weft compare` is intentionally restricted to that original format; it does not automatically rank exercise/planar lab runs. Keep their source/model/task identities and matched evaluation conditions explicit.

## One experiment, one hypothesis

1. Read the current scope and choose a pending acceptance gate in [TODO](../TODO.md).
2. Copy a preset in `experiments/`, give it a new name, and state the expected change and the evidence that could disprove it.
3. Change the relevant module or settings. Preserve evaluation inputs and success definitions when claiming improvement against a baseline.
4. Run appropriate tests and one new experiment directory. Keep failures and rejected proposals.
5. Verify artifacts, inspect replay/state traces, then compare with the baseline. Record mixed results; do not collapse unlike metrics into a winning score.
6. Update the TODO with the run path, source/manifest identities, verification evidence, and remaining limits. Select fresh final evidence before making a final generalization claim.

## Commands

From the repository root with the existing environment:

```sh
.venv/bin/python -m weft experiment experiments/rolling-ball-no-repair.json --out artifacts/my-baseline
.venv/bin/python -m weft experiment experiments/rolling-ball.json --out artifacts/my-candidate
.venv/bin/python -m weft verify artifacts/my-candidate
.venv/bin/python -m weft compare artifacts/my-baseline artifacts/my-candidate --out artifacts/my-comparison.json
```

Each output name must be new. Comparison files must sit outside both input run directories and cannot overwrite an existing file. Open `artifacts/my-candidate/index.html` directly, or use `weft interact --world artifacts/my-candidate/world.json` for a terminal interaction. No server is required.

For an exploratory local source without a preset:

```sh
.venv/bin/python -m weft run --video data/my-video.mp4 --out artifacts/exploration-01 --episodes 15000 --eval-episodes 300 --seed 1729 --eval-seed 938401
```

Only the documented isolated-ball profile is supported. Unsupported inputs fail or remain unverified; do not call a custom file an arbitrary-video capability. Record source rights separately when not using the pinned sample.

## Config contract

Top-level fields are required: `schema_version` (1), `name`, `hypothesis`, `video`, `source_sha256`, and `run`. Unknown fields are rejected to catch misspelled settings. Video paths resolve relative to the config file, not the shell's working directory. Source identity is checked before creating output.

`run` supports:

| Setting | Default | Constraint |
| --- | --- | --- |
| `episodes` | 15000 | Integer 1–50000; training also has a 120-second wall-time cap |
| `eval_episodes` | 300 | Integer 1–2000 per policy |
| `seed` / `eval_seed` | 1729 / 938401 | Nonnegative bounded integers; must differ |
| `provider` | `local` | `local` or explicitly selected `meta` |
| `repair_enabled` | `true` | Boolean; disabling repair keeps all evaluation gates |
| `meta_budget` | `null` | Meta only; finite explicit budget covering the current recorded estimate |
| `tracking` | Profile defaults | HSV thresholds and decode bounds from `TrackingConfig`; cannot expand the hard resource envelope |

Resolved defaults and the original experiment document are preserved in `run-config.json`. Changing tracking measurements also changes the evaluation evidence; the comparator correctly refuses to call that a matched replay comparison. To evaluate a better tracker, first establish an independent labeled perception benchmark.

## Reading the comparison

- **Reconstruction:** same source/measurement partitions, evaluator, and shared physics implementation. Lower development error is not necessarily lower audit error.
- **RL:** additionally requires the same world, task, evaluation seeds, and relevant implementation. Training seed/episode-count variants can be compared if this benchmark stays fixed. Training and evaluation currently share `learning.py`, so edits there conservatively invalidate automatic comparability; review and explicitly version the benchmark rather than bypassing the guard.
- **Time/cost:** descriptive observations, not controlled speed or cost claims. A single concurrent run is not a performance benchmark.
- **No audit observations:** audit metric is unavailable, never a fabricated zero.

Repeated inspection turns the fixed seeds and reserved pass into development evidence. Never tune against them repeatedly and then describe them as a fresh final holdout.

## Run artifacts and recovery

`report.json` is the summary; `evidence.json`, `repair-ledger.json`, `rl-evaluation.json`, `world.json`, and `task.json` retain the underlying records. `manifest.json` pins every artifact; `implementation.json` identifies source files. New runs copy those files into `implementation-source/`, including the local build manifest and dependency lock when available. Credentials and environment files are not included.

To reproduce an old implementation, prepare a separate local scratch checkout, place the snapshot package files under `src/weft/`, restore the two files under `implementation-source/project/` to its root, and use the saved `run-config.json`, source video, and pinned environment. Do not overwrite the current working tree or the old run. Runtime/codec differences may change encoded bytes; compare numerical results and disclose platform changes.

These are append-only folders by workflow, not cryptographically signed records. Hash checks detect changed files but do not authenticate an author who can rewrite the manifest too. The runner preserves partial artifacts and `failure.json` on failure; diagnose that failure before choosing a new run directory. There is no checkpoint-resume or automatic retry yet.

The original `artifacts/run-01` and `iteration-*-01` runs predate source-copy preservation. Keep them as historical records; their implementation hashes alone cannot restore overwritten uncommitted code.
