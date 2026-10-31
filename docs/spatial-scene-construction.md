# Turn a saved object scene into a 3D lab

`spatial-scene` connects existing video reconstruction to the ordinary 3D lab.
It preserves the sealed planar run and builds one explicit 3D hypothesis.
It does **not** infer hidden depth, calibrate a camera or validate physical accuracy.

## Open the guided video result

The explicit direct-video workflow now reaches a lab with a reused learned
policy and a verified mixed learned/manual recording:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/spatial-guided-play-acceptance-02/workflow/lab
```

Replace `lab-console` with `lab` to start the browser interface yourself.
The command performs normal artifact/runtime admission; it does not reprocess
the source or retrain. [Step-by-step index](../artifacts/spatial-guided-play-acceptance-02/workflow/index.html)
and [saved recording](../artifacts/spatial-guided-play-acceptance-02/workflow/lab/sessions/session-9a414378073d/replay.html)
are available in this workspace, not in a fresh clone.

The run completed in **3.429 seconds**, including source tracking, explicit 3D
construction, policy reuse, lab preparation and a recording with **51 learned
actions, three manual actions, one reset and 56 exported frames**. Exact action
replay and saved policy decisions pass. No new training or paid request occurred;
these are short local observations, not a sustained performance benchmark.
The independent saved-output audit passes: source/runtime bindings, all saved
recording events, exact replay receipts and learned-policy identities agree.
That audit performs no new physics replay or source processing.
Tracking still covers only the declared frames 39–60: 21 measured/model slots
at 40–60, with initialization slot 39 preserved as missing. All 157 decoded
source slots remain visible; they were not all tracked.

This ball hypothesis still has authored zero gravity and no floor. In the
separate three-seed study, learned policies achieved **8/12, 5/12 and 4/12**
changed-start successes; a fixed untrained authored controller achieved **12/12**.
The policy used here, seed 1729, was selected before those results. The workflow
is playable, but these results do not show learning superiority, recovered
rolling physics or transfer to a real object.

The first guided attempt remains saved as failed01. It stopped before tracking
because worker startup exceeded inherited CPU limits and lost the frozen package
path. The bounded startup fix respects both; successful02 uses a new output
directory. [Exact receipts and limits](handoffs/combined.md#guided-object-workflow--october-31-2026-904-am-pdt).

## Open the verified local result

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/spatial-scene-lab-acceptance-01/lab
```

Replace `lab-console` with `lab` for the browser interface. The original video,
source evidence and tracking previews are available alongside the 3D sphere.
This manual-only example has authored zero gravity and no floor, not recovered
tabletop rolling. [Saved interactive recording](../artifacts/spatial-scene-lab-acceptance-01/lab/sessions/session-8faf924ec3bc/replay.html).

## Use the saved ball example

This example reuses the manually tracked grayscale video, without repeating
tracking or 2D simulation. Its declared tracking interval is frames 39–60;
the original 157-frame video remains unchanged. The sphere, zero gravity,
initial spin, lateral position and parking task are authored assumptions.
There is no reconstructed floor or wall.

### Guided construction, optional learning and lab

The common builder now connects these existing stages and saves a local index:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --planar-run artifacts/manual-video-acceptance-01/lab/run \
  --config examples/spatial-ball-hypothesis.json \
  --out artifacts/my-object-workflow-01 --train
```

Open the printed `index.html` to see each stage, the exact assumptions, logs,
source identities and saved controller counts. The resulting `lab/` supports
manual or available learned controls, reset and recording. The command prints
both browser-lab and terminal resume commands; it starts neither by default.
Add `--console` or `--serve` explicitly to start one after a successful handoff.
Neither option opens a browser.

Omit `--train` for a manual-only lab. To reuse completed learning without
retraining, replace it with `--learning /path/to/sealed/spatial-policy`.
The policy must match the constructed scene and pass normal lab admission;
the two options cannot be combined. Failed or incomplete requested learning
stops the workflow instead of silently substituting a manual lab. A completed
policy can still fail every task trial: its saved results remain visible and
are not relabeled successful because the lab is playable.

Guided training uses three generations, eight candidates, training seed 1729,
evaluation seed 20261031 and twelve changed-start trials per controller, plus
one nominal trial. These are exposed development settings, not a final holdout.
The default training limit is 60 seconds; `--training-max-seconds` accepts 1–180.
No claim of superiority over authored control follows from completing training.

The launcher supervises every child process: construction is capped at 25 seconds
(20 seconds cooperative inside the constructor), training at its limit plus five
seconds, and lab preparation at 45 seconds. `--max-seconds` bounds the complete
workflow, default 180 seconds from saved input or 300 from video, maximum 360.
Cleanup time is reserved before starting each process; native memory is not
hard-capped. No stage retries, overwrites, paid calls or default promotion occur.
Each run uses a new directory and retains failed stages alongside its receipt.

For a nested saved scene, add `--scene-artifact selected/scene.json`; the exact
path must belong to the source manifest. This does not select a new best candidate.

### Explicit direct-video route

If both tracking settings and matching 3D assumptions are already prepared:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --profile spatial --video /path/to/video.mp4 \
  --tracking-profile appearance --tracking-config /path/to/tracking.json \
  --config /path/to/source-pinned-3d.json \
  --out artifacts/my-video-object-workflow-01
```

Use `--tracking-profile planar` for the existing color-tracking configuration.
This first runs ordinary `lab --no-train --no-serve`, capped at 120 seconds,
then requires the extracted scene's canonical hash to equal
`config.source_scene_sha256`. The launcher never rewrites that pin or invents
depth, materials, floor geometry or a task. A mismatch preserves
`planar-lab/run/scene.json` and stops before 3D construction.

For a first unfamiliar video, use the manual planar workflow below, inspect its
scene, and author the explicit 3D configuration before using `--planar-run`.
That route reuses the saved extraction rather than decoding or tracking again.
The direct route is not automatic arbitrary-video reconstruction.

### Individual stage commands

Choose new output names for every build:

```sh
.venv-lab/bin/python -m weft spatial-scene \
  --run artifacts/manual-video-acceptance-01/lab/run \
  --config examples/spatial-ball-hypothesis.json \
  --manifest-sha256 395df4fb85c7f200b2633cd818120162ad68a84fdf7be64786911dd33391c204 \
  --out artifacts/my-spatial-ball-01

.venv-lab/bin/python -m weft spatial-lab \
  --run artifacts/my-spatial-ball-01 \
  --out artifacts/my-spatial-ball-lab-01

.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/my-spatial-ball-lab-01
```

Use `lab --resume` instead of `lab-console --resume` for the local browser lab.
Construction and lab preparation do not start a server or train a policy.
The examples under `artifacts/` are local saved evidence, not bundled with a
fresh clone.

Try these terminal commands:

```text
controls
record start
action right
step 3
action up
step 2
reset
action brake
step 3
record stop
quit
```

The normal lab saves new sessions separately from the completed `run/`.
Recordings include actions, resets, state and runtime identity; export produces
an interactive replay and video. **Recording evidence** reports exact replay
verification. No learned controller is available until a policy is trained.

## Use another video

First build a supported planar/appearance video lab using the
[existing video workflow](local-lab.md#play-an-object-video-without-training).
Pass that sealed `run/` to `spatial-scene` with a configuration for its exact
scene. The example configurations pin their original scenes and cannot silently
apply to a different video.

The configuration must name every source object and provide:

| Choice | Meaning |
| --- | --- |
| `source_scene_sha256` | Canonical scene identity, computed with `weft.io.canonical_hash` |
| `output_id` | Explicit one-to-one spatial ID; original identity remains in provenance |
| `image_top_z` | Translation from image-down coordinates to upward height, not floor calibration |
| `lateral_center` / box `depth` | Authored unseen placement and box thickness |
| `mass`, `friction`, `gravity`, `contact` | Explicit 3D dynamics hypotheses, not recovered materials |
| `angular_velocity_y` | Required authored spin for planar-v1; planar-v2 spin is preserved |
| `target`, `task` | Authored controls, success tolerances and episode length |

Circle silhouettes become spheres; box width, height and orientation are
preserved. V2 angular motion and source colors are preserved. Objects are never
dropped, resized or moved to repair overlap. `front_plane_only` explicitly
assumes zero lateral velocity and zero X/Z spin. `open_3d_bounds_termination`
removes implicit planar walls: bounds terminate escaped motion but do not
create collision surfaces. Support must come from declared scene objects.

The saved ramp example uses the same constructor with
`examples/spatial-ramp-hypothesis.json`, source directory
`artifacts/source-ramp-reconstruction-01`, and
`--scene-artifact friction-025-mass-1/scene.json`. It preserves the previously
declared positive-offset hypothesis, not a newly selected best candidate.
That source archive has saved frames and complete slot records but no original
video; the missing video stays unavailable.

## Inspect the result

- `scene.json`: the 3D world hypothesis consumed by the existing simulator.
- `inputs/authored-config.json`: exact choices supplied for this construction.
- `construction-report.json`: inherited geometry, changed dynamics semantics,
  assumptions and unresolved evidence.
- `inputs/input-receipt.json`: exact source identities and archive checks.
- `inputs/planar/`: every declared original source member and its original seal.
- `report.json` / `manifest.json`: construction outcome and artifact identities.

A successful construction is **not yet a playable or physically validated
world**. `spatial-lab` performs normal geometry and control admission. Failure
preserves its output instead of selecting another candidate. Missing media,
depth, calibration, hidden state and physical accuracy are never fabricated.

Construction has a 20-second cooperative limit, bounded source/output sizes
and no simulation, inference, training or paid API calls. The CLI runs locally;
the caller should use process supervision for a hard wall-time limit.

For optional learning, bound training and evaluation explicitly:

```sh
.venv-lab/bin/python -m weft spatial-train \
  --run artifacts/my-spatial-ball-01 \
  --out artifacts/my-spatial-ball-policy-01 --max-seconds 90
```

Attach a completed sealed policy using `spatial-lab --learning` when creating
a new lab. The time limit is cooperative; an external supervisor supplies a hard
process limit. The [3D learning contract](spatial-rigid.md) describes independent
seeds and baselines. Training success remains simulated task success, not
evidence that the reconstructed physical world is correct.

## Verified scope

The ball and four-object ramp passed the public construction command in parallel
in **0.588 seconds** total. Their 229/170 declared source members remain archived
unchanged. The ball's original video and 2D evidence are available; the ramp's
missing original video stays unavailable.

The ball then passed ordinary lab preparation and terminal interaction in
**0.738 seconds**: 12 manual actions, one reset, 14 exported frames and 12 exact
replay actions. Including lab control probes, the run used three models,
36 control steps and 1,224 native substeps. No source inference, training,
paid request, server or browser was started. These short local timings are
observations, not a sustained UI performance benchmark.

There are **102 focused passing checks** for input preservation, packaging and
CLI routing. The real acceptance establishes this saved-source construction and
recording path, not unseen-video reconstruction, learned control or physical
accuracy. Exact evidence identities and remaining work are in the
[current handoff](handoffs/combined.md).
