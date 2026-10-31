# Turn a saved object scene into a 3D lab

`spatial-scene` connects existing video reconstruction to the ordinary 3D lab.
It preserves the sealed planar run and builds one explicit 3D hypothesis.
It does **not** infer hidden depth, calibrate a camera or validate physical accuracy.

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
