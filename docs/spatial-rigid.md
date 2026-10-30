# Inspectable 3D rigid-object sandbox

Weft now has an explicit **experimental 3D rigid-object environment** with boxes, balls, contact forces, eight controls and complete action recordings. It runs locally using the existing MuJoCo dependency. It does not automatically recover an accurate 3D world from arbitrary video.

**[Open the inspection page](../artifacts/spatial-ramp-views-01/index.html)** for three ramp hypotheses, a recorded control session and the additional source-camera view. Front, top and angled views show the same recorded state. Zoom, rotate the view or follow the moving objects without changing the simulation. Static supports are wireframes so they cannot hide the objects.

The existing exercise-body labs are unchanged. Muscles remain illustrative; this rigid-object work does not establish human anatomy or contact accuracy.

The later [contact-material study](material-transfer.md) improves prediction on a reserved repeat without retuning it: later center error 61.66→8.51 pixels. It still fails the frozen accuracy check and exposes oversized projected shapes. [Open the newer comparison](../artifacts/spatial-ramp-repeat-01/index.html).

## Interact and record

Run from the repository root with a new recording directory:

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/spatial-ramp-sandbox-02/positive-offset \
  --record artifacts/my-3d-ramp-session
```

Try `right 6`, `forward 6`, `back 6`, `brake 6`, `reset` and `quit`. Nothing advances while you type. `left`/`right` apply horizontal force, `forward`/`back` apply depth force and `up`/`down` apply vertical force. These are bounded virtual forces, not teleportation or automatic flight. `coast` releases control; `brake` opposes velocity.

Your recording folder's `current-frame.png` shows the current state. Quitting exports a video and an offline step-by-step replay, including resets. The existing [control/reset example](../artifacts/spatial-ramp-views-01/terminal-session/replay.html) contains 26 states. The positive-offset example was fixed before comparison, not selected as the best result. Its parking target and controls are authored; no policy was trained here.

### Open in the unified local lab

You can also inspect, control, and record any verified `spatial-rigid-v1` run inside the browser local lab while keeping the source run immutable:

```sh
# Package without starting a server (or generate a standalone analytic fixture first with `weft spatial-fixture`).
.venv-lab/bin/python -m weft spatial-lab \
  --run artifacts/spatial-ramp-sandbox-02/positive-offset \
  --out artifacts/my-3d-ramp-lab-01

# Open the packaged lab on 127.0.0.1 (or use `weft lab --scene-run <run> --out <new-lab>` directly).
.venv-lab/bin/python -m weft lab --resume artifacts/my-3d-ramp-lab-01
```

The lab copies only sealed inputs into `run/inputs/source-run/`, preserves `physical_validation: "unverified"`, exposes synchronized front/top/3D views, full 3D position/quaternion/spin/contact inspection, multi-object dynamic actor switching (`Object to control`), all eight controls, and exact `sessions/` recording/export/replay without altering `run/`. Untrained 3D imports keep learned-policy controls explicitly disabled.

### Bounded 3D RL training and changed-initial-condition evaluation

Train a bounded 8-action linear-feedback and braking policy from an uninformed zero-weight coast prior on a verified `spatial-rigid-v1` run, evaluate it against matched `learned`, `random`, `coast`, and `brake` controllers across exact and collision-free perturbed starts, and package it into a new local lab. New training emits **policy schema 2 / `spatial-policy-v2` / `spatial_cem_v2`**. Saved **schema 1 / `spatial-policy-v1` / `spatial_cem_v1`** policies retain their exact original controller interpretation. Mixed version triples reject; scene identity, actor/order and controller tolerances must match. Settings/evaluation container schemas remain version 1.

```sh
.venv-lab/bin/python -m weft spatial-train \
  --run artifacts/track-b-spatial-lab-01/authored-fixture \
  --out artifacts/my-3d-policy-01

.venv-lab/bin/python -m weft spatial-lab \
  --run artifacts/track-b-spatial-lab-01/authored-fixture \
  --learning artifacts/my-3d-policy-01 \
  --out artifacts/my-3d-trained-lab-01
```

Reset remains exact (`jitter=False`) by default. Opt-in `options={"jitter": True, "position_jitter": ..., "velocity_jitter": ...}` samples bounded initial actor perturbations verified collision-free via `mj_geomDistance >= 1e-4`. In the saved v1 study `artifacts/track-b-learning-changed-init-02`, the learned policy improves simulated return and reduces mean final target distance over the baselines on the authored fixture (`0.588 → 0.142 scene_width` under changed starts; return `+0.2305` vs `-0.3584` coast) and source-linked ramp (`0.319 → 0.071 scene_width` nominal; `1/12` changed-start passes). The authored fixture still has `0/12` changed-start successes. These are not measured metres; legacy `*_m` metric keys retain their historical names. Short-horizon discrete CEM does not reliably satisfy terminal position/speed tolerances. No old study was retrained or relabeled v2; 617 saved v1 actions were checked unchanged. Simulation improvement does not verify physical accuracy.

## What the environment contains

`spatial_rigid.py` implements the separate `spatial-rigid-v1` contract; planar and human profiles retain their own contracts.

- Up to twelve box/sphere objects, with one controlled actor and explicit static/passive roles. Position, orientation, linear/angular velocity, relative mass and three friction coefficients are retained.
- Full-state RL observations: thirteen values per object, three target offsets and episode progress. Actions, rewards, termination flags, contact points and simulated normal forces are inspectable.
- Normalized scene-width distances, relative mass and seconds. Neither centimeters nor actual forces are recovered from the source.
- Validated primitives only: no supplied XML, arbitrary code or external simulator assets. Initial intersections, invalid geometry, excessive speed/spin and excessive integration demand fail explicitly.
- At most 900 steps per episode, 256 integration substeps per action and a 16 MiB recording journal. Reset restores the exact initial state; random geometry perturbation is not supported.

Small computed-state roundoff is accepted within `1e-8`; raw values are retained, not clamped. The observation space covers the declared 200-radian/second spin limit. This avoids rejecting a valid small spinning ball merely because its angular velocity exceeds the old observation bound of 100.

## Additional source evidence

Four original public [Physics101](https://phys101.csail.mit.edu/) clips were selected before acquisition: first-trial wooden-block freefall, a different ramp target, the existing ramp trial's side camera, and a reserved second trial. Only 1,758,074 archive bytes were transferred through twelve bounded range requests; the full 4.81 GB archive was not downloaded. [Provenance and hashes](../data/general/phys101-constraints.source.json) retain the provider citation and local/private-use boundary.

The three construction clips retain all 48, 50 and 37 decoded frames. In this initial study the reserved second trial had container metadata only: **zero frame reads and no pixel inspection**. The later [material-transfer study](material-transfer.md) freezes its method, settings and comparison protocol before exposing that repeat once; it is no longer an untouched holdout. The first preview report failed because `ffprobe` was absent; its images and failure are preserved. Recovery reused every preview and verified its pixels while recording previously unsaved timestamps.

The side camera is 25 fps; the original front camera is 30000/1001 fps. Pairing uses nearest timestamps, not frame numbers, with a maximum 19.9 ms difference and no fitted offset. Provider-declared time alignment remains an assumption.

The green component moves about 94.7 source pixels and its projected edge changes about 23° in the side view. These are color/silhouette image estimates, **not calibrated 3D displacement or yaw**. The freefall camera framing also differs: its pixel acceleration cannot be transferred to the ramp without a verified camera/scale mapping.

## Source-linked experiment and remaining mismatch

The [earlier 2D middle hypothesis](source-ramp-reconstruction.md) supplies front-plane geometry, opening velocity, normalized gravity and relative masses. The new three scenes add assumed object depths and lateral offsets of −.015, 0 and +.015 scene widths; everything else is fixed. Each scene is sealed before grading.

Sliding friction remains .25 at the contact pair. MuJoCo uses the componentwise maximum of equal-priority geom friction coefficients, unlike the earlier Pymunk product rule; the conversion preserves the intended pair coefficient. Soft-contact damping, torsional and rolling friction remain authored assumptions. [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).

All three cases run, but none improves the front-view collision match:

| Fixed hypothesis | Later wood center error, pixels RMSE | Later green center error, pixels RMSE |
| --- | ---: | ---: |
| Earlier 2D middle | 43.42 | 75.39 |
| 3D centered | 45.39 | 76.61 |
| 3D either lateral offset | 45.22 | 76.71 |

The comparison retains all 264 object/frame slots across three scenes, including 48 unavailable pre-initialization slots. Original IDs and unresolved post-proximity identity remain unchanged. Front-image error cannot validate depth. The original trial was already exposed during development; it is not an untouched holdout. **No hypothesis is promoted.**

Next work must constrain table friction, collision shape, relative mass and camera geometry from source evidence. The already-exposed original trial can serve as construction evidence under an explicitly revised protocol; it cannot also become fresh evaluation evidence. Freeze that protocol before opening the reserved repeat. Repeated unconstrained parameter sweeps do not resolve these ambiguities.

## Verification and saved results

As of **October 29, 2026, 9:24 AM PDT**:

- **113 distinct Python checks pass** across analytic freefall, sliding/sticking ramps, off-center momentum exchange, rolling balls, full-state RL/reset, recording/export, legacy paths and view visibility. Saved XML files: `.local/spatial-rigid-integration-tests-01.xml`, `.local/spatial-quaternion-roundoff-test-01.xml`, `.local/spatial-inspection-render-test-01.xml` and `.local/spatial-spin-observation-tests-02.xml`. The last file includes two already-counted affected checks. Earlier failures are preserved, not hidden.
- Scripted Node checks cover frame-request coalescing, worker cleanup, view-only zoom/follow controls, deferred state details, wireframe visibility and hidden-page playback. These are mocked runtime checks, **not browser/device QA**: `.local/spatial-replay-smoke-02.log`.
- Three coast recordings retain 71 states each; the control/reset example retains 26. The corrected views reuse all **239 identical states**, without repeated simulation, inference, training or paid calls. Re-export takes 2.023 s on this computer. [View-export report](../artifacts/spatial-ramp-views-01/report.json).
- A bounded twelve-object fixture executes 600 steps, checks every complete state and restores an exact reset. Step/snapshot/JSON p95 is **.732 ms**, maximum .815 ms; maximum payload is 10,250 bytes. One 1280-pixel inspection render takes 5.61 ms. The 600-step workload takes .462 s; process lifetime peak memory is 108.1 MB including imports. This is one local synthetic workload, not worst-case solver, browser/HTTP, sustained-idle or memory-leak evidence. [Capacity report](../artifacts/spatial-rigid-capacity-01/report.json).

The first 3D study preserves two complete coast recordings and a centered recording interrupted by quaternion roundoff. Recovery reuses the completed cases, reruns only the incomplete one and records the control example. Its [203-file study](../artifacts/spatial-ramp-sandbox-02/report.json) seal is `11430e3025aa88c179bc7c5385476ee11a8a8d75557090bdab296ef95170352c`. Original previews remain sealed; the corrected [43-file view bundle](../artifacts/spatial-ramp-views-01/report.json) seal is `9de4221c31f1047e243603c90f473024fd2cd978d835aa53ce75758ce811ded3`.

No new paid API request, model inference, training, browser session or server was needed for this work. Execution and inspectability are verified; real-world collision accuracy, complete exercise control and broad automatic video-to-environment reconstruction remain open.

Final audit at **October 29, 2026, 9:31 AM PDT**: `.local/spatial-rigid-final-verification-01.json` checks source IDs/clocks, all 264 slots, identical journals, 111 runtime hashes, 32 local links and the saved distinct test results. It repeats no simulation or passing test. The final spin-envelope correction changes validation/observation bounds only; the exported physics states and renderer remain unchanged.
