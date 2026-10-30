# Human-only exercise tasks

This task tests **body-motion tracking while equipment stays stable**, using the [coupled simulator](shared-world.md) without its virtual object-pushing controls. Optional wrist-region contact requirements are explicit, not inferred from a hand near equipment. The older task and recordings remain unchanged.

This is a simulated test, not recovered anatomy, a complete exercise definition or proof of physical grasp.

## Try it

From the repository root, start the verified human-plus-equipment lab:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/shared-exercise-lab-integration-01/lab
```

Open the token-bearing local URL printed by the command. This resumes saved evidence without inference, training or a paid request. The saved task covers **two seconds**, not the complete video; it has three explicitly assisted root controls plus 19 joint controls.

1. Use **Original → Tracks → Geometry** to inspect the video, evidence and initial geometry assumptions.
2. Open **Sandbox**. Choose manual movement, coast, or authored video-following control. Equipment is passive; it has no force controls in this task.
3. Read the task card and interval result. The source reference controller currently **fails equipment stability**; the interface does not turn that failure into success.
4. Choose **Record & start**, interact, and use **Pause** or reset as needed. Reset events remain in the recording.
5. Choose **Stop & save**, then open the exported replay/video/actions. Export also re-executes the actions to verify the saved states.

To prepare another saved shared world, choose a new directory:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/source-equipment-integration-04/candidate \
  --root-assistance \
  --out artifacts/my-human-only-lab-01
.venv-lab/bin/python -m weft lab --resume artifacts/my-human-only-lab-01
```

Preparation seals the exact scene, task, source evidence and runtime. It does not start a server. Omit `--root-assistance` for 19 joint-only controls, or pass `--task <file>` to preserve an explicit task instead. Assistance and task selection cannot change inside a prepared lab. Existing coupled labs keep their legacy task and virtual object controls.

## Compare baseline controllers

For a noninteractive comparison in a new output directory:

```sh
.venv-lab/bin/python -m weft shared-exercise-evaluate \
  --run artifacts/source-equipment-integration-04/candidate \
  --root-assistance \
  --out artifacts/my-exercise-check-01
```

The command compares coast with the authored human reference controller using the same seed/reset, records both, exports videos and re-executes their actions. It does not start a server, train a policy or call an API.

`--root-assistance` permits three explicit virtual forces on the body root; omit it for joint-only control. Neither mode forces equipment. `--jitter` applies the same seeded reset perturbation to both baselines, measuring equipment movement from its actual reset state.

## Inspect the results

| Artifact | Meaning |
| --- | --- |
| `scene.json` | Unchanged physical world and source bindings |
| `exercise-task.json` | Human controls, complete interval, equipment IDs and contact requirements |
| `settings.json` | Frozen gates, seed/reset, implementation hashes, package versions and work bounds |
| `source-verification.json` | Complete copied evidence checks; not physical calibration |
| `evaluation.json` | Separate baseline results, whole-interval metrics, timings and replay checks |
| `recordings/<baseline>/replay.html` | Offline three-view replay with this task's results |
| `recordings/<baseline>/video.mp4` | Every recorded control-boundary frame, including the initial state |
| `recordings/<baseline>/events.jsonl` | Human actions, task metrics, resets and engine snapshots |
| `implementation/`, `manifest.json` | Pinned implementation inputs and output hashes |

Verify integrity and executed actions with the matching evaluator version:

```sh
.venv-lab/bin/python -m weft replay \
  artifacts/my-exercise-check-01/recordings/authored_human_reference \
  --verify-actions
```

A different evaluator is reported explicitly; use the pinned implementation rather than rewriting saved recordings. Incomplete-journal recovery writes a new directory and preserves the interrupted original.

## What counts as success

`shared-exercise-task-v1` pins `shared-exercise-gates-v1` and the exact scene hash. These are authored simulator criteria, not biomechanical thresholds. Reward guides learning; it does not decide success.

All criteria apply across the complete declared interval:

- Body angular/root RMSE ≤0.35 rad / 0.15 model units; per-substep peaks ≤0.70 rad / 0.30 model units.
- Body/floor penetration ≤0.025 model units.
- Every dynamic object: displacement ≤0.03 model units, rotation ≤0.15 rad, speed ≤0.15 model units/s and angular speed ≤0.75 rad/s.
- Each required wrist-region interval: qualifying contact ≥95% of its duration, with no gap longer than 0.10 s.
- Complete finite substep accounting, no prohibited object actuation and no engine failure.

Contact must involve the designated distal forearm near its wrist endpoint and the specified object. Other body impacts do not count. Upward support also requires force opposing declared gravity. The model has no hand/finger grasp mechanism; grip requirements reject as unsupported.

With **no contact requirements**, success means tracking and equipment stability only—not hand support or a completed push-up. The source body starts at frame 2 while equipment observations begin at frame 8; initial equipment placement before observation remains an authored assumption. Nothing is silently trimmed or retimed.

Use `build_shared_exercise_task(..., contact_requirements=[...])` to prepare a task, inspect/save it, then pass `--task <file>`. Each row names `object_id`, `wrist`, `kind` (`wrist_region_contact` or `upward_support`) and half-open `start_step`/`end_step`. Freeze the task before controller optimization. Its JSON selects behavior, not new thresholds. `--task` and `--root-assistance` cannot be combined because the saved task owns that choice.

## Gym and failure handling

`SharedExerciseTaskEnv(scene, task)` exposes seeded reset/step, action/observation contracts, authored actions, metrics, snapshots, rendering and close. Actions contain 19 joints or 22 channels with explicit root assistance. Observations disclose privileged reference/state information; this is not a vision-only learned controller.

Use `shared_exercise_recording.recording_scene(scene, task)` with the existing `Recorder`. Resets retain their resolved settings. Finite partial failures retain actual physics time. When valid current geometry is unavailable, replay/rendering explicitly labels the last valid control-boundary state beside the later failure time; it does not fabricate a new pose.

Telemetry separates contact/site solver state from post-integration state, following the [MuJoCo simulation pipeline](https://mujoco.readthedocs.io/en/stable/programming/simulation.html). It adds no forward solve to live physics. Validators prepare immutable scene/task context once while checking every dynamic snapshot.

## Current limits

The latest source evaluation is `artifacts/shared-exercise-evaluation-02`: 60 steps/960 substeps and 61 video frames per baseline, with exact action replay. Both fail: coast loses body tracking; reference control destabilizes the working proxy. Its actions, engine states and task results match preserved `-01` exactly, apart from the evaluator implementation hash. Preparing invariant validation once reduced measured control/record p95 from 56.8–57.8 ms to 3.0–3.5 ms; total evaluation/export/replay fell from 35.879 s to 4.717 s. Both runs preserve all state checks and evidence. OS caches were not evicted; these are local stage measurements, not browser or sustained-use guarantees.

This is a two-second exposed development case, not the whole exercise or independent physical evidence. Later evaluator versions require their own pinned replay evidence. The current saved replay is `artifacts/shared-exercise-evaluation-02/recordings/authored_human_reference/replay.html`.

Equipment remains solid-box geometry with uncertain depth/materials. One arm-image refinement improved projection but worsened penetration and was rejected; see the [handoff](handoffs/combined.md). This command evaluates baselines, not a new trained policy or physical transfer.

The live integration study, `artifacts/shared-exercise-lab-integration-01`, verifies preparation, manual/coast/reference actions, in-recording reset, export, resume and exact replay: **69 actions, 71 events/video frames**. Its post-reset 60-step reference trial exactly matches the failed baseline above. Mocked-interface checks use these actual saved backend payloads, not invented passing scores. Local six-step control/record/JSON median was **19.26 ms**, p95 **24.18 ms**; first status load/serialization **476.71 ms**, 20 warm status serializations median **5.77 ms**. These are local core/API measurements, not browser/device or sustained-use guarantees.

The live-lab preparation limit is 60 seconds by default, with complete input/output bounds; see the [integration contract](shared-exercise-lab-plan.md). Live recording export uses a 180-second cooperative limit and 128 MiB output guard. Inputs, completed `run/` files and old recordings stay immutable; new interaction recordings live under `sessions/`.

The default 60-second limit is a **cooperative deadline** checked between stages, actions, export frames and replay events; it cannot preempt one native encoder call. Each recording export also has a 64 MiB byte guard. Failures preserve partial files and journals; retries require new directories. Source fit, simulated checks, learning and real physical validation remain separate. The full A+B goal stays open.
