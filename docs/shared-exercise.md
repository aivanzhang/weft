# Human-only exercise tasks

This task tests **body-motion tracking while equipment stays stable**, using the [coupled simulator](shared-world.md) without its virtual object-pushing controls. Optional wrist-region contact requirements are explicit, not inferred from a hand near equipment. The older task and recordings remain unchanged.

This is a simulated test, not recovered anatomy, a complete exercise definition or proof of physical grasp.

## Try it

From the repository root, start the current full-reference lab:

```sh
.venv-lab/bin/python scripts/play_sandbox.py
```

Open the token-bearing local URL printed by the command. The launcher uses this lab's verified frozen runtime while development continues on main. It resumes saved evidence without inference, training or a paid request. The task covers **254 actions / 8.47 seconds**, not only the opening two seconds. It retains 19 joint controls, three explicitly assisted root controls and passive equipment.

1. Use **Original → Tracks → Geometry** to inspect evidence. **Source vs simulation** separates all 257 source slots, the 121 retained poses, the full task window and actual live time. Video playback is independent; its native clock is not silently equated with the pose clock.
2. Open **Sandbox**. Choose manual movement, coast, authored following or **Searched joint corrections**. Root assistance remains programmed; equipment has no force controls.
3. Read the interval result. The task-priority v3 controller and earlier reward-selected policy both complete all 254 actions but fail equipment stability on **3/3 fresh starts**. Worst displacement falls **44.3%**, but remains above the unchanged limit. The earlier two-second policy's 3/3 success does not transfer.
4. Choose **Record & start**, interact, pause or reset, then **Stop & save**. The exports preserve resets, actions and controller identity and verify exact state replay.

The idle object retains the conditional eight-band shape; the working object remains an occluded box. The [offline shape comparison](../artifacts/silhouette-shared-world-03/shape-comparison.html) is separate from full-sequence control. Missing panels in this lab are not replaced with stale fit claims. Its full-reference receipt records the new task identity and unchanged original evidence.

The current [saved replay](../artifacts/feasibility-fullspan-lab-01/sessions/session-04b7c0ac12c4/replay.html) verifies **262 actions /one reset /264 events**, including all 254 post-reset policy steps and 256 learned decisions, with exact state/task replay. The live trial still fails working-object stability. **Explain this trial** reports its final episode's first speed/spin crossing in **(1.233333, 1.266667] s**, source-prior frame 40. These are recorded simulation/reference clocks, not native-video timing or causal contact proof.

To prepare another saved shared world, choose a new directory:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/silhouette-shared-world-03/geometry \
  --out artifacts/my-human-only-lab-01
.venv-lab/bin/python -m weft lab --resume artifacts/my-human-only-lab-01
```

Preparation seals the exact scene, task, source evidence and runtime. It does not start a server. Without `--task`, it preserves a sealed input `exercise-task.json` when present, including its assistance and contact requirements. Only when no input task exists does it create a default: 19 joint controls, or 22 with `--root-assistance`. The flag cannot override a sealed joint-only task; that conflict rejects. `--task <file>` is an explicit validated replacement and cannot be combined with `--root-assistance`. Assistance and task selection cannot change inside a prepared lab. Existing coupled labs keep their legacy task and virtual object controls.

For new saved body/tracking evidence, [assemble the world and lab in one command](equipment-proposals.md): use `shared-exercise-build --human-run <mobile-run-or-lab> --body-scale 1.0 --full-reference` with its equipment evidence/configuration. The direct body-preparation stage is bounded and construction-only; it neither runs a throwaway simulation nor creates an intermediate lab. Existing shared-world bases still use `--run`. Camera, scale and shape assumptions remain explicit; this is not automatic recovery of arbitrary equipment.

The same command now accepts `--video <file> --pose-model <local-model>` instead of a prepared body. Add `--train --evaluation-seed <separate-seed>` to connect the existing bounded search, training-only selection, baseline comparison and one trained lab. [Full command, stage artifacts and limits](equipment-proposals.md#video-to-one-trained-lab). No training or inference is silently enabled for saved inputs.

### Prepare a lab from a geometry alternative

The [authored geometry command](shared-world.md#prepare-an-authored-geometry-alternative) produces a new scene without changing body/reference poses, source evidence, object poses/mass or task parameters. If the input includes an exercise task, it carries that task forward with only the scene hash rebound. Geometry/inertia are authored hypotheses, not recovered equipment or a solved exercise.

After its report and manifest say **ready**, prepare a separate lab:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-geometry-01 \
  --out artifacts/my-geometry-lab-01
.venv-lab/bin/python -m weft lab --resume artifacts/my-geometry-lab-01
```

No `--task` or `--root-assistance` is needed to retain the sealed input task. An explicit task override must match the new scene; it is a deliberate change, not preservation. Old policies/fit reports remain archived and cannot establish success for the alternative. The new scene requires its own evaluation; initial-clearance admission alone checks neither the full movement nor equipment stability.

The earlier [representation-only example](shared-world.md#verified-representation-only-source-example), `artifacts/shared-world-geometry-integration-01/interaction/lab`, retains the original two box collision shapes/inertias as one-part compounds. Its 69 actions/one reset/71 frames replay exactly; the 60-step interval completes and **fails equipment stability**. No policy is attached. The earlier two-second **`silhouette-exercise-lab-01`** instead uses the separately admitted idle-object shape and a newly trained policy. Neither geometry alternative establishes recovered handles or grasp.

## Full-reference episodes

`shared-exercise-fullspan` explicitly promotes a sealed v1 scene and its sealed exercise task into `shared-world-v2`. It preserves the complete input archive, body poses, source clocks, geometry, physics, timestep, assistance and success gates. Only the scene profile, complete interval and corresponding task bindings change. It runs no physics or training and attaches no old policy.

```sh
.venv-lab/bin/python -m weft shared-exercise-fullspan \
  --run artifacts/silhouette-shared-world-03/geometry \
  --out artifacts/my-fullspan-source-01
.venv-lab/bin/python -m weft shared-exercise-train \
  --run artifacts/my-fullspan-source-01 --policy-family phase-residual \
  --generations 2 --population 8 --out artifacts/my-fullspan-learning-01
.venv-lab/bin/python -m weft shared-exercise-select \
  --learning artifacts/my-fullspan-learning-01 --out artifacts/my-fullspan-selected-01
.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning artifacts/my-fullspan-selected-01 --max-seconds 60 \
  --out artifacts/my-fullspan-evaluation-01
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-fullspan-source-01 --learning artifacts/my-fullspan-selected-01 \
  --evaluation artifacts/my-fullspan-evaluation-01 --out artifacts/my-fullspan-lab-01
```

V2 requires one exact 30-Hz reference interval, at most 1,800 actions / 60 seconds. Non-grid-aligned endpoints reject; no clipping, retiming or chunk resets. V1 retains its 180-action maximum. Promotion rejects existing contact requirements instead of inventing how their windows should extend. Missing observations remain missing, even when the simulator interpolates reference targets.

For this 254-action source, the explicit 2×8 search allows at most **8,128 training actions** within 60 seconds. Three fresh starts × four controllers allow **3,048 evaluation actions** within 60 seconds. The generic full-reference evaluation ceiling is 21,600 actions; old-window evaluation remains capped at 720 / 30 seconds. Excess configurations reject instead of shortening the interval. `shared-world-train` remains the separate legacy task; use the human-only commands above.

`artifacts/fullspan-exercise-control-01` completed one frozen comparison in **27.851 seconds**: 28 training episodes / 7,112 actions, then 2,354 evaluation actions. None of 14 saved candidates met every training gate, so selection retained the original reward winner. Learned, authored and coast each completed all three intervals but passed **0/3**; random control failed early. Learned/authored failures were specifically working-object stability. The learned reward improved, but that is not task success. These exposed trials are development evidence, not a future final holdout.

The earlier v2 lab acceptance, `artifacts/fullspan-exercise-interaction-01`, records **262 actions / one reset / 264 frames**, including every one of the 254 post-reset policy actions. All 256 recorded learned decisions and the complete action/state/task replay match exactly; resume worked under its pinned runtime. The live trial fails the same equipment-stability gate. Historical replay: `artifacts/fullspan-exercise-lab-01/sessions/session-73987c5a73d0/replay.html`. Saved UI checks verify 22 controls, the full interval and missing-pose counts. Local six-step control/record/JSON batches measured median **22.539 ms**, maximum **23.451 ms**; these are not browser or sustained-use measurements.

Older runtime-pinned labs/policies remain unchanged and require their archived implementation; no hash is silently replaced. A separate numerical check reproduced the old two-second recording's 68 actions, 70 snapshots, rewards and outcomes exactly under the new core. That check does not authorize old policy loading or imply full-interval success.

## Prioritize task limits during refinement

`shared-exercise-refine` creates a **v3 policy** from a verified original two-generation, eight-candidate phase-residual training package. It imports that package's first generation exactly, without replaying its physics, then runs at most **12 new episodes /3,048 actions** under a **60-second /512-file /128-MiB** bound. It does not accept evaluation results or a v2 selection wrapper as training input.

```sh
.venv-lab/bin/python -m weft shared-exercise-refine \
  --learning artifacts/my-fullspan-learning-01 \
  --out artifacts/my-task-priority-learning-01
.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning artifacts/my-task-priority-learning-01 --max-seconds 60 \
  --out artifacts/my-task-priority-evaluation-01
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-task-priority-learning-01 \
  --learning artifacts/my-task-priority-learning-01 \
  --evaluation artifacts/my-task-priority-evaluation-01 \
  --out artifacts/my-task-priority-lab-01
```

Freeze a fresh `--seed` for evaluation after checking earlier trial history; a default seed is not a promise of fresh evidence. V3 evaluation adds the original reward winner's verified coefficients as a **fifth matched comparison**, alongside coast/random/authored/selected control. This reuses the unchanged actor math, not permission to load an old runtime-pinned policy. Three 254-step starts across five controllers allow at most **3,810 evaluation actions**. Every controller receives the same fresh human-start perturbations; equipment starts, scene, task and gates stay fixed.

Selection first prefers candidates that complete every training interval and pass every limit. Otherwise it minimizes the largest normalized limit violation, then the mean violation, then uses original reward as the tie-breaker. With no passing candidate it retains the least-violating **complete** candidate, explicitly marked infeasible; without a complete candidate it publishes no policy. Reward and its recorded loss are unchanged. The two phase functions, 2×19 coefficients bounded to ±0.25, joint-only residuals, root feedback and passive equipment all remain unchanged. V3 changes the training objective and provenance, not the actor's controls or task definition.

**Verified development result:** `artifacts/feasibility-fullspan-control-01` imports **16 episodes /4,064 actions** and runs **12 new episodes /3,048 actions** in **8.016 s**. It selects imported generation 0/candidate 3; **no new candidate beats it**, and training remains infeasible. The separate five-controller evaluation uses **3,352 actions /12.738 s** on three fresh starts. Selected and previous reward policies both complete **3/3** intervals and pass **0/3**, failing working-object stability; all other controllers also pass 0/3. Worst equipment displacement falls **0.448419→0.249624 model units (44.3%)**, still above the **0.03** limit. Reward decreases **0.719233→0.669190**: reduced gate violation is not increased reward or task success. `.local/feasibility-fullspan-verification-01.json` verifies unchanged scene/task/gates and exact paired evidence. These starts are now exposed development evidence, not a future final holdout.

The initial evaluation admission stopped before creating output or running physics because a valid archived mapping-log path was unsupported. The bounded path-reader correction retained the **same frozen policy and seed stream**; training was not repeated. The rejected admission receipt remains with the study. No new video decode, inference or paid request occurred. The current ready lab/recording at the top uses this versioned policy; old v2 labs remain archived rather than silently repinned.

## Compare baseline controllers

For a noninteractive comparison on the original box-proxy scene, use a new output directory:

```sh
.venv-lab/bin/python -m weft shared-exercise-evaluate \
  --run artifacts/source-equipment-integration-04/candidate \
  --root-assistance \
  --out artifacts/my-exercise-check-01
```

The command compares coast with the authored human reference controller using the same seed/reset, records both, exports videos and re-executes their actions. It does not start a server, train a policy or call an API.

`--root-assistance` permits three explicit virtual forces on the body root; omit it for joint-only control. Neither mode forces equipment. This baseline command's legacy `--jitter` applies the same seeded body/equipment perturbation to both baselines, measuring equipment movement from its actual reset state. It is not the equipment-preserving training/evaluation profile below.

## Train and compare a frozen policy

Both policy families use privileged simulator state and the known reference, not video-only observations. The scene, equipment, declared task interval, reward and success gates remain fixed.

| Training option | Search | Default work |
| --- | --- | --- |
| Omit `--policy-family`, or use `feedback` | v1: grouped feedback gains for all permitted human controls | 3 generations × 8 candidates |
| `--policy-family phase-residual` | v2: 2 fixed phase functions × 19 joint coefficients, each in ±0.25 normalized action units | 6 generations × 12 candidates |

The v2 correction changes only joint actions. Permitted root controls still use authored feedback from the current state; equipment is never forced. Phase is the existing interval fraction, not a retimed reference. Zero coefficients reproduce the authored baseline exactly. Opt in with:

```sh
.venv-lab/bin/python -m weft shared-exercise-train \
  --run artifacts/silhouette-shared-world-03/geometry \
  --policy-family phase-residual \
  --out artifacts/my-exercise-learning-01

.venv-lab/bin/python -m weft shared-exercise-select \
  --learning artifacts/my-exercise-learning-01 \
  --out artifacts/my-exercise-selected-01

.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning artifacts/my-exercise-selected-01 \
  --out artifacts/my-exercise-policy-check-01
```

This input already seals its exercise task. Omitting `--task` preserves its parameters and the new scene binding; do not supply the earlier box-world task or policy.

Both families use two matched training starts, one CPU lane and a 60-second cooperative limit. Identical candidates reuse their existing trials. The v2 action cap is **8,640**, including anchors within the population; cached trials issue no new actions and there are no hidden baseline runs or retries. Its separate evaluation cap is **720 actions** for v1 windows; explicit full-reference scenes use the bounds above. A requested configuration exceeding a cap rejects instead of shortening the task.

The optional **v2-only `shared-exercise-select`** stage chooses the highest original mean reward among saved candidates that passed **every frozen gate on both original training starts**. If none qualified, it explicitly retains the original selection. It performs no retraining, physics actions or evaluation reads, and has a 15-second cooperative limit. The new package preserves every candidate and the exact original policy, training history, settings, manifest, source proofs and actor runtime. `selection.json` records the rule, candidate identity and separate selector snapshot. Training and evaluation APIs/defaults are unchanged; omit this stage to evaluate the original reward-selected policy.

`shared-exercise-policy-evaluate` automatically detects v1/v2/v3. It freezes policy bytes and evaluation seeds before comparing coast, random, authored-reference and selected-policy control on three separate perturbed starts; v3 adds the paired previous-reward comparison described above. The default is 30 seconds, up to 60 seconds for full-reference scenes. Evaluation does not feed results back into training. Commands preserve incomplete runs and failures; a new run always needs a new directory. Without `--task`, training preserves the sealed input task when available. New runs use `shared-exercise-human-jitter-v1`: only human starts vary; equipment keeps exact scene pose/velocity. This does not validate changed equipment placement or guarantee that an arbitrary initial scene has correct support. Existing reset profiles and saved trials remain unchanged.

Inspect `policy.json` for selected gains or residual coefficients and `training.json` for the search history/score, then the **separate** `evaluation.json` for per-seed success/failures. For previously evaluated input, freeze a fresh evaluation seed and check that its reset seeds do not repeat earlier trials. Evaluation is not input to feasibility selection. V1 may select unchanged coast or authored control; v2 may select unchanged authored control or `searched_joint_residuals`. Unchanged anchors are not called learned improvement. A higher reward does not override an equipment or body failure. Three mild changed starts are development checks, not broad generalization or physical validation.

To try and record that exact policy, prepare a **new** lab:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-exercise-selected-01 \
  --learning artifacts/my-exercise-selected-01 \
  --evaluation artifacts/my-exercise-policy-check-01 \
  --out artifacts/my-exercise-policy-lab-01

.venv-lab/bin/python -m weft lab --resume artifacts/my-exercise-policy-lab-01
```

The policy and attached comparison must match the exact scene, task and runtime. `--evaluation` is optional, but omitting it means **evaluation unavailable**, not passed. A failed comparison can be attached when its trials completed; the interface keeps those failures visible. Existing labs are not modified and do not gain a policy automatically.

## Inspect the results

| Artifact | Meaning |
| --- | --- |
| `scene.json` | Unchanged physical world and source bindings |
| `exercise-task.json` | Human controls, complete interval, equipment IDs and contact requirements |
| `settings.json` | Frozen gates, seed/reset, implementation hashes, package versions and work bounds |
| `source-verification.json` | Complete copied evidence checks; not physical calibration |
| `policy.json`, `training.json` | Selected controller and unchanged candidate/reward history |
| `selection.json` | Optional v2 training-feasibility rule, chosen candidate and exact input/output identities |
| `selection-source/`, `selection-implementation/` | Exact original selection documents and separately pinned selector; actor runtime is unchanged |
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

### Current combined geometry and control result

`artifacts/silhouette-shared-world-03` improves the idle object's **conditional mask overlap (IoU) from 0.593595 to 0.666873** with eight shape bands. It loses no previously covered foreground; **55 already-uncovered mask cells remain uncovered**. The occluded working object is unchanged. Full-reference kinematic checks find no increased overlap, but the existing worst overlap remains **0.104829 world units**. These are conditional construction-image and geometry checks, not recovered depth, grasp or a solved full video; inspect the [offline comparison](../artifacts/silhouette-shared-world-03/shape-comparison.html).

`artifacts/silhouette-exercise-control-01` trains on that admitted scene using the same predeclared optimizer and training seeds as the box-world comparison. The workflow takes **22.83212 s**:

| Stage | Verified work/result | Time |
| --- | --- | --- |
| Fresh training | 124 episodes / 7,440 actions | 18.56242 s |
| Training-only selection | Saved generation 2, candidate 6; 193 original files recoverable byte-for-byte; no new physics or evaluation reads | 0.57124 s |
| Independent evaluation | 628 actions; selected policy 3/3, authored/coast/random 0/3 each | 2.86201 s |

Fresh evaluation seeds are **132787259, 656811237 and 1783657337**. The selected coefficients match the earlier candidate, but **scene/task/policy identities are new**; policy hash begins `2b114ccd160ef858`. Reward, gates and task parameters stay fixed. The box-world policy already passed, so this does **not** show that the shape change caused control success. This is one two-second root-assisted simulated task, not new-video or physical generalization.

`artifacts/silhouette-exercise-interaction-01` verifies the latest lab with **68 actions, one reset, 70 frames, all 60 post-reset steps**, task success, exact own replay and resume in **5.7570 s**. Open `artifacts/silhouette-exercise-lab-01/sessions/session-caf49de203d4/replay.html` or its `video.mp4`. Ten six-step API/JSON batches measured median **21.8363 ms**, maximum **22.3967 ms**. Saved-payload UI acceptance rechecked **6,007,393 bytes** and 22 controls; its **zero old source-fit panels** is intentional, with configuration/initial checks available and image comparison separate. These are local API/mock-UI checks, not browser/device or sustained-use measurements.

### Preserved passing box-world policy

`artifacts/shared-exercise-feasible-01` selected generation 2, candidate 6 (zero-based), the best of three feasible candidates among 62 unique saved candidates. Its reward **0.745602768** is lower than the reward-only winner (**0.762395196**) and authored anchor (**0.755012031**). Selection took **0.585506 s** with no new physics, training or evaluation reads; all **190 original training-package files** remain recoverable byte-for-byte. Policy `577392dbf25acd13…` retained the original box-world/task/runtime. Separate fresh seeds **1715624570, 2147129040 and 1762882840** gave **3/3 policy passes, 0/3 for each other controller**, in **661 actions / 2.68616 s**; the workflow took **3.80953 s**. Those results were not used for selection.

The previous **`artifacts/shared-exercise-feasible-lab-01`** retains this passing box-world policy and comparison. `artifacts/shared-exercise-feasible-interaction-01` verifies **68 actions, one reset, 70 frames**, exact own replay and resume in **5.57594 s**. All **60 post-reset policy steps** run and the live task passes with no failed gates. No old-baseline equality was requested. The recording remains `artifacts/shared-exercise-feasible-lab-01/sessions/session-7fafb12a9511`; open its `replay.html` or `video.mp4`.

### Preserved reward-only residual result

`artifacts/shared-exercise-residual-01` completes in **20.5015 s** on the unchanged original scene and frozen two-second task. Training evaluates **124 episodes / 7,440 actions** in **17.3003 s**, with 10 cached candidates. It selects `searched_joint_residuals`, scoring **0.762395196**, **+0.007383165** over the matched authored training anchor. The 38 coefficients, exact source attribution/video, task, runtime and predeclared seed plan are preserved in the package.

The separate fresh-start comparison takes **631 actions / 2.74995 s**: coast, random, authored and selected policy all remain **0/3 task passes**. Against matched authored trials, selected-policy `sam-2` maximum displacement falls **71.6846–79.8258%**. Its body/floor gates pass, but equipment speed remains **0.2493–0.3180 model units/s > 0.15** and spin **1.7854–1.9911 rad/s > 0.75**. These outcomes were not used to retune the policy. This is partial simulator improvement, not a solved task, new-video test, grasp or physical success.

The historical **`artifacts/shared-exercise-residual-lab-01`** retains this reward-selected policy and attached failed comparison. `artifacts/shared-exercise-residual-interaction-01` verifies **68 actions, one reset, 70 frames**, exact own replay and resume in **5.5577 s**. All **60 post-reset steps** run; that live result fails `equipment_stability:sam-2`. No old-baseline equality was requested. The recording remains `artifacts/shared-exercise-residual-lab-01/sessions/session-5440f09a1a94`; open its `replay.html` or `video.mp4` to inspect the unchanged trial.

Ten recorded six-step API/JSON batches measured median **19.9912 ms**, maximum **20.9795 ms**. Actual saved-payload UI acceptance processed **5,292,153 bytes**, exposed 22 controls and three source-fit panels, and retained the distinct training, 0/3 evaluation and live-failure displays. These are local API and mocked-interface checks, not browser/device or sustained-use measurements.

### Preserved earlier results

The earlier nominal source evaluation, `artifacts/shared-exercise-evaluation-02`, has 60 steps/960 substeps and 61 video frames per baseline, with exact action replay. Both fail: coast loses body tracking; reference control destabilizes the working proxy. Its actions, engine states and task results match preserved `-01` exactly, apart from the evaluator implementation hash. Preparing invariant validation once reduced measured control/record p95 from 56.8–57.8 ms to 3.0–3.5 ms; total evaluation/export/replay fell from 35.879 s to 4.717 s. Both runs preserve all state checks and evidence. OS caches were not evicted; these are local stage measurements, not browser or sustained-use guarantees.

This is a two-second exposed development case, not the whole exercise or independent physical evidence. Later evaluator versions require their own pinned replay evidence. Its preserved baseline replay is `artifacts/shared-exercise-evaluation-02/recordings/authored_human_reference/replay.html`.

These older runs retain solid-box equipment with uncertain depth/materials. The latest lab changes only the idle object to the conditional compound described above; the working object and old runs remain unchanged. One arm-image refinement improved projection but worsened penetration and was rejected; see the [handoff](handoffs/combined.md). Baseline evaluation is not policy training or physical transfer.

The live integration study, `artifacts/shared-exercise-lab-integration-01`, verifies preparation, manual/coast/reference actions, in-recording reset, export, resume and exact replay: **69 actions, 71 events/video frames**. Its post-reset 60-step reference trial exactly matches the failed baseline above. Mocked-interface checks use these actual saved backend payloads, not invented passing scores. Local six-step control/record/JSON median was **19.26 ms**, p95 **24.18 ms**; first status load/serialization **476.71 ms**, 20 warm status serializations median **5.77 ms**. These are local core/API measurements, not browser/device or sustained-use guarantees.

The live-lab preparation limit is 60 seconds by default, with complete input/output bounds; see the [integration contract](shared-exercise-lab-plan.md). Live recording export uses a 180-second cooperative limit and 128 MiB output guard. Inputs, completed `run/` files and old recordings stay immutable; new interaction recordings live under `sessions/`.

The first source training run, `artifacts/shared-exercise-learning-01`, completed three generations/36 actual episodes/1,596 controls in 4.303 s, reusing six identical candidates. It selected the **unchanged authored baseline**, with no training improvement. Separate `artifacts/shared-exercise-policy-evaluation-01` ran 12 trials on three new starts in 2.649 s: every controller scored **0/3**. Two random-control trials ended early; other intervals completed. No evaluation feedback was used to retrain.

That evaluation revealed a **reset-protocol flaw**, not a clean controller comparison: equipment receives fixed position/velocity jitter even when human jitter is scaled down. Idle equipment starts 0.0064–0.0120 model units above the floor; one initial speed is 0.15481, already above the frozen 0.15 gate. Its peak speed is identical across all four controllers for each seed. Those results and old reset behavior stay preserved; the separately versioned study below changes neither their evidence nor the gates.

The first policy lab, `artifacts/shared-exercise-policy-lab-01`, remains unchanged. Its `shared-exercise-policy-interaction-01` verifies 68 actions, one reset, 70 frames and exact replay in 5.494 s. All 60 post-reset policy actions/states equal the earlier failed authored baseline. Its ten recorded six-step API/JSON batches had median 19.38 ms/max 20.41 ms.

**Earlier v1 fresh-seed result:** `artifacts/shared-exercise-human-jitter-01` completes in 6.960 s under the human-only reset. Training uses 36 episodes/1,588 controls in 4.3055 s and again selects `authored_unchanged`, with no improvement. Separate evaluation runs all four controllers on three new states: **0/3 each**, 642 controls in 2.6039 s. Every one of the 12 starts retains zero equipment floor gap/speed. Idle equipment stays stable (peak speed 0.0204375 model units/s). Coast keeps both objects stable but fails body tracking; authored/selected control moves `sam-2` at peak speeds 0.6796–0.7302, above the frozen 0.15 gate. Two random trials fail early, including one rigid-velocity-envelope failure. The new reset removes the start-state confound, not these task failures.

The earlier v1 lab **`artifacts/shared-exercise-policy-lab-02`** remains loadable with its unchanged manifest, policy and 0/3 comparison; `.local/shared-exercise-v1-compatibility-01.json` records that compatibility check. Its `artifacts/shared-exercise-policy-interaction-02` verifies **68 actions, one reset, 70 frames**, resume and exact own replay in **5.512 s**, retaining the live equipment-stability failure. It includes 60 post-reset policy actions; no old-baseline comparison was requested. Ten recorded six-step API/JSON batches measured median **20.408 ms**, max **20.476 ms**. Actual saved-payload UI acceptance verifies reset disclosure, policy identity, separate 0/3 comparison, 22 controls and source/geometry links. These are local core/mock-UI checks, not browser or sustained-use measurements. Exact seals, seeds and source audit are in the [current handoff](handoffs/combined.md#current-reset-assembly-and-compound-checkpoint).

The default 60-second limit is a **cooperative deadline** checked between stages, actions, export frames and replay events; it cannot preempt one native encoder call. Each recording export also has a 64 MiB byte guard. Failures preserve partial files and journals; retries require new directories. Source fit, simulated checks, learning and real physical validation remain separate. The full A+B goal stays open.
