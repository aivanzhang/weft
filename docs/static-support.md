# Authored static supports for exercise bodies

Build one shared 3D scene from a sealed exercise body and explicit static-support assumptions. The person remains tied to the original video observations. Support geometry is **authored**, not recovered from the video.

This path does not infer contact, grasp, depth, materials, or biomechanics. A moving weight or an occluded hand is not evidence of a fixed support. Do not turn a moving object into a static support without declaring that change as a simulator hypothesis.

## Build, inspect and play in one workflow

Use the ordinary exercise builder with a support-prior file. In the [configuration below](#configuration), change `profile` to `shared-static-support-priors-v1` and omit `body_source_manifest_sha256`; the builder fills that identity only after preparing the actual body. Keep the exact original video hash, scale, support shapes, load windows and root-assistance choice. These are authored assumptions, not inferred contact.

The simplest entry point builds an untrained scene, saves `report/index.html`,
and enters the terminal only when the lab is ready:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --video /absolute/path/to/exercise.mp4 \
  --static-supports /absolute/path/to/support-priors.json --console
```

Replace `--video` with `--human-run /path/to/construction-only-body` to reuse saved
poses. Add `--train` for bounded phase-based corrections and changed-start
evaluation, or replace `--console` with `--serve` for the local 3D interface.
With neither option, it prints reopening commands. The prior owns scale and root
assistance; do not also pass `--person-only`, `--root-assistance`, equipment
configuration or source mappings. It uses existing local models and never
downloads one. Launcher training uses fixed exposed-development seeds, not a
fresh final holdout.

For explicit training/evaluation settings, use the underlying CLI:

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --video /absolute/path/to/exercise.mp4 \
  --pose-model /absolute/path/to/pose_landmarker_lite.task \
  --static-supports /absolute/path/to/support-priors.json \
  --train --training-generations 1 --training-population 4 \
  --evaluation-seed 924079 \
  --out artifacts/my-support-build-01

.venv-lab/bin/python -m weft shared-exercise-inspect \
  --run artifacts/my-support-build-01 --out artifacts/my-support-inspection-01

.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/my-support-build-01/learning_workflow/lab
```

To reuse existing inference, replace `--video` and `--pose-model` with `--human-run <construction-only exercise-body-source directory>`. Without `--train`, the final lab is `<build>/lab`. Use `lab` instead of `lab-console` if you want to start the local browser interface yourself. Open the inspection's `index.html` to see original media, retained failures, assumptions and results.

The terminal shows body preparation, support binding, shared-world construction, optional training/evaluation, and one final lab. `support-priors.json` preserves your exact input; `support-config.json` adds the completed body pin. No dummy world or person-only inventory is created. Do not combine this mode with `--person-only`, equipment/config/mapping options or scale/root overrides; those settings belong to the support file. Optional source-only crop recovery uses the same existing Full-model recipe, without inferring supports.

Training keeps the full interval and fixed task. Only the phase-residual policy family is supported. The optimizer's frozen choice goes directly to changed-start evaluation: the older secondary feasibility selector is **not applicable**, not passed. An unchanged authored controller stays labeled unchanged; a ready lab does not imply task success. Choose a separate evaluation seed; the example above is not guaranteed unused.

In the console:

```text
controls
record start
motor left_elbow 0.05
step 3
reset 31
mode learned
step 6
record stop
record evidence
files
```

Use `mode reference` instead of `mode learned` when training was omitted. Recording exports and replay evidence come from the same lab. A stopped build remains inspectable and must not be opened as a ready environment.

## Build a candidate

Start with an existing `exercise-body-source-v1` directory whose manifest says `construction_only`. Supply its exact manifest hash and original video hash in a configuration file:

```bash
.venv-lab/bin/python -m weft shared-support-build \
  --body-source artifacts/example/build/body_source \
  --config support-config.json \
  --out artifacts/example-static-support \
  --max-seconds 30
```

The output must be new. A failed attempt remains available for inspection; use another output directory after correcting its cause. No video decode, pose inference, training, server, or paid API call occurs during construction.

The Python API accepts a configuration path or JSON mapping:

```python
from weft.shared_world_support import construct_static_support_run

result = construct_static_support_run(
    "artifacts/example/build/body_source",
    "support-config.json",
    "artifacts/example-static-support",
    max_seconds=30,
)
```

## Configuration

The following is an illustrative prior, not a measured bench or a qualified exercise setup. Replace the two hashes with the exact identities of your input. Placement uses the inherited `x_right_y_depth_z_up` frame and body-scaled world units, not calibrated meters.

```json
{
  "schema_version": 1,
  "profile": "shared-static-support-construction-v1",
  "source_sha256": "<original video SHA256>",
  "body_source_manifest_sha256": "<body-source manifest SHA256>",
  "scale": 1.0,
  "objects": [
    {
      "id": "bench",
      "label": "Authored bench hypothesis",
      "role": "static",
      "shape": "box",
      "position": [1.0, 1.0, 0.3],
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "size": [0.6, 0.4, 0.2],
      "velocity": [0.0, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0],
      "mass": 10.0,
      "friction": [0.8, 0.01, 0.001],
      "color": "#707070",
      "provenance": {
        "mode": "authored_addition",
        "calibration_status": "estimated_prior",
        "physical_validation": "unverified",
        "source_contact_verified": false,
        "assumptions": ["Static shape, pose and material values are authored, not measured."]
      }
    }
  ],
  "support_requirements": [
    {
      "object_id": "bench",
      "human_segment": "left_forearm",
      "direction_world": [0.0, 0.0, 1.0],
      "minimum_load_body_weight_fraction": 0.05,
      "start_step": 0,
      "end_step": 30
    }
  ],
  "root_assistance": false
}
```

Every top-level field is required. Unknown fields and claimed observed, calibrated, or physically verified support provenance are rejected rather than silently relabeled.

| Setting | Contract |
| --- | --- |
| `scale` | Explicit body-to-world prior, 0.5–2.0; never a calibration claim. |
| `objects` | 1–12 static boxes, spheres, or `rigid-compound-v1` objects using the existing shared-world geometry contract. Positions/orientations/materials must be explicit. |
| Box `size` | Full extents along local x/y/z; no automatic fitting or placement. |
| Sphere | Use `radius` instead of `size`. |
| Compound | Use `geometry` instead of `size`, with authored primitive parts and inertia; the existing 32-primitive scene limit applies. |
| `support_requirements` | 1–24 named body-segment/load requirements; only declared static object IDs are allowed. |
| `direction_world` | Explicit unit vector; the constructor does not normalize or infer it. |
| `minimum_load_body_weight_fraction` | Positive authored simulated load threshold, at most 2.0; 0.05 means 5% of modeled body weight. |
| `start_step`, `end_step` | Half-open control-step window `[start_step, end_step)` at 30 Hz, within the complete reference interval. |
| `root_assistance` | Required boolean; use `false` by default. `true` explicitly enables the existing three-axis virtual root assistance. It is not unassisted balance. |

Supported body segments are `torso`, `head`, `pelvis`, `shoulders`, and left/right `upper_arm`, `forearm`, `thigh`, or `shin`. There are no inferred palm, sole, grasp, or ground-plane support requirements. A forearm collision is not automatically a verified hand contact.

## Inspect before playing

Construction emits a `shared-world-v4` **candidate**, not an admitted lab or a successful task:

| Artifact | What it records |
| --- | --- |
| `scene.json` | Original body with explicit static supports; no movable-object goal or object-force controls. |
| `exercise-task.json` | Separate `shared-support-task-v1` load requirements, frozen gates and root-assistance choice. |
| `inputs/support-config.json` | Exact supplied configuration bytes, including formatting when read from a file. |
| `inputs/body-source-manifest.json` | Exact original body-source seal. Every declared source member is copied unchanged at its original relative path. |
| `observations.json`, `source-clock.json` | All original detected, missing and unsampled slots; decoder timestamps are not filled in or retimed. |
| `source-binding-verification.json` | Human-to-observation identity proof. Each support has a null observation identity and `authored_addition` provenance. |
| `reference-feasibility.json` | Saved kinematic floor/static-intersection checks. These are model/forward checks, not dynamics or physical validation. |
| `reconstruction.json` | Exact source/config/scene/task/code identities, limits, assumptions and readiness scope. |
| `manifest.json` | Output byte hashes. Late failures instead receive a failed seal and `failure.json`. |

An infeasible reference is preserved and reported; construction does not move the body, adjust the floor, shorten the interval, or discard failures to make it pass. Existing lab admission must still check initial geometry. Construction alone establishes neither sustained support nor task success.

The constructor has a 30-second cooperative deadline, a 128 MiB input/output envelope, and at most 512 files per package. Original media may be hash-read and copied, but is never decoded here. Native feasibility checks remain bounded by the existing source/reference limits; an outer worker must enforce a hard process deadline when needed.

## Verified local examples

**Guided real-body integration:** [step-by-step inspection](../artifacts/static-support-build-acceptance-01/inspection/index.html). The saved single-leg source completes build→training→comparison→one lab→record/export/exact replay in 17.9 seconds without repeated video inference. All four controllers score **0/3 support-task successes**; one random trial ends early. Original eight source/sixteen interpolated-grid floor failures remain visible. The distant authored box deliberately tests source custody, not a useful exercise support. Nine recorded manual/selected-policy actions replay exactly; the small training-score change is not independent task improvement.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/static-support-build-acceptance-01/build/learning_workflow/lab
```

The analytical support example is playable without a browser or server:

```bash
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/static-support-play-acceptance-01/present/lab
```

The recorded positive control passes its two-second authored task. Lowering only the support by 0.6 world units fails root and sustained-support criteria under the same starting state and rules. Both recordings export and replay exactly: 65 frames in the present case, 61 in the lowered case. Root assistance is disabled, but authored torso actuators remain; this is not free-floating human balance or physical validation. Ground-force requirements, grasp and avatar recording for this profile are not supported.

The genuine saved single-leg body also passes the new **constructor custody check** in `artifacts/shared-support-real-body-01`. Its distant box is deliberately not a recovered support. All original source/body bytes, 235 source slots, 118 references and 234 task steps are preserved. The candidate remains **unadmitted**, with the exact original floor failures: eight source poses and sixteen control-grid poses exceed the unchanged 0.025 model-unit limit; peak penetration is 0.0831304. No rollout or support-task success is claimed for it.

Saved audits and precise bounds are in the [current handoff](handoffs/combined.md#static-support-interaction--october-31-2026-1001-am-pdt). These examples establish bounded construction, declared simulator contact accounting and recording/replay—not arbitrary-video reconstruction, recovered anatomy or measured contact forces.

## Train, compare and record a support policy

Static supports use the existing **phase-residual** learner: it searches small joint-command corrections on top of authored reference feedback. It cannot move supports, add object forces, change task thresholds, shorten the interval, or learn root assistance. Its support-specific policy pins both evaluator files and the exact scene/task. The legacy feedback learners and feasibility selectors do not accept this profile.

The verified example is ready to play without training again:

```bash
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/static-support-policy-acceptance-01/lab
```

Use `controls`, `mode learned`, `record start`, `step 6`, and `record stop`.
For manual control, use `mode manual` and `motor left_elbow 0.05` before stepping.
`record evidence` shows the saved replay check; `files` lists the exports.
The same lab can be opened by the user with `weft lab --resume <lab-directory>`.

This small development run searched four candidates over two training starts
(480 actions), then compared coast, random, authored and selected controllers
on three fresh matched human starts. Supports stayed fixed. The search selected
**the unchanged authored baseline**, with zero training improvement. Selected
and authored controls each passed 3/3 tasks; coast passed 0/3, and random
terminated early in all three trials. These failures remain in the comparison.
The recorded three manual plus sixty selected-policy actions export and replay
exactly. This verifies integration, not improved learning or unfamiliar-video recovery.

To train a different admitted candidate, use new output directories:

```bash
.venv-lab/bin/python -m weft shared-exercise-train \
  --run artifacts/my-static-support --out artifacts/my-support-training \
  --policy-family phase-residual --generations 1 --population 4 --max-seconds 30

.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning artifacts/my-support-training --out artifacts/my-support-evaluation \
  --trials 3 --jitter-scale 0.10 --max-seconds 30

.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-static-support --learning artifacts/my-support-training \
  --evaluation artifacts/my-support-evaluation --out artifacts/my-support-lab
```

Inspect `training.json`, `evaluation.json`, and the lab's comparison before
interpreting success. The task preserves its entire declared interval; larger
inputs can exceed the fixed action/time budget and fail explicitly. Failed
resets must not be replaced with easier seeds. A selected authored baseline is
available through the policy control mode but is not labeled a learned gain.
