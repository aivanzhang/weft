# Authored static supports for exercise bodies

Build one shared 3D scene from a sealed exercise body and explicit static-support assumptions. The person remains tied to the original video observations. Support geometry is **authored**, not recovered from the video.

This path does not infer contact, grasp, depth, materials, or biomechanics. A moving weight or an occluded hand is not evidence of a fixed support. Do not turn a moving object into a static support without declaring that change as a simulator hypothesis.

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

The analytical support example is playable without a browser or server:

```bash
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/static-support-play-acceptance-01/present/lab
```

The recorded positive control passes its two-second authored task. Lowering only the support by 0.6 world units fails root and sustained-support criteria under the same starting state and rules. Both recordings export and replay exactly: 65 frames in the present case, 61 in the lowered case. Root assistance is disabled, but authored torso actuators remain; this is not free-floating human balance or physical validation. Native support learning, ground-force requirements, grasp and avatar recording for this profile are not supported.

The genuine saved single-leg body also passes the new **constructor custody check** in `artifacts/shared-support-real-body-01`. Its distant box is deliberately not a recovered support. All original source/body bytes, 235 source slots, 118 references and 234 task steps are preserved. The candidate remains **unadmitted**, with the exact original floor failures: eight source poses and sixteen control-grid poses exceed the unchanged 0.025 model-unit limit; peak penetration is 0.0831304. No rollout or support-task success is claimed for it.

Saved audits and precise bounds are in the [current handoff](handoffs/combined.md#static-support-interaction--october-31-2026-1001-am-pdt). These examples establish bounded construction, declared simulator contact accounting and recording/replay—not arbitrary-video reconstruction, recovered anatomy or measured contact forces.
