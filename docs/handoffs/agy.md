# agy — Track B handoff

- Assignment: simulation, reinforcement learning, and the unified local lab/recording runtime. Read [full project/task context](../../AGY_HANDOFF.md) and [ownership contract](../parallel-workstreams.md).
- Status: **complete for Track B Milestones 1–4** (cross-track `shared-world-v1` mixed rigid+articulated specification and coordinated doc updates proposed below for Track A / integrator review).
- Recorded: `2026-10-29T18:04:30-07:00 (PDT)` (`America/Los_Angeles`).
- Owner: `agy` (Track B).
- Checkout, branch, revisions: `/Users/ivan/kintsugi/weft`, branch `main` (local commits `f7b5f2f` Milestone 1, `8c501de` Milestone 2, `f914a31` Milestone 3, plus Milestone 4 commit recording this handoff and recovery/provenance work; not pushed to `origin/main` so Track A's concurrent work in the shared checkout is never raced).

---

## 1. Completed work and exact B-owned files

### Milestone 1 — `spatial-rigid-v1` in the unified local lab (`f7b5f2f`)
- `src/weft/spatial_lab.py` (new): `authored_spatial_scene()`, `write_sealed_spatial_fixture()`, `prepare_spatial_lab()` (verifies immutable source + optional policy manifests before writing, rejects failed/rejected or falsely `physics_ready` candidates, probes all 8 actions + moving brake check, seals a new `run/` preserving `inputs/source-run/` and `physical_validation: "unverified"`).
- `src/weft/lab.py`: `SpatialSession` wrapping `SpatialRigidEnv`, `launch_lab` `--scene-run` / `--learning` dispatch, `/api/step` 8-action (`0..7`) support and numerical-failure preservation.
- `src/weft/cli.py`: `weft spatial-fixture`, `weft spatial-lab`, `weft lab --scene-run <dir> [--learning <dir>] --out <new-lab>`.
- `src/weft/assets/lab.html`, `src/weft/assets/lab.js`, `src/weft/assets/lab.css`: synchronized 3-pane front (`x / height`), top (`x / depth`), and rotatable `3D` views, view-only yaw/zoom/follow controls, all 8 manual controls + keyboard (`A/D`/arrows `x`, `W/S`/arrows `y`, `Q/E`/`PageUp`/`PageDown` `z`, `Space` brake), full 3D position/quaternion/velocity/spin/force/contact inspection, honest untrained policy disabling, and `sessions/` recording/export/replay.
- `tests/test_spatial_lab.py` (new): 13 unit, CLI, immutability, numerical-failure, and loopback HTTP auth/recording/reset/replay checks.

### Milestone 2 — Shared capability contract & multi-object dynamic actor controls (`8c501de`)
- `src/weft/capabilities.py` (new): `CAPABILITY_PROFILE = "environment-capabilities-v1"`, `validate_capabilities()`, `describe_capabilities()`, `describe_run_capabilities()` across all 9 supported profiles (`rolling`, `planar-rigid-v1`, `planar-rigid-v2`, `spatial-rigid-v1`, `exercise-v1`, `exercise-v2`, `exercise-v3`, `exercise-imitation-v1`, `exercise-feet-posture-v1`, plus camera-aware untrained body candidates).
- `src/weft/scene.py`, `src/weft/spatial_rigid.py`: `SceneEnv.select_actor(object_id)` and `SpatialRigidEnv.select_actor(object_id)` switching active/passive dynamic bodies in memory without mutating `run/scene.json` or resetting live physics bodies while preserving truncation/escape bounds.
- `src/weft/lab.py`, `src/weft/cli.py`: `PlanarSession.select_actor`, `SpatialSession.select_actor`, `trained_controlled_id` policy guards, `/api/actor` endpoint (locked while recording), `capabilities` in `Lab.summary()`, and `weft capabilities --run <dir> [--out <json>]`.
- `src/weft/assets/lab.html`, `src/weft/assets/lab.js`: `#actor-controls` object selector, `#capabilities-details` contract inspector, 2D multi-object/contact live readout, automatic `trainedActor` restoration before `learned` mode.
- `tests/test_capabilities.py` (new, 4 tests verifying live Gymnasium space parity across all 9 profiles) and `tests/test_multi_object_controls.py` (new, 4 tests covering 2D/3D actor switching, HTTP `/api/actor`, recording lock, and exact multi-actor replay).

### Milestone 3 — Bounded 3D rigid RL & whole-body changed-initial-condition evaluation (`f914a31`)
- `src/weft/spatial_rigid.py`: opt-in collision-free `SpatialRigidEnv.reset(..., options={"jitter": True, "position_jitter": ..., "velocity_jitter": ...})` bounded within `[0, 0.15]` m and `[0, 1.0]` m/s and verified via `mj_geomDistance >= 1e-4` (default `jitter=False` remains exact).
- `src/weft/spatial_learning.py` (new): `POLICY_PROFILE = "spatial-policy-v1"`, `ALGORITHM = "spatial_cem_v1"`, `validate_spatial_policy()`, `policy_action()`, `train_spatial_policy()` starting from an uninformed zero-weight coast prior, freezing `policy.json` digest before independent evaluation across `nominal_exact` and `changed_initial_conditions` against `learned`, `random`, `coast`, and `brake`.
- `src/weft/whole_body.py`: `WholeBodyEnv.reset` `jitter_scale` (`0.0..5.0`) + explicit `qpos_offset` / `qvel_offset` perturbations with joint-limit clipping and initial offset/speed RMSE norms in `info`, preserving bit-for-bit default `{"jitter": True}` at `scale == 1.0`.
- `src/weft/whole_body_learning.py`: `reset_options` support on `rollout()` + `evaluate_changed_conditions()` comparing `learned`, `zero`, `random`, and `authored_feedback` across `0.0x`, `1.0x`, `3.0x`, `5.0x` jitter and 4 named deterministic perturbations (`lateral_root_push_plus_x`, `lateral_root_push_minus_x`, `joint_flexion_offset_0p08rad`, `upward_root_offset_0p03`) under unmodified `SUCCESS_LIMITS`.
- `src/weft/cli.py`: `weft spatial-train` and `weft body-evaluate`.
- `tests/test_spatial_learning.py` (new, 3 tests) and `tests/test_whole_body_changed_init.py` (new, 2 tests).

### Milestone 4 — Non-destructive recording recovery, evidence-vs-simulation provenance & `shared-world-v1` specification
- `src/weft/recording.py`: `recover_incomplete_recording(source_dir, out_dir, *, export=True)` recovering complete validated JSONL events from unclosed, tail-truncated, or partial-export crash directories into a brand-new `out_dir` with explicit `recovery` provenance and SHA-256 digests in `recording.json`, leaving `source_dir` byte-for-byte untouched and unsealed.
- `src/weft/cli.py`: `weft replay <dir> --recover-incomplete --out <new-dir>` (rejecting in-place recovery without `--out`).
- `src/weft/lab.py`, `src/weft/assets/lab.html`, `src/weft/assets/lab.js`: cached `Lab.evidence_provenance()` and `#evidence-layers` summary explicitly separating recorded source evidence (and missing/uncertain frame counts), kinematic pose projection, active controllers (`manual`, `authored_reference_controller`, `learned_policy`), simulated forward dynamics (`physical_validation: "unverified"`), and illustrative muscle display.
- `tests/test_recording_recovery.py` (new, 3 tests covering 3D truncated crash recovery, 2D partial-export recovery + CLI, articulated `exercise-v3` crash recovery, and cached `evidence_provenance`), `tests/lab_ui_smoke.mjs` (+5 checks across Milestones 1–4 -> **38/38 passing**), `docs/local-lab.md`, `docs/spatial-rigid.md`, `docs/whole-body-learning.md`, `docs/handoffs/agy.md`.

---

## 2. Critical decisions, retained failures, and physical-accuracy boundaries

- **Immutable inputs and sealed runs**: Every new workflow (`spatial-lab`, `spatial-train`, `body-evaluate`, `capabilities --out`, `replay --recover-incomplete --out`) validates input manifests before creating a new output directory and never mutates completed `run/` directories or interrupted crash journals in place.
- **Unmodified independent gates**: Neither `SUCCESS_LIMITS` in `whole_body.py` nor `evaluate.py` nor 3D task tolerances were altered.
  - On `squat-seed-11`, `learned` and `authored_feedback` keep `1.0` diagnostic tracking-within-RMSE rate across all jitter scales (`0x..5x`) and explicit perturbations, while the 2 pre-existing floor-intersecting evaluation windows remain explicit failures (`1/3` passes at `0x`, `3/9` at `1x..5x`, `1/3` per explicit perturbation; `6/9` floor-penetration failures, peak `0.21` model units > `0.025` gate).
  - On `authored-fixture` (`8` generations × `16` candidates discrete-action linear CEM), `learned` improves simulated return (`+0.2305` vs `-0.3584` coast) and reduces mean final target distance from `0.58757 m` to `0.14188 m` under changed initial conditions (`0.31866 m -> 0.07147 m` nominal and `1/12` passes under changed initial conditions on `ramp-candidate`), while short-horizon discrete CEM does not stop within the tight terminal speed+position tolerance on every trial.
  - Earlier interrupted script-driver directory `artifacts/track-b-learning-changed-init-01/` is preserved intact with `interrupted.json` provenance rather than deleted or overwritten.
- **What is NOT claimed**:
  - `physical_validation` remains `"unverified"` across every profile (`validate_capabilities` rejects any other value).
  - Muscle bands remain illustrative command-intensity displays, not measured EMG or biomechanics.
  - Whole-body 22-motor control uses 3 explicit virtual pelvis-force actuators and privileged simulator state, not unassisted human balance or recovered intent.
  - Scripted Node checks (`tests/lab_ui_smoke.mjs`, `tests/spatial_replay_smoke.mjs`) use a mocked DOM/runtime harness; no live browser or device QA was performed.

---

## 3. Reproducible commands, verification results, and performance measurements

- Platform & environment: macOS arm64, Python `3.14.7`, `.venv-lab` (`requirements-lab.lock` unchanged; original `.venv` untouched).
- Meta API budget status (`weft meta-status`): `8` settled reservations, `$0.186152` committed of `$5.00`, `$4.813848` remaining, `0` unsettled. **Track B spend: `$0.000000` (zero API calls).**

### Automated test outcomes
```sh
# 1. Core Track B non-HTTP suites (spatial 3D, capabilities, multi-object controls, recovery, planar, whole-body)
.venv-lab/bin/python -m pytest tests/test_spatial_rigid.py tests/test_spatial_recording.py tests/test_spatial_lab.py \
  tests/test_spatial_learning.py tests/test_capabilities.py tests/test_multi_object_controls.py \
  tests/test_recording_recovery.py tests/test_scene.py tests/test_scene_console.py tests/test_lab_recording.py \
  tests/test_whole_body.py tests/test_whole_body_learning.py tests/test_whole_body_changed_init.py -k "not http"
# -> 136 passed in 15.14s

# 2. Loopback HTTP auth, Host/Origin, multi-object /api/actor, and recording/export/replay suites
.venv-lab/bin/python -m pytest tests/test_lab_http.py tests/test_whole_body_lab.py \
  tests/test_spatial_lab.py tests/test_multi_object_controls.py
# -> 58 passed in 3.87s

# 3. Legacy human & foot exercise regression suites
.venv-lab/bin/python -m pytest tests/test_human*.py tests/test_foot*.py -k "not http"
# -> 85 passed in 1.86s

# 4. Deterministic UI and 3D offline replay smoke suites (mocked DOM, no browser)
node tests/lab_ui_smoke.mjs         # -> 38/38 passed
node tests/spatial_replay_smoke.mjs # -> passed
```

### Saved local studies and SHA-256 identities (`artifacts/`, local-only)
1. **`artifacts/track-b-spatial-lab-01/study-summary.json`** (Milestone 1 3D lab integration & latency study):
   - Authored 3D fixture (`authored-fixture/manifest.json` `1afca93a0dc540b10a1c199dd8415f8261419ac0d55a53aad07e190c5a494289`): packaged into `authored-lab`, 3-step MuJoCo control + JSON serialization `p50 = 0.477 ms`, `p95 = 0.601 ms`, `max = 0.892 ms`; 26-event session (`24` steps + `1` reset) exported and verified bit-for-bit (`manifest.json` `0633ef1c17e9eb87d76759ea54bc810c559f968cb2ba956384cb3963c8c00208`).
   - Source-linked ramp candidate (`artifacts/ramp-conditional-depth-01/candidate/manifest.json` `86a77bcf8185733191cd09b68bdc063795c9f36a440971f2950aca9e43e01dfc` unchanged): packaged into `ramp-candidate-lab`, 3-step control + JSON `p50 = 0.827 ms`, `p95 = 1.451 ms`, `max = 1.716 ms`.
2. **`artifacts/track-b-learning-changed-init-02/study-summary.json`** (Milestone 3 3D RL & whole-body changed-initial-condition study; all 4 input manifests verified unchanged):
   - `authored-3d-learning` (`manifest.json` `787ef90986958d36d7898d40af64f251e62c29ae54acbbe1ddc0078513c65995`, `policy.json` `7ec49906135e083fad3c6447b187e502c0821fd9143c42a395139d18188a2efc`).
   - `authored-3d-trained-lab` (`run/manifest.json` `d5ff58ccab42424c15aef4c89b9317e4571b832bc7f4f789d328cdfab28e04b2`) + recorded learned-policy session `sessions/session-03c357759026` (`manifest.json` `dd015c63443eddda95c074cc0c421d90edd674e3fa4de35f2505f59019222b9f`, `32` events / `30` steps / `1` reset / `32` MP4 frames, exact replay verified).
   - `ramp-candidate-3d-learning` (`manifest.json` `25aa10aae0b126e79ec611c218e8ac0fc1dff61c575d05282ae56ea2716300a8`, `policy.json` `769947b4decc28e0dc598b154624ba427190ce11901b8e8be10af5bc46bcb1d8`).
   - `single-leg-squat-changed-eval` (`manifest.json` `e21348b9eb90770b6547ef002b07301f6f8abbe998d8c743bbd0e34d38a24d91`): `learned` passes `30/30` across `0x`, `1x`, `3x`, `5x` jitter and `12/12` across all 4 named perturbations (`0` floor-penetration failures).
   - `squat-changed-eval` (`manifest.json` `b6ec2a23f83842df390400865c40e417f7d3a7fc7dc5c6ca39d7df1a978283ca` / recorded in `study-summary.json`): retains `6/9` floor-penetration failures at `1x..5x` (`0.21` peak model units) under unmodified `SUCCESS_LIMITS`.

### Live processes and locks
- None running (`0` servers, `0` background workers, `0` held locks).

---

## 4. Cross-track interface proposal: Mixed Rigid + Articulated Shared Worlds (`shared-world-v1`)

To support videos where a person interacts with external rigid objects (e.g., kettlebell push-ups, ball throws/catches, box lifts, bench/step supports) **without stitching separate simulators**, Track B proposes the following coordinated contract for Track A + integrator approval before editing shared schemas:

### 4.1 Core invariant: ONE MuJoCo `MjModel` / `MjData`
- Every articulated human segment and every dynamic/static rigid object compiles into **one** procedural MuJoCo XML tree stepped by **one** `mujoco.mj_step(model, data)` clock (`dt = 1/30 s`, bounded substeps `1..256`).
- World coordinates: right-handed 3D (`x_right_y_depth_z_up`), with explicit per-quantity calibration provenance (`"calibrated" | "estimated_prior" | "unverified"`). No silent mixing of normalized image-width units (`spatial-rigid-v1`) and body-height model units (`exercise-v3`): `shared-world-v1` requires Track A to supply an explicit `scale_bridge` (`body_height_to_world_scale`, `origin_world`, and `calibration_status: "calibrated" | "estimated_prior"`).

### 4.2 Proposed `shared-world-v1` scene schema
```json
{
  "schema_version": 1,
  "profile": "shared-world-v1",
  "physical_validation": "unverified",
  "dt": 0.03333333333333333,
  "substeps": 8,
  "gravity": [0.0, 0.0, -9.81],
  "bounds": [4.0, 4.0, 3.0],
  "coordinate_frame": {
    "axes": "x_right_y_depth_z_up",
    "units": {"length": "world_unit_m_prior", "mass": "relative_kg_prior", "time": "second", "angle": "radian"},
    "calibration_status": "estimated_prior",
    "scale_bridge": {
      "method": "two_view_or_monocular_body_height_prior",
      "reference_entity_id": "person-0",
      "uncertainty_note": "Relative person-object geometry estimated from shared camera frame; metric scale unverified."
    }
  },
  "supports": [
    {"id": "floor", "shape": "plane_or_box", "normal": [0.0, 0.0, 1.0], "height": 0.0, "friction": [0.8, 0.005, 0.0001], "provenance": {...}}
  ],
  "articulated_bodies": [
    {
      "id": "person-0",
      "observation_entity_id": "person-0",
      "root_mode": "free_pelvis_6dof_or_3dof_translation_3dof_rotation",
      "proportions": {...},
      "initial_pose": {...},
      "reference_trajectory": "human-reference.json",
      "actuation": "direct_22_or_28_motors",
      "virtual_root_assist_allowed": true
    }
  ],
  "rigid_objects": [
    {
      "id": "kettlebell-0",
      "observation_entity_id": "object-0",
      "role": "active_or_passive_or_static",
      "shape": "box_or_sphere_or_capsule_compound",
      "position": [0.15, 0.35, 0.12],
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "velocity": [0.0, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0],
      "mass": 8.0,
      "friction": [0.6, 0.005, 0.0001],
      "provenance": {...}
    }
  ],
  "control_binding": {
    "mode": "articulated_primary | rigid_actor_primary | composite_vector",
    "active_target_id": "person-0"
  },
  "task": {
    "task_type": "coupled_reference_and_object_goal",
    "authored_addition": true,
    "max_steps": 180,
    "success_limits": {...}
  },
  "assumptions": [...]
}
```

### 4.3 Link to Track A's `scene-observations-v1`
- Every entry in `articulated_bodies[]` and `rigid_objects[]` pins `observation_entity_id` to an entity in `observations.json` (`scene-observations-v1`), preserving:
  1. Original source video SHA-256, frame indices, and native timestamps (never renumbered over dropped frames).
  2. Explicit missing/occluded entity-frame slots (never interpolated into fake ground truth).
  3. Contact-evidence status: visual proximity in 2D/two-view projections must be labeled `"visual_proximity_candidate"`, distinct from simulated MuJoCo `contact_details` (`<person-0:left_hand, kettlebell-0>`, `<kettlebell-0, floor>`).

### 4.4 Migration and proposed contract tests
- **Adapter path**: Pure-rigid `spatial-rigid-v1` scenes map trivially to `shared-world-v1` with `articulated_bodies: []`; pure-human `exercise-v3` / `exercise-imitation-v1` runs map to `shared-world-v1` with `rigid_objects: []`. Existing profile schemas remain supported unchanged.
- **Capabilities extension**: Relax `entities["mixed_rigid_articulated"] is False` in `src/weft/capabilities.py` ONLY when `scene_profile == "shared-world-v1"` and both `articulated_bodies` and `rigid_objects` are non-empty.
- **Required contract tests before enabling**:
  1. Momentum/contact coupling test: dropping a dynamic rigid ball onto an unactuated articulated forearm alters both the ball's 3D rebound trajectory and the arm joint angle within the same `mj_step`.
  2. Initial-intersection preflight test: `whole_body_feasibility.py` extended to check initial `<human_segment, rigid_object>` and `<rigid_object, floor>` penetration without silently lifting either entity.
  3. Exact journal replay test: `Recorder` and `replay.html` round-trip coupled `articulated_bodies` + `rigid_objects` + cross-entity `contact_details` bit-for-bit across steps and resets.

---

## 5. Proposed coordinated top-level doc updates (for integrator)

Track B deliberately left coordinated top-level files (`README.md`, `TODO.md`, `docs/architecture.md`, `docs/parallel-workstreams.md`) untouched to avoid merge conflicts with Track A. Ready-to-apply summary lines for the integrator:
- `TODO.md`: Mark Track B items complete (`spatial-rigid-v1` in unified lab, `environment-capabilities-v1` across all 9 profiles, multi-object `/api/actor` switching, bounded 3D rigid RL `weft spatial-train`, whole-body changed-initial-condition evaluation `weft body-evaluate`, non-destructive `weft replay --recover-incomplete --out`, and `shared-world-v1` cross-track proposal in `docs/handoffs/agy.md`).
- `README.md` / `docs/architecture.md`: Add CLI entries for `weft spatial-fixture`, `weft spatial-lab`, `weft spatial-train`, `weft capabilities`, `weft body-evaluate`, and `weft replay --recover-incomplete --out`, keeping `physical_validation: "unverified"` explicit.

---

## 6. Git state and next safe action

- All Track B source, test, asset, guide, and handoff changes are committed cleanly on local `main` via path-scoped commits; Track A's concurrent untracked/in-progress files in `/Users/ivan/kintsugi/weft` remain completely untouched.
- Local-only verification studies are preserved at `artifacts/track-b-spatial-lab-01/` and `artifacts/track-b-learning-changed-init-02/` (with earlier interrupted driver run preserved at `artifacts/track-b-learning-changed-init-01/`).
- **Next safe action**: Coordinate with Track A / integrator to review Section 4 (`shared-world-v1` scale bridge and coupled person-object observation mapping), apply Section 5 top-level doc summaries, and push `main` once Track A completes its current perception checkpoint.
