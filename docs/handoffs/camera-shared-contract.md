# Camera shared-world admission contract

Updated: 2026-10-31 14:25:46 PDT. Owner: support-learning lane. Implementation and pure checks are complete; shared-world integration and native verification belong to the root lane.

## Scope and ownership

- New helper: `src/weft/camera_shared_contract.py`.
- New analytical checks: `tests/test_camera_shared_contract.py`.
- Root owns `shared_world.py`, native checks, packaging integration, recording, UI and Git. The reconstruction lane owns the camera importer and calibrated-foot consumer.
- This helper validates bounded JSON, declared source identity, coordinate transforms and clocks. It does not construct or validate native geometry, read source files, reconstruct poses, prove camera accuracy or score task success.
- It imports no native human/physics module. Existing camera/root/ground validators remain unchanged. Only grounded `two-view-root-body-v1`, `exercise-v3`/schema3 bodies qualify; other camera methods are rejected before validator dispatch.

## Public interface

```python
build_camera_coupling(body)
validate_camera_coupling(body, coupling, *, dt, coordinate_frame, source_binding)
```

Both return detached canonical JSON dictionaries. They never alter the body, supplied envelope or scene arguments. Rejection raises `ValueError`; there is no repaired fallback.

Exports:

- `PROFILE = "camera-shared-coupling-v1"`
- `SOURCE_CLOCK_PROFILE = "camera-shared-source-clock-v1"`
- `LENGTH_UNIT = "supplied_camera_metre_prior"`
- `MASS_UNIT = "authored_body_mass_prior"`
- `CALIBRATION_STATUS = "estimated_prior"`
- `ROOT_ABS_MAX = 50.1`, `SEGMENT_CONTACT_ABS_MAX = 64.0`
- `MAX_BODY_BYTES = 8 * 1024 * 1024`, `MAX_POSES = 900`, `MAX_SOURCE_FRAMES = 1800`, `MAX_SECONDS = 5.0`

## Exact envelope

The schema1 envelope contains only:

- `profile`, `source_sha256`.
- Canonical pins: `body_sha256`, `camera_binding_sha256`, `ground_sha256`, `root_reference_sha256`, `shape_sha256`, `source_clock_sha256`.
- `world_to_local`: scale `1.0`, identity 3×3 `rotation`, translation `[-origin_world_m.x, -origin_world_m.y, -ground.height_m]`.
- `shared_placement`: scale `1.0`, origin `[0.0, 0.0, 0.0]`.
- `clock`: `control_dt_s`, `source_fps`, `source_time_offset_s`, `first_source_frame`, `last_source_frame`, `total_source_frames`, `retained_poses`, `horizon_steps`.
- `calibration_status = "estimated_prior"`.
- `units`: `camera_length = "m"`, `model_length = LENGTH_UNIT`, `model_mass = MASS_UNIT`.
- `assumptions`: `root_scale = "supplied_camera_scale"`, `body_shape = "unchanged_monocular_estimate"`, `body_mass = "unchanged_authored_model"`, `anatomy = "not_independently_measured"`, `support = "unscored"`.
- `runtime_bounds`: `root_abs_max = 50.1`, `segment_contact_abs_max = 64.0`.

Pins use the existing `weft.io.canonical_hash`, including its JSON serialization conventions. The binding pin covers the whole embedded `camera_reference`, including both views and original evidence hashes. `shape_sha256` uses unchanged `multiview_root.shape_hash` semantics, not a new anatomy estimate. The helper pins declared documents; actual input bytes and sealed source-document proof remain packaging responsibilities.

`source_clock_sha256` hashes this complete original retained clock document:

```python
{
    "schema_version": 1,
    "profile": "camera-shared-source-clock-v1",
    "source_sha256": body["source_sha256"],
    "source_fps": root_refinement["fps"],
    "total_source_frames": root_refinement["source_frames"],
    "control_dt_s": 1.0 / root_refinement["fps"],
    "rows": [{"frame": original_frame, "time_s": original_time,
              "evidence_id": original_primary_evidence_id}, ...],
}
```

Every retained angular, root, provenance and refined-root row must align exactly with the original retained source identity and timestamp. Timestamp/frame arithmetic permits at most `1e-10` seconds of floating-point error. The first retained source frame need not be zero. Missing sampled poses are not filled, moved or dropped; the preserved interval spans the original first and last retained poses. `horizon_steps` is their source-frame difference, not the number of retained poses. Declared total source frames remain distinct from retained pose count.

**The two timesteps are deliberately different:** `body.dt` is preserved, finite and within its existing 20–120 Hz range. Shared `dt` and `clock.control_dt_s` equal `1 / root_refinement.fps`. A body authored at `1/30` with source footage at `30000/1001` fps remains byte-identical; its shared control period is `1001/30000`. An initial implementation conflated these clocks; it was corrected before the first test run and is covered explicitly.

## Scene boundary

The supplied `coordinate_frame` retains the existing shared shape:

- Axis `x_right_y_depth_z_up`; calibration `estimated_prior`.
- Units: length/mass constants above, time `second`, orientation `quaternion_wxyz`, angular velocity `world_radians_per_second`.
- `scale_bridge.body_height_to_world_scale = 1`; `origin_world = [0, 0, 0]`; nonempty bounded `method` and `uncertainty_note`.
- Clock control period and fps match the source; offset equals the first retained original timestamp. `physics_substeps` is an integer in 8–64; `alignment_method` is nonempty bounded provenance. Scene substep equality and the native timestep limit remain root-owned.
- `source_binding.mode = "source_qualified"`; primary hash and original total frame count must match the body. The helper does not replace ordinary evidence-document/entity-binding validation.

Body orientation must already be a proper rotation. Original camera roots remain within ±50 and begin at zero translation. Supplied ground must preserve original base height within 0.01–5 units; there is no floor fitting, shape scaling or source-origin rewriting.

Root must retain the exact original body after full human validation, enforce the new v6 profile and its existing-source custody, and preserve v1–v5 rejection semantics. Objects, gravity, full body geometry, runtime bounds enforcement, policy exclusion and scorer admission are outside this helper. The initial root design allows only static objects, no feet or learned policy, and no scored support claims; this envelope does not establish support, balance or measured anatomy.

## Bounds and verification

All trees are checked before camera validators hash or convert them: plain JSON types only, no cycles/nonfinite values, maximum depth 64, 300,000 nodes and 64-bit integers. Encoded body size is at most 8 MiB; coupling/frame/source-binding inputs are each at most 64 KiB. Pose and source-frame caps are checked before body serialization. Deadline checks occur during traversal/serialization and between validation/hash/return stages. The 5-second bound is cooperative, not a preemptive CPU or memory guarantee. There is no persistent cache or per-frame work.

Command:

```sh
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_contract.py --junitxml=.local/camera-shared-contract-tests-01.xml
```

Result: **40 passed**, pytest summary 0.30s; JUnit 0.275s. The 900-pose/1800-source-frame analytical boundary test took 0.105s for fixture expansion plus build and validation on this machine; this is not a native or production performance benchmark. Native/model/media imports, reconstruction calls, network and subprocess entrypoints were blocked inside these tests. The reused fixture intentionally lacks a physics-valid full human model; it proves contracts only.

Coverage includes detached results, all pins and original clock rows, fractional fps with preserved body period, global world-translation invariance, nontrivial preserved body rotation/root motion, unsupported camera variants, source/frame mismatches, malformed and excessive inputs, immutable assumptions/bounds, and cooperative deadline rejection. No failed test run, source study, native study, model download or paid call occurred in this lane.

Evidence: `.local/camera-shared-contract-tests-01.xml` and `.local/camera-shared-contract-checks-01.json`; the latter pins this module, tests and handoff. No Git command or mutation was performed; the shared worktree contains concurrent owner-managed work. There is no live handle.

## Next safe action

Root can freeze the importer/engine dependencies and run its single bounded native equivalence/admission check using unchanged sealed inputs. Verify geometry, mass, reference trajectories, source-clock horizon, manual controls, failure retention and replay separately. Do not describe the pure contract pass as native admission, successful ground contact or physical validation, and do not reuse the analytical fixture for that acceptance.
