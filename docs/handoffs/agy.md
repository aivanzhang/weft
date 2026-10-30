# agy — Track B handoff

- Assignment: simulation, RL and the unified lab. Read [full project/task context](../../AGY_HANDOFF.md) and [ownership contract](../parallel-workstreams.md).
- Status: active (`agy` Track B session; **Milestones 1 and 2 complete and verified**; proceeding to Milestone 3).
- Current objective: complete Track B acceptance criteria across Milestones 1–4:
  1. Milestone 1 (DONE — `f7b5f2f`): `spatial-rigid-v1` integration into the unified local lab inspect/control/record/replay workflow.
  2. Milestone 2 (DONE): shared capability/task contract (`environment-capabilities-v1`) across all rigid and articulated profiles + multi-object dynamic actor controls (`/api/actor`) and exact multi-actor recording replay.
  3. Milestone 3 (IN PROGRESS): bounded 3D rigid RL learning (`spatial_learning.py`, `weft spatial-train`, opt-in collision-free reset jitter) + whole-body changed-initial-condition evaluation across matched baselines (`learned`, `zero`, `random`, `authored_feedback`) with unmodified gates.
  4. Milestone 4: coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Ownership: B runtime/UI/recording/learning files only; coordinated schemas/dependencies/graders/top-level docs require an explicit interface request in this handoff. Track A is active concurrently in the same checkout on perception/reconstruction; use strictly path-scoped Git staging/commits and never touch Track A files.

## Current checkout and work

- Recorded: `2026-10-29T17:38:00-07:00 (PDT)`.
- Owner: `agy` (Track B).
- Checkout, branch, recent revisions: `/Users/ivan/kintsugi/weft`, branch `main`, HEAD `f6515bb` (Track A track-shape commit on top of B's Milestone 1 commit `f7b5f2f`, zero file overlap).
- Exact changed/added B-owned files in Milestone 2:
  - `src/weft/capabilities.py` (new: `CAPABILITY_PROFILE = "environment-capabilities-v1"`, `validate_capabilities`, `describe_capabilities`, `describe_run_capabilities` across `rolling`, `planar-rigid-v1/v2`, `spatial-rigid-v1`, `exercise-v1/v2/v3`, `exercise-imitation-v1`, `exercise-feet-posture-v1`)
  - `src/weft/scene.py` (`SceneEnv.select_actor`)
  - `src/weft/spatial_rigid.py` (`SpatialRigidEnv.select_actor`)
  - `src/weft/lab.py` (`PlanarSession.select_actor` + `trained_controlled_id` policy guard, `SpatialSession.select_actor` + `trained_controlled_id` policy guard, `Lab.capabilities`, `capabilities` in `Lab.summary`, `/api/actor` and `capabilities` in `Lab.control`)
  - `src/weft/cli.py` (`weft capabilities --run <dir> [--out <json>]`)
  - `src/weft/assets/lab.html`, `src/weft/assets/lab.js` (`#actor-controls` selector, `#capabilities-details` contract inspector, 2D multi-object/contact live readout, `contact_details` readout aligned with `SpatialRigidEnv.snapshot()`, automatic `trainedActor` restoration before `learned` mode)
  - `tests/test_capabilities.py` (new: 4 tests verifying live Gymnasium space parity across all 9 profiles, validation guards, CLI, and immutability)
  - `tests/test_multi_object_controls.py` (new: 4 tests verifying 2D & 3D `select_actor`, live physics continuity, truncation-bound preservation, trained-actor policy guard, HTTP `/api/actor`, recording lock, and exact replay)
  - `tests/lab_ui_smoke.mjs` (+2 UI checks -> 37/37 passing)
  - `docs/local-lab.md`, `docs/handoffs/agy.md`
- Critical design decisions:
  - `select_actor(object_id)` swaps `"active"`/`"passive"` roles in an in-memory validated scene copy without mutating `run/scene.json` or resetting live physics bodies, while re-evaluating `success`/`terminated`/`truncated` so an already-truncated (`steps == max_steps`) or out-of-bounds episode cannot step past declared bounds.
  - Recording locks `/api/actor` until stop/export; starting a recording writes `sessions/<id>/scene.json` with the active `controlled_id`, preserving 100% compatibility with existing `Recorder` and `load_recording` validators.
  - Verified `WholeBodyEnv` observation dimension (`155`) and `FootEnv` observation dimension (`85`) directly against live Gymnasium `env.observation_space.shape` in `tests/test_capabilities.py`.
  - Zero Meta API calls used (`$0.00` spend by Track B; shared $5 ledger untouched).

## Verification and artifacts

- Milestone 2 test results:
  - `.venv-lab/bin/python -m pytest tests/test_capabilities.py tests/test_multi_object_controls.py tests/test_spatial_lab.py tests/test_spatial_rigid.py tests/test_scene.py -k "not http"` -> `46 passed in 2.06s`.
  - `.venv-lab/bin/python -m pytest tests/test_lab_http.py tests/test_multi_object_controls.py tests/test_spatial_lab.py` -> `42 passed in 2.47s`.
  - `node tests/lab_ui_smoke.mjs` -> `37 passed` (mocked DOM/runtime check, not browser/device QA).
- Milestone 1 reproducible study preserved intact: `artifacts/track-b-spatial-lab-01/study-summary.json`.
- Physical-accuracy claims established: none (`physical_validation: "unverified"` enforced by `validate_capabilities`).
- Live job handles, ownership/locks and stop/resume procedure: none running.

## Remaining and next action

- [x] Milestone 1: Integrate `spatial-rigid-v1` into the unified local lab (`f7b5f2f`).
- [x] Milestone 2: Shared capability/task contract (`environment-capabilities-v1`) + multi-object `/api/actor` controls and exact multi-actor recording replay.
- [ ] Milestone 3: Bounded 3D rigid RL learning (`src/weft/spatial_learning.py`, `weft spatial-train`, opt-in collision-free reset jitter) + whole-body changed-initial-condition evaluation (`WholeBodyEnv` perturbation options + matched baselines across perturbation scales with unmodified gates).
- [ ] Milestone 4: Coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Next safe action: commit Milestone 2 path-scoped on `main`, then implement Milestone 3 (`spatial_learning.py` + `WholeBodyEnv` changed-initial-condition evaluation + reproducible study `artifacts/track-b-learning-changed-init-01/`).
