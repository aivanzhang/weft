# agy — Track B handoff

- Assignment: simulation, RL and the unified lab. Read [full project/task context](../../AGY_HANDOFF.md) and [ownership contract](../parallel-workstreams.md).
- Status: active (`agy` Track B session; **Milestone 1 complete and verified**; proceeding to Milestone 2).
- Current objective: complete Track B acceptance criteria across Milestones 1–4:
  1. Milestone 1 (DONE): `spatial-rigid-v1` integration into the unified local lab inspect/control/record/replay workflow.
  2. Milestone 2 (IN PROGRESS): shared capability/task contract (`environment-capabilities-v1`) across rigid and articulated profiles + multi-object dynamic actor controls (`/api/actor`) and inspection.
  3. Milestone 3: bounded 3D rigid RL learning (`spatial_learning.py`, `weft spatial-train`, opt-in collision-free reset jitter) + whole-body changed-initial-condition evaluation across matched baselines (`learned`, `zero`, `random`, `authored_feedback`) with unmodified gates.
  4. Milestone 4: coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Ownership: B runtime/UI/recording/learning files only; coordinated schemas/dependencies/graders/top-level docs require an explicit interface request in this handoff. Track A is active concurrently in the same checkout on perception/reconstruction; use strictly path-scoped Git staging/commits and never touch Track A files.

## Current checkout and work

- Recorded: `2026-10-29T17:18:30-07:00 (PDT)`.
- Owner: `agy` (Track B).
- Checkout, branch, starting revision: `/Users/ivan/kintsugi/weft`, branch `main` (arrived at `982adeb4d79899c8ef8944f8d5440d0bc95a7ed9`; coordinator advanced `main` to `8ca9b60c7e43b9a8711a890d29fbaaa1b2aa1741` with zero B-file overlap).
- Exact changed/added B-owned files in Milestone 1:
  - `src/weft/spatial_lab.py` (new: `authored_spatial_scene`, `write_sealed_spatial_fixture`, `prepare_spatial_lab`)
  - `src/weft/lab.py` (`SpatialSession`, `"spatial-rigid"` dispatch, `learned_policy_available` in `Lab.summary`, actions `0..7` + runtime failure handling + `"spatial-rigid"` recorder profile in `Lab.control`, `--scene-run` / `--learning` in `launch_lab`)
  - `src/weft/cli.py` (`spatial-fixture`, `spatial-lab`, `lab --scene-run` / `--learning`)
  - `src/weft/assets/lab.html`, `src/weft/assets/lab.js`, `src/weft/assets/lab.css` (synchronized 3-pane front/top/3D views, view-only yaw/zoom/follow sync, 8 spatial controls `0..7` + keyboard `WASD`/arrows/`Q`/`E`/`Space=7`, full 3D position/quaternion/spin/contact readout, honest untrained policy disabling)
  - `tests/test_spatial_lab.py` (new: 10 Python unit/HTTP/CLI/replay/failure tests)
  - `tests/lab_ui_smoke.mjs` (added 2 `spatial-rigid` deterministic UI checks; 35 total UI checks passing)
  - `docs/local-lab.md`, `docs/spatial-rigid.md`, `docs/handoffs/agy.md`
- Critical design decisions & rejected approaches:
  - Preserved strict `spatial-rigid-v1` 12-key top-level schema on `SpatialSession.scene` so `Recorder(..., profile="spatial-rigid")` and `spatial_rigid.validate_scene` work unchanged; surfaced `learned_policy_available` on `Lab.summary()` (`/api/status`), `lab-config.json`, and `report.json`.
  - Marked spatial-only manual buttons `6` (`Down (−z)`) and `7` (`Slow down`) with `data-spatial="true"` so non-spatial profiles keep actions `>=6` hidden/disabled except `exercise-v3` mobile controls (`6..9`), and used `document.querySelectorAll("[data-action]")` for full compatibility with both browsers and flat DOM mocks.
  - Kept `authored_spatial_scene()` virtual acceleration at `8.0` m/s² within `spatial_rigid.validate_scene`'s `[0.1, 10.0]` bound; zero imports from `tests/` in production code.
  - Zero Meta API calls used (`$0.00` spend by Track B; shared $5 ledger untouched).

## Verification and artifacts

- Milestone 1 test results:
  - `.venv-lab/bin/python -m pytest tests/test_spatial_lab.py tests/test_spatial_rigid.py tests/test_spatial_recording.py tests/test_scene_console.py` -> `74 passed in 2.33s`.
  - `.venv-lab/bin/python -m pytest tests/test_lab_recording.py tests/test_camera_body.py` -> `24 passed in 1.31s`.
  - `.venv-lab/bin/python -m pytest tests/test_lab_http.py tests/test_whole_body_lab.py` -> `44 passed in 2.24s`.
  - `node tests/lab_ui_smoke.mjs` -> `35 passed` (mocked DOM/runtime check, not browser/device QA).
  - `node tests/spatial_replay_smoke.mjs` -> `passed` (mocked DOM/runtime check, not browser/device QA).
- Reproducible Milestone 1 study (`artifacts/track-b-spatial-lab-01/study-summary.json`):
  - Authored 3D fixture (`artifacts/track-b-spatial-lab-01/authored-fixture`, `scene_sha256: a47b3515b4e2505e4290b5372eac0a98ef80077fdf3b211a7ae3705b0fc225a1`, `manifest_sha256: 1afca93a0dc540b10a1c199dd8415f8261419ac0d55a53aad07e190c5a494289` unchanged before/after packaging):
    - Packaged lab (`artifacts/track-b-spatial-lab-01/authored-lab`, `run` `manifest_sha256: 296ee5e6cfe8b0c57bbabba4efc06b6c45a4a72c3298f85ccb7f87809b07d105`, 137 verified files, `0.0287s` preparation).
    - Responsiveness: initial `/api/status` serialize `8.247 ms`; 100× 3-step MuJoCo control + JSON serialize `p50 = 0.477 ms`, `p95 = 0.601 ms`, `max = 0.768 ms`.
    - Recording/export (`sessions/session-b5c6ec417202`, 24 steps + 1 reset, `manifest_sha256: f8a179862b0c6530bc8cd74985f3c9ad5ed30959772f3d27bd29bb1f1ee27968`): `260.784 ms` export wall time, `run/` seal unchanged, bit-for-bit exact replay verified.
  - Optional saved 3D ramp candidate (`artifacts/ramp-conditional-depth-01/candidate`, `manifest_sha256: 86a77bcf8185733191cd09b68bdc063795c9f36a440971f2950aca9e43e01dfc` unchanged before/after):
    - Packaged lab (`artifacts/track-b-spatial-lab-01/ramp-candidate-lab`, `run` `manifest_sha256: 8a96dc9bc9bdf6ce0eb3eb5bb3bc11b7e1f8d83af509f77febc180deb3122e5a`, `0.0299s` preparation).
    - Responsiveness: 100× 3-step control + JSON serialize `p50 = 0.827 ms`, `p95 = 1.451 ms`, `max = 2.015 ms`; 24-step + 1-reset export (`sessions/session-30661dbc8b34`) `267.66 ms`, exact replay verified.
- Physical-accuracy claims established: none (`physical_validation: "unverified"` preserved everywhere).
- Live job handles, ownership/locks and stop/resume procedure: none running.

## Remaining and next action

- [x] Milestone 1: Integrate `spatial-rigid-v1` into the unified local lab, verify interaction/reset/recording/replay/immutability/latency, and update B guides.
- [ ] Milestone 2: Shared capability/task contract (`src/weft/capabilities.py`, `/api/status`, `weft capabilities`) across rigid and articulated profiles + multi-object dynamic actor controls (`/api/actor`) and exact multi-actor recording replay.
- [ ] Milestone 3: Bounded 3D rigid RL learning (`src/weft/spatial_learning.py`, `weft spatial-train`, opt-in collision-free reset jitter) + whole-body changed-initial-condition evaluation (`WholeBodyEnv` perturbation options + matched baselines across perturbation scales with unmodified gates).
- [ ] Milestone 4: Coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Next safe action: create Milestone 1 Git checkpoint commit on `main`, then implement `src/weft/capabilities.py` and multi-object `/api/actor` controls.
