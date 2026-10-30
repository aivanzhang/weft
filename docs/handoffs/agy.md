# agy — Track B handoff

- Assignment: simulation, RL and the unified lab. Read [full project/task context](../../AGY_HANDOFF.md) and [ownership contract](../parallel-workstreams.md).
- Status: active (`agy` Track B session; **Milestones 1, 2, and 3 complete and verified**; proceeding to Milestone 4).
- Current objective: complete Track B acceptance criteria across Milestones 1–4:
  1. Milestone 1 (DONE — `f7b5f2f`): `spatial-rigid-v1` integration into the unified local lab inspect/control/record/replay workflow.
  2. Milestone 2 (DONE — `8c501de`): shared capability/task contract (`environment-capabilities-v1`) across all rigid and articulated profiles + multi-object dynamic actor controls (`/api/actor`) and exact multi-actor recording replay.
  3. Milestone 3 (DONE): bounded 3D rigid RL learning (`spatial_learning.py`, `weft spatial-train`, opt-in collision-free reset jitter) + whole-body changed-initial-condition evaluation (`weft body-evaluate`) across matched baselines (`learned`, `zero`, `random`, `authored_feedback`) with unmodified gates.
  4. Milestone 4 (IN PROGRESS): coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete --out <new-dir>`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Ownership: B runtime/UI/recording/learning files only; coordinated schemas/dependencies/graders/top-level docs require an explicit interface request in this handoff. Track A is active concurrently in the same checkout (`src/weft/point_flow.py`, `tests/test_point_flow.py`); use strictly path-scoped Git staging/commits and never touch Track A files.

## Current checkout and work

- Recorded: `2026-10-29T17:55:30-07:00 (PDT)`.
- Owner: `agy` (Track B).
- Checkout, branch, recent revisions: `/Users/ivan/kintsugi/weft`, branch `main`, HEAD `8c501de` (Milestone 2 commit on top of `f6515bb` and `f7b5f2f`).
- Exact changed/added B-owned files in Milestone 3:
  - `src/weft/spatial_rigid.py` (opt-in collision-free `reset(..., options={"jitter": True, ...})` bounded within `[0, 0.15]` m and `[0, 1.0]` m/s, verified via `mj_geomDistance >= 1e-4`; default `jitter=False` exact)
  - `src/weft/spatial_learning.py` (new: `POLICY_PROFILE = "spatial-policy-v1"`, `ALGORITHM = "spatial_cem_v1"`, `validate_spatial_policy`, `policy_action`, `train_spatial_policy` starting from uninformed zero-weight coast prior, freezing `policy.json` digest before independent evaluation across `nominal_exact` and `changed_initial_conditions` against `learned`, `random`, `coast`, `brake`)
  - `src/weft/whole_body.py` (`WholeBodyEnv.reset` `jitter_scale` in `[0.0, 5.0]` + `qpos_offset`/`qvel_offset` with joint-limit clipping and initial offset/speed RMSE norms in `info`, preserving bit-for-bit default `{"jitter": True}` at `scale == 1.0`)
  - `src/weft/whole_body_learning.py` (`reset_options` support on `rollout` + `evaluate_changed_conditions` across `0.0x`, `1.0x`, `3.0x`, `5.0x` jitter and 4 named deterministic perturbations under unmodified `SUCCESS_LIMITS`)
  - `src/weft/cli.py` (`weft spatial-train` and `weft body-evaluate`)
  - `src/weft/assets/lab.js` (display `mean_return` and `mean_final_distance_m` on 3D rigid RL evaluation cards)
  - `tests/test_spatial_learning.py` (new: 3 tests verifying opt-in collision-free reset jitter, end-to-end 3D training + trained lab packaging + learned-policy control, and wall-clock timeout refusal to seal)
  - `tests/test_whole_body_changed_init.py` (new: 2 tests verifying `WholeBodyEnv.reset` `jitter_scale`/offset bounds + default parity, and end-to-end `evaluate_changed_conditions` + `weft body-evaluate` CLI)
  - `docs/spatial-rigid.md`, `docs/whole-body-learning.md`, `docs/handoffs/agy.md`

## Verification and artifacts

- Milestone 3 test results:
  - `.venv-lab/bin/python -m pytest tests/test_spatial_rigid.py tests/test_whole_body.py tests/test_whole_body_learning.py` -> `59 passed in 0.76s`.
  - `.venv-lab/bin/python -m pytest tests/test_whole_body_changed_init.py tests/test_spatial_learning.py` -> `5 passed in 11.32s`.
  - `node tests/lab_ui_smoke.mjs` -> `37 passed` (mocked DOM/runtime check, not browser/device QA).
- Milestone 3 reproducible study (`artifacts/track-b-learning-changed-init-02/study-summary.json`, `$0.00` Meta spend, all 4 input manifests verified byte-for-byte unchanged):
  - **Authored 3D rigid RL (`authored-3d-learning`, manifest `787ef90986958d36d7898d40af64f251e62c29ae54acbbe1ddc0078513c65995`, policy `7ec49906135e083fad3c6447b187e502c0821fd9143c42a395139d18188a2efc`)**: under 12 changed-initial-condition episodes, `learned` achieves mean return `+0.2305` and reduces mean final distance to `0.14188 m` versus `coast` (`-0.3584`, `0.58757 m`), `brake` (`-0.3570`, `0.58615 m`), and `random` (`-0.3608`, `0.58993 m`). Short-horizon 8×16 discrete CEM does not stop within the tight terminal speed+position tolerance (`0/12` strict parking passes), reported explicitly without loosening task tolerances.
  - **Packaged trained 3D lab + recorded learned-policy session (`authored-3d-trained-lab`, run manifest `d5ff58ccab42424c15aef4c89b9317e4571b832bc7f4f789d328cdfab28e04b2`; session `sessions/session-03c357759026`, manifest `dd015c63443eddda95c074cc0c421d90edd674e3fa4de35f2505f59019222b9f`)**: 32 events (30 learned-policy steps + 1 reset + 32 decoded MP4 frames) verified bit-for-bit against a fresh `SpatialRigidEnv` replay while `run/manifest.json` remained unchanged.
  - **Source-linked 3D ramp candidate RL (`ramp-candidate-3d-learning`, manifest `25aa10aae0b126e79ec611c218e8ac0fc1dff61c575d05282ae56ea2716300a8`)**: `learned` achieves nominal mean return `+0.5281` (`0.07147 m` final distance vs `0.31866 m` coast) and `1/12` strict passes (`+0.4276` mean return, `0.26311 m` vs `0.28279–0.42572 m` baselines) under changed initial conditions.
  - **Whole-body changed-initial-condition evaluation (`single-leg-squat-changed-eval`, manifest `e21348b9eb90770b6547ef002b07301f6f8abbe998d8c743bbd0e34d38a24d91`; `squat-changed-eval`, manifest preserved in `study-summary.json`)**:
    - On floor-clear `single-leg-squat-seed-11`: `learned` passes `3/3` (`0.0x`), `9/9` (`1.0x`), `9/9` (`3.0x`), `9/9` (`5.0x`), and `3/3` on all 4 explicit perturbations (`joint_flexion_offset_0p08rad`, `lateral_root_push_minus_x`, `lateral_root_push_plus_x`, `upward_root_offset_0p03`) with mean return `0.726–0.728` and `0` floor-penetration failures (`zero` and `random` pass `0/42`; `authored_feedback` passes `42/42` with mean return `0.936–0.940`).
    - On floor-conflicted `squat-seed-11`: `learned` and `authored_feedback` keep `1.0` diagnostic tracking-within-RMSE rate across all regimes, while the 2 pre-existing floor-intersecting evaluation windows remain explicit failures (`1/3` at `0.0x`, `3/9` at `1.0x–5.0x`, `1/3` per explicit perturbation; `6/9` floor-penetration failures, peak `0.21` model units) under unmodified `SUCCESS_LIMITS`.
  - Earlier interrupted script-driver attempt `artifacts/track-b-learning-changed-init-01/` is preserved intact with `interrupted.json` provenance.
- Physical-accuracy claims established: none (`physical_validation: "unverified"`).
- Live job handles, ownership/locks and stop/resume procedure: none running.

## Remaining and next action

- [x] Milestone 1: Integrate `spatial-rigid-v1` into the unified local lab (`f7b5f2f`).
- [x] Milestone 2: Shared capability/task contract (`environment-capabilities-v1`) + multi-object `/api/actor` controls and exact multi-actor recording replay (`8c501de`).
- [x] Milestone 3: Bounded 3D rigid RL learning + whole-body changed-initial-condition evaluation (`artifacts/track-b-learning-changed-init-02/`).
- [ ] Milestone 4: Coherent source-evidence vs simulation inspection, non-destructive incomplete-recording recovery (`weft replay --recover-incomplete --out <new-dir>`), and cross-track `shared-world-v1` mixed rigid+articulated world proposal.
- Next safe action: commit Milestone 3 path-scoped on `main`, then implement Milestone 4 (`recover_incomplete_recording`, coherent provenance inspection, and `shared-world-v1` proposal).
