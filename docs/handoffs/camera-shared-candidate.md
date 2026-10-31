# Camera-body → shared-world candidate

## Ownership and state

- Recorded: 2026-10-31 14:36 PDT. Owner: `/root/exercise_reconstruction`.
- Implemented and focused pure tests passed; product/test frozen for independent review. No source/native execution. Own only `src/weft/camera_shared_candidate.py`, `tests/test_camera_shared_candidate.py`, and this note. Root owns engine/CLI/lab/Git; `support_learning` owns `camera_shared_contract.py`. Foot consumer belongs to its separate repair lane.
- Public API: `prepare_camera_shared_candidate(source_run, out, *, max_seconds=30)`. `out` itself is the new immutable candidate directory, not a lab or `out/run` wrapper. It returns `status`, `sealed`, `candidate_directory`, `manifest_sha256` and scope/pins. Unsafe/existing outputs raise before mutation; later failures return receipts.
- Apply the document-project-todos handoff workflow. No network, paid calls, video decoding, inference, fitting, physics, source study or Git mutation in this lane.

## Approved contract

- Preserve exact grounded `two-view-root-body-v1` body, both pose/camera documents, ground, triangulation/config and original source rights. Reuse camera-coupling, camera/ground validators and observation/evidence registry; do not reconstruct or infer anything anew.
- Primary `adapt_exercise(evidence, scene=None)` observations bind `person-0` to `human`; every decoded slot survives. Clear the adapter's generic fixed-camera assumption to unknown. Secondary source remains separately pinned, never aliased to primary media.
- Preserve original body `dt=1/30`; shared scene `dt=1/source_fps`, including fractional FPS. Full original reference horizon, unit scale and zero outer origin. No inventory declaration, dynamic objects, learned policy, independent exercise-task adapter or contact/absence claim.
- Initial `objects=[]` is an importer capability limit, not proof that the video contains no equipment.
- Named source JSON members ≤32 MiB total; source manifest ≤256 KiB/512 members; primary media ≤64 MiB; runtime ≤8 MiB; package ≤128 MiB/80 files. Thirty-second cooperative deadline, not process/memory isolation. Terminal failure evidence remains sealed when safe; never overwrite input/output.
- Source closure: 16 explicitly named JSON documents, primary media and optional `background.jpg`; source scene/task become `source-run/scene.json` and `source-run/task.json`. The input manifest is copied unchanged to `source-run/manifest.json`. No recursive source enumeration or inherited implementation copy. Exact primary media is copied without decoding; original filenames/rights remain unchanged.
- Runtime whitelist: 26 Python files plus project metadata/locked dependencies, captured before source processing and rechecked before sealing. The lazy `shared_world.validate_shared_world` boundary is mocked in this lane; root owns the actual frozen native-capable schema/admission run. Runtime file hashes are custody pins, not a claim of independent bytecode attestation.
- Success verifies selected source bytes, runtime bytes and every known output before one exclusive staged manifest publication. `candidate.json` remains `prepared_unsealed`; `manifest.json` alone gives terminal status. A failure seal covers only `failure.json`; its `unadmitted_attempted_files` retains expected hashes for untouched partial artifacts without falsely certifying files that may have failed custody checks. Bounded failure journaling may finish after the construction deadline; it never upgrades a failure.

## Saved-source inspection only

- Inspected selected metadata in `artifacts/two-view-root-jog-lab-01/run`; no execution. Manifest SHA256 `325fd37a2a0d856949c4d63513769b318f2af79558d02a65123f6543d3201eed`; 242 registered members. Preserve selected exact members, not copy/claim verification of the entire inherited historical bundle.
- Primary source `b1b362fa7516879c8ae37c48c40972b0090680e6bdcb6933fcf9ccb8c2e2b0cd`; secondary `b6dcf473632483d72f5cd449eeef6d36cfd5bc56d2f5c894dbc82fdf8074a7d7`. Both have 240 decoded slots/120 pose samples, fps `30000/1001`; body retains 112 original poses.
- Source rights: Autelica, CC BY 4.0, moving-camera development clip. Not unfamiliar/holdout or independently measured geometry. The sealed run lacks secondary original media/rights metadata; exact secondary camera/evidence/config custody is available. Root approved explicitly retaining this limitation without fabricating rights or treating media absence as automatic candidate failure.
- Found helper clock mismatch before tests: it initially required body `dt=1/fps`. Helper owner fixed it before these tests; helper SHA256 `503a1ff2f7275b5512c9d4241556fd1f3b89775d63570e72992baac5a3de6bc9` preserves the unchanged-body/source-scene clock split.

## Frozen checks and implementation

- `src/weft/camera_shared_candidate.py`: SHA256 `095555cd8af80eaf527b3d0e66367776e71a350ab76f0c3975d6c415fc34fa67`, 28,613 bytes.
- `tests/test_camera_shared_candidate.py`: SHA256 `25f7281e5cfaa120e4388a816304658f691cb58006920f86d3f1c77ecc76aca7`, 22,331 bytes.
- Initial command: `PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_shared_candidate.py --junitxml=.local/camera-shared-candidate-tests-01.xml`. **39 passed**, 0.471 s JUnit duration. Receipt SHA256 `3cb7dbab1696ed0116849de954b3b21747d5634b752c0a6aeb265c6fb4ac8178`.
- Added the three runtime import dependencies and tightened partial-output/failure custody. Focused command: `PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_shared_candidate.py -k 'source_change_before_terminal or late_final_inventory or late_staged_terminal or post_publication or output_tampering or runtime_snapshot' --junitxml=.local/camera-shared-candidate-focused-02.xml`. **6 passed**, 0.181 s; two new and four affected cases, **41 distinct cases** across receipts. Receipt SHA256 `c9eee7a2d7011800a9b1b43873fb530702270941a498d7c63fab93271353a305`. Unchanged passing cases were not rerun.
- Tests use tiny authored package bytes, actual pure camera/observation/evidence validators and a stub shared-world schema boundary. Native/model/media/network/subprocess access and reconstruction functions are forbidden. Test source `source.mp4` bytes are explicitly **not a video**. No physical/contact/geometry acceptance or real-video importer claim.
- Cases cover fractional FPS/body identity, nonzero original start, full source slots, missing/unsampled rows, both views, absent secondary rights/media, original task limits, source tampering, schema/JSON/manifest bounds, symlink/traversal, native-validator rejection, body mutation, source/output races, late final read/staged seal, post-publication consistency and bounded runtime closure. Scoped `git diff --check` passed.

### Independent review correction

`support_ui` found one integration blocker before any source invocation: fixed eight substeps gave `dt/8 = 1001/240000`, above the engine's unchanged `1/240` maximum at fractional source FPS. The mocked shared validator now enforces the real timestep condition. The importer computes `max(8, ceil(240 * dt))`, rejects values above 64, and preserves the original source `dt` and body `dt`; this jogging clock gets nine substeps. No floor, body, camera or task gate changed. Remaining read-only review found no further concrete blocker.

- Reproduction: `PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_shared_candidate.py -k preserves_exact_body --junitxml=.local/camera-shared-candidate-substeps-red-03.xml`. **One passed, two failed as expected**, 0.149 s. The fractional-FPS cases failed at shared schema validation; receipt SHA256 `e3bde37c1c772bdf901ba105fbd941c490df7e2a9df0bf74654800f891841f5a`.
- Same three affected cases after correction, `--junitxml=.local/camera-shared-candidate-substeps-green-03.xml`: **3 passed**, 0.120 s; receipt SHA256 `e5fdedb16012911f2b17013b1e52d6176032000d256ea8348682fef226dda632`. Both receipts retained; no full-suite/native/source rerun. Distinct covered cases remain 41.

### Integration use (not executed here)

```python
from weft.camera_shared_candidate import prepare_camera_shared_candidate

receipt = prepare_camera_shared_candidate(
    "artifacts/two-view-root-jog-lab-01/run",
    "artifacts/camera-shared-candidate-01",  # must not exist
)
```

`status="constructed"` plus `sealed=True` means an immutable source-qualified native candidate only. It does not mean a successful trajectory, ready lab, learned policy, measured ground/contact, feet admission or generalized-video reconstruction. The existing single-leg foot-intersection and six negative jogging foot gaps are unchanged.

## Next safe action

1. Independent read-only review closed after the substep correction; product/test bytes are frozen.
2. Root completed the saved-source/native check below. Continue explicit task/recording/lab wiring; do not execute the example again or repeat completed tests.
3. If review/admission fails, preserve exact failure evidence and address only the concrete defect; never shift root/floor, change source/body dt, loosen task gates or substitute later poses.

Live handles: none. No Git operations by this lane. Existing floor, projection and foot-admission failures remain unchanged.

## Root integration — October 31, 2026, 2:39 PM PDT

One actual saved-source invocation is complete. Candidate
`artifacts/camera-shared-admission-jog-01/candidate` has 54 sealed files/26,145,833 bytes;
manifest SHA `29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`.
It preserves 240 original source slots, 112 body poses, the complete 238-control
reference interval, original 30-Hz body setting and 29.97-Hz source control clock
with nine physics substeps. Native validation used the actual shared engine,
not the pure tests' stub. No body/root/floor/shape/timestamp changes.

The separate zero-physics feasibility audit passed 351 geometry checks
(112 source poses/239 grid poses), with no initial overlap or floor penetration.
The body still floats above the supplied ground; this is not support, balance,
physical accuracy, control/learning success or a ready lab. Calibrated feet remain
unavailable at the original start. No source decoding, inference or paid call.

Frozen `6a29850` plus this exact importer ran in 0.738501 seconds; full worker
0.820989 seconds/parent 1.250118 seconds, observed child peak RSS 202,326,016 bytes.
Exactly 352 native forward evaluations/352 kinematic resets, zero physics steps.
Plans, runtime pins and parent receipt: `.local/camera-shared-source-*01`;
[precise identities and limits](camera-shared-runtime.md#actual-saved-source-admission--october-31-2026-239-pm-pdt).
