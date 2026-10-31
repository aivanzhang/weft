# Camera shared recording

Status: analytical implementation complete and frozen, 2026-10-31 15:07:42 PDT. Root independently reviewed the implementation without a blocker and owns common recording/receipt/lab/CLI/UI dispatch plus the combined native acceptance. This lane has not run native physics, source/media decoding or reconstruction.

Owned new files: `camera_shared_state.py`, `camera_shared_recording.py`, `tests/test_camera_shared_recording.py` and this handoff. The released task/scorer/engine are frozen and are not edited here.

The wrapper profile is `camera-shared-recording-v1`; it retains the actual v6 scene and camera task without reclassification. Strict state checks independently bind all camera/context/evaluator pins. The reusable `shared_exercise_feet_state._prepare_legacy_metrics_validator` receives only an internal validation-metadata view of the existing human-only kernel arguments. That view is never saved, replayed or passed as a legacy task/scene. Real engine state and public camera metrics remain authoritative.

Root approved the additional runtime pins `shared_exercise_feet_state.py`, `shared_exercise_feet_projection.py`, `shared_support_task.py` required by that helper's imports. No foot geometry or foot task projection executes. Snapshot/event limits are 480/512 KiB, with the existing 128 MiB journal cap unchanged. Preparation freezes camera identities and reuses the actual shared snapshot validator with its prepared coupling hash; per-frame validation does not rehash source modules, revalidate the scene or compile a model. Replay accepts a positive cooperative budget up to 180 seconds, not a hard execution deadline.

Cancellation remains task lifecycle only: no new journal event and no cancel snapshot disguised as a step. User stop closes the journal without invoking task cancellation. Policies, avatars and unsupported controllers are rejected. Replay returns the eight `camera_shared_receipt.IDENTITIES` pins, actual scene/task identity, null policy/learned-action fields and protocol `camera-shared-full-state-replay-v1`. The offline view exposes the same ten pins in `camera_identity`, without substituting the internal kernel-selector identity.

## Public interfaces

- State: `validate_camera_shared_snapshot` and `prepare_camera_shared_snapshot_validator`.
- Recording: `recording_scene`, `validate_recording_scene`, `validate_action`, `validate_snapshot`, `prepare_snapshot_validator`, `validate_event`, `replay_view`, `prepare_renderer`, `render_snapshot` and `verify_action_replay`.
- The wrapper retains `engine_scene`, `exercise_task`, `dt` and an empty `objects` list. Public actions remain 19 body rotations or an explicitly enabled 22 channels including roots. Recorded controllers are only manual, coast and authored reference.
- Strict state checks retain actual engine state, public camera metrics, partial/stale failure boundaries, observed source time, reset provenance and terminal flags. Camera priors and tracking diagnostics do not establish contact, balance or measured anatomy.

## Verification and preserved failures

All 18 distinct analytical cases have passing evidence. The first run passed 12 and failed 6 because the analytical engine fixture emitted NumPy scalars instead of the real engine's JSON-native floats. Only fixture serialization changed; product validation was not relaxed. The six affected cases then passed, while the 12 unchanged passes were reused. Both result files remain intact.

```sh
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_recording.py --junitxml=.local/camera-shared-recording-tests-01.xml
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_recording.py -k 'actual_v6_scene or camera_action_clock or late_recovery or (partial_stale and (finite_failure or bad_clock))' --junitxml=.local/camera-shared-recording-tests-02.xml
```

JUnit times were 0.498 seconds for the first run and 0.162 seconds for the affected-only run. These are analytical check timings, not product responsiveness measurements. The checks execute the real native-free shared snapshot validator and reused metric-validation algebra. Native scene preparation, physics, recording loading and image rendering use analytical stand-ins; network/process/native entrypoints are blocked. They do not prove a saved package or physics model works end to end.

Coverage includes exact camera/source/metric identities, 19/22-channel action routing, immutable views, reset/event admission, preserved finite/stale/bad-clock failures, failure retained after apparent recovery, cancellation not recordable as a step, output bounds and cached hashing. Mocked replay verifies six events, four actions and one reset in each root mode, returns all required pins, and closes its environment on success, mismatch or deadline. The new fixture uses nine substeps at fractional source frame rate; it remains analytical geometry, not a native model fixture.

| Frozen file | SHA-256 |
| --- | --- |
| `src/weft/camera_shared_state.py` | `8ba8190e7fa0a5ebacaa9d6d2ac5a886d589970434a1a107078e8ad9c7a03db4` |
| `src/weft/camera_shared_recording.py` | `96463fe05780532eaffe18a8141076d29fd2abd00ba5f3057650bad72fd4b730` |
| `tests/test_camera_shared_recording.py` | `073d0e315256915aaecc9538551f1c897c4a529322f1a43dce2047ef896613ad` |
| `.local/camera-shared-recording-tests-01.xml` | `d071968f17bfdc564a21f679c39535a790279d8022031ce0c332d92746378b3d` |
| `.local/camera-shared-recording-tests-02.xml` | `f06557e818c7f9485ee239445f37658cdc1b89147659717a34a900a3ee4aa356` |

Machine-readable evidence: `.local/camera-shared-recording-checks-01.json`. No live handle, paid call, native/source job or Git operation exists in this lane. Root owns checkpointing the released files and the next combined native recording/lab acceptance against the saved original-horizon jogging candidate; do not duplicate that run or modify released code while it runs.
