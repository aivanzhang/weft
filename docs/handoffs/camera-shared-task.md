# Camera shared task adapter

Updated: 2026-10-31 14:56:56 PDT. Status: implementation and narrow action-range correction complete; module/tests frozen for root integration. Owner: support-learning lane. No native or source job was run in this lane.

## Ownership and next action

Owned paths are only `src/weft/camera_shared_task.py`, `tests/test_camera_shared_task.py` and this handoff. Root owns the existing engine, protected scorer, recording, lab, CLI and native integration. The reconstruction lane owns the importer. No existing product file or Git state was changed here.

Next: root reviews the frozen adapter, implements strict camera snapshot/recording/lab dispatch and coordinates one bounded native task/recording acceptance. Do not launch another source study or repeat the completed analytical suite without a changed input or specific defect. Native task/recording acceptance remains root-owned and has not run.

## Approved boundary

- Public APIs: `build_camera_shared_task(scene, *, root_assistance=False)`, `prepare_camera_shared_task(task, scene)`, `validate_camera_shared_task(task, scene)`, `CameraSharedTaskEnv`.
- Actual scene remains `shared-world-v6`, with its exact camera/body/ground/source clocks. No legacy scene, profile replacement or camera stripping.
- Initial scored adapter requires `objects=[]`; the engine's static collision support is not silently treated as a qualified equipment task. No feet, object forces, support/contact reward or learned policy.
- Public controls are 19 body joints, optionally 3 explicitly enabled root controls. Default root channels are zero. Observation is the real 155-value human block plus elapsed fraction.
- A distinct `camera-shared-metric-context-v1` pins the unchanged `_ExerciseMetrics` human-only branch selector and minimal kernel arguments. Those arguments are deliberately not a valid legacy task and are never passed to a legacy task validator. The accumulator receives the actual v6 scene; public task, metrics and snapshot profiles remain camera-specific.
- Protected scorer bytes, gates and arithmetic stay unchanged. Envelope pins actual scene, camera coupling, source body/clock, metric context, original scorer and adapter implementation. Read/hash/prepare only during construction, not per frame.
- One real engine supplies immutable substep samples. Preserve full original interval, actual partial-action clocks, stale-boundary failure evidence and existing failure-penalty behavior.
- `cancel()` is idempotent, adds no physics step/time, acts only on an active episode's current observed boundary and cannot relabel terminal success/failure. Cancellation uses the existing failure/penalty semantics. No new learning or training contract is implied.
- Rendering may show the cached actual engine state. It is not proof that strict recording, replay, lab or CLI integration is ready.

## Current evidence and limits

Command:

```sh
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_task.py --junitxml=.local/camera-shared-task-tests-01.xml
```

**43 passed** on the first run; pytest summary 0.33s, JUnit 0.306s, outer command 0.367096834s. These are observed analytical-check timings, not production or native performance claims. Results are pinned in `.local/camera-shared-task-checks-01.json`.

The suite substitutes only the native full-human validation/engine boundary and rendering. It validates the real camera coupling and runs the real protected `_ExerciseMetrics` arithmetic. Fixtures intentionally lack a physics-valid full human model. Native model construction/physics, socket/process entrypoints and camera reconstruction functions are explicitly blocked. No media/model inference calls occur; the suite does not globally instrument every installed media library.

Independent review found that the inherited integer `abs` check overflowed for signed 64-bit minima, allowing an invalid action to overwrite `last_action` before the engine rejected it. The camera adapter now rejects integer values using direct lower/upper comparisons before delegating; the protected lifecycle/scorer remains byte-identical. New 19/22-channel regressions reproduced snapshot corruption (2 expected failures, retained in `.local/camera-shared-task-action-red-02.xml`) and now verify unchanged snapshot, action, clock and active state for integer minima/maxima, booleans, complex and nonfinite inputs. The narrow green selection passed **8 checks**, with 37 unchanged checks reused, not rerun:

```sh
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_task.py -k 'extreme_integer or invalid_actions or one_actual_engine' --junitxml=.local/camera-shared-task-action-green-02.xml
```

Green summary 0.17s; outer command 0.219267708s. Correction receipt: `.local/camera-shared-task-action-checks-02.json`. Preserve the original receipts and failed reproduction. The analytical fixture uses 8 substeps and is not native-admissible at its fractional fps; the actual saved candidate uses 9. Do not promote this fixture to native acceptance.

Coverage includes detached exact identities, selector arguments rejected by the original legacy task schema, unsupported equipment/legacy/policy admission, 19/22 action routing, fractional source/body clocks, every scored value matching the unchanged kernel, failed-gate retention after later recovery, missing/misaligned observations, finite partial failure, nonfinite stale-boundary evidence, reset failures, cancellation before/after actions and all terminal outcomes, invalid actions, cached rendering and no per-step/snapshot hashes or preparation.

Design reads covered the existing task accumulator/lifecycle, reset contract, immutable substep sample and feet adapter pattern. The prior jogging candidate's feasible zero-physics audit is not contact, balance or task-success evidence. There is no live handle, paid call or Git operation.

## Released API and identities

`prepare_camera_shared_task(task, scene)` returns `(actual_v6_scene, detached_task)`. `validate_camera_shared_task` returns only the detached task. `build_camera_shared_task` accepts only an explicit boolean root-assistance choice; it never invents equipment requirements.

Profiles:

| Field | Value |
| --- | --- |
| Task | `camera-shared-task-v1` |
| State | `camera-shared-state-v1` |
| Metrics | `camera-shared-metrics-v1` |
| Observation | `camera-shared-observation-v1` |
| Context | `camera-shared-metric-context-v1` |
| Reset | `camera-shared-reset-v1` |

`implementation_identity()` reads only `camera_shared_task.py` and `shared_exercise_task.py` during preparation. It returns their SHA-256 map and its existing `canonical_hash`. This is an adapter/scoring identity, not the full native runtime closure. Root must separately pin all executed runtime dependencies.

The exact task envelope contains:

- `schema_version`, `profile`, `scene_sha256`.
- `camera_coupling_sha256`, `source_sha256`, `body_sha256`, `source_clock_sha256`.
- `metric_context`, `metric_context_sha256`.
- `kernel_evaluator_sha256` (protected original file), `adapter_implementation_sha256` (new adapter file), `evaluator_sha256` (canonical hash of both file pins).
- Unchanged `gate_profile`, `gates_sha256`.
- Exact full-span `interval`, explicit `control`, honest `scope`.

The context contains its schema/profile, actual scene hash, kernel name and branch, `valid_legacy_task:false`, projection `identity_22_motor_samples_without_equipment`, shared `control_dt_s`, original `source_body_dt_s`, substeps, minimal `kernel_arguments` and their canonical hash. The arguments contain only the human-only branch selector (`shared-exercise-task-v2`), interval, control, empty equipment/contact lists and camera source scope. They lack legacy task schema, scene/gate identities and other required fields; no valid legacy task is created. Input tasks are bounded to 64 KiB serialized JSON and must exactly match the regenerated envelope.

Public metrics retain the kernel's original gates, applicability and `human_tracking_effort_v1` reward metadata, with a distinct camera profile and explicit `kernel_branch`. They add actual scene/outer task/evaluator/context/source pins, camera reset provenance and `observed_source_time_s`. This time is the original source start plus the accumulator's last observed elapsed time; if `clock_status` is unavailable, it does not claim the current native clock is known.

## Lifecycle and recording boundary

- Constructor prepares identities once and creates one actual v6 `SharedWorldEnv`. It never creates a legacy engine or strips `camera_reference`.
- Reset uses the existing `shared-exercise-human-jitter-v1` distribution, with object position/velocity jitter explicitly zero. Reset provenance marks equipment stability not applicable. Failed reset leaves no reusable task accumulator/snapshot, including after a previous successful reset.
- The camera-only integer-range preflight delegates to existing `step`; authored-action and feature routing use contiguous 19/22 human channels. The engine's legacy reward is discarded; public reward comes from actual observed-substep kernel accumulation. Protected gates and formulas are unchanged.
- Snapshot contains its camera profile, actual scene/task/coupling/source/body/clock/context/kernel/adapter/evaluator pins, full actual `engine_state`, distinct `task_metrics`, `last_action`, `engine_state_status`, resolved reset options and task termination flags.
- Finite partial failure retains the actual fractional substep time and current engine state. Runtime exceptions preserve the previous valid engine boundary while the accumulator retains whatever later observations were actually accepted. `engine_state_status="last_valid_control_boundary"` distinguishes this case; the old geometry is not silently presented as current.
- `cancel()` before a valid reset raises. During an active current boundary it sets failure `cancelled`, applies the original failure penalty once, sets task terminated and adds no action/substep/time. Repeating it, or calling it after success, failed gates, runtime failure or another terminal outcome, returns the existing terminal snapshot without changing its label or evidence. Actual engine-state flags are not relabeled; task flags are separate.
- `render()` lazily prepares one cached actual-v6 renderer and passes detached cached engine states. It does not include a new task-result overlay or establish strict recording/replay support. Root must not reuse a legacy task validator or exporter for this shape.

## Frozen pins and remaining work

- Module: `fe877d6e66d4c48bd4aee7042ff95d2ad9f1b27c860cd6ef1179b804bd4bfd99`.
- Tests: `05f375f7af6cadee55e22a8c0f6fb454315adbd3313dfda61b454d574270ba3e`.
- Protected unchanged kernel: `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
- Combined evaluator: `19a19185fb0a9295e374e035d80e6c904917e8100838a950daba864cace11626`.

Native task admission, strict saved-state validation, exact action replay/export, lab/CLI dispatch, policies and equipment/support scoring are not completed by this lane. Root owns the first five integration items. Learning remains explicitly unqualified; equipment and support require a separately versioned contract rather than relaxed gates or silent omission.
