# Human-only exercise task in the local lab

**Status: implemented and locally verified.** This retained design note records the live task and optional frozen-policy contract. The [task guide](shared-exercise.md) provides the current launch, training and evaluation commands; the [equipment guide](equipment-proposals.md) covers one-command assembly from saved evidence. Exact milestone and publication state are in the [handoff](handoffs/combined.md).

New labs explicitly select the human-only task. Existing labs keep their older task; inspection alone never changes the scorer.

## Current basis and review

- Reuse `SharedExerciseTaskEnv`, its frozen gates, distinct recording envelope and offline replay. Do not create another simulator or change scoring criteria for the UI.
- The inspector returns bounded manifest-linked metadata through the authenticated `/api/inspection` endpoint. It leaves source observations, model assumptions and simulator results separate. Read-only review found no new blocking endpoint/path/authentication regression; this is not a general security certification.
- The exercise replay has six passing focused mocked-worker checks in `tests/shared_exercise_replay_smoke.mjs`: task/assistance labels, cumulative metrics, outcome separation, stale failure geometry and recorded controller/policy identity. The generic replay suite covers playback/camera/worker lifecycle. No browser or server was used for this review.
- Reuse `artifacts/source-equipment-integration-04/candidate` as a saved source-linked input, not new perception or physical-validation evidence. The current task interval is shorter than the entire source video; “whole interval” must mean the declared task interval, not the full exercise.

## Explicit selection before launch

The new preparer seals a distinct lab:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/source-equipment-integration-04/candidate \
  --out artifacts/my-human-only-lab-01

# After preparation, use the existing user-started lab launcher.
.venv-lab/bin/python -m weft lab --resume artifacts/my-human-only-lab-01
```

The preparer accepts either `--task PATH` for a validated saved task or `--root-assistance` for an explicitly assisted default task. These options are mutually exclusive. Without either, build the joint-only task with root assistance disabled and no inferred contact requirements. A saved task owns its assistance, complete interval and declared wrist-region requirements.

- Create a new directory; never rewrite an existing lab, task, input or recording. Changing task, assistance, interval or contact requirements requires a new prepared lab. No in-session task mutation endpoint is needed.
- Keep `lab-config.json.profile = "shared-world"`; add `control_profile = "shared_exercise"`. Seal `exercise-task.json` beside the **unchanged** engine `scene.json` and complete source closure. Preserve the old `scene.task`/`task.json`; neither becomes the exercise scorer.
- Verify scene/task identity, source closure and metadata hashes before constructing an environment. Missing, invalid or mismatched task data fails explicitly, never falls back to the legacy session.
- Do not blindly call `prepare_shared_world_lab`: it can rewrite `learned_policy_available` and import a legacy policy. Preserve the pinned scene hash and disable legacy policies through the exercise controller contract, not by editing the scene.
- Preparation copies verified inputs and writes bounded metadata only: no pose/tracking inference, training, provider call, deployment or server launch.

Optional `--learning <sealed-run>` attaches a validated human-only policy to a **new** lab; optional `--evaluation <sealed-run>` attaches its separate complete comparison, including failures. Require exact scene/task/runtime/policy identity. No attachment means unavailable, not passed. Legacy policies cannot be substituted, and existing labs never gain a policy automatically. The ready source example selected the unchanged authored baseline, not improved learned behavior.

## Session and API contract

Only the explicit control profile routes to `SharedExerciseSession`. Authentication, host/origin checks, bounded action batches, single-step serialization and background export remain in place.

| Surface | Required behavior |
| --- | --- |
| Status | Preserve `scene` as the engine scene. Add `control_profile`, `exercise_task` and the authoritative human-only `action_contract`; publish capabilities for the active task. |
| Snapshot | Return the complete `shared-exercise-state-v1` wrapper unchanged. Draw `snapshot.engine_state`; use `snapshot.task_metrics`, wrapper termination flags and `engine_state_status` for task state. Never flatten away either clock or overwrite engine metrics. |
| Actions | Exactly 19 joint channels, or 22 with the three declared root controls, using `env.action_contract()`. Manual vectors, explicit `controller: "coast"`, authored `controller: "reference"`, or `policy: true` only when a validated human policy is installed. No object-force channels or mixed selectors. |
| Rejected actions | Reject object/actor switching, discrete object-force commands, legacy learned policies, wrong-sized/nonfinite vectors and ambiguous controller selectors before advancing physics. |
| Capabilities | One controllable human; equipment remains passive. Expose task action/observation contracts, interval, frozen gate identity and assistance. `learned_policy_available` reflects the validated exercise policy, not the unchanged engine-scene flag. Preserve physical validation as unverified. |
| Reset | Seeded resolved reset options, zero last action, fresh task/engine clocks and accumulators. An advanced coast state cannot be admitted as a start/reset merely because its action is zero. |
| Failure | Retain measured substep time and the last valid geometry's distinct time. Return the task's failure/termination state; do not manufacture a fresh terminal engine pose or substitute the legacy exception envelope. |

The existing `/api/step`, `/api/reset`, `/api/record/start` and `/api/record/stop` remain the control surface. Responses return the same complete snapshot wrapper used by recordings. No task or assistance setting may change during a recording.

## User-facing behavior

1. **Task card:** “Human-only body tracking + passive equipment,” declared interval, root assistance **off** or **three explicit virtual controls**, contact requirements or “none,” and a link to `exercise-task.json`. State that a different task requires a new lab. Existing labs retain their explicit legacy terminal target task label.
2. **Controls:** show the declared 19/22 human motors; hide object selection and force controls. Distinguish manual/coast/authored modes; enable selected-policy mode only for the validated optional policy. Label `authored_unchanged`, `coast_unchanged` or `searched_gains` honestly and disclose simulator/reference features. Training and attached evaluation stay separate; no evaluation is explicitly unavailable. Kinematic source-pose playback remains inspection only and must not generate task scores.
3. **Task result:** show in-progress, passed, not-passed or unavailable from the independent evaluator—not engine target distance, contact-dot count or reward. Null/unmeasured fields are not zero. Early recording stop means an incomplete task interval, not success.
4. **Interval metrics:** show measured/declared duration, valid/observed substeps, body mean/peak errors, peak floor penetration, per-equipment maximum displacement/rotation/speed/spin and each declared wrist-region interval's coverage/gap result. Use bounded summaries and disclosures; do not send source/model arrays into the new panel or rescore them in JavaScript.
5. **Rendering:** reuse the three-view body/equipment renderer but omit the old object-target marker and distance score in exercise mode. Contact dots remain simulated contacts, not automatically qualifying wrist support. Failure prominently labels last-valid geometry beside the later measured failure time.
6. **Recordings:** retain Record & start → Stop & save → replay/video/action links. Pause/navigation never silently closes or discards a journal. Task selection, motor assistance, task hash and source uncertainty remain visible in the saved result.

No-contact task success is tracking/stability only. Neither task mode establishes a completed push-up, grasp, unassisted balance, measured muscles, anatomy or physical transfer. The unobserved equipment prefix and existing source clocks remain unchanged.

## Recording and replay reuse

Use `shared_exercise_recording.recording_scene(scene, task)` with `Recorder(..., profile="shared-exercise")`; record full task wrappers and the exact human action vector. Preserve first/reset seed consistency, resolved reset options, failed attempts, all measured substeps and existing journal/export bounds. Do not emit a `shared-world` recording under a new title.

Reuse the exercise renderer and replay adapter, including their no-target and last-valid-geometry behavior. Record controller intent as manual/coast/authored/learned; policy events retain the exact policy hash and are displayed as recorded policy, not proof of improvement. Old untagged events remain unspecified; start/reset are distinct. Verify saved actions with the matching pinned evaluator through `verify_action_replay`; a new evaluator version is not permission to rewrite old evidence. Interrupted recovery must write a new directory.

## Implementation boundaries

The backend and interface share the fixed wire contract above:

| Lane | Files |
| --- | --- |
| Backend, one owner | New `src/weft/shared_exercise_lab.py`; session/API/record selection in `src/weft/lab.py`; explicit command in `src/weft/cli.py`; task-aware `src/weft/capabilities.py`; new `tests/test_shared_exercise_lab.py`; focused `tests/test_lab_http.py` and capability tests. |
| UI/inspection, one owner | `src/weft/assets/lab.html`, `lab.js`, `lab.css`; optional sealed task-download link in `src/weft/lab_inspection.py`; focused `tests/test_lab_inspection.py` and `tests/lab_ui_smoke.mjs`. No `lab.py` edits in this lane. |
| Existing evaluator/recording owner | `src/weft/shared_exercise_task.py`, `shared_exercise_recording.py`, `recording.py` and their tests. Complete/reset integrity review stays here; UI work must not redefine gates or journal validation. |
| Coordination | Root owns integrated source verification, broader docs and Git. Current ownership is recorded in the [handoff](handoffs/combined.md). |

Generic rendering/export tests are reused; the later optional-policy integration adds per-event controller/hash labels to the existing replay without changing task outcomes.

## Acceptance gates

- [x] **Prepared identity:** new lab seals exact scene/task/source hashes, rejects unsafe/mismatched inputs and legacy policy import, and leaves all inputs unchanged. Resume selects only the explicit new control profile.
- [x] **Control boundary:** direct-session tests and authenticated HTTP checks cover both 19/22-channel modes, valid manual/reference/coast actions, zero object force and rejected selector/actor/policy/vector/override cases without advancement.
- [x] **Independent outcomes:** existing task checks preserve cumulative extrema and success independent of reward. UI checks cover incomplete/pass/fail/unavailable states; the real saved payload displays `equipment_stability:sam-2`, never the engine's target score. Null fields are unavailable, not zero.
- [x] **Reset/record integrity:** advanced-state/reset and first-seed checks pass. The live source study preserves manual/coast/authored tags, an in-recording reset, full wrappers and exact replay across 69 actions/71 events.
- [x] **Failure clocks:** existing task/replay checks retain finite partial-step clocks. Live-lab tests inject unavailable-current-geometry failure, export and replay it exactly; UI distinguishes last-valid geometry from the later task time.
- [x] **UI and legacy:** mocked-DOM and targeted legacy HTTP/recording checks retain explicit task semantics, correct controls, unavailable states and recording lifecycle. No browser/device verification is claimed.
- [x] **Bounded local evidence:** `artifacts/shared-exercise-lab-integration-01` seals preparation → controls/reset → export/action replay → resume; 283 output hashes checked. No inference, paid call or new server. The source and earlier baseline remain unchanged.
- [x] **Optional policy and separate evaluation:** `artifacts/shared-exercise-policy-interaction-01` verifies 68 actions/one reset/70 frames, including 62 policy-tagged actions, exact replay and resume. Actual sealed payload/asset UI checks show the unchanged authored selection, all four 0/3 comparisons and the independent failed live task. All 60 post-reset policy actions/states equal the older authored baseline; no improvement is claimed.
- [x] **Human-only changed starts:** `shared-exercise-human-jitter-v1` retains equipment scene pose/velocity. Fresh `artifacts/shared-exercise-human-jitter-01` starts have zero equipment floor gaps/speeds; idle equipment remains stable. All controllers still score 0/3, and the selected authored weights are unchanged. Old behavior/recordings stay preserved. Changed equipment placement and nominal working-equipment failure remain unresolved.

Receipts: `.local/shared-exercise-lab-backend-01.xml` (initial localhost sandbox bind failure retained), `shared-exercise-lab-http-legacy-02.xml`, `shared-exercise-lab-closure-cache-03.xml`, `shared-exercise-lab-artifact-scope-04.xml`, and `shared-exercise-lab-http-modes-05.xml`. These contain overlapping checks and must not be summed as unique cases. The actual-payload harness is `tests/shared_exercise_actual_payload_smoke.mjs`; it verifies the sealed run's current HTML/JS and real task/source/geometry outputs. Source study seal: `67ad8519c41455592870bf4779ba50c588d4a19635606350088f5da20b07be96`.

Current policy acceptance uses `node tests/shared_exercise_actual_payload_smoke.mjs artifacts/shared-exercise-policy-interaction-02 artifacts/shared-exercise-policy-lab-02`; study seal `f313e300d2288035655ecd188a55a12315482035db5becd52d5824c99a292be5`, lab run seal `acf0fc1665fe99fa7985f5ec8645ab3be5c593b435a99af291f75c9f9a7735a3`. Its 68 actions/70 frames replay exactly; no comparison with the older baseline was requested. The harness checks bounded saved metadata and exact assets, not new simulation or browser interaction.

Preparation bounds: 512 input files/128 MiB, 1,024 output files/192 MiB, 64 MiB per file, 16 MiB runtime snapshot, cooperative 60-second default deadline. Export uses the existing worker with a 180-second cooperative limit and 128 MiB guard; current core/runtime mismatches fail before simulation.

**Completion:** the user explicitly selects this task, sees its true controls and whole-declared-interval result, and saves a verifiable replay in the existing lab. The older task and saved evidence remain unchanged; physical reconstruction and exercise success remain separate unverified goals.
