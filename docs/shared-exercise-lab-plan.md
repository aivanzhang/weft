# Human-only exercise task in the local lab

**Status: planned, not implemented.** Recorded 2026-10-29 22:34 PDT. This is the next integration boundary agreed with the backend review; it does not change the published milestone or existing labs.

The [current task guide](shared-exercise.md) describes the implemented CLI/Gym evaluator and recording path. The [local lab](local-lab.md) still uses the older coupled body/target-object task. Its five-stage inspector does not activate the new scorer.

## Current basis and review

- Reuse `SharedExerciseTaskEnv`, its frozen gates, distinct recording envelope and offline replay. Do not create another simulator or change scoring criteria for the UI.
- The inspector returns bounded manifest-linked metadata through the authenticated `/api/inspection` endpoint. It leaves source observations, model assumptions and simulator results separate. Read-only review found no new blocking endpoint/path/authentication regression; this is not a general security certification.
- The exercise replay has four passing focused mocked-worker checks in `tests/shared_exercise_replay_smoke.mjs`: task/assistance labels, cumulative metrics, outcome separation and stale failure geometry. The generic replay suite already covers playback/camera/worker lifecycle. No browser or server was used for this review, and unchanged suites were not rerun.
- Reuse `artifacts/source-equipment-integration-04/candidate` as a saved source-linked input, not new perception or physical-validation evidence. The current task interval is shorter than the entire source video; “whole interval” must mean the declared task interval, not the full exercise.

## Explicit selection before launch

Add a **new preparer**, proposed command below; it does not exist yet:

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

## Session and API contract

Route only the explicit control profile to a new `SharedExerciseSession`. Keep all existing authentication, host/origin checks, bounded action batches, single-step serialization and export executor behavior.

| Surface | Required behavior |
| --- | --- |
| Status | Preserve `scene` as the engine scene. Add `control_profile`, `exercise_task` and the authoritative human-only `action_contract`; publish capabilities for the active task. |
| Snapshot | Return the complete `shared-exercise-state-v1` wrapper unchanged. Draw `snapshot.engine_state`; use `snapshot.task_metrics`, wrapper termination flags and `engine_state_status` for task state. Never flatten away either clock or overwrite engine metrics. |
| Actions | Exactly 19 joint channels, or 22 with the three declared root controls, using `env.action_contract()`. Manual, zero/coast and authored human feedback are allowed; no object-force channels. |
| Rejected actions | Reject object/actor switching, discrete object-force commands, legacy learned policies, wrong-sized/nonfinite vectors and ambiguous controller selectors before advancing physics. |
| Capabilities | One controllable human; equipment remains passive. Expose the task's action/observation contracts, interval, frozen gate identity and explicit assistance. Learned policy unavailable. Preserve physical validation as unverified. |
| Reset | Seeded resolved reset options, zero last action, fresh task/engine clocks and accumulators. An advanced coast state cannot be admitted as a start/reset merely because its action is zero. |
| Failure | Retain measured substep time and the last valid geometry's distinct time. Return the task's failure/termination state; do not manufacture a fresh terminal engine pose or substitute the legacy exception envelope. |

The existing `/api/step`, `/api/reset`, `/api/record/start` and `/api/record/stop` remain the control surface. Responses return the same complete snapshot wrapper used by recordings. No task or assistance setting may change during a recording.

## User-facing behavior

1. **Task card:** “Human-only body tracking + passive equipment,” declared interval, root assistance **off** or **three explicit virtual controls**, contact requirements or “none,” and a link to `exercise-task.json`. State that a different task requires a new lab. Existing labs retain their explicit legacy terminal target task label.
2. **Controls:** show the declared 19/22 human motors; hide object selection and force controls. Keep manual and authored-follow modes visibly distinct; disable learned mode both in UI and backend. Kinematic source-pose playback remains inspection only and must not generate task scores.
3. **Task result:** show in-progress, passed, not-passed or unavailable from the independent evaluator—not engine target distance, contact-dot count or reward. Null/unmeasured fields are not zero. Early recording stop means an incomplete task interval, not success.
4. **Interval metrics:** show measured/declared duration, valid/observed substeps, body mean/peak errors, peak floor penetration, per-equipment maximum displacement/rotation/speed/spin and each declared wrist-region interval's coverage/gap result. Use bounded summaries and disclosures; do not send source/model arrays into the new panel or rescore them in JavaScript.
5. **Rendering:** reuse the three-view body/equipment renderer but omit the old object-target marker and distance score in exercise mode. Contact dots remain simulated contacts, not automatically qualifying wrist support. Failure prominently labels last-valid geometry beside the later measured failure time.
6. **Recordings:** retain Record & start → Stop & save → replay/video/action links. Pause/navigation never silently closes or discards a journal. Task selection, motor assistance, task hash and source uncertainty remain visible in the saved result.

No-contact task success is tracking/stability only. Neither task mode establishes a completed push-up, grasp, unassisted balance, measured muscles, anatomy or physical transfer. The unobserved equipment prefix and existing source clocks remain unchanged.

## Recording and replay reuse

Use `shared_exercise_recording.recording_scene(scene, task)` with `Recorder(..., profile="shared-exercise")`; record full task wrappers and the exact human action vector. Preserve first/reset seed consistency, resolved reset options, failed attempts, all measured substeps and existing journal/export bounds. Do not emit a `shared-world` recording under a new title.

Reuse the exercise renderer and replay adapter, including their no-target and last-valid-geometry behavior. Verify saved actions with the matching pinned evaluator through `verify_action_replay`; a new evaluator version is not permission to rewrite old evidence. Interrupted recovery must write a new directory.

## Non-overlapping implementation ownership

These are proposed next-work lanes, not authorization to race current source edits.

| Lane | Files |
| --- | --- |
| Backend, one owner | New `src/weft/shared_exercise_lab.py`; session/API/record selection in `src/weft/lab.py`; explicit command in `src/weft/cli.py`; task-aware `src/weft/capabilities.py`; new `tests/test_shared_exercise_lab.py`; focused `tests/test_lab_http.py` and capability tests. |
| UI/inspection, one owner | `src/weft/assets/lab.html`, `lab.js`, `lab.css`; optional sealed task-download link in `src/weft/lab_inspection.py`; focused `tests/test_lab_inspection.py` and `tests/lab_ui_smoke.mjs`. No `lab.py` edits in this lane. |
| Existing evaluator/recording owner | `src/weft/shared_exercise_task.py`, `shared_exercise_recording.py`, `recording.py` and their tests. Complete/reset integrity review stays here; UI work must not redefine gates or journal validation. |
| Coordination | Root owns checkpoint, broader docs and Git. This design note is the only file edited in the current review phase. |

Freeze the wire fields above before the backend/UI lanes implement independently. Reuse current generic rendering/export tests; change replay assets only for a demonstrated integration gap.

## Acceptance gates

- [ ] **Prepared identity:** new lab seals exact scene/task/source hashes, rejects unsafe/mismatched inputs and legacy policy import, and leaves all inputs unchanged. Resume selects only the explicit new control profile.
- [ ] **Control boundary:** authenticated HTTP checks cover both assistance modes, valid 19/22-channel manual/reference actions, zero object force, rejected selector/actor/policy/vector cases and no physics advancement after rejection. Task/capability labels agree with the actual environment.
- [ ] **Independent outcomes:** matching engine target scores cannot pass a failing exercise interval. Test incomplete → complete pass, complete gate failure, partial engine failure and unavailable metrics; null data never becomes zero or passing. Previously failing maxima remain visible after equipment returns to its initial position.
- [ ] **Reset/record integrity:** current evaluator-owner reset-admission and first-seed fixes are verified before integration. Start/reset requires genuinely fresh clocks/accumulators; record/reset/stop/export preserves exact task wrappers and seed options. Exercise action replay reproduces every task/engine state and result.
- [ ] **Failure clocks:** finite partial-step failure and unavailable-current-geometry failure each preserve actual measured duration/substep counts. UI, video and replay distinguish stale geometry time; saving a failed or early-stopped trial does not fabricate completion.
- [ ] **UI and legacy:** mocked-DOM checks show correct motor count, immutable assistance, passive equipment, no old target/score, safe missing states, no hidden learned mode and preserved pause/record/export behavior. Existing generic shared-world, standalone exercise and saved legacy replays retain their task semantics.
- [ ] **Bounded local evidence:** one new saved-candidate lab/trial demonstrates preparation → manual/authored interaction → recording → export/action replay without inference or paid calls. Record hashes, exact commands, failures and representative cold/cached-summary plus action/record latency. Do not repeat unchanged broad suites or claim browser/sustained-use performance from local timings.

**Completion:** the user explicitly selects this task, sees its true controls and whole-declared-interval result, and saves a verifiable replay in the existing lab. The older task and saved evidence remain unchanged; physical reconstruction and exercise success remain separate unverified goals.
