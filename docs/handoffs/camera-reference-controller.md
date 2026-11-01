# Camera reference-controller adapter

Updated: 2026-10-31 19:02 PDT.

## Status and ownership

Pure adapter implemented and analytically checked. It is not yet integrated into lab/trials/recording or run in a native simulator. Root owns that integration and later native acceptance; support_learning owns the camera source loader and reference-trial adapters.

Only this handoff, `src/weft/camera_reference_controller.py`, and `tests/test_camera_reference_controller.py` changed in this lane. No Git operations or live jobs.

The overlay is a non-learned control alternative, not a replacement human model. Original scene, task, scorer, body, camera-reference hashes, roots, ground and public controls stay unchanged. Contact, balance and physical accuracy remain unverified.

## Interfaces

- `camera_reference_action_contract(scene, task)` derives the existing 19-joint / optional 3-root contract with a camera-profile guard; it does not admit a scene.
- `prepare_camera_reference_controller(original_human_bytes, angle_proposal_bytes, report_bytes, *, scene, task, action_contract, runtime_sha256, reference_velocity_index)`
- `prepare_camera_reference_recipe(recipe, *, scene, task, action_contract, runtime_sha256, reference_velocity_index)`

Both factories return the existing immutable `shared_exercise_reference_controller.ReferenceController`: its actual `target_delta`, `action`, `describe` and `recipe` methods are reused. Neither legacy factories nor their allowlists changed. The 40/5 feedback kernel and pre-clipping correction math are not duplicated.

Controller profile: `camera-source-fit-controller-v1`. Recipe: `camera-source-fit-controller-recipe-v1`. Archived interpretation: `archived_execution_only_not_source_fit_admission`.

Provenance uses `angle_proposal_sha256` / `angle_proposal_canonical_sha256`, never candidate-human aliases. Description/recipe retain scene/task/gates/action hashes, all eight camera-task identities, raw body/evidence/report/overlay identifiers, full runtime map, feedback gains and clock binding. Description adds `recipe_sha256` and `controller_implementation_sha256`.

## Caller boundary

The caller must authenticate source/proposal seals and current runtime bytes, validate the actual v6 scene/task, and supply that runtime's trusted `human.reference_velocity_index`. Hash declarations or arbitrary callbacks are not authentication.

The exact body bytes are joined to original scene and overlay/report pins. Evidence/camera raw hashes are declarations here: this API does not receive or authenticate those files. The loader must preserve them and verify original-camera versus supplied-fit-camera equality.

`REQUIRED_RUNTIME_FILES` is the Python closure. Runtime keys are bounded Python basenames; all supplied entries are retained and archive replay must match the entire map. `RUNTIME_SHA256` fixes reviewed human, shared-world, shared reference-controller, protected scorer and camera-task bytes. Current v6 world pin is `19c0f07fc9174bd23d086250bb9d5cc3e2c6ded7c1e84612e120d82772d309df`, intentionally not the legacy factory's old world pin. Other closure bytes and runtime versions remain caller-authenticated.

Archived recipes validate execution bounds/original context, not source-fit qualification again. Raw report/overlay file identifiers remain opaque archived identifiers; changing an opaque identifier can stay structurally valid. Only the external recording seal authenticates its historical meaning.

## Preserved contracts and bounds

Fresh preparation requires all 14 unchanged camera proposal gates, original limits, complete passing/unpromoted status, exact overlay/report/camera/surface joins, original first reference and complete source times. Actual correction, rate and joint limits are checked independently: 0.15 rad / 1.5 rad/s with the existing 1e-8 tolerance.

Original video period, body timestep and physics-substep period stay separate. Fractional FPS and nonzero source start are preserved, with no resampling to body.dt. Full horizon required; roots are never corrected; no object forces. Unsampled and unavailable decoded-tail rows remain unknown, not zero targets.

Caps: body/overlay/scene 8 MiB each, report 16 MiB, task/runtime map 64 KiB each, action contract 16 KiB, recipe 1 MiB; 900 references, 1800 source slots, 128 runtime entries. Preparation has a cooperative 5-second bound, not hard RSS/native-call preemption. Product code performs no file I/O, fitting, native/model/media/network/process work. Actions use prepared immutable tuples.

## Evidence

Receipt: `.local/camera-reference-controller-checks-01.json`. Complete stdout/stderr is retained in each matching `.log`.

```sh
.venv-lab/bin/python -m pytest -q tests/test_camera_reference_controller.py -o junit_family=legacy --junitxml=.local/camera-reference-controller-tests-01.xml
.venv-lab/bin/python -m pytest -q tests/test_camera_reference_controller.py -k 'source_clock or corrections_use or camera_metadata or input_byte or archived_recipe' -o junit_family=legacy --junitxml=.local/camera-reference-controller-tests-02.xml
.venv-lab/bin/python -m pytest -q tests/test_camera_reference_controller.py -k 'not rejects_changed and not scene_task' -o junit_family=legacy --junitxml=.local/camera-reference-controller-tests-03.xml
```

**33 distinct cases passed**, not 47: initial 31 passes; six affected/new passes after strengthening metadata-pin/pre-clip checks; final ten positive/new passes using the exact pinned camera task-envelope producer extracted by AST. The 23 earlier rejection cases are reused; later product changes add validation, while the equivalent task fixture was replaced by the actual producer. No test failures. Earlier intermediate source snapshots were not archived; do not label initial31 as a full run against final bytes.

The fixture uses actual pure coupling validation, the pinned velocity selector and only the pure task `_envelope` function with unchanged gate constants. Native scene admission is deliberately absent. Native/fitting/media/network/process entrypoints are blocked.

Final repeated-action check: **1000 mocked actions in 0.020020792 seconds**, with JSON parsing, reads and coupling rebuild blocked on the action path. This is Python-provider work, not native simulation/browser/physical performance.

Frozen source pins:

- Product: `05907eb0ce4285d8e578cee2b5279b792978acf152e01608e8c5c24149a599b3` (30,083 B).
- Tests: `848d161adadf8401a272a3da138ebc9c17965da2ef602c62caf907ad29675d4e` (23,452 B).

Two tool-format mistakes were rejected before filesystem changes (initial test patch formatting; handoff string delimiter). Neither was a product/test failure.

## Next safe action

Root integrates the existing camera task/lab/trial/recording path with exact recipe/runtime authentication, then performs one bounded native acceptance after caller-side contracts freeze. No separate simulator, task/scene rewriting, learned-policy promotion or body replacement is implied.
