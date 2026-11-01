# Camera angle-overlay controller trials

## Status and ownership

The camera route is implemented with pure custody and mocked orchestration checks. No camera trial, native model, fit, media render, or physical validation was run in this lane. Root owns the separately frozen end-to-end acceptance and publication.

Owned implementation:

- `src/weft/camera_reference_inputs.py`: distinct selected camera-overlay inputs.
- `src/weft/shared_exercise_reference_inputs.py`: explicit `allow_camera=False` opt-in, shared bounded reader/recheck, unchanged saved geometry checks extracted for reuse.
- `src/weft/shared_exercise_reference_trial.py`: camera dispatch inside the existing matched-trial workflow.
- `tests/test_camera_reference_inputs.py` and `tests/test_shared_exercise_reference_trial.py`.

The controller belongs to `camera_reference_controller.py`; root owns `camera_reference_recording.py`, central recording/lab/CLI dispatch. Neither the protected scorer nor the legacy controller was edited here.

## Input contract

`load_camera_reference_inputs(proposal, scene_run, task_path, *, expected_proposal_manifest_sha256=None, expected_scene_manifest_sha256=None, expected_task_sha256=None, max_seconds=10, max_input_bytes=64<<20)` returns immutable `CameraReferenceInputs`. The generic loader accepts this route only with `allow_camera=True`; its default remains legacy-only and its ordinary read cap stays 32 MiB. A separate explicit `camera_max_input_bytes` override activates only after the completed sealed manifest selects the camera family. `allow_camera=True` alone does not increase the budget.

Camera inputs retain common exact-byte fields plus `angle_proposal_bytes` and read-only `proposal_members`. There is no `candidate_human_bytes` or candidate-human identity alias. The receipt profile is `camera-reference-inputs-v1`; its identities distinguish raw bytes from canonical JSON hashes and include all eight camera task identities.

Required producer evidence:

- A completed `whole-body-reference-candidate-study-v1` seal, `angle-proposal.json`, `report.json`, original human/evidence aliases, frozen camera-mode plan, exact selected surface inputs, and the existing report's five camera proof extras.
- Actual `whole_body_reference_report._model` validation, all fourteen unchanged gates passing, complete saved source and 30 Hz geometry-check schedule, no budget exhaustion, and uninstalled/unpromoted overlay labels.
- Exact original evidence aliases and source/surface camera bytes, plan/manifest/member joins, selected-source proof byte counts, and source body/camera/evidence/root/ground/clock identities.

The original scene must be an actual human-only `shared-world-v6`, not a renamed legacy scene. Its seal is ready or an exact `camera-shared-candidate-v1` constructed package. It must retain `human.json`, `evidence.json`, `camera.json`, `ground.json`, and `camera-coupling.json`; missing views are rejected. The four source files must be byte-identical to the proposal inputs. The coupling copy must match canonically. The explicit camera task must match a sealed task when present and preserve the original full interval, source frame denominator, and control clock. The loader does not invent task defaults or original ancestor proof.

These are selected saved metadata checks, not fresh geometry or full world-source admission. Local seals without supplied expected hashes freeze the local identity; they are not external authentication. The trial still performs independent source admission, current runtime checks, numerical controller preparation, and original scene/task admission before reset.

## One existing execution path

`prepare_reference_trial` authenticates the input family before choosing its runtime and environment. Camera inputs select `CameraSharedTaskEnv`, `camera_reference_action_contract`, and root's camera reference recorder. Legacy inputs keep their existing actor and environment.

Both routes share the same loop: two matched resets, original authored actions versus `reference_proposal`, the same original scene/task/scoring targets, full original interval, retained early failures, two exports, and exact action/state replay. Recipe actions preserve explicit root authority and have no object-force channels. Camera trial results keep `shared-exercise-reference-trial-v1` with `controller_family="camera_angle_overlay"`; this is a workflow distinction, not a new task or scorer. A completed comparison does not mean either controller passed.

The recipe profile is `camera-source-fit-controller-recipe-v1`; proposal events use `controller="reference_proposal"` and `reference_controller_sha256`, never `learned`. Root's recording profile is `camera-shared-reference-recording-v1`. The proposal is an external authored target correction, not an installed body, learned policy, recovered contact, or balance result.

The runtime snapshot unions the existing trial closure with `camera_reference_recording.REQUIRED_RUNTIME_FILES`, `camera_reference_inputs.py`, and `source_surface_fit_inputs.py`. Camera recipes retain every archived Python-basename hash; renderer assets remain separately archived and pinned. No archived code is executed. Root's recorder authenticates that exact map again. Legacy feedback pins are unchanged.

The original producer's selected bytes are copied once through `report.proposal_input_aliases`: common human/evidence/report aliases, `inputs/angle-proposal.json`, and remaining proof/plan/surface members below `inputs/proposal/`. The source run, original task, recipe, selected-input receipt, full source proof, plan, and runtime remain sealed separately from recording journals.

## Bounds and checks

- Selected loader: 10 seconds cooperative, 32 MiB legacy / 64 MiB explicit camera aggregate including exact readback; each selected JSON at most 8 MiB, task 64 KiB, manifests 128 KiB. The trial reserves a third read within the selected family's same aggregate budget. Before family admission its report records the possible 64 MiB ceiling plus both conditional limits; completed legacy routing records 32 MiB. The separate preserved-input copy cap remains 32 MiB.
- Existing trial caps remain: 1–120 seconds, at most 900 actions per controller / 1,800 live controls, 128 MiB source read budget, 16 MiB runtime read budget, 128 MiB per export, 512 MiB / 2,048 output files. Replay is additional bounded work already represented by the existing workflow; these are not hard RSS guarantees.
- No models, native dynamics, source inference, geometry, video decode, network, subprocess jobs, or paid calls occurred in these checks. Native/media/process/network entrypoints are blocked in the new tests; runtime and engine dispatch use explicit stand-ins.

Passing receipts:

```text
.venv-lab/bin/python -m pytest -q tests/test_camera_reference_inputs.py tests/test_shared_exercise_reference_inputs.py::test_exact_snapshots_reuse_saved_validator_without_admission_or_mutation tests/test_shared_exercise_reference_inputs.py::test_saved_geometry_uses_pinned_negative_gap_rule_not_all_gap_nondecrease --junitxml=.local/camera-reference-inputs-tests-01.xml
19 passed: 12 camera checks and 7 affected legacy metadata checks

.venv-lab/bin/python -m pytest -q tests/test_shared_exercise_reference_trial.py --junitxml=.local/camera-reference-trial-tests-01.xml
29 passed: mocked loop/routing, custody, failures, reset/export/replay, runtime dispatch and preserved legacy rejection
```

The legacy actor still pins `shared_world.py` to `5400e2312523061dee4015d177f34a49a74b6f565bd79bfd03c6043b67970c10`; current camera world uses `19c0f07fc9174bd23d086250bb9d5cc3e2c6ded7c1e84612e120d82772d309df`. This pre-existing incompatibility is not repinned or hidden. Mocked orchestration now extracts only the exact pinned pure human velocity selector and explicitly stubs the old runtime-admission boundary. A separate test confirms the actual legacy runtime mismatch is still rejected. These checks do not establish current native legacy admission.

## Real short-video input budget

Root's completed proposal is `artifacts/camera-reference-real-01/fit-02/proposal`, seal `12ed22060e2dc699d26139121b73605d95c31a76625c54817395014eae3df8b0`, bound to the genuine jogging candidate and its unchanged original task. Root reports all fourteen producer gates passed; its outer workflow failed only during offline report packaging. No fitting retry is needed. The earlier run remains failed because a missing explicit body/world mapping did not satisfy the pure fitting-input contract. Empty surface declarations plus the existing camera's explicit coordinate identity resolve that input mismatch without supplying heights, ground kinds or endpoint correspondences. Zero available surface terms remain no evidence of surface improvement.

The exact selected-name size audit is `.local/camera-reference-input-budget-01.json` (SHA `bca17e6cc7a8155ef0fd7c62de6fa018a186271665b5c7164a63c02719d686bd`). It read only 23,750 bytes of manifest/plan JSON and used bounded file metadata for sizes, not a fresh full-package admission:

- 25 selected files: 18,825,425 bytes initially, 37,650,850 bytes for loader plus readback, 56,476,275 bytes including the trial's terminal readback.
- The report is 7,856,096 bytes; the exact original/derived/source evidence aliases are each 1,903,436 bytes. Every original alias and identity check remains required.
- A camera-only 64 MiB aggregate ceiling leaves 10,632,589 bytes for this exact three-read volume. Generic legacy requests, mixed families, over-cap requests and members larger than 8 MiB still fail closed. No source slots or evidence are removed to fit the budget.
- The camera loader now parses each selected named JSON once, avoiding repeated expansion of the large report. Raw aliases remain separately read, pinned and rechecked; this is not cross-file identity substitution.

Focused verification: `.local/camera-reference-input-budget-tests-01.xml` has 21 passes; `-02.xml` has eight passes after the parse-cache/report-limit adjustment. Their union is 23 distinct checks: fourteen new budget/routing cases and nine affected existing metadata checks. Logical reads of the measured 56,476,275-byte volume are tested with a stub reader, not large payload allocation or native work. Actual saved metadata validation uses authored fixtures; two trial-family cases use the existing mocked loop. See `.local/camera-reference-input-budget-checks-01.json` for commands and frozen hashes. No native trial was run in this lane.

## Next safe action

Root should review this budget delta, finish the separate saved-report recovery, freeze the updated input/trial runtime, then run the existing native matched-trial/export/replay path once against the completed real proposal. Do not refit, reuse an unrelated scene, fabricate ancestor seals or relax any gate. Input admission still precedes native models. No live job is owned by this lane; no Git mutation was performed.
