# Split arm/leg feedback: saved policy and lab

## Status and ownership

Implemented and frozen for root review; **26 focused analytical/scripted cases pass**. No native training, physics replay, browser or server ran in this lane. Root owns the trainer, CLI, workflow and any separately frozen native acceptance. The actor contract is documented in [split-limb-feedback-actor.md](split-limb-feedback-actor.md).

Owned product paths are `shared_exercise_policies.py`, `shared_exercise_lab.py` and `assets/lab.js`. New tests are `tests/test_split_limb_feedback_lab.py` and `tests/lab_split_feedback_policy_smoke.mjs`. No avatar, scorer, recording implementation, receipt schema or existing test helper was edited. No Git mutation or active job remains.

## Contract and behavior

- Explicit `shared-exercise-policy-v5` uses the existing feedback validator, prepared actor, saved selector and evaluator. No shape-based profile inference or legacy profile reinterpretation is introduced.
- V5 separates torso, shoulder, elbow, hip and knee feedback into five groups with ten bounded settings. The same simulated position/velocity errors are used; joint corrections stay within ±0.25. Any permitted root assistance remains programmed. Object-force controls are not added.
- Exact zero coefficients remain `authored_unchanged`; nonzero coefficients remain `searched_feedback_residuals`. The UI labels the latter **Separate arm/leg feedback**, not learned balance or verified improvement.
- Person-only labs accept the explicit v5 policy alongside their existing v2/v4 contracts. Complete saved feedback proof is required even for zero coefficients, with the caller's existing deadline. Scene/task/source/runtime/evaluator/gate pins, all copied learning evidence and exact policy bytes remain mandatory.
- `training_feasible` is separate from changed-start evaluation and the current trial. Passing training starts cannot overwrite a displayed independent 0/3 result. Missing or inconsistent summary fields keep learned control unavailable while manual/coast/authored controls retain their prior behavior.
- Existing camera, feet and static-support routes are not broadened. They retain their separate qualified task/policy contracts. No automatic attachment to an old lab or runtime compatibility exception was added.

The existing recorder already delegates policy validation and preparation through the common dispatcher. Its exact policy envelope, learned action checks, reset handling and full-state replay protocol therefore need no new recording profile. The new tests exercise that real dispatch with a **mocked engine**, including wrong policy pins, wrong actions and cleanup; this is not evidence of real physics replay.

Actual runtime hashes change with the actor/trainer/dispatcher/lab bytes. Historical policies and saved labs stay immutable and may correctly reject a current runtime. They are not silently repinned. V4 numerical/default-byte compatibility is covered by the actor lane's fixed-identity golden, not by claiming old runtime hashes remain current.

## Public path

Root's opt-in family is `feedback-residual-split-limbs`; the existing `feedback-residual` default is unchanged. The current public routing uses:

```sh
.venv-lab/bin/python -m weft shared-exercise-train \
  --run EXACT_SHARED_RUN --task EXACT_TASK_JSON \
  --policy-family feedback-residual-split-limbs --out NEW_TRAINING
.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning NEW_TRAINING --out NEW_EVALUATION
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run EXACT_SHARED_RUN --task EXACT_TASK_JSON \
  --learning NEW_TRAINING --evaluation NEW_EVALUATION --out NEW_LAB
```

These are syntax pointers, not runs performed here. The root-owned workflow also exposes `--training-policy-family feedback-residual-split-limbs`. Use only a compatible exact scene/task and new output directories. Do not infer improvement from the new family name or nonzero coefficients.

## Verification and evidence

Receipt: `.local/split-limb-feedback-lab-checks-01.json`, SHA `f6476cd0321d327e5e97bc259863c0f176882ecf07133d73daba1fd6f67441d5`. It retains the exact commands, product/test hashes and result pins.

- Python run 01: **14 passed**, 0.688 seconds pytest; `.local/split-limb-feedback-lab-tests-01.xml`.
- Python run 02: **3 additional cases passed**, 0.205 seconds; `.local/split-limb-feedback-lab-tests-02.xml`. The earlier 14 were not repeated.
- New scripted UI: **5 passed**, `.local/split-limb-feedback-ui-tests-01.log`.
- Affected existing v4 UI: **4 passed**, `.local/split-limb-feedback-ui-legacy-01.log`.
- **200 unchanged mocked status updates: 22.304542 ms**, zero policy-label writes, unchanged joint-selector nodes, exactly 200 status requests and no proof-file fetches. This measures the scripted path, not browser performance.

Cases cover coupled/person-only policy metadata and 19/22 controls; full proof dispatch and rejection; zero baseline meaning; resealed coefficient/group/label/runtime tampering; prepare/resume exact copies; bounded saved select/evaluate routing; real prepared actor action checks through mocked recording replay; v4/v2 summary compatibility; recording/reset serialization; invalid summaries; and training-versus-evaluation labels.

Model/data construction, stepping, rendering, media, networking and child processes are denied by the Python fixture during test execution. Mass and floor admission metadata are explicitly authored mocks. XML and complete Node logs are preserved; full Python stdout logs were not separately saved. No failures occurred in this lane's runs. No external spend was incurred.

### Frozen pins

| Path | SHA-256 |
| --- | --- |
| `src/weft/shared_exercise_policies.py` | `2b839f6d3840d8867ffd0bce0c72fabb39cfb40853613efbfe429b4960fc8875` |
| `src/weft/shared_exercise_lab.py` | `63d53ef7a07a63210668fed143e08d5bc65c85fc898709348ffa38bf895f590b` |
| `src/weft/assets/lab.js` | `5c5d62bbd396296b88627b9577ea3c170ee1927f7c0ce425668c24e3f590727f` |
| `tests/test_split_limb_feedback_lab.py` | `30ab98f3528605590b43a0a4994b6b6eacb7e7960f75d9fff8ad3c6fcce0cd6e` |
| `tests/lab_split_feedback_policy_smoke.mjs` | `0e4c48adecc2e41d54fb0a962d3515fd0a2f4f77c7aade3b3d52105083058114` |

Next action belongs to root: independently review the released integration, freeze a coherent complete runtime and run one authorized bounded end-to-end acceptance. Reuse these checks unless a concrete change or failure requires an affected rerun. Report actual task outcomes independently; no unfamiliar-video or physical-transfer claim follows from this UI/dispatch change.
