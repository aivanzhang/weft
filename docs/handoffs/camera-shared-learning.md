# Camera shared learning

Updated: October 31, 2026, 3:51 PM PDT. Status: independent code/delta review closed; root completed one frozen native training/evaluation study. **The separate saved-output audit is still running; no audit pass is claimed here.** Root owns execution and later lab/recording/capability admission; this lane's follow-up is documentation only. All native handles are terminal. No product/test edit, repeated experiment or Git operation accompanies this update.

## Qualification boundary

The actual `shared-world-v6` scene and `camera-shared-task-v1` remain byte-identical to their inputs. Scene `learned_policy_available=false` and task `scope.learning=unqualified` are not changed. The external `camera-shared-policy-v1` artifact records a bounded research search, not retroactive source qualification. Existing camera recording v1 still rejects policies. Attaching a policy to a lab or claiming learned-action replay needs root's separately versioned admission work.

The saved playable jogging case establishes why these distinctions matter: roots off completed the source interval but failed root tracking (mean 5.884286 model units); explicit roots on already passed under authored control (mean reward 0.826209). That existing pass is not learning. Root's first completed search kept the exact roots-on task and tested whether joint feedback improves under changed starts. It does not claim learned locomotion, reconstructed support/contact, unassisted balance, measured anatomy or physical transfer. The roots-off failure remains evidence; it was not hidden or rerun merely to seek a pass.

## Actor and contracts

`camera_shared_policy.py` exports:

- `validate_camera_shared_policy(policy, scene, task)` and `prepare_camera_shared_policy(policy, env)`.
- `validate_prepared_policy`, for an already prepared context; `make_policy`, `coefficients`, `action_sample`, `selection_kind` and `runtime_hashes`.

The actor profile is `camera-shared-phase-feedback-residual-v1`. Twelve coefficients have shape **2 phases × 3 motor groups × 2 feedback features**, each bounded to ±0.25. Existing phase-basis weights blend the existing torso/proximal/distal position/velocity-feedback corrections. The existing feedback kernel reads one prepared feature sample; final joint correction stays within ±0.25 and actions within ±1. Direct feature bounds reject signed-integer minima before the reused kernel's integer `abs` edge case.

Only the 19 joint channels receive searched corrections. Authored bias feedback and any explicitly enabled three root channels remain unchanged. Object-force channels stay absent. The actor reads the existing 156-value camera observation's elapsed fraction and current privileged simulator feedback, never later reference payloads or future evaluation outcomes. Preparation validates and detaches once; actions do no file hashing, model preparation or source I/O.

The policy pins actual scene/task hashes, all eight camera receipt identities, unchanged gate identity, observation profile, action names, feature/basis/bounds, coefficient values, root authority, training seed/score, runtime source hashes and runtime versions. Its explicit scope remains unqualified research. No legacy policy/task validator receives a rewritten camera scene. The source closure includes the reused pure phase/feedback, bounded writer/budget and packaging helpers; CLI source is deliberately not part of policy identity.

## Search and evaluation APIs

```python
train_camera_shared_policy(
    run, out, *, task_path=None, seed=1729,
    generations=2, population=8, max_seconds=60,
)
evaluate_camera_shared_policy(
    training_dir, out, *, seed=924017, trials=3,
    jitter_scale=.10, max_seconds=60, excluded_seeds=(),
)
verify_camera_shared_training(manifest_raw, files, scene, task, *, deadline)
verify_camera_shared_evaluation(manifest_raw, files, scene, task, *, deadline)
```

Input is a sealed untrained camera run or lab `run/`. Use its exact sealed task or an explicitly supplied task file; there is no generated default task or root-assistance override. Settings retain exact task bytes/hash and `task_authority` (`sealed_source` or `explicit_file`). An overridden original task remains in `input-original-task.json`. Prior training/policy/evaluation input is rejected for training.

Training uses deterministic cross-entropy search (CEM): sample bounded weights, evaluate candidates, then update the sampling mean/spread from the best candidates. Both fixed starts (human jitter 0 and 0.05) are reused across candidates. Zero/authored and previous-best anchors are inside the population and control budget. Exact duplicate coefficients reuse saved trials without new actions. Selection uses the existing original mean-reward/completion score, not changed rewards or added success gates. Both starts must produce observed trial data for a candidate to be selectable; reset failures remain in the history, and no eligible candidate means a sealed failed run with no policy.

Evaluation verifies sealed complete training, freezes exact policy/protocol bytes **before its first reset**, and uses a distinct seed stream and fresh changed states. Coast/random/authored/learned controllers receive the same three starts at jitter 0.10 by default. The original full interval is retained; terminal failures and failed resets are not replaced or removed. No evaluation outcome returns to the optimizer. Pairwise differences versus authored are descriptive values, not a new success definition or a physical/generalization claim.

### Output and evidence

Both stages return the existing-style `directory`, `manifest_sha256`, status and stage-summary envelope. Packages preserve original scene/task bytes, source proof/views, parent manifest, input-task bytes and a bounded source-code snapshot. Runtime/source bytes are checked before publication. Output directories must be new and separate from inputs.

Training adds `settings.json`, `training.json`, `policy.json` and bounded per-trial diagnostic JSONL. Diagnostics retain actual feature samples, phase, authored action, correction, applied action delta, reward and observed clocks. History retains initial/final full snapshots, terminal failures and exact seeds. Results distinguish:

- `selected_feedback_kind`: `authored_unchanged` or `searched_phase_feedback_residuals`.
- `nonzero_coefficients`: searched parameter changes, not proof of effective control.
- `selected_applied_residual`: observed/changed actions and maximum actual post-clipping delta.
- `improvement_over_authored_training`: training-only reward comparison, not independent improvement.

Evaluation adds `pre-evaluation-freeze.json`, `evaluation.json`, exact policy bytes, matched-start results and `paired_vs_authored`. Original training diagnostics stay at their original paths; a few replaced training metadata files use explicit aliases. This preserves verification evidence without copying the entire training package twice.

Saved verifiers check every package member hash, exact source views against the immutable parent manifest, source/task/scorer/runtime pins, deterministic CEM samples/updates/deduplication/selection, trial accounting, actual action deltas, frozen evaluation settings, disjoint/matched initial states and result labels. Required source views come from the immutable parent inventory, including the original-task alias when an explicit task changes its raw bytes. Source binding independently requires every observation document and each pinned source-equivalence manifest's complete member set; deleting files and resealing remaining output cannot erase those obligations. They reuse the root-owned internal prepared snapshot validator after pure camera coupling and pinned task-envelope structural checks. They **do not perform fresh native model admission or replay**, and return `native_model_readmission=false`; structural consistency is not physical validity.

## Bounds and executed native envelope

Per stage: at most 60 cooperative seconds, one worker, 128 MiB/512 output files; diagnostic data is additionally capped at 24 MiB total, 8 MiB/file and 8,192 bytes/line. Training permits at most 8,640 controls; evaluation at most 21,600. Every original interval must fit the bound; no cropping or retiming to make a plan fit. Existing output reservation/failure helpers retain partial trials and failed seals. Deadlines are cooperative, not hard atomic limits. External API cost is $0.

Root executed the frozen study once with training seed 1642079, evaluation seed 918364127, 2 generations × 8 candidates × 2 starts, and 4 controllers × 3 fresh starts over all 238 controls at nine substeps. Predeclared upper envelope: 7,616 training + 2,856 evaluation controls = **10,472 controls / 94,248 native substeps**, before deduplication or early failures. Outer bounds were 165 seconds wall, 140 seconds CPU, 256 MiB and 1,800 files. The worker prepared/resumed a current-runtime untrained lab from the original candidate and exact roots-on task, avoiding stale state-validator runtime pins without changing source/task bytes.

Original roots-on run: `artifacts/camera-shared-play-01/roots-on/lab/run`, seal `12ba0a99514e06f6e8ac8a63895a16c3fc739ec30596c236eed0ddf9e8b6391b`; 308 members/54,746,677 bytes. Exact task bytes SHA `3fefe443f56153c0b5a5d50a7c1c318b1a930a01c93f70c298ca18314f98e558`. Root owns `.local/camera_shared_learning_worker_01.py` and `.local/run_camera_shared_learning_01.py`; do not launch a duplicate study.

Root's metadata-only packaging estimate reused two saved native snapshots: the planned full training-history encoding was 3,528,022 bytes against the existing 4,194,304-byte metadata cap (666,282-byte margin). Receipt: `.local/camera-shared-learning-metadata-estimate-01.json`. This is a packaging estimate, not fabricated training evidence or a guarantee that future dynamics fit every bound.

## Terminal native study — saved audit pending

Root reported the terminal result at `artifacts/camera-shared-learning-01`. This section records that execution report without duplicating the independent audit owned by the support-UI lane.

- Reported elapsed time: **15.288462 seconds**; **637 files / 211,948,220 bytes**; observed peak RSS **407,289,856 bytes**.
- Actual work: **8,954 attempted controls / 80,574 native substeps**. Training used 28 episodes, 6,664 controls / 59,976 substeps in 6.6326 seconds, with two cached candidates. Evaluation retained all 12 planned trials and used 2,290 controls in 6.2078 seconds.
- Selected candidate: generation 1, candidate 4 (zero-based); **12 nonzero coefficients**. Its two original training starts changed **474 of 476 actions**, with maximum applied joint-action delta **0.05182765**. This is a searched, behavior-changing controller, not the unchanged authored anchor.
- Training reset seeds: `2008405795`, `2144681633`. Frozen fresh evaluation reset seeds: `1363504319`, `1081726316`, `1995452935`; the two sets are disjoint.

| Controller | Task success | Retained outcome |
| --- | --- | --- |
| Learned | 3/3 | All full intervals completed |
| Authored | 3/3 | All full intervals completed |
| Coast | 0/3 | All full intervals completed; task criteria failed |
| Random | 0/3 | All three terminal velocity failures retained |

Mean reward increased from **0.8262110146 authored → 0.8273682691 learned** (difference 0.0011572545). Mean joint tracking error was slightly lower, but peak joint error and mean root error were higher. **There was no task-success gain.** This is a small mixed-metric development result with explicit authored root assistance, not evidence of learned locomotion, support recovery, broad generalization or physical transfer.

| Artifact | SHA-256 |
| --- | --- |
| Study seal | `1e77872113906953ffb8603fe6c1a45aea3f79e1710621efc3116faf8bbfd682` |
| Study result | `db78f800129163a90596ba52d5d0649a0e072c824cac04d410b8cc0289bc6920` |
| Process receipt | `b1fa281e3262f20bfc809b79c5a4c4cab1b932a0289846d396e8208b2282e147` |
| Frozen policy | `0729a16239c57c7fbd90aaeb59f0e1dfb23f645edbde387d781008eb4dfdef8a` |

Fresh untrained lab: `artifacts/camera-shared-learning-01/lab` (reported seal prefix `63e55be`). It was prepared and resumed; **the searched policy is still not attachable to this lab or camera recording v1**. No live native handle remains. Next: incorporate the independent saved-audit receipt when released; do not rerun training/evaluation or label the audit passed before that evidence exists.

## Analytical verification and freeze

**37 distinct analytical cases have passing evidence.** Initial 30 passed; a new parent-custody regression plus affected orchestration passed (2 checks); direct signed-integer feature protection then passed its 4 affected actor checks. Independent review subsequently found that deleted source members could disappear from a self-consistently resealed inventory. Four new deletion cases reproduced the defect; the original source and failed receipt remain intact. The corrected expected-inventory check passed 8 focused cases (6 new deletion/bundle cases and 2 affected existing cases); 29 unchanged passes were reused. No native/model/media/network/process entrypoint executed, and no source video was read. Existing source semantics/native admission are deliberately substituted only in authored analytical fixtures; real CEM orchestration, unchanged scorer arithmetic, actual native-free snapshot validation, exact fixture proof-file custody and saved verifiers execute.

```sh
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_policy.py tests/test_camera_shared_learning.py --junitxml=.local/camera-shared-learning-tests-01.xml
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_learning.py -k 'actual_cem or reauthorized' --junitxml=.local/camera-shared-learning-tests-02.xml
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_policy.py -k 'exact_authored or both_phase or coefficients_phase' --junitxml=.local/camera-shared-learning-tests-03.xml
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_learning.py -k 'parent_member_cannot or original_task_alias' --junitxml=.local/camera-shared-learning-custody-red-04.xml
OPENBLAS_NUM_THREADS=1 .venv-lab/bin/python -m pytest tests/test_camera_shared_learning.py -k 'actual_cem or reauthorized or parent_member_cannot or original_task_alias or source_equivalence_requires' --junitxml=.local/camera-shared-learning-custody-green-04.xml
```

JUnit times: 1.536, 0.596 and 0.185 seconds for the initial checks; 0.759 seconds for the preserved four-failure reproduction and 0.729 seconds for the focused green correction. These are analytical test timings, not native training or product-performance measurements. The analytical optimizer selected nonzero weights with realized action changes and improved its authored synthetic dynamics; that proves code-path execution only, not useful learning on the saved jogging case. Checks retain random terminal failures, all failed starts, no-policy behavior, partial action-budget failure, unchanged zero-anchor labels, source partition tampering and no hot-path hashing. The shared state hook was implemented/tested by root; protected task/scorer files were not edited here.

Preserved pre-fix source: `.local/camera-shared-learning-before-custody-04.py`, SHA `8400b9f977051d2c61c03c8cd92b0e231f0c3a255435da08d3c4cf14254a5e5a`. Red receipt SHA `5f49544506cd177df1d441101774f29af8dd0058a202cc835bd2d6528cf0cf44`; green receipt SHA `6f15f2bc997b4546c610f818aff838a02f8dad19be3214b8c10e3ccd7d932d4f`.

| Frozen file | SHA-256 |
| --- | --- |
| `src/weft/camera_shared_policy.py` | `b70aa71e29f6fc08abdfd2bdc80409e9256853b04e072e473c3703a03d96a7ba` |
| `src/weft/camera_shared_learning.py` | `77d47aee9240784db51ca203d546e7d8ca7fc82c2e7e26ceabdbd895cee8500a` |
| `tests/test_camera_shared_policy.py` | `b99821d1a99e30a9c35991d69f8e3804d89f051d4ba04247186e44b375383e8f` |
| `tests/test_camera_shared_learning.py` | `520740928cb338d8fba4b061772aefc0dca7c68aaf6cbcbbe8a4dd56429d3085` |

External analytical receipt: `.local/camera-shared-learning-checks-04.json`. Independent code/delta review closed without a remaining blocker before root's one native study. The terminal study above supersedes the earlier pending-execution status; its separate saved-output audit remains pending. Failed trials and mixed metrics remain part of the result. Do not force a winner, revise gates, repeat inspected starts as a fresh holdout or treat this small reward increase as completion of the broader learning goal. Product policy attachment/recording remains unsupported.

## Independent saved audit — October 31, 2026, 3:56 PM PDT

The saved-only audit passes all 24 checks; no native rerun or exact policy replay
was performed. Receipt `.local/camera-shared-learning-saved-audit-02.json`, SHA
`b3dc8e8a0776b71fc8f146f12bff2cb7f259e0f381c1bb3f86df75acdf071d6e`.
All 637 study files/211,948,220 bytes and 248 frozen-runtime files were hashed once.
The first audit's socket/SSL import-guard error is preserved in receipt 01; it
was not a product failure. Continuation reused those hashes only after 885 file
and 3,035 in-tree ancestor change guards, explicitly rereading 211,948,220 bytes.
It called the frozen evaluation verifier once, reusing its embedded training
verification; actual training bytes, task/source/scorer pins, all starts, failures
and native-worker counters agree. Continuation:1.104827 seconds; both attempts
1.682291 active seconds;356,220,928 bytes observed peak RSS.

All handles are terminal. Current product/test bytes remain the frozen reviewed
versions. Root owns publication, TODO/combined notes and the next separately
versioned policy-bearing lab/recording contract. Never repin old v1 recordings
or present this small, mixed control gain as balance/contact generalization.

## Published checkpoint — October 31, 2026, 3:59 PM PDT

Product/test bytes are published at `34d406ef0a1dca2e77f925182a8a479d2396b48f` with exact private
remote verification. Receipt `.local/camera-learning-publication-01.json`, SHA
`7529b3fb4827c0abdf3de402607adfe34f6cf1f5c28a8ed64fe829751f59b9c6`. This follow-up changes notes only. Owned
paths/index were clean and stash empty; all 1,249 unrelated avatar status entries
were preserved. See [combined next steps](combined.md); no study is live or should
be repeated merely to resume work.
