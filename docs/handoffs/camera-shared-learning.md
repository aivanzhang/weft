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

## External-policy attachment — analytical implementation released

The existing study policy is now supported by a **distinct integration contract**.
Native lab acceptance and the independent saved-output audit passed; exact
terminal evidence appears below. Do not retrain or repin study01. Its original policy SHA remains
`0729a16239c57c7fbd90aaeb59f0e1dfb23f645edbde387d781008eb4dfdef8a`.
The actual v6 scene still has `learned_policy_available=false`; the original v1
task still says `scope.learning=unqualified`. Admission enables an external
research controller in the lab, not reconstructed contact, balance, anatomy,
source qualification or physical validation. Original camera recording-v1 still
rejects policies and is byte-unchanged.

### Ownership and stable APIs

- This lane owns new `camera_shared_policy_admission.py`,
  `camera_shared_policy_recording.py`, their two new test files, the minimal
  saved-verifier parameter changes in `camera_shared_learning.py`, and this note.
- Root owns common lab/CLI/recording/receipt/capabilities dispatch and new
  `camera_shared_policy_lab.py`, including evidence packaging, resume and the
  separately bounded native acceptance. No native handle is active in this lane.
- The presentation lane owns new stdlib-only `camera_shared_policy_receipt.py`,
  `camera_shared_capabilities.py`, UI presentation and their focused checks.

Admission exports:

```python
prepare_camera_policy_admission(
    scene, task, *, training_manifest_raw, training_files,
    evaluation_manifest_raw=None, evaluation_files=None, deadline,
)  # complete saved evidence verification; no new native/source admission
validate_camera_policy_admission(admission, scene, task, policy_document)
prepare_camera_policy_sampler(policy_document, admission, env)
execution_runtime_hashes(scene)
```

The sampler freezes original coefficients/groups once and returns
`(action, {phase, authored_action, applied_joint_delta})`. `applied_joint_delta`
contains 19 exact post-clipping differences from the authored action; root
channels, when explicitly enabled, remain authored and unchanged. Sampling uses
one existing observation and one existing feedback-feature read, with no runtime
file reads, source hashes, model preparation or future observations. Nonzero
coefficients and realized action changes are distinct from improvement.

### Runtime bridge and historical evidence

Profile `camera-shared-policy-admission-v1` binds actual scene/task and all eight
camera identities, exact original policy bytes, training/evaluation manifest
hashes, original and execution runtime maps, runtime versions, compatibility,
the coefficient-derived selection label, explicit root authority and
`unqualified_research`. The canonical admission hash is the public identity;
the lab manifest separately hashes the exact saved JSON bytes. No creation time
enters admission, allowing exact regeneration during resume.

Every original actor/physics/scorer/source dependency must still match. Only
`shared_exercise_lab.py`, `recording.py`, and the saved-verifier-only change to
`camera_shared_learning.py` may differ. The admission records exact old/current
hash pairs; every original implementation remains in the retained evidence.
New execution modules are additions, never fabricated original hashes. The
execution closure adds admission, policy recording, policy receipt, policy lab,
camera capabilities, common receipt, `lab.py`, and common capabilities. It uses
the shared receipt helper's exact original/execution sets without recursively
calling the common runtime dispatcher. Runtime reads are bounded to 96 regular
files, 2 MiB per source, 16 MiB total.

The only learner edits add optional `historical_runtime_sha256=None` to the two
saved verifiers, select that map for saved training verification, and forward it
from saved evaluation. Defaults remain current-runtime strict. Training,
evaluation, actor, gates and scoring arithmetic are unchanged. Admission first
compares the **entire** historical/current learner AST, normalizing only those
four declared plumbing changes. Other syntax changes reject. No historical
source is executed. The actual retained study source `77d47aee…` was compared
with current `b9d0a05a…`; normalized AST SHA:
`d83c6d85295843edb9d066d2701462d62206ccb8e2ba33de0dc2e20a070164ce`.

When evaluation is supplied, its exact embedded training inventory/bytes must
match the requested training, then the saved evaluation verifier runs once and
verifies training internally. Without evaluation, the training verifier runs
once. Evidence is bounded to 512 files/128 MiB per complete package. Root retains
one `policy-evidence-run/` package—evaluation when present, otherwise training—
instead of duplicating source-bearing training and evaluation archives. Original
manifests, policy and source proofs are unchanged.

Standalone admission/recording validation checks current runtime, policy and
structural identities; it does **not** independently certify historical search
outcomes. Root's prepare/resume path reconstructs admission from the complete
retained package and compares the exact result. Saved-only receipt inspection
likewise reports its narrower metadata/integrity scope rather than claiming a
fresh source or native admission.

### Policy-bearing journals and exact replay

`camera-shared-policy-recording-v1` retains the actual v6 scene and exact camera
task, plus `learned_policy={document_utf8, sha256}` and `policy_admission`.
It exposes the same recording API as camera recording-v1. No avatars,
reference-proposal controller or cancellation journal event is accepted.

Manual/coast/authored events retain their existing fields. Each learned event
additionally requires `policy_sha256`, `policy_admission_sha256`, and
`policy_sample={phase, authored_action, applied_joint_delta}`. Those fields are
forbidden on other event types/controllers. Root caches each actual sampler
decision, verifies its emitted action and clears the cache at reset. This
adapter uses only a temporary validation view to reuse original non-policy
event/outcome checks; it never saves or replays a learned event as manual.

Replay prepares one actual camera environment and one admitted sampler. It
regenerates every learned action **and** diagnostic exactly, regenerates authored
actions, and compares all rewards, flags and full actual/task snapshots across
manual/coast/reference/learned transitions and resets. Partial failures remain
failures. Protocol is `camera-shared-policy-full-state-replay-v1`; results include
scene/task/eight camera pins, policy/admission identities and integer
`learned_actions_verified` (zero when no learned actions occurred). The engine
closes on success, mismatch and deadline; cooperative replay bound is at most
180 seconds. Existing 480 KiB snapshot, 512 KiB event and 128 MiB journal limits
remain unchanged.

Offline replay exposes the unchanged `camera_identity`, exact full
`camera_policy_admission`, its canonical hash, exact policy hash and bounded
`camera_policy_summary`. Flattened events retain their learned metadata. UI
display is not independent verification of actor arithmetic or physical truth.
Renderer/state preparation is cached once; repeated frames perform no runtime
hashing or model preparation.

### Focused verification and freeze

Receipt: `.local/camera-shared-policy-recording-checks-01.json`, SHA
`8c8b0a9a155a9375b913171799136cbd8507c5421bc7cf05cb5a565afa1d23cc`.
**18 distinct analytical cases pass**, with native/model/media/network/process
entrypoints blocked. The tests exercise real saved-verifier orchestration on a
tiny authored analytical fixture, bounded journal write/load, both 19/22-channel
replay paths, control transitions, reset, manual partial failure, forged policy
decisions/deltas/reward/state, default strict verifier behavior, runtime drift,
missing original source, deadline checks and cached rendering. This is not a
native study, performance benchmark or physical evidence.

The initial 18-case run retained 14 passes and four test-only failures: cleanup
assertions referenced nonexistent `.engine` rather than `._engine`; tamper
rejection itself succeeded. That XML is preserved. Only six affected cases
reran after correcting the assertion and adding the training-only/default
verifier assertions; all six passed. No unchanged suite was rerun.

| Frozen file | SHA-256 |
| --- | --- |
| `src/weft/camera_shared_policy_admission.py` | `390e671c4e3bb27f7aec4561ec665c80f3c986cad857cb95552b3699d253124e` |
| `src/weft/camera_shared_policy_recording.py` | `31440fd3a7d71a05a1d62bb6d86d89ce5c2484e27af26690fcb502f9fd56efd3` |
| `src/weft/camera_shared_learning.py` | `b9d0a05a778759028997c310f4c202972ba6ebc80c4b6a49e310f16e016ac15e` |
| `tests/test_camera_shared_policy_admission.py` | `355039868f1710ad2fe9c23f1014e67941acc11dcae657de824ee363d2ce94f7` |
| `tests/test_camera_shared_policy_recording.py` | `42ec45d5b0ba84a11a10e5fb0772aa06872212c1185da80148a90db6de9d4fa3` |

The receipt contains exact commands, both XML hashes, timing, unchanged original
actor/recording/scorer hashes and the historical AST comparison. Receipt
preparation read 11 files/239,109 bytes in 0.022278 seconds, importing no product
modules and executing no saved code. Root's independent review found no blocker;
the presentation lane's review and the parent-owned native attachment acceptance
remain separate. No Git mutation, paid call, source decode or retraining occurred
in this lane. Next safe action: finish root integration review, freeze the full
execution closure, then let the parent run its single prepared native acceptance.

## Native policy-play acceptance and saved audit — October 31, 2026, 4:26 PM PDT

**Passed:** `artifacts/camera-shared-policy-play-01`. The single parent-owned
worker is terminal; no duplicate acceptance or training is needed. It attached
the unchanged searched study01 policy, opened the existing console, recorded
manual/coast/reference/learned controls and a reset, exported media/replay,
verified exact action/full-state replay, and resumed the policy-bearing lab.
The scene/task source qualification remains unchanged and unverified.

| Evidence | SHA-256 |
| --- | --- |
| Acceptance root manifest | `0738410612c195ff44c89a9ae2d5051e8d32291f9345f3b262a3acbd6eb67739` |
| Acceptance result | `32acd670e6b82f49de67596f49059a88342d0bf0d1b41e2f46695c39f429fe04` |
| Parent process receipt | `a5cb908bd2e18d0f1ee942c37203d198201352692ef8823b1613cfa081297b4e` |
| Frozen execution plan | `7b99ac4f6992da2cfa9447cdb44f712da7ffdfdb14b6574743e774ab347fb663` |
| Frozen runtime manifest | `2d649f20080497ef3d365e0af59f5c862b45ffa6149d361a0468d4f295603f36` |
| New immutable lab run | `a407d79e19ea980b75ea93b75153d06ab9ad5e122a6b1f74ab76c9717b8929dd` |
| Canonical policy admission | `d3e3e2ed44eb4611a0de7080e1196744402405eeca29b90b9059406c56e566c7` |
| Session manifest | `f65f0a28c5f991f2ad8f8c5d9bc2d09c6d19daa0f463cad918af1e34b769d417` |
| Exact replay evidence receipt | `2147faa24f2ce2f389c86c044de3ea60ee936bf66824b18135f83c378c566db6` |

Prepared lab: `artifacts/camera-shared-policy-play-01/lab`. Session and offline
replay: `lab/sessions/session-d730c17b1e7b/replay.html` under that acceptance root.
The recorded journal has **248 events, 246 actions and one reset**: two manual,
two coast, two authored-reference and 240 learned actions. Seed 19's eight-action
exploration is followed by reset 29 and the complete original **238-step learned
interval**. All 240 learned decisions retain exact policy/admission identities,
previous-observed phase, authored action and post-clipping joint correction;
239 caused a realized nonzero joint correction. The three explicitly authorized
root channels stay authored. No object-force channel, new support/contact score,
avatar or physical claim was introduced.

The complete interval retains 2,142 valid substeps and the original
7.9412666667-second source horizon. Original tracking criteria passed, with mean
reward 0.8273664367; this is an interaction/replay check on the existing exposed
development clip, **not** new independent learning evidence or a generalization
claim. It does not alter study01's mixed-result conclusion or its authored-root
assistance limitation.

Parent process: **13.376184 seconds**, 508 files/153,643,253 bytes, observed peak
RSS 585,711,616 bytes. The actual control count is 492: 246 recorded plus 246 replayed.
Each has nine substeps, accounting for all **4,428 native substeps**. The parent
also recorded 620 forwards, 120 native resets and one output-video check; these
include preparation/validation work and are not journal reset counts. No hard
memory-cap claim is made. The run stayed within its planned 120-second wall,
500-control, 4,500-substep and 256-MiB output bounds. No paid calls, download,
source decode/inference or retraining occurred.

### Independent saved-only audit

Final receipt `.local/camera-policy-play-saved-audit-02.json`, SHA
`df7aadb33e587971e004aee12a5061adf2b47e1e3e96d7a2d3086d1129f5745b`.
It joins the root/nested seals to the parent's already-hashed 508-member map,
checks original source/task/policy/training/evaluation identities, compares the
46-file original runtime separately from the 54-file execution runtime, and
streams the 6,124,082-byte journal once. All action modes, decision identities,
exact realized deltas, root authority, reset, full source interval and saved
exact-replay counts agree. Only the three declared integration files differ in
the original runtime; actor, dynamics and scoring pins are unchanged. The full
evaluation bundle retains the 148 original training members through exact aliases;
training and evaluation media were not copied into duplicate evidence packages.

The first receipt `.local/camera-policy-play-saved-audit-01.json` is preserved,
SHA `261b4b71cf227fe020930954888ceb7bf701c783f9a875400fe8ade2ee0aacd8`.
It stopped **before reading the journal** on an auditor assumption: the receipt's
13 exporter/validator implementation pins were incorrectly compared to the lab's
entire 246-file snapshot. All 13 match their corresponding lab entries, and the
separate full 54-file native runtime map matches exactly. The continuation fixed
only that subset assertion, reused completed metadata checks and read the
journal once. This was not a product failure or a repeated native experiment.

Across discovery and both audit passes, 25 unique input files/8,644,049 bytes were
read. The final continuation took 0.042061 seconds; initial metadata discovery
was not separately timed, so this is not total audit latency or a performance
benchmark. Per-member assertions are integrity checks, not independent behavioral
tests. No product imports, actor inference, native replay, media reads, source
reconstruction or network calls occurred in this audit. Exact replay is the
already-sealed native result, not a fresh saved-only replay claim. Reproducible
helpers and metadata checkpoints remain under `.local/audit_camera_policy_play_*`
and `.local/camera-policy-play-audit-inputs-*`.

Owned product/test files remain at the released pins above. Root owns scoped
Git publication and user-facing completion notes; unrelated reconstruction and
avatar work remains separate. This lane has no live handles or outstanding
acceptance work.
