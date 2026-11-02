# Versioned equipment-feedback training

## Implementation milestone and current native result

Implemented a separate policy artifact, training driver and saved-journal
verifier in **new files only**. **38 distinct focused cases pass**: the first
34-case run, then five affected/new checks covering three additional cases,
then two affected checks covering one additional returned-failure case.
All environments and physical outcomes in those tests are authored mocks.
At implementation release 01, no native experiment had run. Independent source
review is closed; the reviewer did not rerun tests or native work. Root has since
completed one native pilot: **0/4 feasible candidates; the unchanged v5 anchor
was selected**. The [native handoff](equipment-feedback-training-native.md) is
the current result authority. No controller improvement or task success was
established. The code/test pins and all original pure-check receipts are unchanged.

Owned files:

- `src/weft/shared_exercise_equipment_feedback_policy.py`
- `src/weft/shared_exercise_equipment_feedback_learning.py`
- `tests/test_shared_exercise_equipment_feedback_learning.py`
- This handoff and `.local/equipment-feedback-learning-01/`.

Root owns Git, integration and native execution. The prior pure actor was
published separately at **`640b39d2c4d4942a662b88fc82c6cd75ea3a311f`** and remains
byte-identical. V5 artifacts/trainer, policy dispatch, task/scorer, source
partitions, prior controllers and public defaults are untouched. No live
handles; $0 external or recurring spend.

## Policy and training contract

The new artifact profile is `shared-exercise-equipment-policy-v1`; it cannot be
relabeled as v5 or selected through the unchanged public dispatcher. It binds:

- Exact original construction-manifest identity, scene, task, evaluator/gates,
  raw **and** semantic pinned-v5 identities.
- The frozen equipment-feature profile, normalizers, 8×2 coefficients, groups,
  channels and original-base `±.25` bounds.
- All 25 original v5 runtime dependencies plus the three new modules, runtime
  versions, original training seed, actual score and unchanged feasibility rank.

Cold validation requires the actual bounded base-policy bytes and an externally
qualified construction-manifest pin. The base file is copied unchanged as
`base-policy.json`. The prepared policy owns no replacement task or optimizer
callback. Before each sample it checks the active environment/engine identities,
compares the supplied reset/step observation bytes with the current cached
observation, and compares that observation's float32 PD prefix with the **single**
current v5 feature sample. The frozen old v5 arithmetic then supplies the base.
The caller still owns serialized environment commands and must not mutate the
admitted model/task. Hashing, cold admission and imports are outside this path.

The explicit, non-routed driver is:

```python
train_shared_exercise_equipment_feedback_policy(
    run, out, base_policy_path=..., base_policy_sha256=...,
    task_path=None, generations=1, population=4, max_seconds=30,
)
```

It requires the **original untrained construction** and an explicit or sealed
task. Policy/evaluation/lab inputs are rejected as construction; no final holdout
or evaluation API exists here. It uses the base policy's original training seed,
the existing two reset protocols, and `_seed_plan(base.training_seed, 2)`.

The CEM search retains the existing algorithm: mean 0, initial standard deviation
0.08, minimum 0.02, clipped `±.25` coefficients, a zero-v5 anchor each generation,
the previous training-best anchor after the first, and exact candidate caching.
The zero anchor is charged within the declared work, not an extra rollout.
Original `optimization_rank`/`_rank_key` and trial validators are reused:
incomplete candidates are ineligible; all original gates precede maximum/mean
normalized excess, then original reward. Prefix gate crossings do not truncate
training; actual termination/failure remains explicit. A complete but infeasible
selected artifact is labeled as such, never task success.

API ceilings are 1–2 generations, population 4–8, at most 60 cooperative seconds
and **8,640 issued controls**. The product includes both protocols and the entire
unchanged horizon; unsupported products reject rather than shorten a task.
The existing 128 MiB /512-file output, 24 MiB aggregate-journal and 8 MiB
per-journal limits remain. Cleanup/sealing reserves up to two seconds within the
requested time. The later native supervisor must enforce CPU, wall time and RSS
independently; cooperative checks are not a native interruption mechanism.

## Lossless journals and resource admission

The journal retains every current observation, exact v5 feature matrix/base,
actual action, all actor diagnostic values, returned reward/flags and original
actual metrics. Numeric arrays use strict base64 of their exact float bytes;
dtype and fixed shapes are explicit. There is **no quantization or field drop**.
Immutable schema, groups, normalizers and candidate coefficients are referenced
once; static metric identities are stored once per original reset protocol and
checked before each dynamic result is recorded. Original source/timestamp
bindings remain in the unchanged construction archive; records distinguish
simulation time from the frozen reference clock.

Action intents are journaled before step admission. Trial `control_steps` counts
issued calls, `returned_controls` counts returned calls, and journal result
records count successfully retained results. A deadline after an intent can
therefore leave an explicit unissued intent, not a fictitious physics step.
Reset/step exceptions preserve partial trials and already-flushed records.
Failed packages cannot qualify a usable policy; an already-written policy file
is not a successful package seal.

The 32 KiB line cap is **not** the planning estimate. Before native source
admission, metadata-only preflight reserves every full-horizon record pair using
exact array sizes, conservative JSON numeric lengths, actual equipment IDs and
one bounded terminal-failure allowance per episode. It rejects file/aggregate
overflow. After source packaging it also reserves total bytes and all possible
journal files before constructing the training environment.

Measured authored full-horizon workflow, **4 candidates ×2 protocols ×254
controls =2,032 issued and returned controls**:

| Quantity | Preflight bound | Actual raw bytes |
| --- | ---: | ---: |
| Eight journals | 13,385,296 | 9,510,896 |
| Largest journal | 1,673,162 | 1,194,704 |

The 2-generation ×8-population version of this fixture rejects before source
admission or environment construction because its journal projection is too
large. The measured result is retained at
`.local/equipment-feedback-learning-01/checks-02/test_mocked_full_254_control_w0/mock-full-horizon-result.json`.
Its selected-feasible label is an **authored test outcome**, not a physical result.

Saved verification reconstructs the lossless arrays and full actor report,
recomputes exact v5/new-action arithmetic, checks current-boundary clocks,
per-control substep bounds, reward-integral deltas, terminal flags and accumulated
peaks, and reproduces CEM candidates/cache/elite updates and original ranks.
`verify_equipment_feedback_training` checks saved arithmetic; the separate
`verify_equipment_feedback_package` first requires the externally pinned seal
and original source-custody collector. Neither is physical replay.

## Focused verification

The first run passed 34 cases in **1.523823 s /136,019,968-byte peak RSS**. A
self-review tightened intermediate saved clock/reward checks and total file
reservation. Five affected/new cases then passed in **1.346526 s
/135,872,512 bytes**; three are additional cases, so the union is **37**, not 39.
Review then found that the existing returned engine-failure branch omits
`legacy_coupled_score_used`. The verifier now preserves and validates that exact
absence instead of requiring or fabricating a flag. Two targeted normal/failure
checks passed in **0.633217 s /121,405,440 bytes**, adding one distinct case for
**38 total**. Every guarded run made zero native or forbidden calls; an
import-time OS probe was blocked before execution. Successful old
actor/dependency tests were reused.

Coverage includes full mocked 254-control episodes; unchanged zero anchors;
training-feasible versus higher-reward selection; exact cached candidates;
same-current-state pairing; stale/replaced-engine rejection; lossless
signed-zero/subnormal arrays; pre-native journal overflow; file-count reservation;
reset/step failures; unissued deadline intents; source-custody ordering; and
resealed policy/source/runtime/record/rank/accounting corruption.

| File | SHA256 |
| --- | --- |
| New policy module | `88d193810bd2c96f30b3f9caf3db38478a7b00d686404730a2eba8f840b85a30` |
| New driver/verifier | `bc06e94648d0dd25b4a2009154ead984b5d9ba42014df4c529a7b3c933799bef` |
| Focused tests | `b710f74ae66ab9e3105b3e69f1af5ff56731775dbb2720a033fd7c66d1650e7f` |
| `process-01.json` | `b37adc1463b8cffa840c93a783cc81292b115524699962b36e08e094ad90f0fb` |
| `tests-01.xml` | `c380623d41aa88c965f3742b5aba33357c748c0ca1132bcee59a09a38117dbbb` |
| `process-02.json` | `98a2d2f8d1d4e39c5ca051e59943ec312b9aec73cc0d32e01f2c6594718ee987` |
| `tests-02.xml` | `a497dcd64eb0bbaeb9969f77f38155ffba7397ef1bab4a6c2d4d7daff6a616e8` |
| `process-03.json` | `a5264f3ab742b4c691f24a0cf4876a8cc887602ed86e87330553f4430e71e117` |
| `tests-03.xml` | `4cb0b3b315101b212baa94320c01dff8af3e666bf47ad99376e09750298458f0` |
| `unchanged-sources-01.json` | `aee82734d05cb10302019ce20d7773ceeb3f26014af3ecd518ea63ea3c7098c0` |

One bounded source comparison confirms **31 frozen sources unchanged**, including
the newly published pure actor. No source-data archive was reread for this check.
Commands and precise selections are preserved in each process receipt and XML.
The runner is `.local/equipment-feedback-learning-01/run_checks.py`; completed
output paths must not be reused or passing checks repeated without a new concern.

## Historical pilot proposal and completed follow-through

At implementation release 01, root authorized preparation, not execution,
of **one** separately supervised
development-training pilot: the previously qualified scene/task/v5 base,
1 generation ×4 candidates ×2 original starts, all 254 controls. That is at most
8 resets, **2,032 actual controls /32,512 physics substeps**, with no extra anchor
rollout, optimizer retry, final holdout, media or public promotion. Keep 30 s
cooperative work including preparation, a proposed 40 s CPU /60 s external wall
envelope and 512 MiB RSS. Preserve the current output ceilings and the resolved
input-specific journal projection; external/API cost remains $0.

Root subsequently froze the original construction/base/runtime, reviewed the
cold native counts and complete resource envelope, and executed the pilot once.
All eight trials completed but failed equipment stability. The saved verifier
passed; there was no task-success or learned-improvement result. See the native
handoff for receipts, measured resources and the proposed next hypothesis.
The failed study does not authorize a shorter interval, wider gate, larger
budget, repeat or new evaluation query.

The original `.local/equipment-feedback-learning-01/release-01.json` is preserved.
Release 02 changes this note's historical/current status and links the separate
native result; it does not change source, tests or their passing evidence.
