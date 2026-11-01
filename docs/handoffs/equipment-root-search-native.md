# One nonlinear root-authority screen

## Current state and ownership

**Root executed once; the search abstained.** All24 measured pairs failed, so
no actual49/50 or fallback was taken. Root owns native work and Git. Independent
review of the worker, supervisor and draft plan closed with no findings. Only
status/review/timestamp metadata changed at freeze; the draft, worker, supervisor
and runtime remain preserved. Source implementation is separately
[released and reviewed](equipment-root-search.md): 50 mocked checks, no native
work or public/default promotion.

Preparation: `.local/equipment-root-search-native-01/`.
Preserved output: `artifacts/equipment-root-search-native-01`.
External/recurring spend: $0. No API, media, source reconstruction or holdout query.

## Actual result

Root's terminal run used **1.293728916 seconds /138,559,488-byte peak RSS**.
Actual work: 24 sequences /48 forecast controls; 48 exact prefix controls;
1,536 substeps, 98 forwards, 11 models, two data objects, one reset and 24 copies.
No repeat was needed because no candidate passed. Search preserved live state
and scorer metrics, and baseline first-boundary identity matches saved base49.
There are no actual validation controls and no full-task completion.

Maximum positive normalized excess improved **7.395916871→4.804166769**, but
the best evaluation (index16) still fails five `sam-2` measurements:

| Boundary | Measurement | Best value | Unchanged limit |
| --- | --- | ---: | ---: |
| First | Speed, model units/s | 0.468175 | 0.15 |
| First | Spin, rad/s | 4.209420 | 0.75 |
| Second | Orientation, rad | 0.170982 | 0.15 |
| Second | Speed, model units/s | 0.751967 | 0.15 |
| Second | Spin, rad/s | 4.353125 | 0.75 |

All24 candidates fail first-boundary speed/spin, not only the held-future
boundary. The best pair's first-boundary speed/spin are worse than baseline
0.392155/3.642698 even though its maximum two-boundary excess improves. The
unchanged pass requirement prevents that tradeoff from being returned as usable
control. This is a failed sparse late root-authority screen, not evidence that
the full legal control space or task is infeasible.

This update uses root's already verified saved audit only; it does not repeat
artifact verification, physics or a candidate search. Root verified five files
totaling145,688 bytes. Receipts:

- `saved-audit-01.json`:
  `1d89d1695dc514355a015ce8abd2a18dbee3d01962988ebfa16acee4a2024fab`.
- Native seal:
  `d7b7f30b32072e3c00cbd7058d1ce4c009630d8fa849cca6b9134117899388f1`.
- `result.json`:
  `fa868e6777565c640605728a68442a1d27c873936fff1f72ac2e7c375cf0c263`.
- `search.json`:
  `f8a020fb1e0276f24e935cc63e4c692d934534fff48219fb02feb77f1aadde8b`.
- Root process receipt:
  `d25beb8294698d9fb12e6300641002dab4c317dcf620ab0acc472760bbb23b14`.

## Exact experiment

Reuse the qualified sequence-control harness, recording, source/task/v5 policy,
25 policy dependency pins, seed29 and no jitter. Reconstruct exactly the saved
48-control prefix, verifying each state/snapshot/task/transition identity without
any prefix prediction. Then search the explicit pair
**`[original_base49, original_base49]`**. The second input is a byte copy of the
first, never the old saved base50. The baseline's first predicted boundary must
match the saved original-base49 endpoint and contact hashes; later second
boundaries are new.

One `RootSearchConfig(26, 52, 5., .25)` decision may change the existing three
root channels in **both** actions. Each original base/dtype and all19 joint bytes
are pinned; roots remain original-base ±0.25 within `[-1,1]`. No object force or
changed physical force limit. All sequence calls and attempted/returned native
operations are independently counted; one owner prevents hidden predictor
lifetimes. Old one-action prediction/projection and sequence projection are
forbidden, as are policy callbacks during the forecast.

If the helper returns a repeated proposal, validate all action/dtype/joint/root
identities and both repeated rows before stepping. Apply actual49, compare its
**full endpoint and every substep contact**, and preserve original raw task
metrics. Stop before50 on nonrecoverable gate failure, metric failure or engine
termination. Only if eligible, apply exactly the planned second action and make
the same comparisons. The reviewed Rollout02 gate-stop function is copied
byte-for-byte; finite incomplete-interval mean tracking crossings stay provisional.
An abstention/unavailable result takes no actual49/50 or fallback.

This is an isolated fixed-pair qualification, **not** later receding-horizon
behavior: a future controller would apply the first action and recompute its
base/search from the new state. Held-base forecasts do not equal recomputed
feedback. Passing endpoint screens cannot certify accumulated substep equipment
peaks or full-task success. Preserve every failed/partial outcome.

## Frozen inputs and bounds

The prepared runtime copies **all295** prior frozen members byte-identically,
adding only `shared_exercise_root_search.py`: **296 files /6,408,720 bytes**.
One bounded pass verified/copied those existing runtime files; no recording,
media or source-proof payload was read during preparation. The original scorer
and all25 policy dependencies remain exact.

| Bound | Maximum |
| --- | ---: |
| Search sequences /forecast controls | 26 /52 |
| Actual controls | 48 prefix +at most2 validation |
| Native substeps | 768 +832 +32 =1,632 |
| Models /data /forwards /resets /copies | 11 /2 /104 /1 /26 |
| Helper active /CPU /external wall seconds | 5 /40 /60 |
| RSS | 512 MiB |
| Output /files | 4 MiB /8 |
| Individual result or diagnostic /each log | 2 MiB /1 MiB |

Worker reserves cleanup within its35CPU/40wall cooperative envelope; the
unchanged supervisor samples child RSS, uses OS high-water accounting, kills and
reaps its own process group on bounds and retains partial output. These are
observed guards, not hard OS memory isolation. Root uses the already reviewed
narrow observer permission for `/bin/ps`; preparation did not launch it.

| Prepared identity | SHA256 |
| --- | --- |
| Frozen plan | `671c9572d5f907c98bcaf37eaaecc58f047c8892a72306b068d9bc36a9864540` |
| Draft plan | `e1f0479e01eff0a14b9d52c505bd5c225584fee25d9adad6bea02630efa99a81` |
| Worker | `0e2035c73452ab038bb0abef1601e59d88e602d3f705230f1ca76b85dac708ee` |
| Supervisor | `45291f55f8e12e63f93329d82e393ac506879d5f619e140d13bda100adebc0cc` |
| Runtime manifest | `983c5615cbba775029b36f14197f63423ef98a7bf1e7965727c3614295cf8d6b` |
| New helper | `f4cedb0c8cbc160542c9a1f11a5d5c07041a57015d745a0cbdd07e434592bdb7` |
| Unchanged gate-stop function | `3756def3695ef50b6510e1800577d109fc3dba4fccec4f7340d59f4a479b027e` |

Historical root execution command (do not rerun):

```sh
.venv-lab/bin/python -B .local/equipment-root-search-native-01/supervisor.py \
  671c9572d5f907c98bcaf37eaaecc58f047c8892a72306b068d9bc36a9864540
```

Do not repeat this experiment, execute the draft or run another unchanged suite.
Worker hashes use the original `canonical_hash` JSON domain;
helper evaluation fingerprints use compact sorted JSON, so do not compare
cross-domain hashes directly.

## Next hypothesis — proposed, not implemented or authorized to run

Move from late action repair to an **equipment-conditioned feedback actor that
coordinates joints and existing root assistance before contact**. The existing
v5 actor sees joint tracking errors, ignores the available equipment state, and
has no residual when those joint errors are zero. It therefore lacks a direct
signal to move away from equipment while still tracking well. Prior authored
clearance was joint-only and saturated; this root-only screen also cannot remove
the immediate first-boundary impact. These results motivate a different
state/control family, not more identical polls or a larger solver budget.

An opt-in actor could add bounded equipment-relative pose/approach-velocity
features to a coordinated joint/root residual, retaining the existing normalized
action/physical limits and forbidden object forces. Evaluate learned candidates
with the **existing full-interval, feasibility-first scorer**, not endpoint-only
acceptance or reward alone. Activation would be state-based throughout the task,
not a hard-coded control49 correction. Existing privileged simulator/reference
inputs and authored forces must stay labeled; this is not learned physical anatomy.

Before any training, root should approve a precise feature/output contract and
bounded work envelope, including exact zero-residual v5 parity and honest
original-gate reporting. No new actor, study, earlier-window query, threshold
change or full-episode run is implemented/authorized by this proposal.
