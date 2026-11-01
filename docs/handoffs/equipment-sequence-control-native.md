# One pre-49 two-action search

## Current state and ownership

Root executed the frozen experiment once. It completed and **abstained**:
all 23 predicted pairs failed measured constraints, so there was no actual49,
actual50 or fallback. The 48-control prefix and original-base49 first-boundary
parity are exact; projection left live state and task metrics unchanged.
No native work was run by this preparation lane. The [experimental
projector implementation](equipment-anticipatory-control-proposal.md) remains
frozen at source `083e629aec3eebc9a22098dfd9c3e8af1b8d1f3e70a36c2710a6c5eb913432ef`.
Root and independent review closed the worker, supervisor and plan without findings.
No extra tests or unchanged native checks were run.

Scratch: `.local/equipment-sequence-control-native-01/`.
Preserved output: `artifacts/equipment-sequence-control-native-01`.
Root's child PID 57047 exited and was reaped. No live handles remain.
Root owns Git integration; $0 external
or recurring spend. No default/public control mode changes.

## Actual result and saved diagnosis

Root measured **1.054703 seconds / 138,264,576-byte peak RSS**. Actual work:
23 sequences / 46 forecast controls, 64 solver sweeps, 1,504 substeps, 96 forwards,
11 models, two data objects, one reset and 23 copies. The final repeat was unused
because no candidate passed. All 48 actual controls were the required prefix.

Outcome: `no_verified_candidate` /
`nonlinear_sequences_fail_measured_constraints`. The solver reached 64 sweeps with
linear residual `1.0240089176605238e-6`, then **all three nonlinear candidate tests
ran and failed**. Do not describe iteration exhaustion as the sole cause or infer
that the nonlinear trust region is infeasible.

The saved-only diagnosis covers every one of the 23 evaluations. Define
`R = max(value / unchanged_limit)` across both boundaries and normalized excess
`V = max(0, R − 1)`. The actual pass tolerance remains `value <= limit + 1e-12`.
Baseline V is **3.85693**; best V is **2.82345**, and the maximum among the 23 rows
is **6.94222**. Only the full-scale candidate passes all control49 measurements.

- At control49, `sam-2` speed and spin fail in **22/23** evaluations.
- At control50, speed and spin fail in **23/23**; orientation also fails in **21/23**.
- No other measured constraint fails. This does not establish unmeasured accumulated
  task-gate safety.

Speed/spin limits are 0.15 model units/s and 0.75 rad/s; orientation limit is 0.15 rad.

| Candidate scale | V across both boundaries | Control49 | Control50 speed / spin | Additional control50 failure |
| --- | ---: | --- | ---: | --- |
| 1 | 2.82345 | Passes measured constraints | 0.479837 / 2.867589 | None |
| 1/2 | 5.37717 | Speed and spin fail | 0.567823 / 4.782881 | None |
| 1/4 | 5.83612 | Speed and spin fail | 0.571789 / 5.127093 | Orientation 0.169329 |

All 23 first-action vectors are exactly the same as the retained original
one-control search, including its three candidate scales. This is an input-vector
comparison, not an additional physical qualification. Pure arithmetic using the
saved base and 19 differences predicts full-scale control50 speed/spin
**−0.011106 / −2.952414**, against measured positive **0.479837 / 2.867589**.
The corresponding normalized spin-response errors at scales 1, 1/2 and 1/4 are
**7.76000 / 5.93369 / 4.20262**. Negative norm predictions and these discrepancies
show a poor local affine response for the tested candidates. Simply increasing
the sweep limit is not supported as the sole remedy.

Only the 36,117-byte projection and small seal were read once for this diagnosis;
subsequent arithmetic used its derived receipt and the prior retained action cache.
No optimizer, new candidate, physics, source edit or repeated geometry work.

Receipts:

- Native seal:
  `63f5c738e13a561f93babad1b33eafff6fcedb64f0a208878b16f57a2a719551`.
- Projection SHA256:
  `771dd9f4e51c4ff8e49780626a312d9e934cfe3b5157fcd347c703c2b2fc0e54`.
- Root's `saved-audit-01.json`:
  `b3ff7364dd478b32d789fecbc1272c3c4b148160c344101f9da471ca53c15f74`.
- This lane's `saved-diagnosis-01.json` (all 23 rows and candidate comparisons):
  `7863d5aad106d8fa31000fdc27c69d4583cee71762903f7292c2d2839beef9f6`.

Audit detail: aggregate forecast controls are nested under `forecast_control_counts`.
Worker trace hashes use `canonical_hash` with default JSON separators; projector
evaluation fingerprints use sorted compact JSON. They are different hash domains
and cannot be compared directly. Root corrected only this audit metadata handling;
no native output or artifact was changed.

## Fixed experiment

Reuse the qualified sequence-parity worker-02, supervisor-02 and existing output
guard. The frozen runtime preserves **all 294 old members byte-for-byte**, adding
only the released two-action projector: **295 files / 6,388,791 bytes**. One bounded
runtime read/copy verified that delta; no recording, media or source-proof scan
was performed during preparation.

Use the same sealed recording and Rollout02 evidence, scene/task/v5 policy, 25
policy-source pins, seed 29 and no jitter. Replay exactly the 48 saved prefix
actions, checking all saved state/contact/snapshot/task/transition identities.
Then call the projector once with **original saved BASE49**, not selected49, and
the **entire fixed saved BASE50**. It is not a recomputed-policy rollout.

The wrapper counts every sequence and native call, prevents hidden predictor
lifetimes, and forbids the old one-action predictor/projector. First-action root
bytes/dtype and original-base ±0.25 joint bounds remain exact; second-action bytes
and dtype never change. The first sequence's complete first boundary must match
the saved original base49 endpoint and substep-contact hashes. Later second
states are new; do not compare them to the old selected49 trajectory.

Returned pairs require the complete repeated two-control proof. Apply actual49
once, compare every endpoint field and substep contact, and record unchanged
scorer metrics. Stop before50 on nonrecoverable gate failure or engine termination.
Only if eligible, apply the unchanged second action once and make the same checks.
Raw mean-angular/root crossings stay provisional for an incomplete interval;
the exact previously reviewed Rollout02 stop function is reused, not a changed
scorer. Unavailable/no-candidate results take **no actual action and no fallback**.
Partial actual-step errors retain the attempted action, observed contacts and
available task metrics.

## Resource and interpretation bounds

- One decision: 24 sequences / 48 forecast controls, 64 solver sweeps, five-second
  active budget, fixed 0.1 differences and 0.25 original-base trust radius.
- At most 1,568 native substeps: 768 prefix + 768 forecasts + 32 actual.
- At most 11 models, two data objects, 100 forwards, one reset and 24 copies.
- 40 CPU / 60 external wall seconds, 512 MiB RSS, 4 MiB / eight output files;
  result/diagnostic individually at most 2 MiB, logs at most 1 MiB each.
- Worker uses a 35 CPU / 40 wall cooperative budget and a seven-second admission
  reserve before the projector. Existing supervisor cleanup/child accounting and
  narrow observer permission are retained.

A completed experiment is not controller or full-task success. Repeated predicted
boundary/floor constraints remain narrower than accumulated task gates. The task
still spans 254 controls; even two successful actual controls do not finish it.
No fitting, training, media, replay, network, paid API, server or promotion.

## Frozen identities and launch

- Plan `plan-frozen-01.json`:
  `61983377b86d458b848d3f34e5781c45c69e1adcd8e00caa4b0357794aad4af1`.
- Preserved draft `plan-draft-01.json`:
  `f43ed902bfe8990ef4ac97eecb7a6deb40d8bcf0b574617853b3b49391879b4c`.
  Only status changed on freeze; no numerical/input/runtime change.
- Worker: `8a79122c1ac553d52c7766326d6a31f3ffaaa214ff3fcf61e546ef9ef18dd96d`.
- Supervisor: `e089334bf5448e662d84dbe86416589e5c3742c4fc76fc21d8f403f068ce8ffb`.
- Runtime manifest: `88f61faa4fcc535e44999a1e1074a580699efebf9583c9be28052ec98f5a2a5b`.
- Parent runtime manifest: `a71c264d135b0dce3af67301fad87151413c99c749c5e79722d8ad32818cf5e1`.
- Byte-identical `actual_gate_stop` function:
  `3756def3695ef50b6510e1800577d109fc3dba4fccec4f7340d59f4a479b027e`.

Root's one-shot launch, not a request to duplicate it:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles PYTHONDONTWRITEBYTECODE=1 \
  .venv-lab/bin/python -B -P .local/equipment-sequence-control-native-01/supervisor.py \
  61983377b86d458b848d3f34e5781c45c69e1adcd8e00caa4b0357794aad4af1
```

Next safe action: choose and review a distinct response-model/contact hypothesis
before any new experiment; do not rerun this unchanged search or merely increase
its budgets. Feasibility remains unknown. Released implementation, all native
evidence and unrelated work stay intact.
