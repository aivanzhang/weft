# Small-scale two-round controller decision

Updated **2026-11-01 12:56 PM PST**. Preparation: `/root/reconstruction_next_step`; review: `/root/equipment_control_review` and `/root`; execution/Git: `/root`.

**Completed; still `no_verified_candidate`.** Root executed the reviewed plan once and reaped the child. All49 prefix controls and the unchanged base prediction matched; no actual control50 occurred. The full task remains incomplete. No production edits, tests, native execution, runtime copies, source/media/API/paid work or Git occurred in this preparation.

## Terminal result

Completed **2026-11-01 12:57:28 PM PST**, PID45480, exit0/reaped: **.985848041 seconds**,138,133,504-byte OS peak RSS,5 files /254,972 bytes; stdout3,258 bytes, stderr0. Actual work:49 exact prefix controls,39 predictions,1,408 substeps,90 forwards,11 models,2 data,1 reset and39 copies. No actual50 or fallback.

Both rounds reported `row_box_conflict` for constraint2 (`sam-2` speed), with **zero solver sweeps**. The new second-round center was evaluation1, a −.025 first-joint probe, with worst normalized violation **2.71296055**. It was not the earlier `.1` search's improved center (**1.12338066**). The fixed-old-center diagnostic therefore did not establish improved search behavior when `.025` also changed round one. A conflict in this local linear approximation does **not** prove physical or nonlinear infeasibility.

Root verified all four sealed members and exact prefix/base parity. Saved global base call73 correctly maps to new local call1; finite-difference parity was neither expected nor claimed. Root's audit is `.local/equipment-action-small-scale-native-01/saved-audit-01.json`, SHA256 `c06a39921243d2fb203619cb02fb2da320958615a2d7549a934838d9d25e8db1`. This handoff update read only the small audit, process receipt and manifest, without repeating payload verification.

- Output: `artifacts/equipment-action-small-scale-native-01`.
- Seal: `fde8aab1176648f0ef16fbd652d8e6f04efd7de7f9fdb1ff020d91e84d5a4fd3`.
- Result: `ae5c3dcf316b871567ec8fe33f7af7f7875acb6b33f2266cf8c4bec01e9e4494`.
- Projection: `10ab1930b8147e0894d9ef690586ee229716fa5c74ff2e41be80797ebc100d9d`.

No further harness, source or native work is assigned here. A saved-only multiscale hypothesis review belongs to the independent reviewer; it does not authorize another run. Root prioritizes launcher acceptance/publication and owns subsequent decisions. No controller success, default promotion or physical-accuracy claim follows.

## Executed protocol and evidence

The [fixed-center diagnostic](equipment-action-local-scale-native.md) improved all three saved candidate errors with `.025` differences at the **old** round-two center. This new trial runs the existing two-round controller with `.025` from round one, so it may choose another center. The earlier result does not establish improved fit or a passing action in this changed search.

Reuse the exact frozen helper `1a5182e83c6a4c8981a5ef30e64c7154878e8ae46fec78060054507098894338` and runtime `.local/equipment-action-relinearization-native-01/runtime-02`, seal `7e4b72d6df93de507755d61246015d78ef6ddc7189f5481aa61308b1dcddcc02`. Predictor and25 original policy dependencies remain unchanged. No default routing or promotion.

Reapply49 exact saved Rollout02 selected actions, preserving every full-state/contact hash, snapshot, transition, root bytes and accumulated protected metric. At exact pre50, use the same-state pinned v5 base action and explicit config:

```python
ProjectionConfig(candidate_strategy='two_round_relinearization',
    maximum_predictions=48, maximum_seconds=5., difference=.025,
    trust_radius=.25, maximum_sweeps=64)
```

Compare only the unchanged base prediction with the saved control50 base: exact action/status/full endpoint/contact hashes and available values. Saved global and new decision-local call counters are recorded separately. Finite differences and derived centers/candidates intentionally change; **no old23-prediction parity is claimed** and no extra comparison prediction is allowed. Missing/partial evidence remains explicit.

A selected action permits at most one actual control50, with exact full endpoint/contact agreement, unchanged dtype/root bytes and original-base trust bounds. Retain the actual accumulated scorer result even if substep gates fail. Abstention takes no actual50 or fallback. Endpoint feasibility is not whole-task success. No full rollout, training, source/task/scorer/clock/geometry changes or physical claim.

## Bounds and custody

Exactly49 prefix controls + at most48 detached predictions + at most1 selected actual50: **1,568 substeps /100 forwards /11 models /2 data /1 reset /48 copies maximum**. One projector decision,5-second active budget,64 aggregate sweeps,40 CPU/cooperative seconds,60 external seconds including5-second cleanup reserve,512MiB sampled/high-water RSS,4MiB/8 output files,2MiB per payload and2×1MiB logs. External/recurring spend: **$0**.

The inherited supervisor retains output/descriptor guards and owned process-group cleanup. Root used the previously justified narrow escalation only for its `/bin/ps` child-RSS observer. Runtime and saved inputs receive execution-integrity checks; no copied runtime or new source proof. Output `artifacts/equipment-action-small-scale-native-01` was absent during preparation. Previous artifacts remain immutable; no live handle is owned here.

## Executed pins

- Plan `.local/equipment-action-small-scale-native-01/plan-frozen-01.json`: `2512289a85a2014522b76e2c26ea567aebc388822459604eeb3a76dd69670f85`.
- Worker `worker-01.py`: `cf149406fbdf819c32c3a0bb1fe492dd06c9b517280bd4910bb845fc7aabde9f`.
- Supervisor `supervisor-01.py`: `94304faaef0f825f6a1e58666c3584f85585edb9393c2d29fba1d0473694bab9`.
- Qualifying scale-study seal: `9ddf6aa7cd8ee615b78bf88b68e63c3b1094cb932c436b1e93656f7977ecf6b6`; root audit `06607356cc967262b915c525d0ae200d50a7b96608d1c8da72e187281355ea7d`.

Root closed resource/side-effect review. Independent review found a global-versus-local prediction-counter mismatch before execution; the correction preserves physical parity checks and reports both counters without equating them. The exact narrow correction was independently reviewed and closed; metadata is frozen. Root owns execution, terminal evidence, TODO/combined and Git.

Recorded command, already executed once with the documented narrow `require_escalated` permission for the child RSS observer; do not repeat:

```sh
/Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P /Users/ivan/kintsugi/weft/.local/equipment-action-small-scale-native-01/supervisor-01.py 2512289a85a2014522b76e2c26ea567aebc388822459604eeb3a76dd69670f85
```
