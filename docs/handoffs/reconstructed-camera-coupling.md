# Reconstructed bodies in shared physics

## Verified integration — November 2, 2026, PST

The joint-image checkpoint is published as `1ef976f`; its exact remote receipt
is `.local/aspset-joint-pixel-publication-01.json`. The full goal remains active.

Root owns the camera coupling/candidate adapter, affected shared-lab runtime
inventory, new focused tests and this handoff. The separate avatar session owns
its renderer, tests and `tools/avatar-assets/` changes. No new acquisition,
inference, paid call, server or deployment is needed for this integration.

## Contract and acceptance

Extend the existing shared-world-v6 adapter to grounded, validated two-to-four
view root/body methods. Preserve the exact reconstructed body, source clocks,
ground, parameters and all source slots. Older two-view-root coupling remains
byte-compatible; new methods use a separately identified coupling version and
explicit reconstruction/assumption pins. Do not substitute a root-only body.

Require the saved original parent and exact reconstruction audit, every
contributing camera/evidence/config document and explicit image-fit settings.
Do not rerun a fitter on admission or infer physical truth from hashes. Pin the
additional validator dependencies in candidate/lab snapshots. Preserve all
existing byte/frame/time bounds, floor gates, task scores and control semantics.

- [x] Qualify two/three/four-view constructed bodies, missing source slots,
  resealed source/audit tampering, fixed ground and rejected penetration.
- [x] Verify native geometry, mass/forces, source time and replay against the
  exact input body; preserve manual versus authored/learned control labels.
- [x] Freeze one actual-source construction, admission and recording run with
  bounded resources. Retain any rejection without shifting the floor, changing
  body parameters or lowering task thresholds.
- [x] Prepare the tested scoped publication; the final remote-verification
  receipt is `.local/reconstructed-camera-publication-01.json` when present.

This is integration work, not unfamiliar-video validation, recovered equipment,
contact accuracy, learned balance or overall goal completion. The frozen source
fit retains two small intersections, 397 image regressions and uncalibrated depth.

## Implementation and qualification

`camera_shared_contract.py` preserves v1 for the original two-view root method.
Other existing two-to-four-view root/body constructors use
`camera-shared-coupling-v2`, with exact method/view/refinement pins and accurate
inferred-shape labels. It reuses the full constructor validators, not just the
root validator. JSON/pose/frame/time bounds are unchanged. Reconstruction and
native simulation are not executed by the coupling check.

`camera_shared_candidate.py` retains each extra view's camera/evidence/config,
the exact root-only parent, complete construction audit and image-fit settings.
Every source slot stays present. `camera_shared_task.py` changes only the new
body's scope label; scoring arithmetic and original gates are unchanged.
`shared_exercise_lab.py` pins eight additional validator dependencies for new
couplings and exposes the saved body audit/settings. No new default or learner.

**210 distinct Python cases pass**, using the latest result per named case:

- `.local/reconstructed-camera-legacy-tests-01.xml`
- `.local/reconstructed-camera-integration-tests-02.xml`
- `.local/reconstructed-camera-integration-delta-tests-03.xml`
- `.local/reconstructed-camera-native-legacy-tests-02.xml`

The new cases include all ten constructor families with two/three/four views,
eleven resealed-input mutations, missing third-view slots, native joint positions,
mass, motor limits, floor rejection, roots-off/on task/recording/export/replay.
Retain `integration-tests-01.xml`: its source fixture lacked authored image `z`
and used the wrong force key. The second receipt retains two wrong replay-result
key assertions, corrected by the delta run. These were fixture/assertion errors,
not relaxed production validation. No UI/browser test is newly claimed.

## Actual source integration

Input: the unchanged `aspset-joint-pixel-body-lab-01/run` from
[the construction checkpoint](joint-pixel-trajectory.md), seal
`64710fc56cd47a02253b397a3840e5ded177b95e71c3f4e5e4f9e392d9cb8674`.
All **188 original slots in three views** remain, with 47 sampled and 141
unsampled slots per view. No source video decode, inference or refitting.

Frozen runtime: `1ef976fa8bb9eb189c8f43ecf239e4bbba949391` plus four
source overlays, 315 files at `.local/reconstructed-camera-source-01/runtime`.
Plan SHA `2c20f7780932074e36202b91d09ec1cadbc7ff8fb7f6e8a549913af019f27b88`;
worker SHA `67c35d7bcd701bd0df993522cb3a761cd85f812dfa0f1b9986a1024c7d090f49`.

| Output | Sealed members | Manifest SHA-256 |
| --- | ---: | --- |
| `artifacts/reconstructed-camera-source-01/candidate` | 67 | `5dffb118e6a679002e2bce682655c19cf4187a8e772b049e9c9842a7cb8c0395` |
| `artifacts/reconstructed-camera-source-01/lab/run` | 394 | `ff3e37a021e3d5fc187a0435b978887057511c7872e680e49f19918e77736f75` |
| `lab/sessions/session-00960d54ca31` under that output | 6 | `e0274233b59157614c2812f6746aeb3553f8c85ff87d4c5dbc4b89b17dc254de` |
| `lab/sessions/session-1af6266e7962` under that output | 6 | `2f3e03965b9d19a4513dbee191986fcb871f309664d4796ecee5d715a7a5809e` |
| Fresh session's `evidence/` | 1 | `88a0ee84a11f48349e6c80b9ef314cea77f8d1d134026224e3938414c9bb6f53` |

The exact inferred body enters one shared-world-v6 engine: **22 controls**, with
three explicitly enabled virtual root forces. The original source clock yields
**184 controls /3.68 seconds**, not the previous standalone 30-Hz inspection grid.
The complete authored episode passes the unchanged simulator gates:

- Angular RMSE .184785641 rad; root RMSE .009217736 model units.
- Maximum simulated floor penetration .001536648 model units (limit .025).
- Mean reward .805956319; mean squared root assistance .038896065.
- No equipment/contact requirements, learned controller or physical validation.

The first recording has six manual actions, one reset and 184 authored actions:
**190 actions /192 events**, all exactly replayed later. It is not independent
changed-start or learned-policy evidence.

### Preserved export-guard failure and separate verification

The first supervisor stopped at **24 model compilations** while the exporter
repeatedly validated native mass/scene metadata. Its video and sealed recording
already existed. Preserve `.local/reconstructed-camera-source-01/failure.json`,
stderr and process receipt; exit1/reaped, 8.169342833 s. Counts: 24 compiles,
3 data allocations, 245 forwards, 1,520 physics calls, 192 writes/193 reads.
No body/task/floor change or reconstruction retry followed. Its original receipt
was not retroactively backfilled. Repeated native mass compilation is an
optimization opportunity, not evidence of a physics failure.

Separate `.local/reconstructed-camera-verification-01` reused the same saved
runtime/lab, independently replayed all 190 prior actions and created a **new**
12-action/one-reset/14-event recording through normal lab export. Its saved
receipt is available and reports exact task/state/action replay. The old
recording, all 394 run members and ready lifecycle remain unchanged after close.

Plan SHA `88c0d368498be19bb812dbc78194efd5bf9423f0c08b561a5631815b7e3406dc`;
worker SHA `43d098f5dcb0e0708be603e16b3d5a12e77556a35cfe2f85faf9647732a45464`.
Bounds: 45 s external /35 s CPU /768 MiB RSS /48 compiles /16 data allocations /
1,000 forwards /2,500 physics calls /300 reads /32 writes /64 MiB new output.
Actual: **6.192199292 s /241,106,944 B RSS**, 34 compiles/3 allocations,
260 forwards/1,952 physics calls, 14 writes/15 reads. Supervisor exit0/reaped,
6.299888084 s, nine new files/2,810,930 B. Report SHA
`622122790b07067cfb5f7a0a5acd213b897e756cd00ef176a2110b1b54d3ee53`.

Thirty in-process action/state/JSON calls: median **1.482084 ms**, p95
**1.582544 ms**, max **1.593417 ms**, max payload 48,656 B. Not HTTP/browser or
sustained-idle verification. All root processes are terminal/reaped; provider
spend $0. No key/ledger access, server, deployment or destructive cleanup.

## Try it and continue

```sh
PYTHONPATH="$PWD/.local/reconstructed-camera-source-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/reconstructed-camera-source-01/lab
```

Use `controls`, `motor left_knee 0.1`, `step`, `reset 29`, `mode reference`,
`record start`, `step 6`, `record stop`, `files` and `record evidence`. The user
may replace `lab-console` with `lab` to launch the visual interface. No server
has been started here. The source/runtime paths are local evidence, not provided
by a fresh clone. Future source changes must not silently repin this saved lab.

Next: bounded real learning and matched changed-start evaluation on this exact
source-qualified world, plus unfamiliar-source/calibration qualification of the
frozen reconstruction. Preserve the root assistance declaration and original
scores. Independent dynamic ground/equipment accuracy, single-video calibration,
missing geometry and unassisted balance remain unresolved. Never count this
one exposed development clip as a fresh final holdout or overall completion.

Publish only the four source paths, new test and relevant guide/handoff/TODO
edits. Preserve all separate avatar work. The final publication receipt records
exact commit/remote, owned-path/index/stash state and evidence hashes.
