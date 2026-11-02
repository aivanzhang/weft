# Fixed-length source body fit

## Active — November 1, 2026, PST

Previous goal turn is **progress**: verified and published `0a05e0c`, reusable
three/four-view construction, 163 Python /69 mocked UI checks, actual manual
record/replay, and the retained real-source floor regression. Root-owned paths,
index and stashes are clean; the independent avatar session owns current dirty
avatar runtime/tests and `tools/avatar-assets/` files. No root job is live.
The combined generalizable video-to-coupled-world/RL goal remains active.

### Saved diagnosis and changed hypothesis

Read only the sealed three-view source/model artifacts. At frames0/76/80/84/88/
120, every available leg's model direction matches the triangulated segment
direction to floating-point precision (largest inspected angle about 8.54e-7°).
This is not a rotation-sign bug. At frame 80, image-estimated left thigh/shin
lengths are .3194/.2880 m and right .3287/.2762 m; the fixed simulated lengths are
.3899/.3751 m. Hip-position differences are small. Copying image segment
directions into these longer chains accounts for most ankle discrepancy.
Do not change lengths per frame or move ground to hide that inconsistency.

Implement an opt-in fixed-length limb position fit within the existing camera
body constructor/lab. Keep root, floor, dimensions, force settings and all source
slots fixed. Use supported knee/ankle or elbow/wrist positions only, not floor
scores, C3D references, contact labels or task success. Reuse the existing spatial
hinge conventions; verify analytic forward geometry/Jacobians against native
model controls. Missing complete limbs keep their explicit prior; no inference
of a missing endpoint is relabeled observed. Preserve old construction exactly.

Proposed bounded construction contract, frozen before real-source execution:
four angles per limb, at most .30 rad change per angle, unchanged 0–2.6 bend
limits, at most .10 rad update, 12 iterations and four backtracks per iteration.
Squared knee/ankle position residuals plus original-angle regularization 1e-3
and adjacent-correction regularization 5e-4, only across gaps ≤ .25 s. Stop/reject
nonfinite/nondecreasing trials; retain optimizer outcomes. Source-position
objective must not worsen; this is not independent physical accuracy.
Fixed shape/root make an angle adjustment meaningful, unlike shortening bones
to pass the already observed floor check. No evaluation limit changes.

### Bounds and next actions

- [x] Implement/review the reusable pure-numerical fit and native analytic
  controls, then wire an explicit new method through body validation, CLI/lab,
  compact labels and recording. Preserve old methods/admission guards.
- [x] Freeze runtime and one real-source candidate before unchanged floor and
  image assessment. Keep failures. No dynamic task success/default promotion.
- [x] Verify actual controls/recording if the candidate is produced, preserve
  all source artifacts and update durable context before authorized publication.

Local pure preparation/tests: existing dependencies, bounded small matrices,
zero API cost. Before native/source execution, record exact runtime/input hashes,
work/time/memory/output limits and process ownership in a new directory. No new
source acquisition, inference, server, browser, deployment or key/ledger access.

## Implementation and first native plan

`limb_position_fit.py` reuses the spatial Z/Y/X + signed bend convention and
provides analytic position derivatives. Native control verifies all four limb
chains with a translated root/rotated torso; finite differences verify both
hinge signs. `camera-body --fit-limbs` opts into `camera-body-length-fit-v1`
for two/three/four-view direction-enabled construction. Root, dimensions, ground,
source slots and force settings remain unchanged. The source's Euler branch
history is kept separate from fitted angles; fit corrections cannot change the
next raw pose's chart branch. Missing limbs reset the temporal correction prior.

93 focused Python checks pass in 7.50 s. A subsequent 22-case changed-path run
passes after the explicit Euler-history isolation; 70 mocked UI checks pass.
Receipts: `.local/camera-limb-fit-tests-01.xml`,
`camera-limb-fit-branch-tests-01.xml`, `camera-limb-fit-ui-tests-01.log`.
Old construction methods/defaults and shared-world/learning guards stay intact.

Next native job `.local/aspset-fixed-limb-lab-01`, output
`artifacts/aspset-fixed-limb-body-lab-01`. Freeze committed runtime plus only
root-owned fitter/integration files. Reuse original left body/ground/root-view
and right-view input, same `--body-view both`, adding only `--fit-limbs`.
Maximum 30 s external/25 s CPU, 768 MiB sampled RSS,  six compilations/64 data
allocations/1,000 forwards; zero dynamics, original video decoding, inference,
network or API. The new pure fit additionally has a 20 s cooperative deadline.
All 47 retained references /188 source slots are required. Publish/seal the
candidate before auditing its unchanged 47 source/111 grid floor checks and
all qualified source image projections. Retain regressions; do not select a
new floor/length/radius based on those results. Provider cost $0.

## Native candidate and next verification

New lab `artifacts/aspset-fixed-limb-body-lab-01/run`, 378 files, seal
`e3988f89a7638a3a34f87fc71ab97f5f284b145bdb1bfa16081470c13b8aa3f4`.
Worker 1.593685 s /152,158,208 B; supervisor 1.663644 s, terminal/reaped.
Four compiles/53 data allocations/253 static forwards; zero dynamics/model/media.
All 163 supported limb instances receive bounded angle corrections ; 29 reach the
iteration cap and 134 stop with no descending trial, not a convergence guarantee.
Maximum correction .30 rad. All 47 source poses pass the unchanged .025 floor
limit, but three still have smaller intersections. Four of 111 interpolated grid
poses fail; maximum penetration .055751. Candidate remains globally infeasible.

Next: saved-only original/new source-model and image projection audit, plus actual
manual record/replay in `.local/aspset-fixed-limb-interaction-01`. Reuse the
committed+six-overlay frozen runtime. Same 45 s external/35 s CPU/768 MiB RSS, 
six compiles/16 data allocations/3,000 dynamics calls/1,000 forwards; 39 actions,
exact replay, 60 in-process latency samples, ≤64 video writes/256 export reads.
No source-video decode, inference, network/API, browser or server. Provider cost $0.
Preserve all original artifacts and keep new sessions outside the immutable run.

## Verified results and publication boundary

Actual recording: `artifacts/aspset-fixed-limb-body-lab-01/sessions/session-d79d024a6b5f`,
seal `e0d6c42baf5c0b8533b27e23d37e45369d7d221bc813f32ccb5adc186180dacb`.
All 39 actions/one reset/41 states, rewards and completion flags replay exactly;
41 video frames exported. Worker 1.025796 s /161,759,232 B; supervisor 1.129608 s,
terminal/reaped. Two compiles/two data allocations, 1,104 physics calls/145
forwards, 41 video writes/42 verification reads. Original run and lifecycle
metadata unchanged. Sixty in-process action/state/JSON calls: median .830292 ms,
p95 .854206 ms, max .918459 ms, max 35,522 B. Not HTTP/browser or sustained-use QA.
No source-video decode, inference, API/key/ledger access, server or deployment.

Saved audit `artifacts/aspset-fixed-limb-audit-01`, four members, seal:
`195eb49ae03f752b00582bd5055974917ef391473b61a4fe5dce87310382b65e`.
It verifies 755 baseline/candidate members, 163 supported fits, 1,697 retained trial
losses and 6,768 image slots: 1,620 compared, 72 low-confidence/outside, 5,076
unsampled. Root/ground/shape/forces and triangulation/source bytes stay exact.
Supported-limb position SSE .976595→.564435 m²; source fit, not physical error.

| Camera | Compared points | Mean pixel error before→after | RMSE before→after | Regressed points |
| --- | ---: | ---: | ---: | ---: |
| left | 497 | 10.1899→9.2281 | 14.4156→13.5326 | 55 |
| mid | 561 | 8.9665→8.2954 | 12.6640→12.1694 | 97 |
| right | 562 | 9.6492→9.2751 | 15.8909→15.7960 | 126 |

All 278 regressions are preserved; no available slot is dropped to improve a
summary. Audit 0.206960 s with no new solving/native/media work. All 158 saved
floor checks retain the unchanged .025 limit. Four failures occur at 1.633333,
1.666667,1.700000,1.733333 s, all bracketing source frames 80/84/88.

The [saved geometry figure](../../artifacts/aspset-fixed-limb-inspection-01/comparison.png)
was visually checked: two frame 80 leg projections and the complete floor-check
timeline. Figure/input package seal:
`afe728154328d1f5eaea0fde54bfb3a409816cbf924ff8680d0cada69901e39a`.
It uses saved positions and floor results only, not source-video decoding or new
physics. Plot/source positions remain inspectable; it is not measured anatomy.

Final scoped union: **96 passing Python cases** from the 93-case initial suite,
22 changed-path cases and seven final admission cases (three new rejected
status/source claims). Native construction used the six overlays in
`.local/aspset-fixed-limb-lab-01/plan.json` over `0a05e0c`. After that run, only
closed source/status-label validation was tightened in the two fit modules;
math/limits/output schema are unchanged, and the saved candidate passes current
validation. Tests: `.local/camera-limb-fit-tests-01.xml`,
`camera-limb-fit-branch-tests-01.xml`, `camera-limb-fit-admission-tests-01.xml`.
**70 mocked UI checks** pass in `.local/camera-limb-fit-ui-tests-01.log`.

Reproduce construction using the prior exact left/camera/ground/root-view/right
inputs and adding `--fit-limbs` to the [multi-view command](../multi-view-body.md).
Use a new output directory; never overwrite this candidate. Saved-only audit:

```sh
.venv-lab/bin/python scripts/studies/verify_aspset_limb_fit.py \
  --out artifacts/aspset-fixed-limb-audit-new
```

The verifier intentionally pins the existing original/candidate pair. Prototype:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-fixed-limb-body-lab-01
```

Use `controls`, `action`, `step`, `record start`, `record stop`, `files`. No learned
policy exists. `lab --resume` starts the visual server only when the user runs it.
Publication will be independently checked under standing approval and recorded
in `.local/aspset-fixed-limb-publication-01.json`; do not stage avatar changes.

### Next safe action

`.local/aspset-fixed-limb-transition-diagnosis-01.json` retains exact frames 80/84/
88. Left-hip twist is −.1572 /+.8784 /−.2201 radians. Frame84 hip/knee are supported
but ankle is unavailable; the whole-leg monocular prior is retained. Qualify
multi-view supported-proximal-segment transport and its interaction with fixed
length fitting before changing interpolation/control or fitting the floor.
Missing ankle/twist/bend evidence must remain explicit; preserve all originals,
failed intervals and independent scorer limits. The full combined goal remains
active, including coupled equipment/contact, learning and unfamiliar videos.
