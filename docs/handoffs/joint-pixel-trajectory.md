# Joint image trajectory — verified checkpoint

November 2, 2026, PST. Previous goal turn is **progress**: `8b9c7ff` is
remote-verified, with 234 focused Python /74 mocked UI checks, independently
audited depth alternatives, and exact actual recording/export. Its original
source candidate selects no depth and retains 1/47 source +2/111 floor failures.
The full combined goal remains active; do not repeat that frozen branch study.

## Changed hypothesis and frozen contract

The preceding fit assumes an exact upper-bone axis/knee and fixed distal-length
sphere; 44/70 original rays do not meet that sphere. Fit knee/elbow **and**
ankle/wrist original pixels together, allowing all four existing limb angles
to change. Couple neighboring angle corrections across source time instead of
treating each limb pose independently. This challenges the fixed-knee
assumption, not the source evidence or independent scorer.

Keep root, torso, shape/lengths, ground, forces and original observations fixed.
Use the existing analytic four-angle limb geometry and camera projection.
No inference, native simulation, branch selection, source decoding or paid call
belongs inside the fitter. It operates on one to four calibrated source views
and at most 900 retained references, keeping every missing image slot in the
adapter audit. Reference gaps over .25 s and frames without any usable image
break temporal coupling; unobserved frames remain unchanged.

Use a new, explicit uncertainty JSON: image fraction `.0025`, angle-prior
standard deviation `.3` rad and correction-speed standard deviation `4.5`
rad/s. These are development assumptions, not calibrated detector noise or
physical motion confidence. The temporal penalty uses actual elapsed time and
differences of **corrections**, not a claim that the exercise itself is slow.
No angular velocity or acceleration of the physical body is measured here.

Bound corrections to .30 rad per existing scalar angle, updates to .10 rad,
hinges to [0,2.6], 12 iterations/four backtracks. Use a block-tridiagonal
four-angle solve, not a dense 900-frame matrix. Cap one trajectory at 10 s and
the body stage at 45 s. Retain every trial/correction/loss and per-view image
regression; no floor/contact target enters the objective. Original complete
position fits and partial-direction transport remain separately inspectable.
No change to the preceding v1/v2/branch contracts or default behavior.

## Ownership and acceptance

- [x] Implement/qualify analytic joint-image derivatives and bounded trajectory
  solve; test known geometry, noisy knees, missing views, timestamp gaps,
  rank deficiency, image resizing and all-view retention.
- [x] Integrate an explicit `camera-body --fit-joint-pixels SETTINGS.json`
  path with the existing fixed-length/partial-direction constructor, source
  bindings, honest lab labels and recording. Reject stacking with v2/branches;
  preserve the entire original pre-fit reference and metadata in binding checks.
- [x] After controlled qualification, freeze at most one original-source
  candidate with unchanged cameras, evidence, ground and floor/task gates.
  Record source/runtime/resource pins before execution, audit all outcomes,
  verify actual interaction/export/exact replay and after-close integrity.
- [x] Update guide/TODO/combined handoff and prepare a scoped reviewed checkpoint.
  The publication receipt below records the final remote-verification status.

Root owns joint-trajectory source/body/CLI/lab/test/doc paths. Preserve the
separate avatar session's `src/weft/assets/avatar_replay.js` and
`tests/avatar_replay_smoke.mjs` (last observed dirty). Starting revision
`8b9c7ff7a17c24c5a2c0c3aaa6e6397918088c77`; preceding root jobs are terminal,
index/stash empty. Recheck current Git state before publication. Reuse saved
source packages and dependencies; no acquisition, API/key/ledger access,
server/browser, deployment, clipboard/TXT file or avatar authoring work.

## Qualification before original-source execution

**269 distinct Python cases /75 mocked UI checks pass**: 32 pure trajectory,
27 native integration/source-binding/CLI-recording and 210 affected legacy
cases. Receipts `.local/joint-pixel-trajectory-{core,integration,compatibility}-tests-01.xml`
and `.local/joint-pixel-trajectory-ui-tests-01.log`. The native fixtures verify
both knee/elbow and ankle/wrist positions against actual engine snapshots for
two, three and four views, without changing the old constructor/reference fits.

Full-size pure performance controls `.local/joint-pixel-trajectory-performance-01`
are terminal/reaped, exit0: 900 frames/four views/7,200 constraints, seven
iterations, **2.484977333 s fit /2.576625417 s total /51,314,688 B RSS**.
Fit JSON is 1,627,944 B; corrections max .147245014 rad. This is one synthetic
limb trajectory, not a whole-body/browser performance claim. The block solver
also matches an independent dense system on small inputs and verifies the full
900-frame residual without a dense matrix. No native/inference/API calls.

Frozen original ASPset candidate:
`artifacts/aspset-joint-pixel-body-lab-01`, with existing cameras/ground and
`examples/joint-pixel-uncertainty.json`. Construction limits: 60 s external,
55 s CPU, 768 MiB sampled RSS, 6 compiles/64 data allocations/1,000 static
forwards/zero dynamics; 64 MiB per file and 400 MiB/3,000 files post-exit.
Compare both against its original partial-direction construction and the
preceding v2 candidate; do not tune settings using the result. Preserve every
source/solver failure and every image/floor regression. Provider spend: $0.

## Verified original-source result — November 2, 2026, 2:59 AM PST

The same ASPset `04ac-0003` source/cameras/manual ground now passes all
**47 source +111 interpolated floor samples** under the unchanged .025 limit.
Worst penetration is `.01549837045304999` model units; two smaller intersections
remain. These are fixed sampled-floor consistency checks, not collision-free
motion, continuous-time guarantees, dynamic contact or measured anatomy.

Four trajectories adjust all 188 limb poses using 1,056 qualified endpoint/view
constraints. Assumed standardized image loss falls `1615.858148624279`→
`1450.082242916398`; its absolute value is not calibrated statistical accuracy.
All 163 original complete-limb fits and partial-direction transport stay
separately recorded. There are 134 weakly image-constrained limb poses.
Both arm fits reach 12 iterations; legs stop for no descent after 7/11 iterations,
and the left leg reaches the .30-rad correction bound. Do not claim convergence
or increase these frozen bounds in response to this clip.

| Comparison | Partial-direction input | Previous v2 | Joint-image trajectory |
| --- | ---: | ---: | ---: |
| Source floor failures | 1/47 | 1/47 | 0/47 |
| Grid floor failures | 3/111 | 2/111 | 0/111 |
| Maximum penetration | .077483313 | .049339568 | .015498370 |
| Left mean error, px | 10.664219752 | 10.436842641 | 10.340614801 |
| Middle mean error, px | 6.611042111 | 6.673992558 | 6.345838530 |
| Right mean error, px | 6.197058046 | 6.352132999 | 6.273927968 |

All 6,768 original image slots remain: 1,620 compared, 72 low-confidence/outside
and 5,076 unsampled. Keep **399 individual regressions versus the input /397
versus v2**. The right image mean worsens against the partial-direction input.
This is exposed development/source-fit evidence, not a fresh final holdout.
No default, learned policy, independent contact or physical-accuracy promotion.

### Source and frozen runtime

Reuse the exact input packages pinned in the
[preceding handoff](distal-depth-branches.md#frozen-inputs-and-construction):
original body/left evidence/root-view input, third-camera input, mid evidence
and manual ground. No original-video decode, new inference or source acquisition.
Settings bytes are copied into `run/joint-pixel-uncertainty.json`; canonical
settings hash `b4039733780e83af352cd50ee3a19cad409dba53a9abb5ba3fec535fe9f7c426`.

- Runtime: `8b9c7ff7a17c24c5a2c0c3aaa6e6397918088c77` plus seven owned overlays,
 315 members, `.local/aspset-joint-pixel-lab-01/runtime`.
- Construction plan SHA-256
  `37b812d15838863dec888ea260572771bc51242bdfff193155204ec361132a01`;
  worker `5f8b25c31c59d1df1f1019e1fc6f915a6ae6eb517c1d0a835e7fd8cc780d748a`.
- Candidate `artifacts/aspset-joint-pixel-body-lab-01/run`: **385 sealed members**,
  seal `64710fc56cd47a02253b397a3840e5ded177b95e71c3f4e5e4f9e392d9cb8674`.
- Worker **2.458211958 s /159,924,224 B RSS**; supervisor 2.538663791 s,
  exit0/reaped. Four model compiles, 53 data allocations, 253 static forwards,
  zero dynamics/decode/inference/API. Including lifecycle: 388 files/46,024,414 B.

Exact construction command, only with a **new** output directory:

```sh
.venv-lab/bin/python -m weft camera-body \
  --run artifacts/aspset-camera-body-input-01/left \
  --camera artifacts/aspset-manual-ground-01/camera-0.json \
  --ground artifacts/aspset-manual-ground-01/ground-0.json \
  --root-view artifacts/aspset-camera-body-input-01/root-view.json \
  --additional-view artifacts/aspset-third-camera-input-01/view.json \
  --body-view both --fit-limbs --partial-segments \
  --fit-joint-pixels examples/joint-pixel-uncertainty.json \
  --out artifacts/my-new-joint-pixel-lab
```

Do not rerun this successful unchanged study. The native supervisor deliberately
rejects duplicate execution. Its source candidate is manual-only; no new server
or learning job starts from construction.

### Independent audit

`scripts/studies/verify_aspset_joint_pixels.py` independently recomputes
articulated positions, all original-image projections, finite-difference image
information, actual-time correction penalties, 49 trial losses and exact native
before/after positions. It imports only saved-data/math/hash helpers, not the
joint optimizer, video/inference or simulation runtime.

- `artifacts/aspset-joint-pixel-audit-01`: four sealed members, seal
  `a6374b6b86bdac7ccfbf07dc78d825dbce05185054e0000b6268da5011d0698f`.
  Checks 1,144 baseline/v2/candidate members, 45 identical original source files,
  all 163 preserved complete fits, every original image slot and all 158 unchanged
  floor identities/gates. All failures/regressions remain visible.
- Controls `.local/aspset-joint-pixel-audit-01`, plan SHA-256
  `95e799c061eaeeed97b10afaa1448e0dc907a338a29c7386fd6b4cc35cc45a82`,
  worker `5faf3a72f7eb304c32c733e576c240253225bff06f1f3c46b33c318ef3a15c31`.
- Worker 1.511192500 s /104,235,008 B RSS; supervisor 1.552568208 s, exit0/reaped.
  Limits 30 s external /25 s CPU /512 MiB RSS /16 MiB per-file and total output /
  eight files. Zero new optimizer/native/inference/API calls.

### Actual interaction, export and after-close integrity

Controls `.local/aspset-joint-pixel-interaction-01`, plan SHA-256
`23a9b2743953a6f0b69fe9e316faa10e944ea5ccebe1d922a538da9e8ff05241`,
worker `3f7896e983cdbd05ff6dbaefe8109173be65b4379b23e72043c2e22c15f5bb05`.
Recording `artifacts/aspset-joint-pixel-body-lab-01/sessions/session-512ff00a4618`,
six sealed members, seal
`c9e5774181d81043589a6d486b90354608f1f0758bd1a3dcde05a4d5db419e1e`.

**39 actions /one reset /41 exactly replayed states, rewards, flags and video
frames**. Two compiles/two data allocations, 145 forwards/1,104 physics calls,
41 writes/42 reads. Worker 1.092033542 s /164,151,296 B RSS, 1.125135000 s including
close; supervisor 1.185695834 s, exit0/reaped. Seven files including manifest,
1,907,643 B. Same bounded 45 s external /35 s CPU /768 MiB RSS /6 compiles /
16 allocations /1,000 forwards /3,000 physics calls /64 video writes /
256 reads /64 MiB output as the previous verified interaction.

Sixty in-process action/state/JSON calls: median .838312786 ms, p95 .870894385 ms,
max .900417101 ms, max response 35,610 B. Not HTTP, browser painting or sustained
idle verification. `after-close.json` rechecks all 385 run members and both
lifecycle hashes; status remains **ready**, result **manual_candidate**.

## Publication and next safe action

Root owns 16 paths: two new source modules, five body/CLI/lab/UI source edits,
two new Python tests/the UI test, uncertainty example, independent audit,
guide/TODO/combined handoff/this handoff. Preserve the separate avatar renderer,
renderer test and `tools/avatar-assets/` changes. Index/stash were empty before
this checkpoint; remote fetch completed at 2:58 AM PST. Final publication and
owned-path/remote/integrity verification belong in
`.local/aspset-joint-pixel-publication-01.json`; absence means unfinished.

All root construction, audit, interaction and full-size performance processes
are terminal/reaped. No API budget/key/ledger changes, provider spend, server,
browser, deployment, submissions or destructive Git cleanup.

This goal turn is **progress**, not completion. Next preserve this frozen method
while qualifying unfamiliar videos and admissible coupled whole-body
control/learning. Current shared-camera admission still requires the original
two-view-root method, so do not bypass it by changing a method string or
substituting a root-only model. Explicitly verify the reconstructed body's
reference/source/camera/ground custody, dynamics and unchanged task gates when
extending admission. Root/shape/calibration uncertainty and independent dynamic
ground/equipment-contact accuracy remain unresolved; passing source floor
samples does not satisfy those requirements or the full generalized goal.
