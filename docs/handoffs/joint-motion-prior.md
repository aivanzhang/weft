# Source-image motion refinement

## Verified optional path — November 2, 2026, 9:18 AM PST

Source-constrained motion refinement improves tracking on the original task,
but worsens image fit against the preceding image-only candidate and introduces
small floor penetrations. **Keep it opt-in; no default or learned policy is
promoted.** Both tracks and the complete goal remain active.

### Try the saved body

From the repository root, use the tested frozen runtime:

```sh
PYTHONPATH="$PWD/.local/aspset-motion-prior-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/aspset-motion-native-01/lab
```

Enter `controls`, `mode reference`, then `step 6`. Use `record start`, advance
controls, and `record stop` to save/export a new recording. `mode manual` and
`motor NAME VALUE` select one bounded motor; other motors become zero. Nothing
moves until `step`; `quit` saves an active recording. No server, browser or
training starts. The actual console check accepted reference mode and advanced
one .02-second control in 1.623 s; its immutable run and lifecycle stayed exact.

This lab uses the **new body's own authored reference**, not the original-target
comparison below. It has explicit root assistance, no equipment and no learned
policy. For the frozen original-target experiment, inspect the
[recorded replay](../../artifacts/aspset-motion-comparison-01/lab/sessions/session-af2236ab8beb/replay.html)
or [video](../../artifacts/aspset-motion-comparison-01/lab/sessions/session-af2236ab8beb/video.mp4).
The console and saved artifacts require this checkout's local evidence; they
are not bundled into a fresh Git clone.

## Implementation and fixed assumptions

`joint_pixel_trajectory.py` retains schema 1 byte-for-byte behavior. Optional
schema 2 penalizes acceleration of the **actual angle coordinates**, whereas
schema 1 only penalizes correction size/speed. It differences adjacent angular
velocities at their true midpoint spacing. These are scalar Euler-coordinate
accelerations, not measured physical accelerations or a force guarantee.

The two-neighbor block solver uses linear storage. The fitter keeps all image
observations, root, torso, lengths, clocks, camera/floor/force assumptions,
.30-rad total/.10-rad step bounds and 12 iterations/four backtracks. Missing
images or gaps over .25 s break acceleration coupling. Accepted trials cannot
exceed the **pre-fit constructor's** complete image loss; that does not promise
improvement over the separate schema-1 fit or every individual image point.

The existing `camera-body --fit-joint-pixels` option accepts
[`examples/joint-motion-uncertainty.json`](../../examples/joint-motion-uncertainty.json).
The file freezes the original image/angle/correction-speed assumptions plus
60 rad/s² acceleration standard deviation before source execution. Multiview
binding validates the selected version/assumptions; the lab names the prior
and its limitations. No thresholds, reference-overlay admission gates, force
limits or source observations were relaxed.

## Source and original-task results

ASPset `04ac-0006` retains three supplied cameras, all 555 source slots,
47 body poses and 853 qualified joint-image constraints. This is a different
motion from the earlier example, but the same subject/cameras and exposed
development evidence. Source regions and ground remain explicit assumptions.

| Construction check | Previous image-only fit | Motion prior |
| --- | ---: | ---: |
| Complete weighted image loss | 598.280 | 648.376 |
| Actual acceleration loss at the same scale | 2,519.202 | 1,114.352 |
| Failed sampled floor checks, source/grid | 0/47; 0/111 | 0/47; 0/111 |
| Maximum sampled floor penetration, model units | .020439 | .018524 |
| Source poses with any floor penetration | 3/47 | 4/47 |

Both image losses improve over the pre-fit constructor's 1,218.512. Relative to
schema 1, schema 2 worsens image loss **8.37%** and lowers acceleration loss
55.77%. Of 1,417 comparable image slots, 506 regress; all 6,660 slots retain
their missing/unsampled statuses. All 185 changed limb poses remain weakly
constrained. Two groups hit the iteration cap and two stop without descent.

For dynamics, **the original body, target, task, scorer, force limits and starts
stay fixed**. A study-only provider computes legal manual joint commands toward
the frozen refined angles, keeping the original feedback gains and identical
root commands at the same current state. It does not attach a fitted overlay,
change the environment or claim a learned policy. Seeds: nominal 29, plus
7421163, 2034436424 and 199742131 with the unchanged
`shared-exercise-human-jitter-v1`, `jitter_scale=.1` reset. Each paired episode
executes all 184 controls /3.68 seconds.

| Mean across four paired starts | Original authored commands | Refined-target commands |
| --- | ---: | ---: |
| Original task successes | 4/4 | 4/4 |
| Angular RMSE, radians | .214421 | .183876 |
| Endpoint RMSE against original constructed target, model units | .155100 | .121055 |
| Root RMSE, model units | .028195 | .027472 |
| Mean reward | .690488 | .748027 |
| Mean episode peak floor penetration, model units | 0 | .001991 |

Angular error falls **14.25%** and endpoint error 21.95%, with **no task-success
gain**. All new episodes have small floor penetrations (largest .002684).
These are simulator metrics, not measured physical accuracy or balance.

The unchanged-force diagnostic still rejects exact force feasibility. Of
constraint-free midpoints, 14/44 exceed caps (maximum 2.129×), versus 23/45
(4.095×) before. All 42/42 constraint-free 20 ms transition probes exceed a cap
(9.866×), versus 43/43 (10.560×). Counts differ because active constraints differ;
the transition probes are finite-window velocity jumps, not exact derivatives.
Contact-constrained extremes remain, and authored dynamics are unvalidated.

### Separate post-freeze published-reference assessment

After commands and trials were frozen, the nominal recording was compared
with the previously exposed published joint reference. No reference coordinates
were used for construction, tuning, alignment or selection. On the same 479
available joint slots, simulated endpoint RMSE is **125.352 mm**, versus the
earlier 156.583 mm; original constructed-body RMSE stays exactly 69.433 mm.
Across all 2,220 simulated slots, error is 130.051 mm versus 160.965 mm.

Reference accuracy and joint-definition equivalence remain unverified. This is
development assessment, not a fresh final holdout, measured ground/equipment
contact or proof of physical transfer. An independent saved-only calculation
checks all 6,660 rows, 479 matched slots and 421 source members without physics.
[Assessment report](../../artifacts/aspset-motion-reference-assessment-01/index.html).

## Verification, integrity and bounded work

**91 distinct Python cases and two focused mocked-UI checks pass.** Coverage
includes dense independent solver comparison, a 900-frame residual check,
nonuniform clocks, missing/gap boundaries, exact-image preservation, image-loss
rejection, source bindings, native geometry and actual CLI recording/replay.
Three complete v1 outputs retain exact old-source byte/hash parity. All changed
runtime files match the tested frozen snapshot.

Receipts: `.local/joint-motion-{core-tests-01,integration-tests-01,image-guard-02}.xml`,
`.local/joint-motion-legacy-parity-01.json`; UI checks are
`jointImageTrajectoriesKeepPriorFitsAndUnknownDepthVisible` and
`jointMotionPriorNamesItsAssumptionWithoutClaimingFeasibleForces` in
`tests/lab_ui_smoke.mjs`. A test-only fixture typo was fixed before source work.

| Job | Actual elapsed / peak RSS | Hard wall / CPU / RSS bounds |
| --- | --- | --- |
| Synthetic 900-frame/four-view fit | 2.090 s fit; 53,395,456 B | 30 s /25 s /192 MiB |
| Source reconstruction | 2.387 s /178,094,080 B | 60 s /55 s /768 MiB |
| Coupled lab + force probes | 3.089 s /280,395,776 B | 60 s /55 s /768 MiB |
| Paired controls + recording/export | 11.566 s /429,391,872 B | 60 s /55 s /768 MiB; 16,000 substeps |
| Saved-action replay recovery | 2.206 s /179,814,400 B | 30 s /25 s /384 MiB |
| Published-reference assessment | .371 s /66,715,648 B | 30 s /20 s /384 MiB |

These are local measurements, not browser performance or a measured speedup.
Source construction and coupled diagnostics evaluate native geometry without
stepping; the replay/comparison explicitly step it. No new model inference, paid API,
credential/ledger read, deployment or browser use occurred.

Preserved harness incidents:

- All eight trials, recording and export completed before an unnecessary second
  replay reached the aggregate 16,000-substep cap. The original unsealed study
  directory and exit-1 logs remain intact. A new replay-only job verified all
  185 events/184 actions exactly without rerunning trials or encoding.
- The original study read guard omitted ancestor-directory opens needed by the
  secure receipt writer. Export/replay succeeded, but the ordinary UI receipt
  is absent. A saved-result/runtime binding check with corrected read-only
  traversal passes; it does **not** backfill or relabel the original recording.
- Two saved-audit writers accidentally listed their just-opened manifest as
  an empty member. All other member hashes verified. Keep both `*-audit-01`
  bundles unchanged and use new `*-audit-02` copies with valid seals and
  `seal-recovery.json` provenance. No numerical results were recomputed.

## Exact local artifacts

Seals are SHA-256 of `manifest.json`; member counts exclude the manifest itself.

| Artifact under `artifacts/` | Members | Seal |
| --- | ---: | --- |
| `aspset-motion-prior-body-01/run` | 384 | `30f0c6e036d76b3fdcc4f8ccee65a1804218c4acd03cedc2aa882052bd7d9920` |
| `aspset-motion-native-01/candidate` | 67 | `9a3b9bdab56f6c6f42b1449d54f28a75391b58027194adc7ae3603d6faba1f94` |
| `aspset-motion-native-01/lab/run` | 395 | `e08fb8aceca34fcfdeeaec0f94dd452de95eed674884dd9133f8a23f22813b3f` |
| `aspset-motion-native-01/dynamics` | 70 | `61ac8ac5f4a7dab61cf819fa0d54577696bcb8aa61d2e738a5ead69f63cb56f1` |
| `aspset-motion-replay-verified-01` | 6 | `fa21c6acfc5aa09448a3a1eaf823e4114d65b5ed3b520aa007571314ed8435ed` |
| `aspset-motion-saved-audit-02` | 5 | `20693c1cbda0cbd59780a4e3aa483c452677dc639f767d2cff905b3b3c5d9bf5` |
| `aspset-motion-comparison-audit-02` | 6 | `c7be5b75ccac9a182cc86c2f69245b4495670a4ccfc590c767b339bdbae048f5` |
| `aspset-motion-reference-assessment-01` | 10 | `c7f15cfce8f1759f9975c1b8448925c76c094632f6b0b2cca921f11faf6601af` |

The original-task recording is
`aspset-motion-comparison-01/lab/sessions/session-af2236ab8beb`; its manifest seal
is `1c65607f7a532d30de6ca0128238c75583ed45fecc397fb047c11b5c1166d032`.
Its failed parent study's 416 files are pinned by
`.local/aspset-motion-replay-01/source-index.json` and independently checked in
the comparison audit. The construction audit verifies 1,305 input members,
46 retained trial losses, 6,660 image slots and all 158 floor checks. The control
audit verifies 1,472 actions, identical paired resets and 185 recorded events.

Study plans, workers, snapshots and terminal process receipts live under
`.local/aspset-motion-{prior,native,comparison,replay,reference-assessment,console}-01`;
additional saved verification is `.local/aspset-motion-postfreeze-audit-01`.
Original source body/camera seals and full construction command are in the
prior plan and `worker-result.json`. All launched jobs are terminal/reaped;
no server or paid resource remains running.

Root owns the fitter, multiview binding, lab label, tests, example and guides;
preserve the separately owned avatar edits. Publication uses standing approval,
with exact owned paths and remote verification recorded in
`.local/joint-motion-prior-publication-01.json` after push. The preceding root
checkpoint is `435fea6`; concurrent avatar commits may also be ancestors. The
stash was empty when refreshed.

## Next work and completion boundary

- [ ] Qualify the frozen motion prior on different subjects/viewpoints and
  visibility patterns; preserve source errors, rejected proposals and original
  task comparisons. Do not tune 60 rad/s² from this exposed reference result.
- [ ] Resolve the remaining force/transition and ground/equipment discrepancies
  from source-qualified geometry/uncertainty. No blind gain/force sweep or gate
  relaxation; approximate tracking does not establish exact feasibility.
- [ ] Demonstrate full-task learned behavior under the same independent
  comparisons, not this authored manual-action study or a new easier target.
- [ ] Complete fresh final evaluation and independent physical/contact evidence
  across the broader goal. Current tests and this one development clip do not
  complete generalized video reconstruction.
