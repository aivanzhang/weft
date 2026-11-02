# Native endpoints and conditional equipment reconstruction

## Current checkpoint — November 2, 2026 (PST)

Both tracks remain active. Parallel owners completed selective-crop timing,
non-grid person-only simulation/recording, an unfamiliar-pose comparison and
conditional equipment-mask geometry. Root integrates the portal and independent
review. These close software gaps, not generalized physical reconstruction.
No API call, deployment, submission or new dependency was needed.

## Test the product

```sh
.venv-lab/bin/python -m weft portal
```

The user starts this local server and opens its printed authenticated link.
Project `22a5644f9ea648d18e117893e470f561` (`native-timing-exercise.webm`) remains
available with reference/manual/learned controls and recording. Existing Muse
candidate comparisons remain on project `4e0f1fe859664bb5ba788214e129451d`.
Both retain their original simulations; the trusted-release list now includes
`b0917e7`, `64f4bef` and `5819ba4`. Actual read-only admission takes 2.58/1.95 s,
changes no run bytes and advances no physics. See
`.local/portal-three-release-admission-01.json`; 21 focused checks pass.

Inspect the new conditional equipment world through the ordinary local lab:

```sh
.venv-lab/bin/python -m weft lab --resume .local/mask-geometry-play-01/build/lab
```

It retains the original full task, offers manual/authored controls and recording,
and has no trained policy. The candidate is experimental; the working-object
stability failure below remains visible, not treated as successful reconstruction.

The new person-only native example is
`.local/native-nongrid-real-01/interaction/lab`; current-runtime admission was
verified in .413 s with zero physics steps:

```sh
.venv-lab/bin/python -m weft lab --resume .local/native-nongrid-real-01/interaction/lab
```

Its saved
`authored-export/replay.html` and `video.mp4` are siblings of `lab/`.
It supports inspection/manual/authored control, **not a trained policy**: its
original pose-floor checks fail. The successful assisted simulation is not a
reason to hide that source-reference failure.

## Exact source interval through simulation and export

`native_interval.py` introduces `native-terminal-control-interval-v1` only for
verified native person-only `shared-world-v3`. The final action is shortened to
the unchanged source endpoint, with 16 observed physics substeps and nominal
timestep restored afterward. Task/source-clock hashes, selectors, policy runtime
closures, duration-weighted scoring, failed substeps and exact replay carry the
same interval. Other profiles keep the strict grid rule; no equipment, feet,
static-support or camera extension is silently enabled.

Inspection shows the exact task duration and final action. Interactive replay
schedules adjacent recorded task times, including partial failures whose geometry
remains at the last valid boundary. MP4/GIF remain explicitly labeled
**constant-rate event previews**, not native-time video. Their new timing metadata
is checked against the unchanged journal; legacy exports retain their format.

### Real licensed squat acceptance

Source `data/exercise/bodyweight-squats-danielflefil-500.mp4`, SHA
`ec636e28b4711887ecd9ff7014b64801d16055faf2f2b0bf6ed9221ea8280118`, is the
complete 68-frame, 25 fps GIF derivative of a CC BY-SA 4.0 exercise, not raw camera
footage. All 34 saved poses are reused, with zero inference. Frame 67 remains
unsampled; the complete retained reference ends at frame 66 / **2.64 s**.

- Source preparation: 1.16 s /291 MB, one OpenCV and one FFmpeg decode.
- Construction/interaction: 2.66 s /219 MB; 79 normal actions plus a
  **0.006666667 s** terminal action, 1,280 substeps and 81 journal events.
- Exact action/state replay passes. Assisted authored reward is .94165125,
  angular error .07899396 rad and floor penetration .00143655 model units.
- Original reference fails 13/34 source and 30/80 grid poses, maximum
  penetration .073841875. **No training/evaluation ran.**
- Export: 1.68 s /208 MB, no new rollout; all journal bytes unchanged.
  The video is 81 frames /2.7 s; recorded simulation is 2.64 s.

Evidence: `.local/native-nongrid-real-01/source-result.json` and
`interaction/{result,process-result,export-result,export-process-result}.json`.
The initial interaction enforced CPU60 but its RSS watchdog was denied; wall/RSS
were measured, not hard-enforced. Export used working CPU/wall/self-RSS bounds.
Body seal `20fb5d931fe49814981f7504b09dc22b43dcf9b2ab47d2cce6278936fc8ef3c9`;
lab seal `004dbe19376f075716555256e2eb0330d084c5a2007d118d2a4ec8bb2b31262b`;
export seal `edbb67d4dbc96caf25a1962bc6759d4683b45d583b8541cba023f1b78add137f`.

Verification: 22 dedicated native/runtime-export cases plus scoped task, policy,
recording and feet regressions; counts overlap, do not add them. Receipts live in
`.local/native-nongrid-01/`. Four saved legacy snapshots match the previous
runtime exactly, a narrow compatibility check. Inspection has 60 passing cases;
native/legacy replay and lab-window DOM checks pass; exact current asset pins and
commands are in `.local/native-interval-integration-01/ui-verification.json`.
No browser/device QA claim.

Saved-only floor diagnosis localizes all 13 failures to the left shin capsule's
ankle endpoint. At .64 s, root descent of .33901 model units exceeds leg
shortening compensation of .26017; the ankle falls .07884 from its starting
clearance. Its source visibility remains .817–.955. This points to coupled
monocular root/limb inconsistency, not a uniquely wrong joint or proven controller
failure. Nine knee clamps occur on the other side and outside failing windows.
No new decode, inference, forward dynamics or rollout was used. Evidence:
`.local/native-nongrid-real-01/source-floor-diagnosis/`, 14 pinned inputs, seal
`749e06809bd70e02a3abc29ec5fd4643f4b4ba80b1ec13b86b1cb6467b08a8ce`.
Next test a confidence-weighted root/lower-limb fit under declared camera/scale
and conditional contact hypotheses; do not shift the floor to conceal errors.

## Selective crop with original native timing

The ordinary exercise builder now accepts `--pose-recovery selective-crop` with
`--native-timing`. Original baseline decode callbacks preserve every frame before
inference. Only after unchanged recovery and quality validation does one FFmpeg
capture bind native timing to selected poses; it does not repeat pose inference.
Saved capture accepts only verified `exercise-pose-recovery-v1` input. Raw poses,
crop decisions and source hashes remain unchanged. 235 focused/regression checks
pass; receipts are `.local/selective-native-01/receipt.json` and its two XML files.

Real fresh Full/IMAGE preparation in `.local/selective-native-real-02` takes
18.04 s /464 MB. It retains 257 decoded /259 metadata frames, 1/1000 native ticks
and the signed **−3 ms decoder/native offset**. Two original/crop source decodes,
129 baseline inferences and 12 crop inferences are retained; one additional native
decode performs zero inference. Twelve crop candidates yield five
accepted source-only proposals. All 20 copied recovery/source artifacts match;
body has 61 sealed payloads, seal
`f46b64266be8f5e3ac39ae605d15f3014db80c4326b1c9552187dce16cf31e8e`.
This is body preparation, **not equipment-free scene admission**. Do not use
`--person-only` for the kettlebell clip. Original equipment preflight is retained.

Preserve `.local/selective-native-real-01`: old recovery evidence correctly
failed the current runtime hash check. Never repin it. The successful attempt
used fresh recovery under the actual current runtime. CPU180 was enforced;
wall/RSS were measured with cooperative stage deadlines, not a hard RSS guard.

## Unfamiliar exercise: heavier model did not recover the gap

`pose_model_comparison.py` requires the same complete source/frames/clocks and
runtime, independently recomputes pose quality, and reports regressions rather
than merging frames. Twelve focused tests plus ten custody/native checks pass.

`.local/unfamiliar-recovery-03` preserves one bounded existing Heavy/IMAGE model
attempt: 1,107 decoded frames /554 inference calls, 13.16 s /365 MB. Usable poses
fall **353→341 /554** (63.7184%→61.5523%); six gains do not offset 18 regressions.
The 91-sample /6.038 s gap remains entirely unusable. Even a prohibited frame
union would only reach 359, below the required 361. No body is admitted; no
threshold reduction, frame splicing, training or paid call follows this failure.
Saved source previews show self-occlusion, changing camera and the person crossing
the left frame boundary. A larger model alone does not restore absent evidence.

## Conditional equipment-mask geometry

`mask_track_hypotheses.py` streams exact original-cell masks with fixed mutual-IoU
and ambiguity gates, bounded pixels/work and no provider-ID matching preference.
Gaps, withheld slots or ambiguity permanently end a chain; no revival or
interpolation. It preserves original unresolved identities and expressly does
not prove physical identity/contact. Its source adapter checks source/raster
hashes; encoded-to-raster derivation is a separate check. 39 focused tests pass.

`scripts/studies/qualify_mask_associations.py` verifies the saved source case in
.67 s /117 MB: 257 frames /771 slots /747 masks, three conditional chains from
frame 8 through 256. All 752 consumed files remain unchanged. Four study tests
pass. `.local/mask-association-source-01` seal:
`f4a7ba1194e2a4799d8ecb5ceea208a9a4363295b99bba8a07f770f23540a6bd`.

`equipment_association_geometry.py` independently recomputes the chains and
checks all 747 original encoded masks against saved rasters once. Initial
object/mask correspondence and static-camera/static-object assumptions remain
unverified priors. Same-label uninterrupted construction evidence is required;
withheld relationship metadata is filtered from derived reports. Raw groups stay
unchanged in sealed originals. Legacy strict temporal admission remains unchanged.

`.local/mask-geometry-02` takes 15.71 s /235 MB with zero physics, inference,
source-video decodes or API calls. All 759 source files remain unchanged:

- `sam-2`: no improvement; exterior mismatch 549→549, no replacement config.
- `sam-3`: conditional candidate; exterior mismatch 4,273→3,166 (25.9% lower),
  volume 11.3% smaller, no new positive-mask loss and original envelope preserved.

These are image-fit results, not verified shape/contact or a promoted repair.
Study seal `dbd7a3b002e4624b84bc50184f029c131f08b441ee5935359913584bcdc50168`.
Preserve `.local/mask-geometry-01`, which failed before decode/fit because its
isolated package omitted the palette asset; the second attempt fixes packaging
only. The all-construction study is unaffected by the later withheld-metadata
fix; mixed-partition reuse must use the fixed implementation.

## Original-task equipment admission and comparison

`association_geometry_inputs.py` validates complete archived proof, selected
geometry, original source/task bytes, ordinary admission, rebound task and both
trial identities. The ordinary `exercise_geometry_inputs` reader dispatches the
new sealed profile; no bare-config shortcut, changed gate or inherited policy.
A deterministic stored ZIP retains all proof/source/runtime members without
extraction or execution. Cumulative read reservations, byte/file limits and
partition guards preserve the existing 512-file candidate-package limit.
**50 intake/archive regressions, nine association-geometry and 34 legacy temporal
checks pass**. Independent review caught and closed candidate-selection, original
task, nested trial identity and aggregate-read gaps before publication.

`.local/mask-geometry-admission-02` passes ordinary admission with only `sam-3`
replaced. The original task changes only its scene hash. One nominal authored
trial per world, seed 1729 and no jitter, completes **254 actions /4,064 substeps
/8.4667 s**. Both fail solely `equipment_stability:sam-2`; body errors are identical
(.15076954 rad angular, .01961757 model-unit root RMSE). The changed idle object
stays stable. **No task-success improvement or default promotion.**

The bounded process takes 4.05 s /371 MB; output is 221 MB within its 256 MiB cap.
No fit, decode, training or API call is repeated. Source hashes stay unchanged.
Study seal `78220fed2508a67ad1e9fcd2c1b2dd2626693a12279015dcdbb92c37420011ed`;
candidate seal `1c6ca8f69fefd0549c3507633812991f840efb1b767e52687c0e6e62a85c230e`.
Ordinary product intake verifies in .305 s /175 MB, zero physics; receipt:
`.local/mask-geometry-admission-intake-01.json`.

The complete ordinary geometry-study builder and `Lab.summary()` admission also
pass in `.local/mask-geometry-play-01`: 2.63 s /497 MB peak RSS under a 45 s /
512 MiB bound, zero additional controls/training/inference. Its exact candidate
scene and original rebound task remain unchanged; all 868 lab members verify,
seal `40b2162bd5b4f57211e385212db810f8237cd2727f20b3ce695200585941746e`.
The complete proof archive stays inside the lab. Lightweight metadata snapshots
retain its path/hash instead of duplicating the 26 MB archive twice under their
16 MiB cap; signatures and original archive bytes remain verified. The final
intact-snapshot assertion passes in `.local/mask-geometry-snapshot-tests-01.xml`.

Preserve `.local/mask-geometry-admission-01`: proof/config validation completed,
but its harness omitted the required baseline output directory and stopped before
native admission or any physics. Failure seal
`c9d04869f5714f0cb1063bab4688d6082e389fc725188db2490abd31f9a8c45a`.
The second attempt fixes packaging only and retains the candidate/task/seed.

## Integration state and next actions

All three delegated lanes, independent review and root integration are complete
for this checkpoint. No live worker, server or agent job remains. The overall
generalizable reconstruction goal stays active, not complete.
Separate avatar edits in `tools/avatar-assets/`, `assets/avatar_replay.js` and
`tests/avatar_replay_smoke.mjs` are not part of this checkpoint.

Next qualify source-constrained ground/root/contact reconstruction and broader
independent full-task learning. Idle-object image fit does not address the failed
working-object control/contact case.
The heavier unfamiliar model is a rejected unchanged hypothesis, not a retry
target. Hosted multi-user operation is still undeployed/unauthorized. Preserve
both API ledgers; this checkpoint consumes no provider budget. Main last fetched
equal to `origin/main` at `b0917e7`, stashes empty. The publication receipt is
`.local/native-interval-integration-01/publication.json` after verified push;
recheck worktree/remote before continuing. Only the explicitly owned paths in
that directory's `owned-paths.json` belong to this checkpoint.
