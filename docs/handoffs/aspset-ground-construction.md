# ASPset source-qualified ground construction

## Active — November 1, 2026, PST

Current checkpoint is **progress**, not full-goal completion. The reusable
`stereo-ground` CLI, report, conditional source plane and frozen-ground body
construction are verified; 121 distinct scoped cases pass. The body retains a
source-frame-80 floor failure, caused in the current model by a left-leg prior
where two-view knee/ankle support is unavailable. No plane, gate, default or
policy was changed to hide it. All root children are terminal/reaped. Preserve
the independent avatar worktree changes. Publication receipt for this checkpoint:
`.local/aspset-stereo-ground-publication-01.json` (absence means not yet published).

Previous goal turn is **progress**: `a4b4dc6` publishes explicit person regions,
183 passing cases, both saved-body constructions and calibrated source assessment.
Remote publication was independently verified. Full goal remains active; these
model-derived bodies do not qualify ground/contact or complete task learning.

Current source: `data/aspset/04ac-0003-01`, two188-frame50fps4K views with publisher
pinhole cameras. Published joint references remain assessment-only. The camera
world is originally left-camera aligned; declared mm scale does not establish
gravity or a horizontal floor. Never feed the published C3D coordinates into
ground fitting or select placement using body clearance or task scores.

Existing `camera_ground` requires an explicit horizontal world plane. Existing
two-view body/root methods preserve that binding and do not invent ground.
Before extending that pipeline, test a source-only background-surface hypothesis.
Reuse `regional_features.regional_sift`, the existing calibrated two-ray primitive
and the earlier carpet study's bounded fit/comparison design. Do not change the
floor-target calibration contract or repeat failed controller tuning.

## Frozen foreground-texture screen

Owner: root. Output `artifacts/aspset-ground-texture-01`; worker completed/reaped.
Inputs: already saved960×540 inspection JPEGs of original frame0 in both views,
their hashes, original cameraJSON and original-video hashes. The declared raster
transform is exactly one-quarter scale; this screen does not claim the JPEGs are
lossless or use new source-video decoding. No body/pose/reference input.

Attention masks: foreground rectangle `[0,348,960,540]` in each inspection image.
Exclude the visible cone rectangles `[405,365,460,410]` (left) and
`[20,340,60,380]` (mid). These are reviewer-declared apparent ground regions,
not automatic segmentation or measured plane labels. Masks bound feature centers,
not full descriptor support. Preserve every candidate correspondence/rejection.

Bounds: one worker, one OpenCV thread,1,200features per view,0.7 mutual descriptor
ratio; existing two-ray depth/ray-angle gates plus at most0.75inspection-pixel
reprojection error (3 original pixels). Freeze every-third-match development split
before geometry. Fit at most512seed17 plane hypotheses/2,048draws from construction
points,1.5cm inlier distance. Retain the previous independent geometry gates:
12training inliers,80%training inliers,6comparison matches, complete comparison
geometry,2cm comparison95th-percentile/4cm maximum plane error,0.08m² spread and
at most45° from nominal camera-up. No threshold tuning on this result.

Resource envelope:30s external/20s cooperative,768MiB sampled peak RSS,32MiB output,
32files. No inference, physics, network, API, key/ledger or installation; external
and recurring cost$0. If texture cannot qualify a plane, retain the rejection and
choose a genuinely different source constraint; never substitute a fitted body
floor while calling it independently reconstructed ground. Preserve avatar work.

Result: 1,068 left /739 mid features, **zero mutual matches**, no triangulated
points and no plane. All eight admission checks fail. Worker 0.151426 s,
supervised 0.446957 s, peak RSS 126,107,648 B. Seal:
`d4b7ae87e00d57be94a15c50e24e98613bd0641539cef660a241001823126bb3`.
Do not repeat this unchanged screen or relax its gates.

## Manual-marker construction — changed hypothesis

Source-only inspection of the already saved original frame-0 JPEGs identifies
three colored markers. The distant orange marker's left-view base is partly
hidden. Explicit manual apparent-base centers, not cone apexes, are frozen in
`artifacts/aspset-ground-markers-01/plan.json` before triangulation. Pixel ranges
are reviewer sensitivity assumptions, not measured error bounds. The marker
footprints, ground planarity, camera scale and calibration remain uncertain.

Three points pass the unchanged 3-original-pixel reprojection screen; maximum
error is 0.973914 px. Triangle area is 64.0116 m² and its normal is 0.684885° from
nominal camera-up. This is only a **conditional manually guided plane**: three
construction points provide no independent plane/ground/gravity validation.
Seal `4ebee242abe4b9ccae7f287ac503adde5802dee34f90b8d5c2442399730fc19c`.
No pose/body/reference input, media decode, model, physics or API call. The tiny
feasibility calculation ran synchronously (0.000499 s, 35,012,608 B peak RSS);
its declared time/RSS envelope was observed, not a separate hard supervisor.

Next root-owned change: reusable explicit stereo-ground annotations and CLI.
Require source/calibration/image/frame identities, bounded points, no discarded
construction labels, fixed reprojection/degeneracy/planarity checks, and reported
one-coordinate-at-a-time label sensitivity. Preserve an explicit unavailable
independent-check status when no assessment points exist. Reframe cameras with
one proper rotation so the conditional plane becomes horizontal; never fit a
body, alter source pixels/clocks, or call this measured gravity. Bound to 16
points, 128 sensitivity perturbations, two supplied camera streams, existing
camera limits and small local files; no paid/recurring cost or new dependency.
After analytic/CLI checks, construct one new source-bound camera/body candidate
against this frozen plane and retain failed floor/coverage checks without repair.

The reusable `stereo-ground` CLI now passes 39 focused cases; 82 unchanged
camera-ground/root/body compatibility cases also pass. The source-specific
preparation script creates `artifacts/aspset-ground-inputs-01` (seal
`e38b93ae84a6aa5d2208d8a51c80af1d8d6860071fba742bdc137ad81930fb71`).
Actual CLI output: `artifacts/aspset-manual-ground-01` (seal
`bdf11ee7ddd7b74fce3420c6ee36902acc35b9c57d15117536671bf6745f23f2`).
All 24 single-coordinate perturbations are retained; maximum angular change
0.146197° and center-plane change 0.008641 m. These are not confidence bounds
or combined-label/calibration uncertainty. Independent assessment is unavailable.
CLI process: 0.218618 s /121,896,960 B peak RSS; two saved-JPEG decodes, no source
video/model/physics/API activity. Runtime sources and inputs are sealed together.

Both saved pose streams have 47 available camera roots and 141 unsampled slots;
the prior two-view candidate has both hips for all 47 retained source poses.
Next bounded native job: one left-primary camera/body lab, explicit `right_knee`
manual inspection selection, supplied frozen ground, two-view root and `both`
body directions/dimensions. Keep missing limb priors explicit; no body/floor
repair, reference C3D, dynamics, inference, video decode, server or API.
Freeze runtime from published `a4b4dc6`, excluding ongoing avatar edits. Native
limits: 30 s external, 25 s CPU, 768 MiB sampled RSS, at most 6 compilations,
12 data allocations and 1,000 static forwards; zero physics steps. New input
`artifacts/aspset-camera-body-input-01`, output `artifacts/aspset-camera-body-lab-01`,
control/runtime `.local/aspset-camera-lab-01`. No job started at this note update.

## Native construction and saved-result verification

The first native worker stopped at its 12-data-allocation guard after four
compilations and 212 static forwards. `HumanEnv.reference_snapshot` creates one
scratch `MjData` per retained preview pose; 47 such previews exceed the original
harness allowance. No lab was published. Preserve `.local/aspset-camera-lab-01`.

The corrected harness reuses the exact frozen runtime and inputs, raises only
that declared operation allowance to 64, and uses new control/output directories.
`.local/aspset-camera-lab-02` completed/reaped: **four compilations, 53 data
allocations, 253 static forwards, zero dynamics**. Worker 1.162085 s /134,021,120 B
peak RSS; supervised 1.259906 s; pipeline construction 0.629109 s. No source
decode, inference, reference C3D, API, server or deployment. The source video and
raw pose bytes are unchanged. Input seal:
`0e63ca94b07bd68b1e76bae8fe2e8987737c9e4289f84013519f07cdb457739b`.

New manual candidate: `artifacts/aspset-camera-body-lab-02`, run seal
`99ac31f00f82f9c9a7e347aa8e6bc3b17196f523683911db331515a7629fde29`.
It retains all 47 original references /188 source slots and the exact supplied
ground. Root and partial body construction use the supplied two-view estimates;
missing directions remain explicit priors. Left leg: 19 two-view /28 prior poses;
left arm: 26 two-view /21 prior poses; torso/right leg/right arm: 47 two-view each.

The unchanged 0.025 floor limit fails **one of 47 source poses and three of 111
control-grid poses**. Maximum penetration is 0.0666618937 model units at original
frame 80 /1.6 s, `left_knee_geom`. At that frame, both the left knee and ankle lack
usable two-view image evidence; the left-leg direction is a monocular prior.
The separate descriptive surface-gap audit has four penetrating /43 positive-gap
poses, range −0.066662 to 0.074030 and median 0.039071 model units. Positive gaps
are not proven missing contact, and nominal ground is not independent truth.

Independent saved-artifact audit:
`.local/aspset-stereo-ground-acceptance-audit-01.json`, SHA-256
`f1079a2e2ce3c61d6e0c2a30c6ed20c28d090581b8fbbb1e6321bfce048fa1fe`.
It verifies 456 unique sealed members in six packages, all six marker projections,
376 camera transforms/clocks, exact body/source/ground bindings, source/grid
failure counts, signed-clearance aggregates and eight report links. No repeated
native or inference work. Runtime 0.156623 s.

Tests: 39 new core/package/CLI cases plus 82 existing camera-ground/root/body
cases. XML receipts are listed in the audit. One new envelope test initially
used a tilted plane whose perpendicular offset still fell inside the limit;
the corrected analytic fixture uses a horizontal out-of-envelope plane. This
test correction did not alter production thresholds. The earlier allocation
guard failure is likewise a harness failure, not a successful model result.

Maximum numerical-capacity check: 16 points, 1,800 camera rows per view and 128
single-coordinate perturbations, 0.155158 s /39,157,760 B process peak RSS.
Receipt `.local/stereo-ground-capacity-01.json`. It is analytic capacity evidence,
not sustained UI performance, source accuracy or a full-goal pass.

## Reuse and next safe action

Read [the public CLI guide](../stereo-ground.md) and the saved ground report at
`artifacts/aspset-manual-ground-01/index.html`. Preparation is reproducible with:

```sh
.venv-lab/bin/python scripts/studies/prepare_aspset_ground.py --out artifacts/NEW-ground-inputs
.venv-lab/bin/python -m weft stereo-ground \
  --camera artifacts/NEW-ground-inputs/camera-0.json \
  --camera artifacts/NEW-ground-inputs/camera-1.json \
  --image artifacts/aspset-person-region-01/left/background.jpg \
  --image artifacts/aspset-person-region-01/mid/background.jpg \
  --annotations artifacts/NEW-ground-inputs/annotations.json --out artifacts/NEW-ground
```

Do not rerun these unchanged studies simply to rediscover the saved result. The
user may inspect the manual candidate with `weft lab --resume`; no server or new
candidate interaction/recording was exercised in this checkpoint. The established
equipment console/replay remains unchanged and already verified separately.

- [ ] Resolve missing left-limb source support before any new controller/floor
  sweep. The publisher has a third camera; only the earlier incomplete prefix
  was acquired, never extracted or inferred. Inspect existing archive metadata
  before bounded additional acquisition, or use another genuinely new source
  constraint. Preserve original clocks, unavailable slots and raw observations.
- [ ] If adding additional-view reconstruction, build a reusable source-bound
  multi-view method with independent projection/degeneracy checks; never drop
  missing slots or use the published C3D assessment to choose body/ground values.
- [ ] Qualify ground/equipment contact, independent physical accuracy, meaningful
  full-task learning and remaining goal dimensions. Three manual markers and
  passing software tests do not satisfy those completion gates.

No key/ledger access or new spend. The $5 shared Meta authorization is unchanged.
Root-owned files are the two `stereo_ground` modules/tests, CLI entry, study input
adapter and linked documentation/TODO/handoff. Avatar edits belong to the other
session and must not be staged, reset, stashed or discarded here.
