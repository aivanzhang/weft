# Three- and four-camera body candidates

Optional [fixed-length fitting](#optional-fixed-length-angle-fit),
[partial directions](#optional-partial-limb-directions) and
[distal image constraints](#optional-distal-image-constraints), with
[retained depth alternatives](#optional-depth-alternatives), provide separate
saved comparisons. The newer [joint-image trajectory](#fit-knees-and-ankles-together)
passes the current sampled-floor tolerance. It does not validate physical
accuracy or replace the default.

Weft can now use **two or more supporting cameras per joint** from three or four
supplied views. This is an optional manual sandbox, not automatic camera recovery
from one video. Camera calibration, synchronized clocks, shared coordinates and
ground must be provided explicitly. Muscles remain illustrative; no learned
controller or physical-accuracy claim is added.

## Try the three-view baseline

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-third-camera-body-lab-01
```

Enter `controls`, choose an `action`, then `step`. Use `record start`, `record stop`
and `files` to save and inspect actions. Nothing moves until you step. To start the
visual lab yourself instead:

```sh
.venv-lab/bin/python -m weft lab \
  --resume artifacts/aspset-third-camera-body-lab-01
```

This candidate **fails the floor check**. It is useful for inspection and manual
interaction, not a successful exercise reconstruction. The existing equipment
sandbox and all defaults are unchanged.

- [Recorded manual interaction](../artifacts/aspset-third-camera-body-lab-01/sessions/session-c3c8e7838e80/replay.html)
  and [video](../artifacts/aspset-third-camera-body-lab-01/sessions/session-c3c8e7838e80/video.mp4).
- [Joint support and conflicts](../artifacts/aspset-third-camera-body-lab-01/run/two-view-triangulation.json),
  [body geometry](../artifacts/aspset-third-camera-body-lab-01/run/two-view-body.json),
  [floor failures](../artifacts/aspset-third-camera-body-lab-01/run/reference-feasibility.json).
- [Saved comparison audit](../artifacts/aspset-third-view-audit-01/report.json) and
  [model versus image-estimated joints](../artifacts/aspset-third-view-audit-01/model-image-joints.json).

These local artifacts are not included in a fresh Git clone.

## Add another calibrated view

Build from the same sealed original mobile body and [two-view root inputs](two-view-root.md#build-a-separate-candidate).
Add `--additional-view third-view.json`; repeat once for a fourth camera.

```json
{
  "schema_version": 1,
  "name": "third-camera",
  "evidence": "third-evidence.json",
  "camera": "third-camera.json",
  "clock_provenance": "Describe how these exact source frames align with the other views"
}
```

Paths resolve relative to this JSON. The original root configuration's
`world_provenance` must describe the shared coordinate system for **all** views.
Matching frame rate alone does not prove physical synchronization.

```sh
.venv-lab/bin/python -m weft camera-body \
  --run path/to/original-mobile-run \
  --control-joint right_knee \
  --camera first-camera.json --ground ground.json \
  --root-view root-view.json --additional-view third-view.json \
  --body-view both --out artifacts/my-three-camera-lab
```

Omit `--body-view` for root position only, or choose `directions`/`dimensions`
separately. No model call, training or server starts. Additional views can be
combined with `--partial-segments`.

The input can also be a construction-only body prepared with
`exercise-body --evidence-run path/to/sealed-poses --out artifacts/my-body`.
Pass that output as `--run`. `--control-joint` selects the first joint for manual
inspection; it does not infer a task, reward or policy. The selection is saved
separately, and the source remains unchanged. Without this flag, an existing
source task/policy selector is required.

Grounded reconstructed bodies can now enter the [shared-world workflow](local-lab.md#import-a-camera-based-body-experimental)
through its existing source and floor checks. Failed candidates remain available
for inspection; they are not automatically admitted or trained.

## What is retained

- Every original source/frame/joint slot, including unsampled and conflicting
  observations. Every original retained body pose still requires both hips.
- Exact evidence and camera hashes, declared clocks, original ground and input
  files, per-joint contributing views and rejected estimates.
- Existing bounded pixel-error solver: every qualifying view participates; no
  favorable-pair selection, outlier removal or guessed missing depth.
- Explicit nearest-proper solver rotations for decimal-rounding error, bounded
  to 1e-5 per matrix element. Original camera matrices stay unchanged. This is
  numerical preparation, not calibration fitting.
- Missing complete limb directions retain labeled single-view priors. Fixed
  body-length, angle, floor and optimization limits remain unchanged.

New methods are `multi-view-root-body-v1` and `multi-view-body-v1`. Historical
`two-view-*.json` artifact filenames remain for compatibility; their contents
identify the actual method and all three/four views. The legacy two-view path is
unchanged.

## Real-source result: improved coverage, worse floor consistency

ASPset clip `04ac-0003` retains all 188 source slots and 47 body poses. The same
frozen manually labeled ground is used for both candidates.

| Check | Two views | Three views |
| --- | ---: | ---: |
| Complete left-arm estimates | 26/47 | 47/47 |
| Complete left-leg estimates | 19/47 | 22/47 |
| Failing source floor poses | 1/47 | 4/47 |
| Failing interpolated floor poses | 3/111 | 7/111 |
| Maximum floor penetration, model units | 0.066662 | 0.092148 |

Frame 80's missing knee/ankle now have mid/right image support, but its model
ankles still differ from those image estimates by about 0.112/0.140 m under the
supplied scale. The model and image-estimated joints are different artifacts;
neither establishes measured anatomy or contact. All regressions remain saved,
and the candidate is not promoted.

The actual lab recorded **39 actions, one reset and 41 frames**, with every state,
reward and completion flag replayed exactly. Construction took 1.687 s and
150.8 MB peak RSS; interaction/export/replay took 1.050 s and 160.8 MB. Across 60
in-process action/state/JSON calls, median latency was 0.832 ms and p95 0.880 ms;
these are not browser, HTTP or sustained-use measurements.

The saved-only audit checked 776 manifest members, all 6,204 joint slots and 4,247
projections. It retained 84 conflicting joint estimates and all missing slots.
No paid API call was used. Exact source, runtime, test and process evidence is in
the [handoff](handoffs/aspset-third-view.md).

## Optional fixed-length angle fit

The input joints imply inconsistent limb lengths in some frames. `--fit-limbs`
adjusts supported joint angles while keeping body lengths, root position and
ground fixed. It uses the same body constructor and lab, with no new inference.
Add the flag to the construction command above; `--body-view directions` or
`both` is required. It can be combined with the optional partial-limb mode below.

Try the saved fitted candidate:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-fixed-limb-body-lab-01
```

Use `lab --resume` instead of `lab-console --resume` to start the visual interface
yourself. The same manual controls and recording commands apply.

| Check | Unfitted three-view body | Fixed-length angle fit |
| --- | ---: | ---: |
| Failing source poses | 4/47 | 0/47 |
| Failing interpolated poses | 7/111 | 4/111 |
| Maximum floor penetration, model units | 0.092148 | 0.055751 |

All 47 sampled poses pass the unchanged 0.025 floor limit; three still have
smaller intersections. The full reference remains **infeasible** between frames
80–88. Frame 84 lacks a supported left ankle, so its original single-view leg
prior remains unchanged. No default or learned-policy promotion occurs.

Mean image error improves in each supplied camera, but **278 of 1,620 compared
joint/view points get worse**. This is construction agreement, not independent
accuracy. The fit preserves all 6,768 source-image slots and all missing evidence.
Corrections are limited to 0.30 radians per angle. Its 12-iteration cap and
no-descent exits do not establish optimizer convergence or physical accuracy.

- [Before/after geometry and remaining failures](../artifacts/aspset-fixed-limb-inspection-01/comparison.png):
  solid chains are the left leg; dashed chains are the right leg.
- [Recorded manual replay](../artifacts/aspset-fixed-limb-body-lab-01/sessions/session-d79d024a6b5f/replay.html)
  and [video](../artifacts/aspset-fixed-limb-body-lab-01/sessions/session-d79d024a6b5f/video.mp4).
- [Independent saved audit](../artifacts/aspset-fixed-limb-audit-01/report.json),
  [complete image comparisons](../artifacts/aspset-fixed-limb-audit-01/source-images.json),
  and [bounded proposal histories](../artifacts/aspset-fixed-limb-body-lab-01/run/two-view-body.json).
- [Exact implementation, verification and next work](handoffs/aspset-limb-fit.md).

Actual recording again verified 39 actions, one reset and 41 frames exactly.
Construction took 1.594 s /152.2 MB peak RSS; interaction/export/replay took
1.026 s /161.8 MB. No paid API call, source-video decode or new inference was used.

## Optional partial-limb directions

Add `--partial-segments` to use an available shoulder-to-elbow or hip-to-knee
direction when a wrist or ankle cannot be reconstructed. This now supports two,
three or four cameras and can accompany `--fit-limbs`. Each joint retains its
actual contributing cameras; an unavailable endpoint never gains invented views.

Complete limbs keep the same fixed-length fit. For incomplete limbs, the upper
segment turns toward the supported direction while the unobserved bend and twist
remain explicit single-view priors. This transport is **not** covered by the
complete-limb fit's 0.30-radian bound. Source positions, ground, lengths and forces
stay fixed; missing points remain missing.

The saved exercise comparison is mixed, not a replacement default:

| Check | Fixed-length fit | Fit plus partial directions |
| --- | ---: | ---: |
| Failing source poses | 0/47 | 1/47 |
| Failing interpolated poses | 4/111 | 3/111 |
| Maximum floor penetration, model units | 0.055751 | 0.077483 |
| Mean left-camera error, px | 9.2281 | 10.6642 |
| Mean middle-camera error, px | 8.2954 | 6.6110 |
| Mean right-camera error, px | 9.2751 | 6.1971 |

All 24 supported partial limbs match their reconstructed upper-segment direction;
all 163 complete-limb fits remain unchanged. All 6,768 image slots remain visible,
including 41 worsened comparisons. Frame 84 now has a floor intersection at the
ankle end of the left lower-leg shape. Correct direction alone does not establish a correct fixed-length
body, unseen ankle, contact or physical accuracy.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-partial-limb-body-lab-01
```

Use `controls`, `action`, `step`, `record start`, `record stop`, and `files`.
No learned policy is available. Use `lab --resume` instead to start the visual
interface yourself.

- [Manual replay](../artifacts/aspset-partial-limb-body-lab-01/sessions/session-4f6d03008f86/replay.html)
  and [exported video](../artifacts/aspset-partial-limb-body-lab-01/sessions/session-4f6d03008f86/video.mp4).
- [Independent comparison](../artifacts/aspset-partial-limb-audit-01/report.json),
  [all source-image slots](../artifacts/aspset-partial-limb-audit-01/source-images.json),
  and [partial-limb checks](../artifacts/aspset-partial-limb-audit-01/partial-limbs.json).
- [Verification and remaining work](handoffs/aspset-partial-limb.md).

123 Python checks and 71 mocked UI checks pass. The real 39-action/one-reset
recording replays all 41 states and frames exactly. This verifies the interaction
workflow, not the reconstruction's physical accuracy. No default was changed.

## Optional distal image constraints

Add `--fit-partial-pixels --fit-limbs --partial-segments` to the construction
command, with `--body-view directions` or `both`. When an ankle/wrist lacks a
supported 3D estimate, any qualified original images can still constrain it.
This opt-in method changes bend and rotation around the upper bone by at most
0.30 radians each. Upper-bone direction, root, lengths, ground, forces and
complete-limb fits stay fixed. **Distal depth remains inferred**, not observed.
All contributing/excluded views and rejected proposals remain inspectable.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-distal-pixel-body-lab-01
```

Use the same controls and recording commands above; `lab --resume` starts the
visual interface when you run it. No learned policy is available for this method.

| Check | Partial directions | Added distal pixels |
| --- | ---: | ---: |
| Failed source poses | 1/47 | 1/47 |
| Failed interpolated poses | 3/111 | 3/111 |
| Worst floor penetration, model units | 0.077483 | 0.072313 |
| Left-camera mean image error, px | 10.6642 | 10.6286 |
| Middle-camera mean image error, px | 6.6110 | 6.5939 |
| Right-camera mean image error, px | 6.1971 | 6.2201 |

This is a mixed construction result: 37 individual image comparisons worsen,
and the floor check still fails. It is not a completed reconstruction or a new
default. All 24 fits, 163 unchanged complete fits and 6,768 source-image slots
were independently checked from saved artifacts. The 39-action, one-reset
recording replays all 41 states and video frames exactly.

- [Manual replay](../artifacts/aspset-distal-pixel-body-lab-01/sessions/session-9f4a8ded64d7/replay.html)
  and [video](../artifacts/aspset-distal-pixel-body-lab-01/sessions/session-9f4a8ded64d7/video.mp4).
- [Independent comparison](../artifacts/aspset-distal-pixel-audit-01/report.json),
  [all image slots](../artifacts/aspset-distal-pixel-audit-01/source-images.json),
  and [fit histories](../artifacts/aspset-distal-pixel-body-lab-01/run/two-view-body.json).
- [Exact verification, limits and next work](handoffs/aspset-distal-pixels.md).

176 distinct scoped Python cases and 72 mocked UI checks pass. Construction
took 1.654 s /151.4 MB peak RSS; interaction/export/replay 1.035 s /161.9 MB.
No paid API call, original-video decode or new inference was used. Browser
painting and sustained responsiveness are not established by these measurements.

## Explicit image and pose uncertainty

The optional v2 fit replaces the fixed image-versus-pose penalty with declared
error scales. Add `--partial-pixel-uncertainty examples/partial-pixel-uncertainty.json`
to the construction command with `--fit-partial-pixels --fit-limbs --partial-segments`.
The [example settings](../examples/partial-pixel-uncertainty.json) assume image
error of 0.25% of the larger image dimension and pose/adjacent-correction scales
of 0.30/0.45 radians. **These are assumptions, not measured confidence.**
Image resizing preserves the fit; a more distant, smaller image motion has less
influence. All source views and the original .30-rad correction bound remain.

Try the separately saved candidate:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-distal-uncertainty-body-lab-01
```

The lab identifies weak image directions and keeps depth inferred. Compared
with v1, failing interpolated poses fall from 3/111 to 2/111, but 1/47 source
poses still fails. Worst floor penetration falls from 0.072313 to 0.049340
model units. Mean left-camera agreement improves; middle/right worsen, with
39 individual regressions retained. This is not a replacement default or a
successful exercise reconstruction.

- [Manual replay](../artifacts/aspset-distal-uncertainty-body-lab-01/sessions/session-c5c1fedf9c4f/replay.html)
  and [video](../artifacts/aspset-distal-uncertainty-body-lab-01/sessions/session-c5c1fedf9c4f/video.mp4).
- [Independent comparison](../artifacts/aspset-distal-uncertainty-audit-01/report.json),
  [all image slots](../artifacts/aspset-distal-uncertainty-audit-01/source-images.json),
  and [settings/proposals](../artifacts/aspset-distal-uncertainty-body-lab-01/run/two-view-body.json).
- [Verification and remaining geometric ambiguity](handoffs/distal-pixel-uncertainty.md).

213 scoped Python cases /73 mocked UI checks pass. Actual 39-action/one-reset
recording replays all 41 states and frames exactly; immutable run and lifecycle
metadata stay unchanged after closing. No new inference or paid API call.

## Optional depth alternatives

Add `--partial-depth-branches` to the v2 construction command above. Keep
`--partial-pixel-uncertainty`, `--fit-partial-pixels`, `--fit-limbs`,
`--partial-segments` and `--body-view directions` or `both`.

An image can place an ankle along a ray without identifying its depth. This
stage saves possible ankle/wrist positions where each ray meets the fixed-length
limb, including missing intersections and rejected alternatives. It can select
a position only when agreement with **all** available images improves, nearby
supported poses favor it, and another depth is not similarly plausible.
That smooth-motion assumption does not make depth measured.

This is a separate geometric stage after the unchanged small local fit. It can
rotate around the upper bone by up to pi radians and retains the existing
0–2.6-radian knee/elbow range. Root, upper-bone direction, limb lengths, ground
and complete-limb fits stay fixed. It never selects a depth to pass a floor check.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-depth-branches-body-lab-01
```

Use `controls`, `action`, `step`, `record start`, `record stop`, and `files`.
Use `lab --resume` instead of `lab-console --resume` to start the visual lab
yourself. No learned policy is available for this candidate.

**This source case remains unresolved.** All 24 partial-limb poses abstain:
15 lack nearby supported observations on both sides; nine have no qualified
alternative. All 52 alternatives and 44 missed ray/limb intersections remain
saved. Body positions and image errors are identical to v2; floor failures stay
1/47 source poses and 2/111 interpolated poses, worst penetration 0.049340 model
units. No default, learned-policy or physical-accuracy promotion occurs.

- [All depth alternatives and reasons](../artifacts/aspset-depth-branches-body-lab-01/run/two-view-body.json).
- [Independent saved-data audit](../artifacts/aspset-depth-branches-audit-01/report.json)
  and [all source-image slots](../artifacts/aspset-depth-branches-audit-01/source-images.json).
- [Manual replay](../artifacts/aspset-depth-branches-body-lab-01/sessions/session-7a63eab3930f/replay.html)
  and [video](../artifacts/aspset-depth-branches-body-lab-01/sessions/session-7a63eab3930f/video.mp4).
- [Exact bounds, tests, failures and next work](handoffs/distal-depth-branches.md).

234 focused Python cases /74 mocked UI checks pass, including selected depths
on controlled two-, three- and four-camera geometry. Actual 39-action/one-reset
recording replays all 41 states and video frames exactly. Construction took
1.688 s /151.5 MB peak RSS; interaction/export/replay 1.047 s /162.1 MB. These
verify the manual workflow, not source accuracy or browser responsiveness.

## Fit knees and ankles together

The earlier distal fits kept the knee/elbow position fixed. This optional fit
adjusts knee/elbow and ankle/wrist images together, linking nearby corrections
by their actual source times. It still keeps the root, torso, body lengths,
ground and forces fixed. Missing observations remain missing; depth is inferred.

Start the latest saved manual candidate:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-joint-pixel-body-lab-01
```

Use `controls`, `action`, `step`, `record start`, `record stop`, and `files`.
For the visual interface, start `lab --resume` instead of `lab-console --resume`
yourself. This candidate has no learned policy.

| Check | Previous v2 candidate | Joint-image trajectory |
| --- | ---: | ---: |
| Failing source floor poses | 1/47 | 0/47 |
| Failing interpolated floor poses | 2/111 | 0/111 |
| Maximum penetration, model units | 0.049340 | 0.015498 |
| Mean left-camera image error, px | 10.4368 | 10.3406 |
| Mean middle-camera image error, px | 6.6740 | 6.3458 |
| Mean right-camera image error, px | 6.3521 | 6.2739 |

The unchanged floor limit is 0.025 model units. Two smaller intersections
remain, so **passing these samples does not mean collision-free motion**.
All 6,768 image slots remain saved, including 397 individual regressions versus
v2. Of 188 limb poses, 134 have a weakly image-constrained angle direction;
both arm fits reach their iteration cap. This is one exposed development clip,
not independent physical validation or proof that unfamiliar videos will work.
Against the earlier partial-direction baseline, the right-camera mean gets
worse (6.1971→6.2739 pixels); that comparison is also retained in the audit.

To construct a separate candidate, add
`--fit-joint-pixels examples/joint-pixel-uncertainty.json` to the camera-body
command with `--fit-limbs --partial-segments --body-view both`. It cannot stack
with `--fit-partial-pixels` or `--partial-depth-branches`.
The [settings](../examples/joint-pixel-uncertainty.json) explicitly assume image
error of 0.25% of the larger image dimension, an angle scale of 0.30 radians,
and correction-speed scale of 4.5 radians/second. That speed describes changes
in corrections, **not measured body motion**. Each angle correction is capped
at 0.30 radians; no floor or contact score enters this fit.

- [Manual replay](../artifacts/aspset-joint-pixel-body-lab-01/sessions/session-512ff00a4618/replay.html)
  and [video](../artifacts/aspset-joint-pixel-body-lab-01/sessions/session-512ff00a4618/video.mp4).
- [Independent audit](../artifacts/aspset-joint-pixel-audit-01/report.json),
  [all image comparisons](../artifacts/aspset-joint-pixel-audit-01/source-images.json),
  and [all original fits and new proposals](../artifacts/aspset-joint-pixel-body-lab-01/run/two-view-body.json).
- [Exact inputs, bounds, checks and remaining work](handoffs/joint-pixel-trajectory.md).

269 focused Python cases and 75 mocked UI checks pass. Actual interaction
records 39 actions/one reset and exactly replays 41 states and video frames;
the completed run and lifecycle stay unchanged after close. Construction took
2.458 s /159.9 MB peak RSS; interaction/export/replay 1.092 s /164.2 MB.
These are local pipeline measurements, not browser responsiveness. No new
source-video decode, inference or paid call was used.

### Use full-body controls

The exact reconstructed body also works in the
[shared physics/RL lab](local-lab.md#import-a-camera-based-body-experimental),
with 19 joint motors and explicitly enabled body-position assistance. Its
184-step programmed-reference episode passes the unchanged simulator task;
190 recorded actions replay exactly. This is a separate optional lab, not a
learned policy or validated real-world contact. All three cameras, original
body estimates, uncertainties and source slots remain preserved.
