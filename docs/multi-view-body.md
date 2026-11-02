# Three- and four-camera body candidates

The optional [fixed-length angle fit](#optional-fixed-length-angle-fit) now improves
this saved exercise candidate. It still has four failing in-between poses.

Weft can now use **two or more supporting cameras per joint** from three or four
supplied views. This is an optional manual sandbox, not automatic camera recovery
from one video. Camera calibration, synchronized clocks, shared coordinates and
ground must be provided explicitly. Muscles remain illustrative; no learned
controller or physical-accuracy claim is added.

## Try the saved exercise candidate

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
  --camera first-camera.json --ground ground.json \
  --root-view root-view.json --additional-view third-view.json \
  --body-view both --out artifacts/my-three-camera-lab
```

Omit `--body-view` for root position only, or choose `directions`/`dimensions`
separately. No model call, training or server starts. Additional views cannot yet
be combined with `--partial-segments`. Shared-world/learning adapters that require
the older two-view method remain unavailable; they are not silently bypassed.

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
