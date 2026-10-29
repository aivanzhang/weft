# Fresh full-body geometry diagnosis

**Do not promote the static shape corrections.** All three tested alternatives worsen the later source-frame comparisons. No model, reference, floor, root trajectory, or acceptance threshold changed.

## Evidence and method

The [sealed diagnostic](../artifacts/fullbody-geometry-diagnosis-01/report.json) uses the original fresh squat baseline: 34 retained poses from a completely decoded 68-frame, 500×500, 25 fps public-source derivative. Input hashes and exact raw pose arrays are retained. All poses were compared; none were skipped.

Actual MuJoCo joint positions are centered at the pelvis, removing root translation and floor-height offsets from the comparison. Each limb endpoint error is decomposed into attachment error, accumulated bone-length error, and direction error. These vectors sum to the endpoint error; their RMS magnitudes must not be added as scalar contributions. No dynamics, floor residual, floor fitting, or manual labels enter the diagnostic.

Four fixed alternatives were declared before examining results: original geometry, side-specific static lengths, static torso-local attachments, and both changes. Confidence-weighted medians use only frames 0–40, the first 60% of source timestamps. Length changes are capped at 15%; attachment changes at 0.03 model units. Hip attachments stay opposite around the unchanged pelvis. Frames 42–66 are reserved for these shape comparisons. This is a development split, not independent physical ground truth or a final holdout; the inherited baseline scale and original reconstruction used the full clip.

## What causes the disagreement

Across 272 elbow/wrist/knee/ankle comparisons, RMS attachment, length, and direction terms are respectively 0.02917, 0.02949, and 0.00904 model units. Most model directions reproduce the inferred world-landmark directions to numerical precision. The issue is not a general rotation-order error.

- Inferred left forearm median length changes from 0.1900 in the prefix to 0.2533 in later poses, approximately 33%. This is evidence of pose-estimate variation, not changing anatomy or a reliable static subject measurement.
- The largest endpoint discrepancy is the left wrist at frame 26: 0.12641 model units despite reported landmark confidence 0.862. Confidence does not establish geometric accuracy.
- Eight source poses, frames 26–40, have right-knee estimates above the authored 2.6-radian model limit. At frame 38, inferred flexion is 2.825; the right ankle's vertical discrepancy is −0.0474, comprising attachment −0.0050, length +0.0068, and clipping-related direction −0.0493. Its source ankle confidence is 0.4085. This does not justify relaxing the joint bound.
- Right ankle vertical discrepancy changes from +0.0809 at frame 0 to −0.0474 at frame 38. Left ankle frame 42 differs by −0.0178, mostly accumulated length mismatch. Geometry disagreements therefore exist independently of the root and plane, but this study does not assign all floor conflicts to geometry.

## Later-pose checks

Errors below cover all 12 hip-centered joints on the 13 later poses. Image comparisons use source pose-model outputs, not independently labeled anatomical landmarks. Projection retains the original uncalibrated scale and camera x/−z mapping.

| Geometry | World endpoint RMS, model units | Image endpoint RMS, pixels |
|---|---:|---:|
| Original | 0.03031 | 10.637 |
| Static side-specific lengths | 0.03329 | 11.442 |
| Static attachments | 0.03590 | 11.992 |
| Both | 0.03642 | 12.296 |

Some individual endpoints improve, but aggregate and other-joint regressions remain. No alternative was selected or applied. Static shape changes also cannot repair projected segment-direction errors while angles stay fixed.

## Follow-up hypothesis — tested, not promoted

The [four-source pixel-length experiment](pixel-body-shape.md) tests static segment lengths against confidence-weighted **pixel vector magnitudes**, using unchanged projected model axes and frozen scale, with a weak original-length prior and the same first-60% split and 15% cap. Poorly projected segments are unidentifiable. This differs from the earlier normalized-direction reference fitter: it tests shape magnitude, not per-frame angle adjustment or raw inferred 3D length medians. Three clips improve later-pose image errors, but all candidates still fail floor checks; none is promoted.

Require later-pose projection checks and available independent source labels before considering dynamics. Do not resize bones per pose, lift roots, refit the floor, loosen gates, or claim recovered anatomy. Camera calibration and additional views remain unresolved alternatives.

## Verification

Twelve analytic tests cover exact forward kinematics, known asymmetric shape errors, split isolation, crop/confidence handling, caps, decomposition closure, missing evidence, identity/time bounds, and prohibition of physics steps or floor fitting. The real diagnostic took 0.138 seconds, made no paid calls, and preserved source/code hashes. The 15-file seal is `d9efa79be82375d0f6e66ee5c686094b85adf21fbb580b90a931dc0a2cea42b3`.
