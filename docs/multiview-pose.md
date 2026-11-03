# Two-view pose reconstruction: better agreement, incomplete body

Two actual camera views now constrain 3D joint positions without using the pose model's guessed depth. On the same compared joints, agreement with a third camera improves substantially. **This is a separate development experiment, not a replacement sandbox body or a claim of anatomical accuracy.** Hidden joints remain missing, and the previously inverted pose is rejected rather than repaired.

Later work adds a [Full-model root-only sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-root.md): triangulated hips position an otherwise unchanged monocular body. It does not reconstruct the missing two-view limbs or replace the original experiment described here.

A subsequent [partial body adapter](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-body.md) applies available torso/limb directions and shared dimensions while labeling missing directions as single-view priors. It retains all unavailable slots, mixed results and the original studies; it does not recover hidden anatomy or validate contact.

## Inspect the result

- [Frame 200 across all three cameras](../artifacts/multiview-jog-two-view-01/frame-0200-comparison.png): the single-camera estimate is displaced/inverted; inconsistent two-view joints are absent.
- [Third-camera comparison at frame 120](../artifacts/multiview-jog-two-view-01/view-3-frame-0120.png): cyan is image evidence, purple is the old single-view estimate, orange is two-view reconstruction.
- [Results](../artifacts/multiview-jog-two-view-01/report.json), [all 2,880 major-joint comparison slots](../artifacts/multiview-jog-two-view-01/comparison.json), and [all 7,920 prediction slots](../artifacts/multiview-jog-two-view-01/predictions.json).

This original Lite study did not produce a new body. The subsequent [Full-model study](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/full-pose-model.md) recovers the bad torso and provides a newer, separately constructed single-view manual candidate, while retaining coverage regressions. To inspect the original supplied-ground body:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/camera-ground-jog-lab-01
```

That original body supports manual interaction and recording, but still contains the documented single-view body errors. Neither experiment changes the usual single-video upload workflow or its Lite default.

## Source, split and clock assumption

The [Autelica jogging-turn sample](https://autelica.com/sample/) publishes three moving-camera videos with reconstructed camera parameters. The [sample license](https://autelica.com/sample/files/LICENSE.txt) is CC BY 4.0; attribution is **Autelica (autelica.com)**. Separately licensed hands/pilot data is excluded. The original drone1 clip and all earlier results remain unchanged.

Two additional original clips, the source page, license and viewer JavaScript total **30,300,639 bytes**. The JavaScript was inspected as text, not executed. An initial sandbox DNS failure is retained in `multiview-jog-extra-views-01`; the bounded authorized download succeeded in `-02`, with no paid request or automatic retry. Both clips fully decode to 240 frames, 1920×1080, 30000/1001 fps. [Supplemental attribution](../data/exercise/autelica-jogturn-extra-views.source.json).

The split was declared before acquisition/inference:

- **Drone1 + drone2:** raw 2D image joints and supplied camera transforms for construction.
- **Drone3:** image-joint projection comparison only.
- **Provider body/UV/contact data and monocular world landmarks:** excluded from construction. Provider body joints are opened only after predictions are written and hashed.

**Clock alignment is an explicit assumption.** The provider's preview viewer seeks all videos to equal media time, and the original files have matching decoder clocks. That does not independently establish synchronization of the original MP4s. Recovered offsets of 0, −0.70174 and +0.50146 seconds are preserved but not applied; there is no offset fitting, resampling or interpolation. The resulting image agreement is descriptive evidence, not independent synchronization proof.

Camera parameters were themselves reconstructed using multiple views. Camera3 is outside this reconstruction's fitting inputs, but it is **not an untouched independent final holdout**. Height-based scale and provider joint positions are not laboratory truth.

## Method and availability

`multiview_pose.py` intersects two calibrated image rays. Each joint requires confidence ≥0.5 in both images, actual image bounds, ≥2° ray separation, forward camera depths between 0.2 and 100 model metres, and reprojection error ≤1% of the image's larger dimension in each construction view. These limits were fixed before the real run. Camera inverses are computed once per view; no model inference is repeated during geometry construction.

Source hashes, complete camera frames, exact decoder times, explicit clock provenance and distinct camera/source identities are required. The method preserves every frame/joint slot; it does not interpolate, borrow provider joints, fill occluded limbs, fit floor scores or reassign identities.

Drone2 has 104 partial tracked poses, 15 low-confidence poses and one absent pose. Drone3's 120 samples all fail the legacy whole-body visibility gate because the wrists are hidden, despite confident shoulders/hips/legs. Its raw evidence is retained for per-joint comparison; the old gate's downstream motion error is not evidence that the video is static. No rejected view is silently promoted to a complete body.

| Availability across all 240 source frames | All 33 landmarks | Twelve major joints |
| --- | ---: | ---: |
| Estimated | 2,875 | 1,033 |
| Image evidence unavailable, including unsampled frames | 4,944 | 1,819 |
| Inconsistent image rays | 101 | 28 |
| Total retained slots | 7,920 | 2,880 |

Each view was sampled on 120 frames. The other 120 frames remain unsampled, not invented. The right wrist is not reliably visible in drone2 and has no two-view estimate. A complete articulated body cannot be claimed from this output.

## Paired comparisons

Both methods use exactly the same available joint slots within each row below. There is no fitted alignment, scale, reflection, time shift or identity swap during comparison. RMSE is root mean square error, which gives larger errors more weight.

| Development comparison | Paired slots | Single-view estimate | Two-view estimate |
| --- | ---: | ---: | ---: |
| Absolute joint-position RMSE versus reconstructed reference | 1,033 | 0.4004 m | 0.0843 m |
| Third-camera image projection RMSE | 935 | 98.61 px | 23.66 px |
| Third-camera mean projection distance | 935 | 83.14 px | 17.80 px |

Remaining maxima are **0.539 m** against the reconstructed reference and **122.01 px** in the comparison camera. These conditional improvements do not establish full-video accuracy. All missing/rejected slots remain in the reports.

At source frame 200, the old torso points downward (vertical direction −0.761), while the reconstructed reference points upward (+0.990). The new method cannot produce a torso: both shoulder observations disagree across image rays and are explicitly rejected. This is useful failure detection, **not a corrected upright pose**. The frame was selected from an already observed failure, not a fresh holdout.

## Verification and resource bounds

**64 scoped tests pass**: 38 new two-view checks and 26 existing camera-root checks, in `.local/multiview-pose-tests.xml`. Analytic cases cover known 3D geometry, positive/negative distortion, moving cameras/bodies, world-coordinate changes, ignored contradictory monocular depth, occlusion/missing samples, parallel/antiparallel rays, invalid depth, contradictory images, and source/time/calibration binding. No unrelated full suite or UI tests were repeated.

The actual 240-frame construction took **0.282 s**, with the comparison and saved panels taking **1.789 s total**. A separate maximum-envelope screen retained 1,800 source frames, 900 samples/view and 59,400 slots: **2.258 s geometry**, **0.095 s JSON serialization**, 15,626,963 serialized bytes and 221,986,816 bytes process peak RSS including synthetic fixture construction. A 901-sample input was rejected. This one-process screen is not sustained memory, browser, HTTP or idle-performance evidence.

| Evidence bundle | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Additional sources | 8 | `4e7b973c003e373e0028e93a2a64c6d2d2de988d7b14ffee64151cef41e42d17` |
| Drone2 raw inference | 114 | `e8496c50441982c7a8e07fcf80bade7a862b8cc98791c64be9b1808b151427e6` |
| Drone3 raw inference/rejection | 114 | `52683b836b2a938adda15182dd8ef17e8b209f5ed1aec840d3512510af7cecdb` |
| Two-view construction/comparison | 31 | `05f05cf13cea18f2b5d20878578df788466ab94294a1ee78b626d8329fefb07a` |
| Maximum-envelope screen | 2 | `23ad1706fed0f48a20888e48887fdfc0a77ed32ce0d1b67f34f75430d4575273` |

The reproducible study command is:

```sh
.venv-lab/bin/python scripts/studies/check_jog_two_view.py \
  --out artifacts/multiview-jog-two-view-new
```

Use a fresh directory; preserve all original studies. This command reuses saved inference and makes no API call. Do not edit snapshotted implementation while it runs.

## Still required

Establish the original clips' cross-view timing and resolve remaining image/calibration disagreement without changing fixed comparisons or filling hidden joints. The later Full model corrects frame 200's raw inversion, and the partial body adapter supports manual recording with explicit missing-direction priors. The [upper-segment v2 experiment](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-segments.md) now uses supported partial limbs, but hidden endpoints, substantial individual regressions and independent geometry/contact validation remain open. No learned jogging policy, measured muscle activation or physical-transfer result is claimed.

At **October 28, 2026, 11:25 PM PDT**, the shared Meta ledger remains six settled calls, **$0.166152 of $5**, **$4.833848 remaining**, no unsettled reservation. This entire experiment used local inference/geometry, with no new paid API call, browser interaction, server startup or deployment.
