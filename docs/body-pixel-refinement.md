# Image-position fitting: inspect the rejected candidates

The new bounded fitter makes all five bodies agree more closely with their pose model's image coordinates. **It is not a reliable physical repair:** four candidates still intersect the floor, the Heavy squat's deepest intersection worsens, and the push-up loses a passing limb-direction check. No working sandbox, default model or acceptance threshold changes.

Open the [offline source/body comparison](../artifacts/body-pixel-refinement-02/index.html) to inspect the original image beside both bodies. It needs no server or API call. The [Heavy squat sandbox](heavy-pose-model.md) and earlier [squat/push-up sandboxes](selective-crop-exercises.md) remain unchanged.

## What was tested

Five existing bodies cover four distinct videos: Heavy and Full on the same barbell squat, Full on kettlebell push-up, Lite on single-leg squat, and Lite on a separate full-body squat. This is not five new videos or a fresh final holdout.

The new `body_pixel_refinement.py` fits **joint positions**, rather than only normalized limb directions. It uses twelve image joint estimates centered on the source pelvis and the original fixed torso scale. The actual MuJoCo model supplies forward positions and their angle derivatives. World-coordinate pose estimates, floor distances, manual labels and policy rewards are excluded from the loss.

The fit retains:

- Every original source sample and all 484 body poses; no missing pose is filled or discarded.
- Fixed bone lengths, body anchor, root movement, camera-scale assumption, floor, motor limits and joint limits.
- Original image confidence, which is not reinterpreted as fitted-body accuracy.
- Unavailable targets: missing, invalid, cropped or confidence-below-0.5 joints do not drive the fit. An interpolated pelvis cannot become an observed image origin.

Each rotation correction is bounded to 0.15 radians, with at most twelve iterations and four backtracks per iteration. The robust position loss uses a 0.05-model-unit transition, original-angle regularization 0.001 and correction-only temporal regularization 0.0005. Per-fit limits are 900 frames and 60 seconds. Root/scale are recomputed from the original evidence and must match the model exactly; supplied-camera bodies and stacked reference fits are rejected.

These bounds are fixed before real fitting. All five candidates are sealed before floor or label grading. The original first-pose floor offset is preserved; it is explicitly **not refitted to the candidate's changed first pose**.

## Results

Image error below is confidence-weighted agreement with the **construction model's own image estimates**, not independent source labels. Floor intersections use the unchanged 0.025-model-unit limit.

| Existing body | Image error, px | Failed source floor poses | Maximum floor intersection | Fixed limb-label passes |
| --- | ---: | ---: | ---: | ---: |
| Heavy barbell squat | 30.17 → 22.32 | 45 → 40 / 104 | **0.1332 → 0.2042** | 15 → 15 / 16 |
| Full barbell squat | 29.68 → 21.81 | 55 → 52 / 107 | **0.1789 → 0.1940** | 11 → 12 / 16 |
| Full push-up | 17.22 → 13.78 | 0 → 0 / 121 | 0.00056 → 0 | **10 → 9 / 12** |
| Lite single-leg squat | 6.97 → 4.24 | 8 → 3 / 118 | 0.0831 → 0.0544 | No matching labels |
| Lite full-body squat | 11.98 → 7.17 | 13 → 9 / 34 | 0.0738 → 0.0479 | No matching labels |

The 30-Hz timeline checks retain their original counts: failures change 95→87/213 for Heavy squat, 111→108/213 for Full squat, 0→0/255 for push-up, 16→6/235 for single-leg squat and 30→23/80 for full-body squat. Across all **1,480 paired source/grid checks**, 60 recover but **15 newly fail**. No checks disappear.

On fixed development labels, Heavy squat's mean angular error improves 8.58°→5.95° without changing its 15/16 pass count. Full squat gains two passes but loses the right forearm at source frame 10. Push-up loses the right upper arm at frame 30 and worsens mean angular error 8.48°→10.78°. Two squat variants reuse the same labels and must not be counted as independent source diversity. Four unobservable push-up label slots remain unavailable.

## Why image fit is insufficient

The Heavy candidate's deepest collision occurs at **source frame 80**. Its left ankle is below the fixed confidence threshold (0.241); the right ankle estimate lies outside the image despite confidence 0.895. Neither drives the fit. Left-ankle image disagreement grows 115.73→144.71 px while the fitted visible points improve. Inspect the [unchanged source frame](../artifacts/body-pixel-verification-01/heavy-worst-source.png) and [complete diagnostic](../artifacts/body-pixel-verification-01/report.json).

Missing ankle targets are not the entire explanation: newly failing Heavy frame 70 and full-body-squat frame 52 each have all twelve targets above the confidence threshold. An occlusion-only filter would therefore not resolve every regression. Pose confidence, uncalibrated scale, perspective, hidden depth and fixed body proportions remain distinct uncertainties.

This continues the earlier [independently checked Meta body/contact hypotheses](meta-body-hypotheses.md). Their first-pose floor construction is reproducible, but neither those proposals nor this image fit identifies the physical cause. Gravity remains unchanged. Muscle bands remain illustrative; this experiment provides no measured activation, contact force or exercise-load evidence.

## Verification and reproducibility

**Twenty scoped analytic checks pass** in `.local/body-pixel-refinement-tests-02.xml`. They cover exact nonsquare projection, a known correction, numerical agreement of actual MuJoCo derivatives, opposite-depth ambiguity, missing/cropped/extreme coordinates, source/scale/time binding, fixed physical assumptions, joint limits and deadline preservation. No physics step or floor refit is allowed during fitting.

The five fits took 0.102–0.389 seconds each with two bounded CPU workers. All 495 original sampled slots and 484 original references remain. The first report assembly failed on a helper argument mismatch **after every fit was sealed**. Recovery reused all fits and the completed first case's floor/label checks; no successful fitting or pose inference was repeated. The recovery comparison took 1.302 seconds, excluding earlier fitting; it is not an uninterrupted end-to-end latency claim.

Sequential decoding verifies all **773 unique source frames** and exact source-image pixels in all **15 comparison panels**. Source/body panels were inspected locally; no browser or development server was started. The successful comparison bundle occupies 46,684,476 bytes. No new model inference, training, dynamic rollout, paid call or video upload ran.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Preserved reporting failure and five completed fits | 182 | `4212c2079d63c73925414d6669c67e41ac2c32cadc856d4658266ac4db272642` |
| Completed five-body comparison | 218 | `edc76d63dae053a5f92f62fb3c5bc460b8748cd4ef2a1a3d9feb1d0fc91218e3` |
| Source-panel and binding verification | 3 | `2a261ac9be5f4c9c9ee5df0a1f03fcb795ef156549c85dfee9c4bad616f9a336` |

To reproduce the study after its inputs or implementation change, use a fresh output directory:

```sh
.venv-lab/bin/python scripts/studies/check_body_pixel_refinement.py \
  --out artifacts/body-pixel-new
```

The `--reuse-fits` option requires exact matching source/model/runtime/contract/label hashes and a verified prior bundle. Reuse the existing results for inspection; do not repeat these unchanged fits or tune them against exposed labels. Additional source-supported camera/ground/contact constraints and independent changed-action evidence remain necessary. Broader ball and general-video reconstruction are still open work.
