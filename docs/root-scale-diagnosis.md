# Root-scale diagnosis: foreshortening matters for one clip, not a universal repair

The current root scale divides the fixed 3D torso length by its median image length. It does **not** account for the torso's shorter projection into the image plane. That omission is measurable in the push-up clip, but negligible in the squat and single-leg clips. A multi-segment scale improves construction pixel residuals for squat and push-up, yet would **increase squat root-motion amplitude**. It is not an established floor repair.

Evidence: [diagnostic report](../artifacts/root-scale-diagnosis-01/report.json), [fixed diagnostic contract](../artifacts/root-scale-diagnosis-01/pre-diagnostic-contract.json), and [script](../artifacts/root-scale-diagnosis-01/diagnose.py). Original models, root audits, raw pose evidence, per-vector observations, source hashes, and decoded dimensions are retained. No models, roots, floors, labels, or evaluators changed. No floor checks, candidate fitting, dynamics, or API calls ran.

## Construction-only calculation

For every original reference pose, the diagnostic compares eleven vectors: eight limb segments, torso, shoulder width, and hip width. Model vectors come from actual MuJoCo forward geometry projected as **x/−z**. Image vectors use normalized landmark differences multiplied by the verified source width/height.

Let `p` be an image-pixel vector and `m` its projected model vector:

- Length ratio: `k_length = ||p|| / ||m||`, in pixels per model unit.
- Direction-sensitive ratio: `k_dot = (p · m) / (m · m)`, reported separately, including nonpositive values.
- Fixed diagnostic length scale: exponential of the confidence-weighted median log length ratio.
- Current scale: `k_current = median(image torso length) / fixed 3D torso length`.

Weights are the minimum endpoint visibility/presence, with **no positive-confidence cutoff and no angular-mismatch exclusion**. Missing/nonfinite points, zero confidence, endpoints outside the actual image, observed vectors shorter than two pixels, or model projections below 10⁻⁵ units are unusable for estimation. All rows and reasons remain in the artifact. Actual image bounds `[0,1]` are stricter than the legacy `[-0.05,1.05]` tolerance; both flags are recorded. These rules were fixed before calculating results.

Coverage: 345 original model poses and 3,795 vector slots. Single-leg has 1,279 usable vectors and 19 short image projections; squat has 1,076 usable and 24 outside-image vectors; push-up has 1,397 usable. Other original raw frames without model references remain explicitly listed. Dimensions were verified by decoding the source videos: 480×640, 1280×720, and 640×480 respectively.

## Fixed-scale estimates and pixel residuals

| Clip | Current k | Global length k | Global dot k | Weighted pixel-vector RMSE, current → length scale | Root multiplier if applied |
|---|---:|---:|---:|---:|---:|
| Single-leg squat | 123.603 | 125.269 | 124.201 | 5.887 → **5.893** | 0.98670 |
| Squat | 472.282 | 435.446 | 434.362 | 28.510 → **25.810** | 1.08460 |
| Push-up | 160.272 | 188.189 | 185.699 | 13.732 → **9.579** | 0.85165 |

All k values are pixels per model unit, not calibrated physical camera scale. Residuals compare the same usable vectors before and after substituting the diagnostic scale in the algebra; they are construction agreement, not independent image labels. No new body or root trajectory was created.

The conditional root multiplier is `k_current / k_diagnostic`, because model-units-per-pixel is the inverse of k. Thus the global estimate suggests almost no change for single-leg squat, **8.46% larger movement for squat**, and **14.83% smaller movement for push-up**.

## Isolating torso foreshortening

Using exactly the original torso-scale sample IDs separates projected shortening from changing which body parts contribute to the estimate.

| Clip | Median projected torso / 3D torso | Projected-torso k | Conditional root multiplier | Torso-only weighted pixel-vector RMSE |
|---|---:|---:|---:|---:|
| Single-leg squat | 0.99239 | 125.416 | 0.98554 | 2.118 → 2.008 |
| Squat | 0.99255 | 476.031 | 0.99212 | 17.926 → **18.299** |
| Push-up | 0.90033 | 179.534 | 0.89271 | 8.887 → **3.806** |

The current formula was independently recomputed from the retained observations and matched the saved scale. The push-up torso is shortened by roughly 10% in projection; the other two are shortened by less than 1% at the median. Therefore ignoring foreshortening is **not the main explanation for the large squat floor conflict**. Even the more geometrically appropriate torso ratio need not lower raw vector RMSE: robust length-ratio estimation does not minimize squared vector error, and direction/length errors remain.

## Why one global camera scale is uncertain

Per-segment estimates disagree substantially:

- Squat median k ranges from about **346** for the right upper arm to **683** for the right forearm, while torso k is **476**.
- Push-up hip-width k is **103**, torso k **180**, and left-thigh k **214**.
- Weighted global ratio ranges (10th–90th percentiles) are approximately **98–148**, **319–619**, and **116–223**.

These differences can reflect inferred shape/pose error, projection/perspective, unobserved depth, or landmark uncertainty. They are not proof of true anatomical proportions or camera movement. The diagnostic retained 45 nonpositive dot ratios in single-leg and three in squat, rather than silently removing direction disagreements; push-up had none.

## Implications for floor inconsistency—conditional only

With unchanged angles and geometry, a different fixed scale would multiply root x/z displacement relative to its original origin. It would not move or refit the floor.

- **Squat:** projected-torso correction reduces movement by only 0.79%; the all-segment estimate instead increases movement by 8.46%. At negative-root-z poses, the latter would lower every body surface further. Better aggregate image fit is therefore not a sufficient reason to expect better floor consistency.
- **Push-up:** both projected-torso and all-segment estimates reduce descent magnitude, so scale mismatch is a plausible contributor. But the original root z spans −0.5396 to +0.2457: reducing positive displacement also lowers some poses. New intersections remain possible. No candidate floor result is claimed.
- **Single-leg squat:** the scale change is small and slightly worsens aggregate pixel-vector RMSE. This does not provide evidence for a useful global correction.

Any next implementation needs a separately declared, bounded scale contract, fixed construction inputs, and unchanged independent direction/floor checks. It must preserve uncertain depth, the original root origin, and rejected results; no per-pose root lifting or floor-based scale selection. This diagnostic does not authorize default promotion or establish physical calibration.
