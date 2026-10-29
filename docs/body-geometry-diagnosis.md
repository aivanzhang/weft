# Body geometry diagnosis

**The 19-axis rotation mapping is consistent with MuJoCo. The largest position errors come from fixed model lengths and attachment geometry differing from the retained pose estimates, not reversed axes or hinge order.** Some source-world limbs collapse dramatically between frames despite high reported confidence. Preserving their directions while restoring fixed lengths can move an ankle substantially downward.

Evidence: [diagnostic report](../artifacts/body-geometry-diagnosis-01/report.json), [reproduction script](../artifacts/body-geometry-diagnosis-01/diagnose.py), and per-pose geometry files for [single-leg squat](../artifacts/body-geometry-diagnosis-01/single-leg-squat/geometry.json), [squat](../artifacts/body-geometry-diagnosis-01/squat/geometry.json), and [push-up](../artifacts/body-geometry-diagnosis-01/kettlebell-pushup/geometry.json). Input hashes and original model copies are retained. This compares construction evidence with reconstruction, **not physical ground truth**.

## Scope and method

The audit covers 345 existing model-reference poses: 118 single-leg squat, 100 squat, and 127 push-up. It compares twelve shoulder, elbow, wrist, hip, knee, and ankle positions after transforming retained MediaPipe world landmarks by `CAMERA_TO_SIM` and subtracting each pose's hip midpoint. The reconstructed body is centered on its pelvis site. This removes root translation and floor placement from the comparison.

The actual MuJoCo model receives the original 19 angles and original root position. Only forward geometry runs; `mj_step` is disabled. No model fitting, source edits, training, contact-based objective, root lifting, or floor changes occur. Every reported reference pose remains, including low-confidence, cropped, clamped, and assumed-twist cases. Raw visibility/presence and the existing crop-aware effective confidence are recorded separately. High-confidence summaries use the existing 0.5 threshold; confidence is not physical accuracy.

Three analytic probes also pass synthetic rotated/bent landmarks through the extractor **and the actual MuJoCo body**, including a gimbal-lock torso orientation and a nonidentity fitted anchor. All twelve positions agree within **5.6×10⁻¹⁶ model units**. The Z→Y→X rotation composition, elbow +X bend, and knee −X bend agree with the authored model.

## Position error and its components

Distances below are in the shared model/world-estimate coordinate units, not independently calibrated metres.

| Case | All 12 joints RMSE | Eight non-base endpoints RMSE | Attachment component RMS | Length component RMS | Direction component RMS |
|---|---:|---:|---:|---:|---:|
| Single-leg squat | 0.06238 | 0.07387 | 0.02754 | 0.05819 | ~9×10⁻¹⁶ |
| Squat | 0.05033 | 0.05871 | 0.02653 | 0.05988 | 0.000561 |
| Push-up | 0.04809 | 0.05303 | 0.03625 | 0.03696 | 0.000008 |

For each elbow/knee/wrist/ankle, the audit verifies the exact vector identity:

`position error = attachment offset + accumulated length mismatch + accumulated direction mismatch`

The component vectors add; their RMS magnitudes do **not** add. Length and attachment errors can partly cancel. Direction mismatch is essentially numerical zero except the explicit bend clamp and one nearly straight assumed-twist case.

The eight-endpoint RMSE remains 0.08011, 0.05894, and 0.05380 respectively when restricted to chains whose required image landmarks all have effective confidence ≥0.5. Filtering on visibility alone therefore does not resolve these geometry errors.

## Concrete largest errors

### Squat: transient world-limb shortening

At source frame **150**, the left ankle differs by **0.297799** model units, with chain confidence **0.7632**.

| Segment | Retained world-estimate length | Fixed model length |
|---|---:|---:|
| Left thigh | 0.172956 | 0.389759 |
| Left shin | 0.258890 | 0.370622 |

The accumulated length-error vector has z=**−0.301606**; the attachment component magnitude is 0.026717 and the direction component is approximately 7×10⁻¹⁶. Thus the large downward endpoint displacement is reproduced by extending these unusually short estimated segments to the fixed model lengths. It does not require a knee-axis error or a learned action.

### Single-leg squat: confidence does not prevent collapse

At frame **16**, the estimated left shin is **0.069134**, versus fixed length **0.353698**, despite chain confidence **0.9344**. The left ankle error is **0.333904**. The absolute largest endpoint error, 0.400195 at frame 164's right ankle, has lower chain confidence 0.3980 and remains explicitly retained rather than hidden.

### Push-up: both lengths and attachments matter

At frame **2**, the left ankle error is 0.232983. Estimated thigh/shin lengths are 0.271422/0.263757 versus fixed 0.388959/0.414332; chain confidence is 0.5018. Across the clip, attachment and length RMS contributions are similar. This does not establish that the actual person's limbs change length: these are monocular model estimates.

## Fixed shape restrictions

The model uses one shared left/right length per segment, symmetric attachment positions, and a shoulder/hip layout constrained to its torso frame. The extractor instead recovers directions between independently estimated endpoints.

Examples of persistent asymmetry in the construction evidence:

- Single-leg squat thigh medians: left 0.2929, right 0.3986; both model thighs use 0.3483.
- Squat upper-arm medians: left 0.2206, right 0.1488; both use 0.1838.
- Squat forearm medians: left 0.1417, right 0.2223; both use 0.1849.

These asymmetries may reflect pose-estimation/perspective errors, not anatomy. The torso-length component alone is smaller: shoulder-center RMS errors are 0.01480, 0.01637, and 0.00880. Source hip spans also depart from the model's torso-local lateral axis; their maximum off-axis components reach 0.1156 in single-leg squat. Fixed shoulder endpoints cannot reproduce all observed shoulder-height differences either. All per-pose spans and segment statistics are retained.

One squat bend is explicitly clamped: frame **70**, left elbow, inferred 2.685789 radians → authored maximum 2.6. Its forearm direction differs by 0.085789 radians, producing at most 0.015857 endpoint direction error. It does not explain the leg errors. No other bend clamp occurs in these three models. The push-up's small nonzero direction residual occurs on a low-confidence, nearly straight right leg whose twist was already labeled assumed.

## Bounded correction to test next—not implemented here

1. **Keep fixed lengths, root references, and floor unchanged initially.** Refit the 19 rotation angles against hip-centered Cartesian joint targets using a bounded robust loss. Cartesian fitting minimizes actual endpoint differences, unlike independently matching segment directions from displaced attachment points.
2. Use the existing endpoint visibility/crop confidence, but retain geometric inconsistency diagnostics: high confidence did not prevent severe segment shortening. Keep original targets, original angles, and per-endpoint residuals. Do not silently delete difficult poses or call a fitted point measured.
3. Bound optimizer work, angle limits, per-frame changes, and temporal regularization. Use the original rotations as the prior/initialization; disclose clamped or unidentifiable coordinates. Report positional improvements and any worsening of segment-direction agreement separately.
4. Only then consider a **single clip-wide**, bounded shape fit—segment lengths, modest bilateral asymmetry, or attachment offsets—with strong priors and confidence-aware robust aggregation. No per-pose bone-length changes. A global shape fit cannot explain transient world-limb collapse and must not infer anatomical asymmetry from monocular bias.
5. Freeze the objective before new runs. Fit only construction landmarks, not floor-overlap scores, evaluator labels, learned-policy returns, or held-out trials. Run unchanged geometry/floor/dynamics checks afterward; an endpoint fit is not guaranteed to improve contact feasibility and may worsen it.

This is a diagnosed reconstruction limitation and a proposed local experiment, not a completed physical repair, anatomy recovery, or balance result.
