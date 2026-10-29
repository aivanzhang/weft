# Two-view limb correspondence diagnosis

The v2 upper-segment regression is not explained by missing bends alone. **Confident but inconsistent left/right image labels can supply the wrong upper direction.** No labels, body geometry, thresholds or comparisons were changed in this diagnosis.

A later [parent/temporal correspondence experiment and every-frame inference study](dense-exercise-poses.md) make zero corrections: the source lacks the required uninterrupted confident context. Every-frame estimates also expose an inconsistent hip pose that prevents building the denser sandbox. The earlier playable body remains unchanged.

## Evidence

The sealed [diagnostic report](../artifacts/two-view-segment-evidence-02/report.json) reuses all 240 source frames. It retains 720 hip/knee/ankle pair slots and both same-label and second-view-swapped hypotheses, without selecting either. Provider body projections are reconstructed development references, not independent anatomical truth; shared scale and packaged-video timing remain assumptions.

- [Drone2 at frame 104](../artifacts/two-view-segment-evidence-02/view-2-frame-0104.png): image knee/ankle labels lie near opposite-side reference projections. The v2 left upper-leg direction error worsens **4.95°→42.60°**, despite image confidence 0.926. This supplies a wrong upper direction, not only an uncertain ankle bend.
- [Drone1 at frame 148](../artifacts/two-view-segment-evidence-02/view-1-frame-0148.png): ankle labels disagree with the opposite-side projected feet. The left upper-leg direction improves **21.25°→7.48°**, but the absent endpoint and retained distal orientation still leave the largest body-joint error.

Across the 79 corrected limb/pose pairs, direction error improves in 61 and worsens in 18. Direction RMSE falls 29.21°→10.68°. These averages retain the problematic poses; none is removed or given a fabricated endpoint.

## Why minimum image error is insufficient

For frame 104 knees, same/swapped two-view reprojection RMSE is **10.43/5.00 px**; for ankles it is 15.82/4.43 px. Frame 148 ankles give 40.72/5.85 px. A swapped correspondence is a plausible hypothesis, not proof of correct anatomical side: it cannot establish which camera's original label is wrong.

Across all 720 pair slots, 425 lack the required image points. Of the 295 comparable pairs:

| Existing ray checks | Pair slots |
| --- | ---: |
| Both label assignments supported | 206 |
| Same labels only | 76 |
| Swapped second view only | 5 |
| Neither supported | 8 |

The existing confidence, ray-angle, depth and reprojection limits are unchanged. Both hypotheses pass for most comparable pairs, so silently choosing the lower error would hide genuine ambiguity. Hip-connected anatomy, temporal evidence or separately supplied labels are needed to resolve side identity; later comparison labels must not become fitting inputs without a newly declared split.

## Reproducibility

```sh
.venv-lab/bin/python scripts/studies/diagnose_two_view_segments.py \
  --out artifacts/two-view-segment-evidence-new
```

Successful 11-file seal: `54e9036a0af8f6c7d9bec8d7dcb681a84b7571f82df5a4d6f83bf485b981ec8c`. All sources/candidates remain unchanged; no inference, API request, training or simulation repair occurred. The first harness tried to iterate null groups on unavailable body frames and failed before results; its five-file failure seal `f1acb5d451053dba367ce85fd1db8cb2704c18e8894bfd0c61e2914d2323fa1e` is retained. The corrected diagnostic explicitly preserves unavailable frames and skips only absent direction comparisons.

The [v2 sandbox](two-view-segments.md) remains experimental and opt-in. Its ground gaps, distal/twist uncertainty, hidden right arm and physical-validation limitations are unchanged.
