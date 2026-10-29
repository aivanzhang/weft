# Inspectable foot references

Weft builds optional fixed-size ankle/heel/foot-index triangles from pose evidence. These remain separate diagnostic artifacts: this builder adds no mass, collisions or controls. A later opt-in [ankle/foot sandbox](foot-sandbox.md) consumes the triangles and adds explicit capsule feet and motors; default interactive bodies remain unchanged, and the candidate still fails floor checks.

Open the [eight-frame comparison sheet](../artifacts/foot-reference-01/foot-comparison-sheet.png) or [full report](../artifacts/foot-reference-01/report.json). Green shows raw pose estimates, purple the fixed triangle, and orange approximate source silhouette axes. The overlays use source-ankle alignment and an uncalibrated scale; they are not a complete camera/body reconstruction.

## Method and missing evidence

`weft.foot_reference.build_foot_reference(spec, evidence, source_metadata)` returns an artifact without mutating the body. Source metadata must identify the exact video, decoded frame count, dimensions and rate.

The first 60% of source time determines each side's fixed triangle from eligible pose estimates. Each later sample supplies triangle orientation, while all three edge lengths remain fixed. At least three eligible prefix samples are required. Confidence, image bounds, identity, geometry and timing checks apply; no gaps are interpolated. Raw image/world landmarks, rejected samples and unsampled frame indices remain available.

Anatomical axial roll and sole normal remain null. An inferred landmark plane is not a measured sole, floor contact or support force.

## Four-source result

All 388 raw pose frames and 776 side slots are retained: **526 available, 250 unavailable**.

| Source | Left available | Right available | Raw pose frames |
|---|---:|---:|---:|
| Single-leg squat | 118 | 115 | 118 |
| Cropped squat | 0 | 0 | 107 |
| Kettlebell push-up | 127 | 109 | 129 |
| Full-body squat | 34 | 23 | 34 |

The cropped squat has insufficient visible prefix feet; high model confidence does not override the crop. Twenty analytic tests pass. Triangle-length variation is at most 6.94e-17 model units. Processing took 0.862 seconds with no physics steps or API calls.

Eight source-only frames supplied 16 anonymous foot slots before fitting: seven approximate visible silhouette axes and nine unavailable due to crop, overlap or self-occlusion. These reused-development annotations have 5–9 px endpoint uncertainty and are not anatomical heel/index ground truth. Conditional proxy differences span **7.36–24.10 px** and **12.0–57.8 degrees**. Visual inspection confirms short/tilted predictions on the full-body squat. No accuracy or collision-geometry promotion follows.

Source-label bundle: ten-file seal `96e701322ba21a843523efa0f452881d4d56614fa1d0158638d23e0cd4b3578a`. Study: 53-file seal `5f3061fc13a087574aad3df492478b27fc8ad18b4043bd95c2601212eaac994a`.

The later [ankle/foot candidate](foot-sandbox.md) preserves these projection uncertainties, unknown sole orientation, missing samples and the existing floor checks. Its added physics is not an accuracy promotion of these reference estimates.
