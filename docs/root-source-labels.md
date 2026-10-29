# Source-only exercise root labels

Twelve frame slots were selected and frozen before viewing pixels, then annotated from the original video images only. These are **assistant visual estimates, not expert anatomical ground truth**. No pose/model/candidate outputs or root-motion implementation were inspected.

## Fixed selection

For each source, frame IDs are zero-based `floor(fraction × (reported_frame_count − 1))`, at fractions 0.10, 0.50 and 0.90. Source hashes came from `benchmarks/temporal-identity-v5.json`; no candidate information from that manifest was used. Lunge is excluded as requested.

| Clip | Source size | Reported frames | Frozen frame IDs | Pelvis / torso measurements |
| --- | --- | ---: | --- | --- |
| Single-leg squat | 480×640 | 236 | 23, 117, 211 | 3 approximate |
| Squat | 1280×720 | 213 | 21, 106, 190 | 3 approximate |
| Kettlebell pushup | 640×480 | 259 | 25, 129, 232 | 3 approximate |
| Jumping jacks / burpees | 640×480 | 1,357 | 135, 678, 1220 | 2 approximate; frame 135 null |

The selection was frozen October 28, 2026, **2:48:02 p.m. PDT**; labels were frozen at **2:49:58 p.m. PDT**. Selection SHA-256: `8705780c606182b8accbd71a976d9192934f5b0221fa4a620832294fe92c9240`.

The decoder returned exactly 12 selected source images, with ordinal positions checked before and after each read. Native codec seeking may decode an internal keyframe prefix; no full-video decode or extra frame selection was performed. Original image dimensions and black letterbox borders remain unchanged. No frame was replaced because it was awkward or occluded.

## Measurement meaning

Each non-null slot records an estimated projected pelvis midpoint between hip joints and an estimated projected midpoint between shoulders, in original image pixels with top-left origin. Many far-side joints are hidden and all hips are clothed: estimates rely on visible body outlines, not exposed joint centers. The confidence field explicitly says `assistant_visual_estimate_low`.

Uncertainty is a subjective radial pixel allowance, not a calibrated statistical confidence interval:

- Single-leg squat: pelvis ±12 px; shoulders ±10 px.
- Squat: pelvis ±30 px; shoulders ±25 px.
- Kettlebell pushup: pelvis ±16 px; shoulders ±12 px.
- Jumping jacks / burpees: pelvis ±14–16 px where estimable; shoulders ±12–16 px.

Projected torso length is the 2D distance between these estimated endpoints. Its uncertainty allowance is their summed endpoint allowances. It is **not** a measured physical torso length or camera/scale calibration. For frame 135 of jumping jacks / burpees, hips are hidden behind the foreshortened torso/head: pelvis and torso length are null; the selected slot and original image remain present.

These sparse frames can check image-root location at those times only. They do not establish a complete motion path, worst-case drift, depth, full-body balance, anatomical correctness or physical accuracy. Periodic exercises can place several fixed samples in a similar pose. Do not fit camera, physical scale or model parameters to these labels and then call the same labels independent validation.

## Files and use

`data/root-motion-labels/index.json` identifies four per-clip JSON files. Each file pins source/video dimensions, frozen selection hash and image hashes; its `annotations` list retains three slots. Key fields are `frame`, `pelvis_midpoint_px`, `pelvis_uncertainty_radius_px`, `shoulder_midpoint_px`, `projected_torso_length_px`, and `projected_torso_length_uncertainty_px`. Null measurements must remain unscored/unknown, not zero or silently omitted from coverage.

`artifacts/root-source-labels-01` contains the pre-view selection record, extraction audit, twelve original PNGs with decoded RGB hashes, source-only three-frame sheets and separate uncertainty overlays. Overlay circles visualize annotation uncertainty only. The manifest seals the images, labels and this document before candidate scoring. No inference was run to generate these coordinates.
