# Actual-atlas CPU mip review

**Modeled cross-cell bleed is demonstrated; the actual GPU cause remains unmeasured.** This single screen used the actual unique-cell atlas/UVs and previously saved source RGB at9,600 nominal tips. It did not decode the original4K source texture again, build a model, or run a renderer.

## Control and measurements

Sample heights are0.10/0.35/0.50/0.85 along each fin, at integer LOD0–5. The model decodes sRGB8 RGB, generates straight-RGBA linear-light2×2BOX mips with8-bit re-encoding, then bilinearly samples with CLAMP. Its isolated-cell control samples the **same mip pixels**, clamping neighbor indices to the original cell rectangle. The64×32 cell boundaries align with BOX blocks throughLOD5: BOX generation does not cross cells at these levels, but bilinear footprints can.

A source-dark tip has original linear luminance≤0.05. A contamination observation below is a modeled-visible sample (alpha≥0.18) receiving **more than0.05 added luminance from neighboring cells**. Counts across four heights are observations, not unique fins.

| LOD | Current cross-cell observations | Color-coherent order | Current within-cell brightening observations>0.05 |
|---|---:|---:|---:|
|0|0|0|0|
|1|0|0|0|
|2|0|0|0|
|3|0|0|6|
|4|2383|110|27|
|5|364|23|3|

AtLOD4,height0.10 specifically,2383 of4769 source-dark modeled-visible tips are contaminated; the cell-clamped control removes that cross-cell contribution. Reordering lowers this count to110. Maximum neighbor-added luminance remains substantial: currentLOD4/5 maxima0.1940/0.3494; reordered maxima0.1921/0.3326. The order reduces frequency, not every worst case. Deep within-cell source-color mixing is a separate mechanism and cannot be fixed by packing alone.

The current fin-index order has4728 dark/bright adjacent pairs among9461 horizontal/vertical neighbors; the tested order has42. Cell labels use the mean of the two source-tip colors, not an anatomical or complete material classification. Two-pixel gutters shrink to0.125 mip pixels atLOD4 and0.0625 atLOD5.

## Conditional color-coherent control

The observed bleed triggered exactly one deterministic permutation:32 luminance bins, then linear R/G/B, then source-fin ID. Every64×32 RGBA cell was copied byte-for-byte. No source RGB or alpha texel changed. Per-fin UVs moved with their cells; LOD0 samples stayed exactly equal. No new model or packed PNG was created.

`packing-only-permutation.npz` contains:

- `source_fin_at_destination[4800]` — source cell to copy into each row-major destination;
- `destination_for_source_fin[4800]` — inverse order;
- `simulated_uv[4800,4,2]` float64 — remapped UVs in unchanged fin order.

The source atlas is `../panda-fringe-color-atlas-v1/source-color-atlas.png`,64 columns×75 rows. Its RGBA-array SHA256 is `8fb4b02b24c4c43060c120658585d6db4fa076b2d8eeb7cb6c77299d7ac9d885`; the predicted packed RGBA-array SHA256 is `e4246ab44f50e0835c4622eb9d7f38787d51f443840aebb7634b2d60becb4363`. The permutation NPZ SHA256 is `97a62ea8c4171737872fc100e45c529c6ddf8b0a830468f000e28562cc01a677`. Any actual derivative must recheck float32 UV export and preserve geometry/skin/source bytes independently.

## Important non-equivalences

The GPU's derivatives, actual or fractional/trilinear LOD, anisotropic filtering, visibility, occlusion, lighting and antialiasing were not measured. These results do not establish that the observed EEVEE flecks useLOD4/5 or that source colors are corrupted. Original-cell alpha bytes are preserved, but **filtered alpha visibility changes** with neighbor order: atLOD5 modeled-visible observations change2189→2352, and dark neighbor-induced visibility changes27→60. That is not zero visual change or automatic quality approval.

## Custody and bounds

One CPU process completed in1.302s with sampled peak RSS284,704,768B under30s/512MiB/32MiB. Exit0 and owned-child exit were verified. Original inputs and prepared code were checked unchanged after execution. No retry, native/provider/AI call, actual-model mutation or live child remains.

`result.json` contains all24 level/height rows per control, residuals and worst samples. `current-samples.npz` and `packed-samples.npz` preserve global/cell-clamped RGBA samples and the saved source RGB. `prepared.json`, `config.json`, `screen.py` and `job.json` pin the exact calculation. Parent owns the one subsequent matched actual derivative/render; this directory is filtering evidence only.
