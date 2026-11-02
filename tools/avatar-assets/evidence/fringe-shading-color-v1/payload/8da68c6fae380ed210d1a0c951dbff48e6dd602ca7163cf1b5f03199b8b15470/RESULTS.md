# Analytic per-fin color atlas result

**Passed one bounded CPU screen. No avatar or production file was changed.** The helper samples source colors along each fin's original UV edge instead of interpolating only its two endpoint colors. The same 4,800 area-weighted analytic fins, their geometry and skin arrays remain unchanged.

## Measured color result

Errors below are absolute linear-RGB differences at visible, fractional LOD0 samples: 256 fins × 257 queried positions, with 16,942 visible samples. They are CPU measurements, not GPU observations.

| Fixture | Mean error: old endpoints → new atlas | Maximum error: old → new |
|---|---|---|
| Sharp black/white bands | 0.285454 → 0.000837885 | 1.0 → 0.0475963 |
| Colored gradient, including its REPEAT boundary | 0.00605339 → 0.000853512 | 0.421467 → 0.00518106 |

At atlas texel centers, maximum new errors were 0.00344419 and 0.00424028; the larger between-texel stripe error is retained explicitly. The colored-gradient source is not periodic across its boundary: its comparison includes that REPEAT seam, not only a smooth interior field. No source markings are whitened or suppressed. A local diagram shows the worst sampled source/old/new visible strips; it is not an avatar render.

## Representation and preservation

- Atlas: **4,096×2,400 RGBA** / 9,830,400 pixels for 4,800 unique 64×32 cells, each with a 60×28 interior and two-pixel padding.
- Maximum tested horizontal sample pitch: 0.21717 mm; a legal 20 mm source edge would have a coarser 0.33333 mm pitch and is not represented by this particular uniform fixture.
- Color: encoded source RGB → per-texel sRGB decode → REPEAT texel-center bilinear sampling along the original edge → sRGB8 atlas encoding. Opaque source RGB supplies the color; existing two-tip alpha supplies shape.
- The previous physical UV mapping is preserved, but resampling is **not alpha-identical**. Three sampled spans near 9.00/9.42/13.03 mm changed mid-height tip widths by at most 0.12524/0.08051/0.00318 mm. This is reported, not described as zero morphology change.
- All fin-array hashes, including positions, normals, indices, joints, weights and root correspondence, remain unchanged. The helper returns new UVs, atlas pixels and a report only.
- A complete repeated 4,800-fin build produced byte-identical atlas and UV arrays. Seven known 2×2 REPEAT color cases and the known sRGB encodings 0/188/255 passed. Six invalid/oversized input cases rejected before unsafe work.

The consumer must not multiply the baked colors by the former fin vertex colors again. Use neutral material multiplication and omit/neutralize **only that fin primitive's old COLOR_0**; preserve every original source primitive and its materials. No normals, culling or lighting are changed here.

## Bounds and unresolved limits

The single monitored child exited0 with verified ownership cleanup after **2.084 s**; sampled peak RSS was **467,845,120 B (446.17 MiB)** under the 512 MiB cap. Output before this sealing note was 1,970,000 B under the 32 MiB cap. The source images were small (512×64 and 512×128); the full 16 MP source-input limit was not exercised. Actual 4K-model integration requires its own bounded memory assessment. No retry or live child remains.

Sixty horizontal samples and 8-bit encoding cannot preserve arbitrary source detail. Color extends from the source edge along the fin's height; it is not a recovered 3D coat texture. Two-pixel gutters do not establish isolation at deep mip levels or under anisotropic filtering. No GPU, alpha-to-coverage, perspective or actual-avatar appearance was tested. The previous 512px phase-dependent merging, 256px unresolved-tip limit, geometric fin-normal lighting difference and intersections are not fixed by this color-only change.

## Reusable sources and evidence

- `color_atlas.py`: `build_color_atlas(...)`; exact API and input conventions are in `README.md`.
- `prepared.json`: immutable inputs, source hashes, settings and exact command.
- `result.json`: complete numeric reports and retained limitations.
- `verification.json`: source custody, geometry/skin hashes and deterministic output hashes.
- `job.json` / `cpu-monitor.json`: actual bounds, exit and sampled resources.
- `analytic-color-comparison.png`: viewed analytic color diagram.

No native, browser, provider, AI-role or actual-avatar call occurred. This result supports testing the color-only hypothesis; it is not quality acceptance or release of an actual model.
