# Per-fin color atlas — prepared analytic screen

No actual avatar, native renderer, provider or AI-role call is part of this experiment. Production files and prior outputs remain unchanged. Execution requires the parent session's release.

## Reusable helper

```python
build_color_atlas(fins, material_ids, textures, silhouette,
                  cell_width=64, cell_height=32, padding=2, batch_fins=128)
```

`fins` is the bound area-weighted profile `area-share-edge-fin-analytic-v1`: four vertices per fin in root-A/root-B/tip-A/tip-B order, exact source UV endpoints, original physical silhouette UVs, and inherited four-joint skin arrays. The helper validates 1–4,800 fins, 8.5–20 mm root spans, finite arrays, normalized skin, and the unchanged physical UV crop. It reads positions/weights without altering them and returns only the new atlas, UVs and measured settings/report. No intermediate anchor or weight is created.

Source images are top-down encoded RGB(A)8; source basecolor factors are linear RGB. Each source edge is interpolated in its original UV coordinates, then sampled with normalized texel centers, REPEAT and bilinear filtering after per-texel sRGB decoding. The sampled linear color is encoded to sRGB8 atlas RGB. No source marking is whitened, suppressed, or replaced. Color extends from the edge along fin height; that extension is an authored surface-color assumption, not recovered volumetric fur.

Each 64×32 cell has a 60×28 interior and two transparent border pixels on each side. Its RGB gutters extend the nearest interior colors; alpha resamples the **unchanged** 1,024px physical two-tip atlas. The new UVs map the fin to its own cell interior. At 4,800 fins the atlas is 4,096×2,400 (9,830,400 pixels), below 4,096 per dimension / 16 MP. Source image decoding is separately limited to an aggregate 16 MP; sampling batches never exceed 128 fins. A 20-second internal builder deadline sits inside the 30-second whole-screen monitor.

## Required integration semantics and limits

- Apply the returned UVs only to the added fin primitive; preserve its positions, normals, indices, joints and weights, and every original source primitive.
- Use a neutral white basecolor factor and omit or neutralize the old **fin-only** vertex-color multiplier, because the actual source colors/factor are already baked into this atlas. Applying the old colors again would darken the coat. This does not whiten the atlas or change source markings.
- Keep the existing MASK cutoff 0.18. This screen does not change culling, lighting, geometry, spacing or the previous display-size limits.
- Sixty horizontal samples, sRGB8 quantization and between-texel interpolation lose sufficiently fine markings. This is not exact texture equivalence. The benchmark explicitly measures between-texel loss.
- Two-pixel padding does not guarantee isolation under deeper mipmaps, anisotropic filtering, alpha-to-coverage or perspective. Those remain unverified. No GPU claim follows from the CPU sampler.
- Existing four-corner skinning controls the fin. More color samples do not create extra skin anchors; the source edge is assumed to supply color for the protruding fin.
- Previous 512px phase-dependent tip merging and 256px unresolved groups are not repaired by adding RGB detail.

## One analytic screen

The frozen previous uniform curved-surface fixture rebuilds the same area-weighted arrays and verifies them against its saved root indices/UVs/centers before using them. A sharp black/white striped transition tests markings that endpoint interpolation misses. A smooth colored gradient tests quantization/preservation without claiming the new method must improve an already-nearly-linear field. Reported errors use linear RGB, visible alpha samples, texel centers and 256 fins × 257 fractional positions. Additional checks cover physical alpha widths, deterministic full-atlas bytes, unchanged geometry/skin/source-image hashes, known 2×2 REPEAT color samples and invalid/oversized inputs.

After release, execute once:

```sh
.venv/bin/python -B outputs/fringe-color-atlas-analytic-v1/execute.py
```

The existing runner bounds the owned CPU child to 30 seconds, 512 MiB sampled RSS, 32 MiB output and one numerical-library thread; normal bounded cleanup grace is separate. Inputs/code are hash-checked before/after. No automatic retry is prepared; any failure remains in this versioned folder.
