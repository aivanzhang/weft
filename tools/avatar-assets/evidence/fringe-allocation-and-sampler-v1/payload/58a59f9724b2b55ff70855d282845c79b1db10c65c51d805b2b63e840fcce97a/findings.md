# Edge-fringe allocation: deciding result

**The fixed 8.5–10 mm edge range distributes roots unevenly because it follows source tessellation.** The later scoring/spacing stage changes that distribution little. These are source-space allocation measurements, not visible fur coverage or quality approval.

## Measured imbalance

Surface area is estimated per color/grid bin using three equal-weight barycentric samples per eligible triangle; the total triangle area is exact. Linear source-luminance bands were fixed before execution, with no anatomical labels.

| Source luminance | Eligible area share | Old fin share | Old roots/m² |
| --- | ---: | ---: | ---: |
| Low, <0.10 | 46.34% | 76.39% | 1999.8 |
| Middle, 0.10–0.50 | 0.28% | 2.29% | 9953.3 |
| High, ≥0.50 | 53.38% | 21.32% | 484.7 |

Low-luminance surfaces receive **4.13× as many roots/m²** as high-luminance surfaces. The small middle band is especially sensitive to texture transitions and quadrature; it is not a separate material or anatomical region.

The old 3,231 roots and the 3,409 available narrow-range edges differ by only **0.01375 total variation** across a fixed 4×4×4 grid (0 means identical allocation, 1 means disjoint). Narrow-range availability differs from surface-area allocation by 0.38217; the final roots differ by 0.37597. This supports the length filter as the main observed allocation bottleneck, not a claim about every source of visual quality loss.

## Range comparison, unchanged safety exclusions

| Root-edge range | Safe whitelist edges | Available after existing face buffer | Low/high available-root density ratio | Grid allocation difference from area |
| --- | ---: | ---: | ---: | ---: |
| 8.5–10 mm | 3,415 | 3,409 | 4.15× | 0.38217 |
| 6–10 mm | 9,150 | 9,122 | 4.37× | 0.40236 |
| 6–20 mm | 25,480 | 25,437 | 2.23× | 0.24428 |

Lowering only the minimum to 6 mm adds candidates but does not correct the color/area imbalance. Extending the maximum to 20 mm provides substantially more evenly distributed candidates, though its allocation remains biased. Two grid cells containing at least 0.5% of eligible area each have no old roots and no 6–10 mm candidates; they have 24 and 65 candidates under 6–20 mm.

**Next hypothesis, not implemented:** retain the fixed fin/vertex budget, use broader safe-edge availability, and allocate roots by eligible surface area rather than raw edge counts. Do not make every long edge a wide fin by default; decoupling visible tuft width from source edge length would require new attachment/UV/weight proofs. Wider availability alone is not an accepted geometry or appearance improvement.

## Reproducibility and bounds

- Inputs: source GLB `e0488074965a89a9f7eda7788ebb26e7a8f14f98bbb44e1d9856fa091915e5d5`, existing candidate `e113fa5372ebddbf04a9860876734d53265ded9611d1ffecd1ae7e2bd87bc2eb`, frozen safe-edge whitelist and correspondence.
- The 53-face unsafe-edge buffer was independently reproduced; 2.663122 m² remain eligible. Recorded cap material targets remain excluded.
- One source image decode. Exact candidate-v2 UV0 sampling: normalized texel centers, REPEAT, LOD0 bilinear filtering after per-texel sRGB decoding, then material factor. Recomputed endpoint colors match the actual old GLB colors byte-for-byte after float32 serialization.
- Grid coordinates use original whole-body normalized Blender Z-up XYZ; glTF `[x,y,z]` converts to `[x,-z,y]`. Color/grid bin areas use fixed three-point quadrature, not exact triangle clipping or full texel integration.
- Command already executed once: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python outputs/edge-fringe-allocation-census-v1/execute_cpu.py`.
- Observed: **0.523 s**, **157,876,224 B sampled direct-child RSS**, PID 11811 exited. Limits: 30 s / 512 MiB / 16 MiB. All inputs and frozen scripts unchanged. No retry, geometry, native render, provider call, or live handle.
- No nearest-neighbor scan, screen-space coverage, lighting, GPU, animation, or appearance claim. Alternative-range counts are available source anchors, not a newly selected coat.

`allocation.json` SHA256 `eb57b1bb91d287069146a86c810b461d745f03804725b839fe494b0bd72035c6`. Execution manifest SHA256 `3a12f92afb15bde7c900f03894c8f0dd2bc13b4972193851121cab5afc9282dc` was sealed before this summary; `findings-receipt.json` binds the append-only prose to it.
