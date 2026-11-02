# Analytic allocation result — not an accepted avatar

The reporting-only v2 screen passed. V1 remains failed because a NumPy integer prevented final JSON serialization; its invalid partial report and traceback were not edited. V2 changes only JSON scalar conversion and serializes before exclusive file creation. All other screen functions are AST-identical. Both helpers, sampler, RNG, fixtures, atlas and thresholds are unchanged; all six computed root-array archives and both images are byte-identical across the two runs.

| Equal-area fixture | Old → new roots by band | Density variation (CV), old → new | Area-distribution error (TV), old → new |
|---|---|---|---|
| Uniform | 1582/1609/1609 → 1591/1592/1617 | 0.00795 → 0.00752 | 0.00375 → 0.00354 |
| Coarse/dense/medium | 600/1280/1896 → 1461/1520/1819 | 0.42053 → 0.09795 | 0.17444 → 0.04563 |
| Reordered, alternating diagonals | 1240/1896/640 → 1561/1748/1491 | 0.40752 → 0.06780 | 0.16879 → 0.03083 |

Old uneven fixtures yielded 3,776 fins; the new allocation yielded 4,800. Both uniform versions yielded 4,800. The new method reduces, but does not eliminate, bias on these three analytic surfaces. All three use the same actual surface area and 2,993 source vertices. The worst projected total, including six corners per fin and the 4,096 reserve, is 35,889 vertices. Minimum selected root spacing is 3.375 mm; maximum root attachment error across five two-bone bending poses is 1.452e−8 m. Safety masks, finite geometric normals, inherited skin, source immutability and three invalid-input cases passed.

## Display-size limitation

Physical middle-height tip widths are 1.2759/1.2670 mm with 4.9835 mm center separation. These values and all sampled screen masks remain identical across 8.5/10/15/20 mm source-edge spans; crop-edge alpha is zero. Wider anchors therefore do not widen these tip groups in the tested atlas mapping.

| Nominal display size | Two-group result across eight horizontal pixel phases | Longest opaque middle-row run |
|---|---|---|
| 720 px | 8/8 retain two groups | 2 px |
| 512 px | 5/8 retain two groups; 3/8 merge | 3 px |
| 256 px | 0/8 retain two groups; all merge | 2 px |

These are CPU BOX-filtered middle-line measurements at an assumed 1.43 m camera scale and the unchanged 0.18 alpha cutoff, not GPU shading, full-surface visibility or visual fur acceptance. Individual hairs are not claimed at 256 px; separation is not guaranteed at 512 px. This screen does not establish fin intersection safety or actual-character appearance. Exact endpoint LOD0 colors do not match every intervening source-texture sample across longer edges. No source markings, actual models or production files were changed.

## Resource and custody result

- V1: failed final reporting, 2.345 s, sampled peak RSS 113,721,344 B, owned child exit verified.
- V2: passed analytic screen, 2.077 s, sampled peak RSS 112,803,840 B, owned child exit verified.
- Aggregate monitored runtime: 4.422 s of the 20 s envelope. V2 ran under the remaining 17.655 s / 256 MiB / 16 MiB bounds.
- No live child remains. No native, provider, AI-role, browser or actual-avatar job was run.

`result.json` contains the complete per-fixture and phase measurements. `verification.json` binds the reporting-only correction and unchanged v1 outputs. `prepared.json` pins every source and every predecessor file. Parent review is required before any actual derivative; this analytic result is not such a release.
