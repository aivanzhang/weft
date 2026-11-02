# Experimental area-share allocation screen

Prepared only; execution requires the parent session's release. No avatar, renderer, provider or AI role is used.

## One changed hypothesis

The unchanged baseline helper ranks safe 8.5–10 mm source edges by fixed-view orientation. The experimental helper admits 8.5–20 mm source edges, assigns each representable face's area equally to its eligible incident edges, then uses seeded exponential ranking weighted by that area and a bounded 0.5–1 orientation preference. Both retain 3 mm minimum root-center spacing, an 8 mm height, maximum 4,800 fins, six exported corners per fin, a 4,096-vertex reserve and the 100,000-vertex total ceiling. No positions are welded and no intermediate skin weights are authored.

The atlas has a 20.8 mm physical domain including gutters. Two tip centers are fixed at ±2.5 mm with maximum radius 0.8 mm, contained within the central 6.6 mm. UV crops reflect each actual source-edge length, so an extended anchor span does not stretch the visible tip groups. Six-mm roots are not promoted: the earlier analytic screen recorded systematic merging at 512 px.

Three identical polygonal curved surfaces have equal actual area and equal vertex counts: uniform axial rows, coarse/dense/medium equal-area bands, and reordered density with alternating diagonals. Metrics compare old/new per-band root counts, density variation and area-allocation error. Independent checks cover source immutability, indexed edge adjacency, masks, normalized inherited skin, five bending poses, geometric normals, spacing, finite output, invalid inputs and vertex limits. The atlas screen records fixed physical widths/separation, clipping, phase-dependent 720/512/256 px middle-line alpha footprints and contiguous runs at the unchanged 0.18 cutoff.

Endpoint colors use the exact LOD0 REPEAT/texel-center/sRGB-before-filter sampler copied from candidate-preparation-v2. Interpolation between these endpoint colors is **not exact whole-fin texture matching**, especially across longer source edges; this experiment does not hide or recolor source markings. Alpha occupancy, visible-group counts and controlled tessellation improvement do not prove actual-avatar fur quality, GPU appearance, disjoint fins, dense fur or resolved hairs at 256 px.

## Command and bounds

After release, run once:

```sh
.venv/bin/python -B outputs/fringe-allocation-analytic-v1/execute.py
```

The existing resource monitor owns the single child: 20 seconds, 256 MiB sampled RSS, 16 MiB output, one numerical-library thread. Its normal bounded cleanup grace is separate from the computation deadline. Source hashes are checked before and after execution; failures and the single-start marker are retained. No retry is prepared. The three input surfaces are analytic fixtures, not generated assets.
