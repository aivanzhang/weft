# Portable coat: measured cause and one different experiment

Status: actual-source CPU screen only. No model, texture, native render or public
preset changed. The current exported front/side images were directly inspected;
their hashes and the root appearance rejection are pinned in `prepared.json`.

## Why the present coat is sparse and bristled

- The 3,600 actual fins have 8 mm straight lifts, all about 10.008 degrees from
  the outward surface normal. Each has exactly two narrow cutout columns. This
  is a radial outline representation, not a surface-covering coat.
- Actual atlas occupancy above its 0.18 cutoff is only 15.60% at the median.
  The sum of all unoccluded opaque fin area is 0.0670 m² against 1.7606 m² of
  eligible source area. These differently oriented areas are not an image
  coverage measurement, but show the small amount of opaque material available.
- The 720px front camera gives a median half-height stroke width of 0.552 px
  versus a 3.395 px projected lift. At 256px these become 0.196 and 1.207 px.
  Tall subpixel blades explain readable bristles with weak body coverage.
- Even on broad front-facing body regions, the fin plane's median camera-facing
  factor is only 0.304 front / 0.320 side: much of the card is seen edge-on.
- Actual alpha mips retain all 7,200 nominal tip samples through modeled level4,
  but only 858 at level5 and 60 at level6 remain above cutoff. Median alpha is
  0.2078 at level4 and 0.1255 at level5. The 256px front footprint estimates
  median axial minification levels4.80 horizontally /4.54 vertically. These
  BOX-mip/bilinear calculations are not measured EEVEE derivatives or flicker.
- Source-colored packing fixes neither missing opaque area nor edge-on geometry.
  The normal-map undercoat changes lighting detail, not strand geometry or
  silhouette density. Configured native groom fibers are a different renderer
  representation; they do not exist as those fibers in the exported GLB.

The older broad `tuft-v1` already failed for dash/block speckling and a spiky
crest. The later shorter regional-fin candidate improved outlines but reduced
summed projected alpha area. Neither should be repeated as a density solution.
No quantitative Muse-image fiber count or Muse-equivalence claim is made here.

## One changed representation: shallow paired-face micro-clumps

Keep the original body, rig, actions and all 3,600 root edges. Convert 2,700
existing roots into outward-facing, single-winding surface patches; retain the
remaining 900 actual contour fins unchanged for the first isolated comparison.
Each converted patch follows the two exact source triangles incident to its root
edge. The edge endpoints remain fixed; the two opposite source vertices receive
at most a1.2 mm outward lift. No nearest-surface projection or invented rest skin
is needed: every corner has an exact source vertex and triangle owner.

Use source-triangle barycentric UV sampling through the existing color sampler,
not an extruded single root color. A new alpha profile distributes multiple
connected, tapering micro-strokes along the existing groom field across each
patch. Target approximately45% coverage within the triangular footprint; this is
an experimental target, not a fabricated measured atlas. Keep the same packed
4096×1824 image,64×32 cells, color-space convention and source palette. Geometric
triangular footprints and tapered alpha must avoid opaque rectangular ends;
deep-mip blobs, temporal shimmer and a tiled/shingled look remain rejection risks.

The source screen proves all3,600 roots have exactly two indexed incident faces
and coherent current material ownership. A deterministic, nonoverlapping2,700
pair subset covers5,400 actual triangles /0.6704 m²,38.08% of eligible area.
Exact selectors are in `selection-screen.json`; no faces were changed.

At the hypothetical45% patch coverage, summed unoccluded alpha-area proxies are
5.30× current front /4.75× side. This is the reason to try a different orientation
and surface footprint, **not** a prediction of accepted visible fur density.
Occlusion, alpha shape, offsets, normals, skin and motion still require actual
verification before any renderer experiment.

## Fixed proposed bounds and unsupported cases

- 27,000 worst-case exported card corners and9,000 card triangles: whole-model
  bound82,805 vertices including the existing4,096 reserve, and71,581 triangles.
  Current actual model has94,909 vertices /76,981 triangles.
- Original atlas stays7,471,104 pixels; no larger texture, more roots, new paid
  generation, native-groom render or normal-map-only fur substitution.
- Safe source UV/corner-normal/skin correspondence must be reconstructed from
  the raw `_AUTHOR_VERTEX_ID` export and native snapshot before prototype output.
  Current screen establishes positions/faces only. Missing or ambiguous evidence,
  part/fur boundary crossings, degenerate/inward patches and new intersections
  are unsupported—not permission to guess or silently reduce the requested count.
- Existing cheek/shoulder folds remain visible and separately rejected. A coat
  comparison cannot accept those defects or change the approved assets.
- This remains a short-pile component hypothesis. Dense native-fiber lighting,
  long hair, all viewing scales, continuous motion and Muse parity are unproved.

## Execution

`execution.json` records the exact frozen command; `screen.py` and selected source
snapshots are retained. One CPU child completed in0.262 s, PID70167 exited.
No RSS samples were captured before exit, so sampled memory is unknown/null,
not zero. The guard was30 s /512 MiB /16 MiB output. All input hashes match after
execution. No native, provider or AI-role calls occurred. The next safe action is
the proposed bounded CPU prototype, then one root-owned matched native component
comparison only if its correspondence/geometry gates actually pass.
