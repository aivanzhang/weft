# Portable edge-fringe experiment

**A technical pass and partial outline improvement, not a promoted fur finish.**
The actual panda has softer ear/tail/paw edges, but its white coat is still mostly
smooth and the original cheek/shoulder defects remain. The approved assets,
production defaults and automatic quality gates are unchanged.

## What changed

The generic prototype roots small upright two-triangle fins on safe shared mesh
edges. Each root endpoint keeps its original position, texture coordinate and
four bone weights; its tip inherits the same weights. The mesh does not rotate
toward the camera. A neutral two-tip alpha atlas, geometric normals and explicit
two-sided MASK material replace the old surface-lying coat ribbons.

Selection uses recorded fur/material ownership and an independent edge census.
Authored repair caps, moving-part seams, boundaries, coincident edge groups and
incoherent normals are excluded. No species coordinates or bone names are built
into the fin generator. This experiment reuses a manually prepared panda recipe;
it is **not a fresh hands-off prompt success** or a released CLI profile.

The matte undercoat, body/companion geometry, original texture images, nineteen
joints and four-second `dance_flight` action remain unchanged. Unused old binary
coat data is retained for byte-preservation evidence; it still counts toward
the file limit. Only old coat mesh bindings are removed, and the new primitive
is attached to the original body node/skin.

## Deciding evidence

- The preceding visibility audit found that sparse geometry and unresolved
  fibers dominate the old coat's limitations; backface culling is not the main
  cause. CPU projections are not GPU measurements.
- The native color probe confirmed encoded, bottom-up sRGB byte-image pixels
  for the tested path. Keep `pixels_are_linear=False`; no global gamma fix.
- Independent review found a different issue: the prototype's endpoint-grid
  sampler was not glTF's normalized texel-center REPEAT sampler. The candidate
  adapter corrects this before exporting colors. Eight tiny color oracles pass.
  It samples linear LOD0 colors, not actual GPU mip/derivative filtering.
- A curved two-bone analytic fixture checks 4,800 fins, attachment, spacing,
  masks, normals and deterministic output. Nominal 9mm tips separate at 512px in
  the analytic filter, but all tested spans merge at 256px.
- The actual source offers 41,345 conservative shared edges; 9,150 are 6–10mm.
  The chosen 8.5–10mm range and 3mm spacing produce **3,231 fins**. The range is
  fixed before execution, not expanded after seeing the render.

## Actual result and bounds

| Step | Observed | Fixed bound |
| --- | --- | --- |
| Analytic screen | 0.783s; 120,815,616 B sampled RSS | 15s / 256 MiB / 16 MiB output |
| Source census | 0.512s; 202,080,256 B sampled RSS | 30s / 512 MiB / 16 MiB output |
| Candidate build and CPU checks | 1.279s; 495,222,784 B sampled RSS | 30s / 512 MiB / 96 MiB output |
| Independent native reload/render | 15.275s; 1,040,498,688 B sampled RSS | 120s / 2 GiB / four threads / 128 MiB output |

Memory is sampled direct-child RSS, not a guarantee of peak or descendant usage.
Each child exited and its exit was verified. The native preview retained
78,224,680 B, including its editable scene. No native hair rebuild, full video,
provider generation or pipeline-role AI call ran for this experiment.

The GLB has **64,291 vertices / 69,043 triangles**, one real nineteen-joint skin
and the unchanged action. Root attachment error is at most **1.070e−7m** across
102 numeric times: all 97 frames at 24fps plus the original review timestamps.
Standard structure, surface-frame and weighted-motion checks pass.

An independent byte/attribute audit also verifies all 33,487,168 original BIN
bytes and the exact JSON edit allowlist, including the disclosed generator
metadata. Every fin's root position/UV, normalized weights, geometric normal and
safe-edge membership passes. The 0.262-second child exited before an RSS sample;
its memory is unknown, not zero. It does not repeat the color filtering or
whole-motion audit. [Independent evidence](../outputs/panda-edge-fringe-audit-v1/audit.json).

Four rest views and eight poses were re-imported/rendered at 512² with sixteen
samples. They are opaque RGB PNGs. A 256px static fallback and labeled pose
contact sheet are derived from those images; neither is relabeled animation.
The separate native inspector does not certify animation; CPU skin checks supply
the bounded motion evidence. No browser, collision, hair-physics or seamless-loop
claim follows. The HD model remains outside Weft's diffuse UV0-only/20k-vertex
consumer profile; its importer and physics are unchanged.

## Immutable local artifacts

- [Actual model](../outputs/panda-edge-fringe-v1/model.glb):
  SHA256 `e113fa5372ebddbf04a9860876734d53265ded9611d1ffecd1ae7e2bd87bc2eb`,
  34,524,020 B.
- [Front preview](../outputs/panda-edge-fringe-preview-v1/front.png),
  [flight pose](../outputs/panda-edge-fringe-preview-v1/renders/dance_flight/0003.png),
  [256px pose sheet](../outputs/panda-edge-fringe-review-v1/contact-sheet-256.png),
  [motion-free fallback](../outputs/panda-edge-fringe-review-v1/static-256.png).
- [Editable preview scene](../outputs/panda-edge-fringe-preview-v1/source.blend),
  [generator/config/source pins](../outputs/edge-fringe-candidate-preparation-v2/prepared.json),
  [appearance review](../outputs/panda-edge-fringe-review-v1/appearance-review.json).
- [Compact earlier diagnosis/analytic evidence](../evidence/portable-fur-diagnostics-v1/README.md).
- [Compact actual fringe experiment](../evidence/edge-fringe-experiment-v1/README.md):
  84 files / 4,619,940 B, including all twelve actual previews, editable generator
  snapshots, exact settings, independent audit and preserved preparation history.

Historical commands below have already run once; their append-only guards reject
reuse. A new experiment requires a new pinned output directory and changed,
recorded hypothesis. The v1 candidate preparation remains unexecuted and
superseded by v2's texture-sampling correction, not erased.

```sh
# Run from tools/avatar-assets. Do not repeat these completed jobs.
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B outputs/edge-fringe-candidate-preparation-v2/execute.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B outputs/edge-fringe-preview-preparation-v1/execute.py
```

The next local hypothesis is allocation under nonuniform mesh triangulation,
with a cheap analytic check before another avatar render. It is not implemented
or a justification to raise geometry/resource limits. The separate fresh-prompt
Studio acceptance tests still await their saved spending authorization.

Original rights/provenance remain inherited from the pinned source manifests:
user-authorized generated geometry and locally authored hidden surfaces, rig,
weights, finishing and motion. Metadata is not independent rights verification.
Large experimental GLB/Blend files remain local; compact evidence is not a remote
backup of those binaries. See [current ownership and handoff](../TODO.md).
