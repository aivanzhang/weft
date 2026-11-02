# Edge-fringe analytic screen v1

**CPU-only analytic contracts passed; appearance is unverified.** No panda,
Blender process, generated avatar, provider or AI-role call was used. No production
code, default or approved asset changed.

One monitored child ran in **0.783 s**, sampled peak direct-child RSS
**120,815,616 bytes**, and retained **327,038 bytes** at process completion. Its
exit was verified. Limits were 15 s / 256 MiB / 16 MiB; cleanup grace is recorded
by the existing runner. Source snapshots and the executed command are in
`prepared.json`; `screen.py` contains the actual verification, not mock success.

## Result

- A curved two-bone fixture with 4,961 source vertices and 9,600 triangles
  produced **4,800 fins**, 19,200 geometric vertices / 9,600 triangles.
  Conservative exported-fin cost is 28,800 corners; with this analytic base and
  4,096 reserve, the projected count is 37,857, not an actual GLB count.
- All roots use exactly two source faces. Explicit non-fur, authored-closure,
  material/part/UV seam exclusions were exercised. Open and nonmanifold edges
  are never welded or treated as valid roots.
- Minimum independently checked root-center spacing: **6.515 mm**, versus the
  configured 3 mm minimum. Fin normals match their geometric winding; minimum
  outward direction dot product is 0.98478.
- Five authored linear-blend-skinning poses, from −25° to +60°, retain root
  attachment within **2.18e−8 m**. Endpoint colors match their source UV samples
  within **2.88e−8**. A separate known 2×2 color example verifies sRGB-before-
  interpolation and material-factor behavior. Original arrays remain exact.
- Same-seed repeated small builds match array-for-array. Excess fin/source/total
  vertex counts and invalid weights reject explicitly.
- At 512 px, nominal **9 mm** fins retain two alpha groups in **16/16** tiles
  after the analytic BOX filter, at unchanged **0.18** cutoff. The 6 mm case
  merges; **every tested span merges at 256 px**. No failed size is promoted.

The atlas test uses integer, aligned footprints and a middle-row group count.
It does not simulate GPU sampling phase, perspective, occlusion, culling, AA,
lighting or visible coat coverage. Alpha occupancy is not quality. The analytic
face-on projected extrusion is approximately 0.50 px at 512 px for 15% lean;
this is not a measured absence of block-like marks. Real outline allocation,
two-sided lighting, card intersections and source-specific deformation remain
untested. No hair physics or native-groom equivalence is claimed.

## Provisional pure-data API

`edge_fringe.build_edge_fringe(points, faces, corner_normals, uv0, materials,
parts, fur_faces, non_fur_faces, closure_faces, joints, weights, flow_directions,
textures, design_views, settings, *, source_export_vertices)`

Inputs use meters and one shared coordinate frame. Shapes are N×3 points,
F×3 indexed triangles, F×3×3 corner normals, F×3×2 UV0, per-face material/part
IDs and Boolean masks, N×4 joints/weights, N×3 flow directions, and at most
16 fixed design-view vectors. Texture descriptors contain encoded RGB(A)8
arrays in **top-down/image-V-down** order plus a linear RGB material factor.
Do not pass Blender's bottom-up float image buffer directly or pre-linearize it.
Sampling decodes each sRGB texel before filtering; root opacity is not inherited.

Settings: exact keys `seed`, `min_fins`, `max_fins`, `length_m`,
`min_spacing_m`, `min_edge_m`, `max_edge_m`, `comb`. This screen used 8 mm length,
8.5–10 mm source edges, 3 mm spacing and 0.15 lean. Selection is bounded to
32,000 examined candidates, 4,800 fins and a ten-second internal deadline;
the separate child monitor enforces total runtime, sampled RSS and output.

Outputs are detached geometry/UV/color/skin arrays, source endpoint/face/corner
correspondence, directions and exclusion/budget reports. Geometry is static and
skinned, never camera-billboarded. `two_tip_atlas(seed)` returns the bounded
1024² neutral-white RGBA8 atlas. This is a prototype, not a live CLI profile.

## Next safe action

Root decides whether to build **one data-only existing-model candidate** and
independently screen actual visibility/skin/color before any native preview.
Preserve this directory; it is already executed and must not be rerun in place.
The original command was:

```sh
.venv/bin/python -B outputs/edge-fringe-analytic-v1/execute.py
```
