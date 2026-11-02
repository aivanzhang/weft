# Furry multi-material source selection

**Use the existing card-free fox baseline:** `outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb`, SHA `f34aa798252355fc6e44a03f78cdc7f31eab682a20b591a93c9dcb35ed82f8a3` (27,708,644 bytes). It has one mesh, 30,418 referenced vertices, 47,776 triangles, a real 20-joint skin and four authored clips. This is an existing procedurally authored fox, not new neural generation.

| Furry material | Baseline primitive / material ID | Base image | Normal image | Base image size | Factor |
| --- | --- | --- | --- | --- | --- |
| coat | 0 / 0 | 1 | 0 | 2048² | omitted → [1,1,1,1] |
| cream | 1 / 2 | 4 | 3 | 2048² | omitted → [1,1,1,1] |
| dark | 7 / 9 | 7 | 6 | 1024² | omitted → [1,1,1,1] |

All routes use UV0, distinct images, no texture transforms, REPEAT wrapping, linear/trilinear filtering. Three selected base images total9,437,184 pixels. Existing negative REPEAT UVs must remain valid.

Mesh node20/skin0 is parented to identity AvatarRig node21. Joint slots are **not** node IDs; the full exact map is in `review.json` (root node19, torso18, head2). Metre scale, +Y up/+Z forward and rest anatomy are authored assumptions, not measured animal anatomy or Weft compatibility.

Preserve and exclude the scarf, eyes, nose, inner ears, book and pencil. The12 active constant-color materials are exclusion/preservation coverage—not permission to grow fur on them. Unused legacy card materials1/3/10 remain in the header; do not treat them as live emitters. No new model or emission occurred.

The generated raw panda (`f3b8c6f…47188`) is a valid single-image furry control, not a multi-image proof. Its shoulder dragon requires an explicit non-fur mask. The existing rigged panda fur demo has three meshes and is not selected.

For future material-owned native emitters, preserve source skin/actions and enforce one **aggregate** fiber budget. The old fox's5,000 parents and30 children imply at most155,000 fibers total, not per material. This note does not establish native execution, memory fit, dense fur or visual quality.

Only saved JSON and bounded GLB headers were read. Source hashes come from pinned existing custody evidence; no image pixels, accessor arrays, native/provider/AI-role/browser calls or repeated inventory jobs were used.
