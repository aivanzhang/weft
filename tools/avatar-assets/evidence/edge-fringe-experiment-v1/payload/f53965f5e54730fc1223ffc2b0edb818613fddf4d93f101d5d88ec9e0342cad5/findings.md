# Safe-edge source census

Status: one read-only CPU pass completed. No geometry, render, model call, or production change. This is source feasibility evidence, not fur-quality approval.

## Selection and result

- Source model: `../panda-adaptive-undercoat-finish-v2/model.glb`, SHA256 `e0488074965a89a9f7eda7788ebb26e7a8f14f98bbb44e1d9856fa091915e5d5`.
- Actual undercoat binding: native source `body_surface` → final material 5 `body_surface__short_pile_adaptive_v1`, primitive 5, 21,919 exported vertices / 31,353 triangles.
- Excluded `body_cap__short_pile_adaptive_v1` because its recorded source `body_cap` is a `normal_continuity.corrections[].materials` target. Reference material `body_surface` remains eligible. UV1 is not used as a cap classifier. Other non-undercoat materials and both old portable coats are unselected.
- 52,683 unique indexed edges: 41,376 shared by exactly two faces, 11,307 boundaries, zero indexed nonmanifold, degenerate, or shared winding-conflict edges.
- 41,345 conservative candidates survive exact indexed adjacency, winding, geometry, normal-coherence, and no-other-exact-position-edge checks. Of these, 9,150 are 6–10 mm long; median candidate length is 12.497 mm. This establishes available source edges, not final fin placement or visibility.
- Three eligible faces contain seven corner normals opposed to their geometric face normal. Their adjacent edges are excluded. A further 25 shared edges have endpoint-normal angles over 60°; filter categories overlap.
- Boundary records include 10,796 same-primitive index splits and 503 cross-primitive coincidences; 16 have no exact position-edge match. Categories overlap. 171 boundary records belong to exact-coincidence groups larger than two: no claim of globally manifold geometry. No coincident vertices or edges were welded into candidates.

## Derivative handoff

`candidate-indices.npz` contains only primitive-local exported vertex-index pairs and their two triangle ordinals, aligned by row. Arrays `primitive_5_edges` and `primitive_5_faces` are little-endian int32. These are not anatomical or pre-export source indices. Model hash, accessor IDs, full material definitions, UV0 ranges/schema, skin slot ordering, inverse-bind hash, all node local/world transforms, and concrete witnesses are in `census.json`.

Body node 19 / mesh 0 / skin 0 has 51,367 base exported vertices, 62,581 triangles, and 19 joints. Joint-world × inverse-bind differs from mesh-world by at most 3.592e-7 in rest matrix components. Rest positions use glTF +Y-up authored meters; Blender coordinates are `[x,-z,y]`. UV0 is stored glTF UV data. A future derivative must copy source UV0 and normalized joint weights exactly at each chosen source root, preserve material/part masks, and retain the original model. This census does not authorize that derivative or establish animated edge behavior.

## Bounds and artifacts

Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python outputs/edge-fringe-source-census-v1/execute_cpu.py` (already executed once; append-only outputs prevent reuse).

Observed: 0.512 s wall, 202,080,256 B sampled direct-child RSS, PID 7290 exited. Limits: 30 s / 512 MiB / 16 MiB. Original inputs and frozen scripts remained hash-identical; zero paid calls. The worker decoded 1,009,919 scalar values within the unchanged 2,000,000 cap.

The execution manifest is sealed before this prose: `manifest.json` SHA256 `a13b3a808d2d049e19d4463016e3cfe2064bb66e59a999894c4a5ebbcbba41f0`; report SHA256 `1368b0c67c1d46888cf18dc51fd40258801456799d5ea398dbd691bf55728435`; candidate indices SHA256 `9f69aa0aee97f3a3775864020238449597290ef4c7faf1322424ff649c326f6a`. `findings-receipt.json` binds this append-only summary to that manifest.

No silhouette, density, groom-direction, animation, self-intersection, physics, or appearance approval is implied. No retry or live handle remains.
