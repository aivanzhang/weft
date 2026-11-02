# Adaptive short-pile surface detail

This optional generated-model finish adds fine normal-map detail underneath an
existing groom. A normal map changes lighting, not the mesh silhouette. It is
**not actual fur, denser hair coverage, geometry repair or Muse-quality proof**.
The default finishing policy is unchanged. Native hair and portable fur cards
still need separate appearance review.

## Inputs and routing

Add this optional object to a validated `finish_generated` design that already
contains `groom` with direction/comb and `groom_flow`:

```json
{
  "fur_undercoat": {
    "profile": "short-pile-v1",
    "resolution": 2048,
    "seed": 4517,
    "pitch_m": 0.004,
    "strength": 0.22,
    "roughness": 0.92,
    "min_faces": 1,
    "max_faces": 40000,
    "uv_policy": "material-adaptive-v1"
  }
}
```

UVs are the coordinates that place a texture on a surface. Each selected material
is measured separately. Adequate original UV0 coordinates stay unchanged. A
region whose coordinates overlap distinct surfaces or have insufficient density
can receive an authored normal-only atlas: UV1 for a single-layer source, or UV2
when original UV1 already serves source textures. Color textures remain on their
original coordinates. The new atlas uses 1024 pixels; retained UV0 uses the
requested 1024 or 2048 pixels. The report records this per-material choice.

The selection comes from the existing measured fur mask, not a panda-specific
rule. Only selected faces use cloned materials with replaced normal response.
Existing linked roughness/metallic maps are preserved; configured constants apply
only to unlinked sockets. Original material graphs, images, positions,
weights and animation data are not intentionally modified by this adapter.
The exporter still runs independently and must be verified.

New atlas bakes declare `mikk-unshared-triangle-v-up-v2` in their plan and material
reports. This matches Blender's per-triangle texture orientation even when authored
corner normals point to opposite sides of a face. The exported correction requires
that exact saved bake policy, clone name, face count, UV index and PNG hash;
missing or changed evidence fails. Original source UV1 tangents are not selected
just because they use UV1. Rebuild the normal
textures and tangents together, never reinterpret an old PNG by changing tangent
signs alone. The [MikkTSpace implementation](https://github.com/mmikk/MikkTSpace/blob/master/mikktspace.c#L913-L926)
defines this per-face orientation; native fixtures check Blender's actual result.

Omitting `uv_policy` retains the prior single-UV baker; omitting `fur_undercoat`
disables the feature. The AI author may propose this public object, but cannot
set the private export budget or raise resource limits.

## Bounds and explicit refusals

- At most four selected material regions; at most 2,048 faces in any region
  needing an authored atlas. Selected polygons must be triangles.
- The coordinator budgets exact source-export vertices, all portable cards at
  twelve exported corners each, and a 4,096-vertex reserve. The adapter adds
  conservative UV/material seam counts. The total must stay within 100,000.
- Aggregate analysis/baking is bounded by 32 million raster candidates and
  90 seconds, within the existing native worker limit of 600 seconds, 2 GiB
  sampled direct-child RSS and 512 MiB output. These are ceilings, not latency
  predictions or whole-process-tree memory guarantees.
- Flow directions are computed once per call and shared read-only across
  material bakes. One float atlas is retained at a time; completed PNGs are
  packed without keeping each decoded float buffer resident.
- One or two original UV layers are supported, with active/render UV0. Three
  source layers, conflicting layer names, unsupported material graphs,
  ambiguous polygon ownership, invalid frames or insufficient budgets fail
  explicitly. No source repaint, hidden remesh, budget increase or paid retry
  follows from failure.

## Run and inspect

From `tools/avatar-assets`, use a **new** output path:

```sh
.venv/bin/python -m avatar_assets.finish_generated path/to/finish-design.json \
  --out outputs/my-undercoat-candidate-v1
```

The result includes the original design, resolved private configuration, pinned
backend sources, PNG normal maps, an editable Blender source, animated GLB,
portrait/pose evidence and reports. `finish-report.json` records regional UV
routing, hashes, shared-flow work and export-budget assumptions. A successful
build is `needs_visual_review`, not an accepted delivery. Failed attempts retain
their logs and available outputs in the same folder; do not overwrite or blindly
rerun them.

Inspect native and independently loaded GLB views, actual state poses and final
media. Unit-length tangents alone do not prove that normal lighting uses the
correct UV directions. The native analytic fixture deliberately rotates UV1 and
checks exported tangent direction and both handedness signs against geometry.
It also checks unchanged source color pixels, geometry, UV0, excluded faces and
active/render UV settings. Analytic triangles are not generated avatar examples.

**Weft compatibility is unchanged:** its diffuse UV0-only importer does not accept
this HD normal-mapped/extra-UV representation. Do not strip detail silently or claim
consumer acceptance from the authoring validator.

## Current verification

The [UV2 preservation increment](undercoat-uv2-preservation.md) passes 111 unique
focused checks and an actual analytic Blender bake/export. Original UV1 normal
and roughness/metallic maps survive; only the explicitly authored tangent bytes
change in the export derivative. This is component evidence, not full finishing
acceptance of the repaired panda or a fresh Studio delivery.

The integration passes **682 software tests**, an analytic native/exported frame
proof, and an actual existing-source panda finish and reload. It has not passed
the user's portable-fur quality target or fresh text-driven acceptance.

The first native fixture caught a UV-layer handle invalidation; reacquiring the
layer preserved UV0 in fixture v2. V2 then exposed the exporter's UV0-based
tangents on a UV1 normal texture. The first actual panda build exposed a second
issue: corner-dependent signs disagreed with MikkTSpace on opposed normals.
All failed fixtures/builds remain immutable. The new policy rebakes normal PNGs
and corrects only the derivative export's selected tangent storage.

Normal PNGs retain their baked green-axis orientation. The checker must preserve
that basis rather than interpreting a flipped image-coordinate V axis as a
request to invert normal lighting. The [glTF normal-map example](https://github.com/KhronosGroupArchives/glTF-Sample-Models/blob/main/2.0/NormalTangentTest/README.md)
documents its OpenGL normal convention; the [specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes)
defines how the stored tangent sign reconstructs the bitangent. Verification uses
known direction vectors and the actual native bake coordinates, not just a
passing unit-length check.

Six native analytic cases include smooth, mirrored and opposed corner normals.
The final CPU verifier reproduces the exporter's float32 normal quantization
exactly, records signed-zero differences separately, and retains the unchanged
direction thresholds. Maximum tangent error is 0.002761° and sampled baked-PNG
world-normal error is 0.003507°. Only allowed tangent bytes change. See
`outputs/adaptive-undercoat-verifier-v6`; analytic triangles are not avatars.

Actual `outputs/panda-adaptive-undercoat-finish-v2` completed in **122.283 s** at
**2,085,765,120 B sampled direct-child RSS**, close to the unchanged 2 GiB ceiling.
Its model has 75,548 vertices, 19 skin bones and one four-second dance/flight clip.
The independent reload produced four 720px views and eight actual poses in
**22.214 s / 1,266,188,288 B sampled RSS**. Those poses are not a new MP4 or a
seamless-loop proof. The portrait remains an opaque static fallback.

Sampled review retains the soft native hair but finds substantially smoother
portable fur and visible cheek/shoulder marks after dragon takeoff. Result:
**partial component, not promoted**. No approved model, gallery or default changed.
Exact hashes, failures, bounds and commands are in [the handoff](../TODO.md) and
`outputs/panda-adaptive-undercoat-review-v2`. All native workers exited; this
feature check made zero new AI, Meshy or image-generation calls.
