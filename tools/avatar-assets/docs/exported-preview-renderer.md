# Exported-model preview renderer

Exported GLB review now requires bounded **EEVEE Next**. Native hair and portrait
renderers remain Cycles. This change preserves the preview API, framing, actions
and output paths; it addresses a measured rendering difference, not a quality
promotion or a general renderer benchmark.

## Why the renderer changed

glTF requires single-sided materials to cull backfaces. Double-sided materials
show both sides and reverse back-facing normals before lighting.
[Official glTF specification, §3.9.5](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#double-sided).
The installed Blender importer sets the material flag, but that alone does not
prove a renderer applies it.

A tiny actual GLB with four controls was imported once and rendered from front
and back in both engines. Cycles showed all 81 sampled interior pixels of each
single-sided backface that should be hidden. EEVEE showed zero. Both sides of the
double-sided control remained visible with reversed normals. For the opposed
single-sided triangles sharing one authored normal, EEVEE preserved that normal;
Cycles changed it and produced near-black shading in the tested views.

Evidence: `outputs/fringe-culling-native-control-v1/findings.json`, SHA256
`2e91babb1a9ddb34dfa567bc3901fb540ffc0d800771fa6260f38803c0e3b450`.
One invocation: 6.240s / 340,951,040 B sampled RSS, under 60s / 1 GiB /
four threads / 16 MiB. The process exited; no repeated native test or provider
call ran. These are 128px analytic controls, not generated avatars or motion.

## Bounds and failure behavior

`scripts/blender_provider_preview.py` selects `BLENDER_EEVEE_NEXT` and verifies
that `scene.eevee.taa_render_samples` preserves the requested integer in 1..64.
Missing/ignored engine or sample controls fail explicitly; there is no Cycles
fallback. `native-inspection.json` records actual engine and requested/actual
samples under policy `gltf-single-sided-culling-eevee-next-v1`.

Existing wall-time, sampled process-memory and output guards remain. Sampled
child RSS does not measure all driver/GPU allocations. The test covers this
installed Blender/backend, not every platform, shader, shadow or glTF feature.
Headless environments without usable EEVEE remain unsupported for this review
path; they must not silently produce misleading culling evidence.

Six focused setup/metadata checks pass in
`outputs/export-preview-engine-tests-v1`; 49 surrounding provider/guide/pipeline
checks pass in `outputs/fringe-preview-integration-tests-v1` (0.849s).

Four actual model variants—baseline, source normals, source-color atlas, and
both—completed matched EEVEE previews: four rest views and four actual action
poses each, 512px/16 samples. Total native time was 15.037s; each child stayed
within 120s/2 GiB sampled RSS/128 MiB output and exited. Peak sampled child RSS
was 1,087,225,856 B; total retained preview output was 317,982,046 B, including
editable scenes. No full video or browser comparison ran.

Parent review of six images finds that source-surface normals remove most broad
white-body speckles. The color atlas alone does not, and the combined variant
shows new light flecks on dark regions despite passing its LOD0 color gate.
Texture minification is under investigation; no fur profile is promoted.
`outputs/fringe-four-way-review-v1/review.json` binds the matched source hashes,
identical bounds, observed metrics and exact inspected images. Old Cycles images
are not treated as a same-renderer before/after comparison or a speed benchmark.

## Texture-packing follow-up

A bounded CPU screen isolated neighboring-cell filtering from within-cell color
variation. Its modeled dark-tip contamination fell from 2,383 to 110 at LOD4
after sorting unchanged cells by source color. This is a simplified filtering
model, not a measurement of GPU-selected texture levels or actual visibility.

An actual derivative copied all 4,800 RGBA cells byte-for-byte and remapped their
UV coordinates; geometry, normals, skin and actions stayed unchanged. Its 9,600
nominal-tip samples still pass the original color gate after float32 UV export
(p95 .003572, p99 .004260, maximum .019307; zero errors above .1).
The model is `outputs/panda-fringe-packed-v1/model.glb`, SHA256
`502b92030c037e8e97a95befc16a0809473e4dbb6f567a8a1b13d3e788c9e304`.

One matching EEVEE preview rendered eight images in 3.629s / 1,087,504,384 B
sampled child RSS, retaining 82,342,882 B including the editable scene. The child
exited within the same bounds. Direct review of five before/after images finds
most black-region bright speckles removed, while the pale cheek marks and
mottled shoulder exposed during takeoff remain. Packing changes both color
filtering and filtered-alpha visibility; this does not isolate color bleed as
the sole cause. Dense fur, deeper mip isolation and fresh-prompt quality remain
unproved. No default, approved asset or automatic profile changed.

[Packed front preview](../outputs/panda-fringe-packed-eevee-preview-v1/front.png),
[takeoff pose](../outputs/panda-fringe-packed-eevee-preview-v1/renders/dance_flight/0001.png).
Exact review: `outputs/fringe-packed-visual-review-v1/review.json`, SHA256
`6f4efdcfcf8cfb8026998d62134bc51c1c6a69ab87442ab841689276465edcbc`.

Compact receipts, recipes and previews are saved in
[renderer/color evidence](../evidence/fringe-shading-color-v1/README.md) and
[packing follow-up](../evidence/fringe-packing-followup-v1/README.md).
The independent exported-file audit passes; its initial checker-only provenance
allowlist failure remains preserved. Large native/model binaries remain local.

The renderer correction alone does not repair source color-interpolation defects,
create dense fur, establish Muse parity or complete fresh Studio acceptance.
Approved assets, source archives and Weft's consumer/physics remain unchanged.
