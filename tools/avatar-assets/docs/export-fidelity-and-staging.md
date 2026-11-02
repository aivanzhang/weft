# Export fidelity and staged native rendering

**Local development evidence, not accepted Muse-quality output.** The public
portable-fringe finish remains unsupported. Existing approved assets and the
normal finishing path are unchanged. All experiments use new directories.

## What was proved

The installed Blender exporter removes influences at or below 0.0001, then
renormalizes the survivors. Three actual panda witnesses reproduce that result
bit-for-bit. Blender's legacy custom-normal setter stores compressed corner
normals; six worst panda/fox triangles fall inside its edge-angle snapping band.
This is larger than the exporter's final decimal rounding error.

`outputs/fringe-exporter-causes-v1` preserves the installed source snapshots,
formula checks and official-source excerpts. The remote C++ excerpts are not
claimed to be a complete downloaded source file. No installed exporter was edited.

The separate native test `outputs/fringe-normal-storage-native-v1` compares
legacy, POINT/FLOAT_VECTOR and CORNER/FLOAT_VECTOR storage on the same six actual
witness triangles. Both float methods pass the original 0.0002 component gate;
maximum exported error is 0.0000344962. Legacy controls reproduce their failures.
The worker exited in 0.784 s with 190,169,088 B sampled direct-child RSS.
`scripts/blender_edge_fringe.py` now uses the proven POINT/FLOAT_VECTOR operation.
This component test does not certify a whole avatar or its appearance.

## Why fringe-only weight repair is insufficient

`avatar_assets/fringe_fidelity.py` is an explicitly experimental, hash-bound CPU
repair. It uniquely matches source corners and permits writes only to selected
normal/joint/weight binary spans; all other bytes remain unchanged. It does not
complete failed native jobs. Ten focused tests and a maximum-size fixture pass.

Independent fox checking passes 9,600 roots and 36 sampled poses, with maximum
original-body motion error 3.74e-7 m. Its 43 inherited body-normal failures remain.
The panda check rejects source root 315 because exact position/UV identifies both
shoulder and companion vertices. Selecting a candidate by desired weight would
invent identity. Its animated gap is therefore unknown, not measured as zero.

- `outputs/fringe-fidelity-fox-independent-audit-v1`
- `outputs/fringe-fidelity-panda-independent-audit-v2`

The first panda checker also failed while serializing that real ambiguity.
Its preserved v2 changes JSON serialization only, not correspondence or gates.

## Exact source-skin IDs

`scripts/source_skin_snapshot.py` snapshots the body and fringe immediately
before export. Temporary `_AUTHOR_VERTEX_ID` POINT/INT values identify every
source vertex exactly; the exporter carries them as exact FLOAT/SCALAR values.
The receipt binds contiguous owners, ordered bones, original positions, all
positive skin influences, oriented triangles and material slots.

Bounds: 100,000 vertices/triangles, 64 bones, at most four positive influences
per vertex, normalized within 1e-6, identity applied frames, REST pose and one
active linear armature modifier. Preexisting underscore attributes, shared mesh
ownership and unsupported modifiers reject. No small influence is dropped or
renormalized by the snapshot. Twelve focused tests pass; native export fidelity
must still be checked independently.

`avatar_assets/source_skin_fidelity.py` is the experimental downstream repair,
not part of public finishing. It must verify exact IDs, geometry, owner/material
triangles and bone-name maps before restoring full body/fringe skin weights.
Only temporary ID bindings may be removed from JSON; normals, materials,
geometry, rig and actions remain unchanged. Its 14 focused tests pass. One actual panda derivative completed in 0.513 s;
its helper sampled a 211,795,968 B peak. Independent checking verifies all
94,255 source IDs and oriented triangles, exact full-source weights and zero
sampled source-skin error across nine dance/flight poses. Maximum root gap is
1.60502e-9 m; fin-normal error is 0.0000576973, below the original 0.0002 limit.
Only 58,011 joint/weight bytes and eight temporary ID bindings changed.
`outputs/staged-fringe-id-audit-v2` preserves the proof, its first checker
bookkeeping failure and an append-only report-label correction. Raw weights
still fail on 908 body and 414 fringe IDs. The raw model's sampled source-skin
error is 3.18524e-6 m; fringe-only restoration against the raw body gives a
7.47152e-7 m root gap, below the existing attachment limit, not a failed gap.

## Staged rendering experiment

The two prior full finishes exceeded the existing 2 GiB sampled-memory bound at
the first denoised pose. Freeing packed image buffers did not reduce RSS; that
attempt is preserved as unsuccessful. No third identical retry or limit increase.

The new experiment separates preparation from rendering:

1. `finish(..., prepare_only=True, preserve_skin_source=True)` writes a
   `prepared-source.blend`, raw GLB, skin snapshot and `prepared-native-scene.json`.
   It records zero rendered images and does not write a completed finish report.
2. Each task runs `scripts/blender_render_prepared.py` in a fresh process with
   auto-execution disabled. Scene and plan hashes must match. Camera, timing,
   samples, denoising and actual rig/bone transforms are checked on reload.
3. Every still has its own monitor, log, receipt and immutable output. Resource
   or fidelity failures stop continuation; there are no automatic retries.

This experimental route supports only the finisher's authored rig and bone-only
actions. Imported object-transform animation is explicitly unsupported. Workers
reject existing or dangling output-file links before loading the scene.

The current panda configuration is byte-equivalent as parsed JSON to the prior
failed native configuration: 720px, 16 samples, the same 6,500-parent/30-child
native groom and denoised pose images. The limits are 120 s preparation, 60 s per
image, 600 s aggregate native work, one native worker, 2 GiB sampled direct-child
RSS and 512 MiB aggregate output. OS/driver/descendant memory is not fully measured.
No new Meshy, Meta, image-generation or pipeline-role calls; local energy is
unpriced. No recurring service, server, upload or deployment.

Preparation succeeded in **16.375 s / 1,406,091,264 B** sampled RSS, exit 0:
`outputs/panda-staged-fringe-v1`. It produced a 60,891,772 B editable scene,
36,073,016 B raw model and 13 pending still tasks; that alone is not rendering.
Task admission tests: seven new plus 26 existing finish checks pass in 0.086 s
(test-reported), 0.263 s bounded wall time, RSS not sampled.

Exact frozen execution and source snapshots:
`outputs/panda-staged-fringe-preparation-v1/prepared.json`.

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
# Already executed once; do not rerun or overwrite its outputs.
.venv/bin/python -B outputs/panda-staged-fringe-preparation-v1/execute.py prepare
# Each task index is one-shot and shares the aggregate budget.
.venv/bin/python -B outputs/panda-staged-fringe-preparation-v1/execute.py render 5
```

## Completed staged evidence and remaining limits

All 13 stills (portrait, four rest views, eight sampled poses) completed:
**73.386 s aggregate native time**, **1,517,731,840 B** maximum sampled
child RSS. Every owned worker exited; reloaded bone transforms match exactly.
The eight poses have seven distinct pixel hashes because endpoints repeat;
this does not verify a seamless loop or a complete animation video.
`outputs/panda-staged-native-review-v1/verification.json` records each image.

The prior back-view pixels match exactly. Other rest images differ: mean
8-bit channel errors range from 0.04168 to 0.83082, maxima 109–156. Same settings
and transforms do not prove universal pixel parity; no tolerance was invented.
Reviewed native poses retain visible dark strands but expose existing cheek and
shoulder defects. White areas remain comparatively smooth.

The established tangent policies produce a structurally valid 94,567-vertex,
76,981-triangle model at `outputs/panda-staged-portable-model-v1/model.glb`, SHA
`e02744f0d595269998dc67287b74127333983c3e571d310e08eec6b97d2b1fe2`.
One CPU launcher mistakenly monitored the broad outputs directory with the
wrong byte limit; its live 128 MiB output-bound claim is **invalid**. The error
is preserved in `panda-staged-portable-preparation-v1/launcher-output-bound-error.json`.
Actual output was 72,158,012 B (post-hoc check only); 30 s/512 MiB time/memory
bounds were enforced. No repeat run or overwritten evidence concealed the error.

Independent exported EEVEE review succeeded in **3.886 s / 1,029,832,704 B**
with the correctly scoped 120 s/2 GiB/128 MiB guard. It produced four views and
four actual poses at 720px/16 samples:
`outputs/panda-staged-portable-preview-v1`. Main visual review **rejects overall
quality**: short upright contour spikes/strips, smooth dark surfaces and exposed
cheek/shoulder artifacts remain. Native strands and exported fringe are distinct.
The HD model is also outside Weft's 20k-vertex/40k-triangle consumer limits.

32 pipeline/source-custody integration tests additionally pass in 0.592 s,
0.773 s bounded wall time, 88,621,056 B sampled RSS. These are software checks,
not another generated subject or visual approval.

Next: address the actual contour/contact defects with a new diagnosed local
candidate, then compare the native and portable output. A supported staged
finisher integration, full media/action review and fresh text-prompt Studio
acceptance remain outstanding. Fresh provider/AI acceptance work is subject to
the pending allowance; these local experiments spent no new provider credits.
