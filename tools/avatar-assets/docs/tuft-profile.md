# Portable tuft experiment

Status: **opt-in finisher integration tested; the first actual candidate failed
the unchanged2GiB memory bound. Not visually accepted.**
`legacy`, `combed-v1`, `fine-v1` and `distance-v1` retain their previous outputs.
No existing model, texture, source or result folder was changed.

## Adapter contract

```python
from blender_fur_cards import card_geometry, attach_fur_material, tuft_export_budget
from groom_flow import prepare_flow
from fur_root_sampling import RootColorSampler

# Measure the base export and include every band's count in this one guard.
budget = tuft_export_budget(base_export_vertices, sum(b['count'] for b in bands))
field = prepare_flow(low, span, native_direction, native_comb, groom_flow)
sampler = RootColorSampler(source_pixels, selected_triangle_uvs, pixels_are_linear=False)
data = card_geometry(vertices, selected_triangles, weights, settings, normals,
                     profile='tuft-v1', prepared_flow=field, root_color_sampler=sampler)
material, material_report = attach_fur_material(
    name, {'color': [1, 1, 1, 1], 'fur_atlas': 'tuft-v1'}, output)
```

The finisher now validates the opt-in, selects the new atlas, and multiplies its
neutral color by the existing supported `FurRootColor` export graph. It requires
exact source root colors and the native regional field; the global-only helper
fallback is not accepted by this integrated profile. The helper does not modify the base
mesh, native groom, masks, rig, actions or other materials. It does not implement
short-pile normal-map detail or a separate coat carrier.

`prepared_flow` is the existing `groom_flow.prepare_flow` result. Its bounds and
coordinate basis must match the native emitter's regional field. It is evaluated
at each actual sampled root; portable direction, comb and regional length scale
are not inferred from the subject. Without a field, `settings.direction` and
`settings.comb` (default0.65) give an explicit global field with length scale1.
That fallback is supported math, not proof of regional parity.

The adapter pairs every eligible loop triangle with its actual UV corners and
validates one borrowed float32 pixel buffer for all bands. Per-band triangle
ordinals are mapped back to that sampler. It retains the existing declared
encoded-sRGB interpretation of `image.pixels`; this change does not independently
prove Blender's decoding convention. Cards finish before `add_native_groom`, so
the up-to256MiB root-sampling array is released before the dense-hair stage.
Normal, color-correction and exact-part exclusions remain in force. Both helper
sources are snapshotted, and `tuft-export-budget.json` records the whole-band
allowance before construction.

`root_color_sampler(triangle_ordinal, barycentric)` is a trusted adapter callable
returning one finite **linear RGB** triple in0..1. The ordinal refers to the exact
input triangle list, even when a degenerate triangle is skipped. Its three UV
corners must come from that triangle's real loops, not each vertex's first UV.
The helper calls it once per root and copies that RGB to all six positions.
Passing interpolated per-vertex `colors` to `tuft-v1` is rejected. Neither optional
input is silently applied to older profiles.

## Representation and bounds

- Seeded equal-area face strata allocate roots by eligible triangle area.
- A shuffled2:1 mix of short broad pile and longer tufts creates two geometric
  scales, without duplicating a full-body shell or camera-facing avatar planes.
- Each card has six positions, four outward-oriented triangles, one atlas tile
  and at most four normalized inherited skin influences.
- The two centerline segments use the native comb's direction formula. Regional
  length scale is applied before layer variation and explicit clamps. Final
  centerline length is1–40mm; width is0.3–20mm. Clamps are counted in diagnostics.
- The new1024² atlas has sixteen seeded tiles, each with three separated tapered
  alpha fibers and neutral white RGB. It uses single-sided `MASK`, cutoff0.22,
  roughness0.90 and metallic0. Old atlas profiles are unchanged.
- Input maxima are100k source vertices/100k triangles/64 influences per vertex;
  output is at most5,000 cards. The caller must additionally run the aggregate
  export-budget guard across all bands before construction and verify the actual
  GLB afterward. Twelve exported corners/card plus a default4,096-vertex reserve
  must fit100k vertices; this guard never silently reduces counts or decimates.
- Diagnostics contain compact counts/ranges and at most128 evenly spaced root
  records. The helper does not decode or copy source images; the adapter retains
  existing image bounds and should release its sampling array before dense groom
  rendering. Existing native runtime/memory/output limits remain unchanged.

## Verification and next action

Command:

```sh
.venv/bin/python -m unittest tests.test_tuft_fur_cards tests.test_combed_fur_cards tests.test_fur_cards -q
.venv/bin/python -m unittest tests.test_finish_quality_contracts tests.test_tuft_fur_cards tests.test_fur_root_sampling -q
```

Analytic tests cover exact old-profile hashes, seeded topology/skin/UV behavior,
shared regional flow, distinct UV islands, dimensions, invalid/bounded inputs,
aggregate exported-corner limits and compact diagnostics at5,000 cards. A BOX
downsample of each atlas tile to8px keeps three separated alpha groups in the
tested middle row. The alpha area above cutoff is approximately42.17%; this is
**atlas occupancy, not visible body coverage or GPU texture filtering evidence**.

Next: investigate the retained failure before another native build. A separately
authorized diagnostic exported preview could screen the already-written raw GLB
without repeating grooming. No such preview has run. Actual ordinary-size
pile/silhouette, markings, highlights, flecks, contact seams and motion attachment
still need review. More cards, a larger area proxy or passing math tests do not
establish convincing fur or repair hidden geometry.

## Prepared actual comparison

Recipe: [panda-portable-tuft-v1.json](../examples/meshy/panda-portable-tuft-v1.json),
SHA256 `d3bf765350839680d139bb1e0bc1f8e96cfefbc70e61db5898c3cc40981038ed`.
It derives from the accepted dance/flight pack's exact source design,
SHA256 `42fa109957109ad7afaa503b7269f7e9069a81674bbc98eb0d9a6ec4d54e8909`.
Source GLB SHA256 remains
`734ebdbbd06c0428a57f817d1a5deed8ff7078016ab262abb313513d94f293ca`.
Its read-only preflight measured51,055 vertices and62,581 triangles.

Only identity of this experiment, its exact edit prompt, portable coat and review
render controls differ:3,600 cards split1,980/1,620 in the existing bands,
18mm base length,10mm base width,720px native portrait/views/poses at16samples.
The original6,500-parent/30-child groom, source geometry/materials, masks,
regional field, rig, weights, normal repairs and exact dance/flight keys remain.
The conservative exported ceiling is98,351 vertices including the4,096 reserve.
The retained raw GLB passed a read-only format preflight at87,333 vertices,
76,981 triangles, one skin and one `dance_flight` animation. Full finishing remains incomplete. The separately rendered portable appearance was
subsequently rejected; see the sampled review below.

Commands (finishing failed; a separate raw-GLB diagnostic later succeeded):

```sh
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/panda-portable-tuft-v1.json --out outputs/panda-portable-tuft-v1
.venv/bin/python - <<'PYCODE'
from pathlib import Path
from avatar_assets.provider_model import preview
preview(Path('outputs/panda-portable-tuft-v1/model.glb'),
        'outputs/panda-portable-tuft-diagnostic-v1', resolution=720, samples=16,
        motion_states=[{'id':'dance_flight','seconds':[0,1,1.18,1.36,2.36,2.55,2.8,4]}])
PYCODE
```

Both stages are serial and use4CPU threads/2GiB sampled direct-child RSS.
Finishing:600s/512MiB output; exported preview:600s/256MiB output. No full video,
new Meshy/model call, paid retry or old-output overwrite. New provider cost:$0;
local energy cost is not priced. The shared heavy-worker slot was released before
execution. Root/independent review must judge the actual new images before promotion.

On2026-11-01 after13:57 PST, session16284 ended with wrapper exit1 after the runner
terminated/reaped native PID60066 (exit-15). `job.json` records27.428s,
2,148,352,000 bytes sampled peak direct-child RSS,34,596,327 retained output bytes,
`resident-memory budget exceeded`, `process_exit_verified:true` and
`manual_stop_required:false`. The limit stayed2,147,483,648 bytes. No worker remains
active, no retry was started, and no gallery was promoted.

The log localizes the crossing to Cycles setup for the second native pose
(frame24). Portable geometry/export, native portrait, four rest views and the
first native pose completed first. This does not prove why process RSS grew;
the image-sampling allocation was no longer in the function's live scope.
`source.blend`, final finish report and tangent postprocessing were not completed.
The raw GLB is diagnostic evidence, **not a completed avatar pack**.

Immutable evidence:

- Failed manifest SHA256:
  `0ce03c8209d331893cce9bc9156a7ad5536c0ac5c244a996b5cd20afa57e6006`.
- Job receipt SHA256:
  `ef031e62ad35ccb3a4b04c9e76447b9af4c1621ec5fc8b1a6788afcaa59032a5`.
- Raw model SHA256:
  `3da9e96520c6fd47ee19f5160db6e24385217da2b999d63d3e1fc0362fcaa6da`.
- Full failure and exact source snapshots:
  [panda-portable-tuft-v1](../outputs/panda-portable-tuft-v1/manifest.json).

Keep that folder unchanged. An appearance diagnostic or a changed memory strategy
must use a new sibling output, retain these failure receipts, stay within the
existing limits, and get the shared heavy-worker slot before execution.

## Read-only follow-up

The retained [native front](../outputs/panda-portable-tuft-v1/views/front.png),
[back](../outputs/panda-portable-tuft-v1/views/back.png),
[portrait](../outputs/panda-portable-tuft-v1/portrait.png) and
[first pose](../outputs/panda-portable-tuft-v1/renders/dance_flight/0000.png) preserve
recognizable proportions and markings, with fine speckling/noise and a rear-head
ridge still visible. These images hide the portable cards. They **cannot assess
the new exported tufts**, and the native baseline used different resolution and
sample settings. The first pose alone cannot assess the departing contact area.

Read-only GLB/embedded-atlas comparison:

| Property | Previous `combed-v1` | New raw `tuft-v1` |
| --- | ---: | ---: |
| Portable exported vertices | 24,181 | 36,196 |
| Color attributes per coat primitive | `COLOR_0` only | `COLOR_0` only |
| Atlas RGB range above cutoff, encoded8-bit | 192–255 | 255–255 |
| Actual atlas alpha occupancy above cutoff | 27.087% | 42.169% |
| Alpha cutoff | 0.18 | 0.22 |
| Roughness | 0.78 | 0.90 |
| Double-sided | false | false |

The new `COLOR_0` values contain actual dark/light source colors, not dummy-white
values or a second unused color attribute. This confirms the intended data path,
not visible coat density, color matching or convincing fur. Baseline model SHA256:
`693f8af27d9d18b5beeb344e6168f9c79471f82b9660780867b8d71956ff2c67`;
candidate raw model SHA256:
`3da9e96520c6fd47ee19f5160db6e24385217da2b999d63d3e1fc0362fcaa6da`.

### Memory hypothesis, not a measured fix

Export finished in3.110s before rendering. All five undenoised rest renders log
the same Blender-tracked peak542.56MiB. The first720px denoised pose logs578.30MiB;
the process RSS bound crossed during setup/sample0 of the next pose. Those tracked
figures are not total process RSS. The runner saved an aggregate sampled peak,
not a complete per-render RSS series, so neither a leak nor enabled persistent
Cycles data is established.

The worker currently shares one process across glTF export and all native renders.
The installed glTF exporter also retains module-level gather caches until settings
change; their retained byte size was not measured. The source sets no explicit
`use_persistent_data` value. Do not infer that it was enabled or alter undocumented
exporter cache internals based on these observations.

After independently reviewing a separate raw-GLB preview, the smallest proposed
memory experiment is to set `scene.render.use_persistent_data=False` explicitly,
free only owned `RENDER_RESULT` image buffers after each successfully saved PNG,
and record process RSS before/after every render. Keep source texture buffers,
samples, resolution, groom, geometry, framing and saved images intact. The image
buffer operation and resulting memory behavior still need verification in that
new bounded worker; no implementation or retry has run.

If the memory headroom remains insufficient, use a hash-verified early saved
`source.blend` and fresh serial render processes, with the same aggregate wall,
memory and output allowance. Process isolation is the fallback, not permission to
multiply the deadline, reauthor the model, lower quality or raise2GiB. No memory experiment has run. The completed appearance diagnostic below used a
fresh import/render worker and did not repeat or alter the failed finish.


## Actual portable review: rejected

The independent raw-GLB preview completed in24.035s with1,207,336,960bytes sampled
peak RSS and74,525,877output bytes. Four rest views and eight actual exported
`dance_flight` poses are720px/16samples. All57 animation channels match the
accepted source exactly (maximum numeric difference0); all25 preview artifacts
passed their hash inventory. This is preview evidence for an incomplete parent,
not a completed finish or a recovered animation.

The parent agent independently inspected the actual exported
[front](../outputs/panda-portable-tuft-diagnostic-v1/front.png) and
[1.36-second flight pose](../outputs/panda-portable-tuft-diagnostic-v1/renders/dance_flight/0003.png).
**Rejected:** conspicuous dash/block speckling, a spiky crest, and a body that still
reads as plastic rather than dense soft fur. These are sampled visual findings,
not user approval or a complete video review. Do not promote or rerun this profile
unchanged. More or larger sparse cards did not solve the coat appearance.

- [Immutable sampled review](../outputs/panda-portable-tuft-review-v1.json),
  SHA256 `e730cba8046f68fd4e749f29600d3f19f4501086f76dc1f384d2db48daa45ac3`.
- [Diagnostic context](../outputs/panda-portable-tuft-diagnostic-v1.context.json),
  SHA256 `072efce6fdd6ea9673c85f3e37bcc5539f5ded336adf6200543f47c45353ea27`.
- Preview manifest SHA256
  `f1cdfb829f4927d59a69f7684d67eefa46aa23c036e43c756d5b83a021f8cd01`.

The next separate hypothesis is an opt-in short-pile normal texture on measured
fur-bearing faces, with only subtle silhouette fibers. Normal detail changes
surface lighting, not geometry or actual hair density. Preserve the source palette,
UVs, non-fur masks, rig and motions; leave the old HD generator byte-identical.
The memory hypothesis remains a separate future experiment, not a silent budget
increase or an untraceable combined retry.

## Opt-in short-pile undercoat experiment

`scripts/blender_fur_undercoat.py` adds `fur_undercoat` to the finishing adapter;
it does not change `blender_surfaces_hd.py` or enable a default quality policy.
Settings are exactly `profile:short-pile-v1`, resolution1024/2048, bounded seed,
pitch2–20mm, strength0–0.5, roughness0.5–1, and an explicit selected-face range.
The adapter requires the existing native groom and regional field. Only triangles
whose three vertices pass the final measured groom mask receive cloned materials.
Their base-color graph and UVs remain; their previous normal input is explicitly
replaced by the authored undercoat and their roughness/metallic are declared.
Excluded surfaces and original material graphs stay intact.

The baker uses the existing `PreparedFlow`, original loop UV0/tangent frames and
deterministic rest-space noise. It biases small normal tilts across the comb
direction; it does not repaint color, displace geometry or claim recovered hair.
UVs must be in0–1. Different surfaces overlapping more than1% of covered texels
are rejected, and degenerate selected UV triangles are unsupported. The limits
are100k vertices/triangles,32million candidate raster samples,90seconds,2048px,
and chunked temporary arrays. The outer native worker still owns the2GiB bound.
Six pure baker tests cover deterministic chunking, non-fur exclusion, unit/tilt
bounds, input immutability, overlap rejection, malformed/oversized inputs and
raster/time limits. Combined with adapter/rest-evidence tests:27passed in1.809s.
These analytic tests establish neither visible softness nor coat density.

### First diagnostic: correspondence failure, not an appearance result

The first prepared comparison kept the accepted dance/flight GLB's existing
portable fibers unchanged. It bound model SHA256
`693f8af27d9d18b5beeb344e6168f9c79471f82b9660780867b8d71956ff2c67`,
design `42fa109957109ad7afaa503b7269f7e9069a81674bbc98eb0d9a6ec4d54e8909`,
and measured-density witness `source.blend`
`3537d88b09c6b9ef59ee0e77d89378ecae8b9961a182b2551d4e63383822727e`.
Settings:2048px,4mm pitch,0.22 strength,0.92 roughness,seed4517;
no fiber, rig, motion, color or UV change. The proposed append-only GLB patch
retains the entire original binary prefix and copies original geometry rows only
where material ownership requires splitting. A temporary analytic patch fixture
verified all62,581 original body-triangle attribute hashes, skin/motion data and
portable meshes exactly; it was not an appearance asset.

The authorized native bake in
[panda-portable-undercoat-v1](../outputs/panda-portable-undercoat-v1/manifest.json)
stopped before creating a texture or patched model. Session91195/nativePID70286
exited1 and was reaped in2.343seconds, with429,162,496bytes sampled peak RSS and
3,090,719output bytes. No resource limit was raised or crossed. The combined
native-import/source-GLB correspondence predicate failed; individual deltas were
not saved. No patch, preview, automatic retry or visual acceptance followed.

[Read-only follow-up](../outputs/panda-undercoat-import-audit-v1.json) records
SHA256 `e91c787e9c370dcf08e06ae6aa844ebf814bb802eaec9b834edc190a5e731bb9`.
Blender's importer skins vertices into bind pose using float32 arithmetic.
Recomputing the actual source's rest transforms produces2.3841858e-7m position
differences, beyond the initial2e-7m threshold on2,486 body vertices. This is a
specific plausible cause, not a measured native mismatch. A changed probe must
save shape/index/UV/position deltas first, retain exact triangle ordering, and
allow at most1e-6m import-rounding correspondence while keeping original GLB
position bytes exact. Preserve the failed folder; any next native probe needs a
new sibling and the shared worker release. New provider/model calls:$0/zero.

### Sampled deformation evidence

The adapter now captures actual base-body rest arrays before grooming for both
new authored rigs and existing skins. `deformation_measurement` binds little-
endian float64 XYZ/int32 edge hashes, actual bone names, measured meter-scale
bounds, route and sample count. Each state binds the rest hash and measured
sample count. Every evaluated pose must retain vertex count, edge count and
exact edge endpoint ordering. Unsupported/missing evidence cannot become an
empty-list success. Two focused analytic tests cover the actual capture helper;
this undercoat diagnostic does not execute or verify a pre-rigged motion job.
