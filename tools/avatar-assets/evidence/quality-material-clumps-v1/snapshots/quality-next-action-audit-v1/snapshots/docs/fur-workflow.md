# Furry avatar workflow

The existing [`generate` pipeline](prompt-pipeline.md) accepts text, optional
references, requested states and output presets. Its author can now use regional
native grooming, combed browser cards and bounded local color corrections.
These are generic controls; a new subject still needs its own measured masks,
bone positions, proportions and action review.

## Separate the three representations

| Representation | Purpose | Important limit |
| --- | --- | --- |
| Clean generated mesh and texture maps | Shape, markings, face and accessories | A single picture does not reveal the back or a rig. A 4K map is not proof of detail. |
| Editable native hair in `.blend` | Fine coat in portraits and rendered animation | Not exported as hair in GLB; no simulated hair dynamics. |
| Skinned cutout cards in GLB | Lightweight visible coat in the browser | Less dense than native hair. Review color, silhouette and motion separately. |

For text generation, request a continuous undercoat surface rather than carved
fur ridges or spikes. For reference generation, inspect the actual guide first.
Meshy's image-mode texture prompt does **not** replace its geometry guide. An
optional image edit can prepare a clean-undercoat guide while preserving identity;
preserve both images and the exact edit prompt. This raster-preparation step is
currently agent-assisted, not an automatic built-in step of `generate`.

Never disguise a failed mesh with more hair. Review the bare source from front,
side and back before rigging. Keep the user's original reference unchanged.

## Reusable finishing controls

Use a new `finish_generated` JSON and a new output directory. The implementation
in `avatar_assets/finish_generated.py` validates the complete schema; the AI
author contract in `avatar_assets/pipeline_prompts.py` describes the same fields.

- `groom`: bounded native parent/child count, length, radius, seed, roughness,
  direction and comb strength. Increase strand footprint only after inspecting
  normal display size, memory and silhouette.
- `groom_flow.regions`: up to twelve named ellipsoids with `center`, `radius`,
  `feather`, `direction`, `comb` and `length_scale`. Positions are normalized
  against the source bounds in Blender coordinates: X left/right, +Y back, +Z up.
  Regions blend in order; roots and skin remain fixed. Facial hair can be shorter
  than cheeks or body. Length scales are 0.25–2; final strands remain at most 8cm.
  **This regional flow currently applies only to native hair.**
- `groom_exclusions` and `fur_max_chroma`: keep eyes, nose, inner mouth and
  non-furry accessories visible. Color alone is not sufficient for all subjects.
- `groom_shader`: native longitudinal/radial roughness. This controls sheen, not
  additional geometry or invented texture detail.
- `portable_fur.settings.profile: "combed-v1"`: flatter, tapered ribbons aligned
  with the cards' global direction. `sample_root_color: true` uses source-texture
  root colors converted from sRGB to linear vertex colors. The source image must
  explicitly be sRGB. The original `legacy` profile remains available.
- `color_corrections`: at most four explicitly authored surface-color repairs.
  A face must satisfy both a spatial region and a dominant-channel test; bounded
  face counts reject accidental broad matches. Original texture pixels, UVs and
  geometry remain unchanged. Corrected vertices are excluded from fur because
  their original root-color texture has not changed. This is not a texture repaint.
- `weight_constraints` and ordered rig layers: prevent a moving limb from
  pulling unrelated body vertices. Review actual peak poses and accessory seams;
  neutral-pose appearance and a valid skin do not prove good motion.

The panda recipes are examples, not species-independent defaults. The current
source mesh is calibrated to an authored 1.2m, glTF +Y up/+Z forward. It has
sixteen authored bones plus a shoulder-owned `CompanionAnchor`. The supporting
arm stays neutral; the companion is not an independently simulated character.
Exact exported node IDs, rest matrices and ownership are saved with each model.

## Commands

From `tools/avatar-assets`, with the existing isolated `.venv`:

```sh
# Prepare a different character without an AI/provider call.
.venv/bin/python -m avatar_assets generate --prompt 'A small russet fox with a soft short coat, cream muzzle and green scarf; idle, look around, wave and celebrate' --preset preview --max-credits 35 --prepare-only --out outputs/fox-request-next
# Execution still requires an actual authorized ledger allowance.
.venv/bin/python -m avatar_assets generate --resume outputs/fox-request-next --execute
# Local-only finishing after a measured recipe change; choose a new output.
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/muse-panda-smooth-fur-v3.json --out outputs/panda-finish-next
.venv/bin/python -m avatar_assets reload3d outputs/panda-finish-next --out outputs/panda-reload-next
.venv/bin/python -m avatar_assets.provider_model outputs/panda-finish-next/model.glb --out outputs/panda-export-preview-next --resolution 1024 --samples 32
```

Start full media only after sampled poses and portable appearance pass. A
bounded HD look-development export can use `export2d --preset preview --size 720
--portrait 1600 --seconds 2 --samples 8`. It retimes the complete four-second
actions into two seconds; do not advertise it as original eight-second Muse
timing. Static fallbacks are actual first frames. Backgrounds are opaque and
seamless loops remain uncertified even if endpoint poses match.

## Current panda evidence and costs

Selected native fur-demo model: [`muse-panda-smooth-fur-v3`](../outputs/muse-panda-smooth-fur-v3/manifest.json).
Its [portrait](../outputs/muse-panda-smooth-fur-v3/portrait.png),
[GLB](../outputs/muse-panda-smooth-fur-v3/model.glb) and editable
[`source.blend`](../outputs/muse-panda-smooth-fur-v3/source.blend) are real outputs,
not analytic fixtures. All 26 saved key poses passed sampled review for these
explicitly authored alternatives:

| State | Visible action |
| --- | --- |
| `idle` | Small settling/sway |
| `look_around` | Gentle alternating head turns |
| `head_tilt` | Alternating curious side tilt |
| `tail_sway` | Tail lifts and dips twice; not a horizontal wag |

The prior wave/celebrate trials remain rejected; these states are not relabeled
completion of those actions. The native portrait is showable fur progress. The
[portable preview](../outputs/muse-panda-smooth-fur-export-preview-v3/three-quarter.png)
is a lightweight review candidate: it no longer has widespread white cards on
black surfaces, but its sparse tufts remain visibly less plush than native hair.
Native sampled acceptance is not browser-fur parity or final human acceptance.

Model SHA256: `a5664da6dd59e4f9fed0e9154c7af75e198d9a9750edce74d258687a98c3ab77`.
It has 74,600 exported vertices, 71,717 triangles and seventeen skin joints.
Source includes 4K basecolor/normal and 2K metallic/roughness maps. Seven native
flow regions preserve strand roots; longest stored parent strand is 21.601mm.
The 6,500 parents / 30 children are configuration counts, not a visual score.
The GLB instead contains 2,400 cutout cards with inherited weights.

Matching compact media:
[`muse-panda-smooth-fur-media-v3-portable-v1`](../outputs/muse-panda-smooth-fur-media-v3-portable-v1/manifest.json).
It contains a1600px opaque portrait, four720px/24fps/two-second silent H.264
clips with48 distinct decoded thumbnails each, exact first-frame static fallbacks
and contact sheets. The full four-second actions are retimed into two seconds.
The filtered portrait retains its raw render; filtering softens fine strands.
Native rendering took560.204s with1,683,128,320B sampled peak RSS, within the
unchanged1200s/2GiB limit. The10,294,024-byte compact artifact set is byte-preserving;
196 full frames and the render log remain in the complete local media folder.

From Weft, `./tools/avatar-assets/run gallery` now shows this panda first, followed
by the unchanged approved dragon and prior examples/reference. The former default
is preserved as `browser/gallery-before-panda-fur-v1.json`. The focused two-model
catalog is `browser/gallery-panda-fur-v1.json`; use it for the complete pipeline
browser test, which requires actual models and matching media rather than2D references.

Build: 58.857s / 1,825,210,368B sampled child RSS. Independent GLB preview and
reload passed in 17.780s / 2.592s. Residual maximum sampled edge extension is
19.20mm near the companion during look-around, 8.52mm during head tilt and
8.42mm during tail motion; mild tail squashing remains. No collision, every-frame
or seamless-loop claim follows from these samples.

### Rig and consumer handoff

- Units: authored meters, 1.2m total height; right-handed glTF +Y up / +Z forward.
- `Head`: node 0; `CompanionAnchor`: node 5, parent `RightShoulder` node 6;
  `Root`: node 16 under `CreatureRig` node 18. The anchor is skin ordinal 9:
  **`JOINTS_0` indexes `skin.joints`, not node IDs.**
- The companion is a rigidly weighted region of the shared mesh, not a separate
  dragon skeleton or independently simulated actor. Its 25,803 selected source
  vertices become 26,028 exported rigid-anchor vertices after splitting. A small
  card boundary retains partial anchor influence; inspect seams when changing actions.
- Rest pose is the exported node hierarchy plus inverse bind matrices. Root/object
  scales are identity; maximum measured rest-times-inverse-bind error is 3.48e-7.
  Full local/world transforms, skin order, materials and source hashes are in the
  [actual-model audit](../outputs/panda-fur-focus-v1/actual-model-color-and-rig-audit.json)
  and [exported nodes](../outputs/muse-panda-smooth-fur-v3/exported-nodes.json).
- The native anchor's display bone is authored three meters long; this is a helper
  display length, not companion size or recovered anatomy. GLB stores transforms
  and inverse binds, not this editable-bone endpoint.
- `rig-contract.json` targets local controls, **not Weft**. Its empty rigid-object
  attachment list does not describe this skinned region; the region selector and
  actual parent are recorded separately in `exported-nodes.json`/the audit.
- [Unchanged-fidelity Weft conversion is explicitly unsupported](../outputs/panda-fur-focus-v1/weft-unsupported-v1/unsupported.json).
  Keep prior compatible panda/robot outputs unchanged. No ten-body-frame mapping,
  importer change or physics/action/reward change was made.

One built-in image edit produced the smooth guide in
`outputs/panda-fur-focus-v1/geometry-guide`; its exact prompt, original reference
and image hashes are preserved. The tool did not expose a model version or
account charge. There was no image retry.

The user approved one additional 35-credit Meshy job, bound to the frozen plan
in `outputs/muse-panda-smooth-base-v2`. Task
`01a1984c-da9c-70f4-959f-b765d6c752ff` completed in 248.43 seconds: one submission,
35 credits, no paid retry or purchase. The raw GLB SHA256 is
`f3b8c6f48222ef2e588ef1798f788a79909a660f5467b92719d2242b7ad47188`.
The canonical ledger then recorded sixteen submissions / 202 credits against
210 aggregate allowance; eight base credits remain. Dollar cost per credit is
unknown. The pricing estimate was 30 credits for 7.1 image/4K texture plus five
for 4K geometry; see [official Meshy pricing](https://docs.meshy.ai/en/api/pricing).
No Meta credentials/budget, new subscription or deployment was used.

The original ZIP remains unchanged with SHA256
`cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`.
Generated-media metadata is provenance, not independent rights verification.
User-supplied reference ownership/license and the original 3D source are unknown.

Preserved rejected attempts:

- `muse-panda-fur-flow-v1`: directional hair cannot hide the older coarse mesh.
- `muse-panda-smooth-fur-v1`: much better native portrait, but rear-belly arm
  weights stretch by 93.61mm and a missed companion detail stretches by 32.69mm.
- Its portable preview shows white flecks: Blender's exporter stored the real
  root colors in `COLOR_1`, leaving white in the glTF base-color channel
  `COLOR_0`. Exported-channel verification is required, not just attribute presence.

The replacement source also invents a long rear tail stalk. A small authored
color repair changes its green surface to charcoal; it does not fix tail shape
or establish original anatomy. All hidden surfaces, scale, rig, weights and
gestures are authored/generated assumptions. Muse-quality parity, facial/finger
controls, prop interaction and faithful frozen-profile Weft conversion remain
unverified or unsupported. This detailed mesh exceeds Weft's 20k-vertex limit;
do not change the consumer to admit it.

See [TODO](../TODO.md) for the current selected candidate, live handles, tests,
local bounds, publication receipt and next safe action. Creation manifests remain
immutable; later review records must bind their exact hashes.
