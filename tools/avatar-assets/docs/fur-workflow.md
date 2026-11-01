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
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/muse-panda-smooth-fur-v2.json --out outputs/panda-finish-next
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
