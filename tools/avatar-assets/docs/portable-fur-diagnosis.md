# Portable fur: measured diagnosis

Status: **the current exported coat is not accepted as convincing fur**. This is a
read-only diagnosis, not a new asset or successful repair. No code, source assets,
provider jobs or renders were changed for it.

## Evidence inspected

- [Exported motion at 1.36 seconds](../outputs/studio-exported-motion-proof-v1/renders/dance_flight/0003.png),
  plus the saved [front](../outputs/studio-exported-motion-proof-v1/front.png),
  [side](../outputs/studio-exported-motion-proof-v1/side.png) and
  [back](../outputs/studio-exported-motion-proof-v1/back.png), all 720×720.
- [Native portrait](../outputs/muse-panda-dance-flight-v2/portrait.png),
  [recipe](../outputs/muse-panda-dance-flight-v2/finish-config.json),
  [finish report](../outputs/muse-panda-dance-flight-v2/finish-report.json) and
  [actual atlas](../outputs/muse-panda-dance-flight-v2/textures/PortableCoat_1-fur-atlas.png).
- The immutable [card implementation](../outputs/muse-panda-dance-flight-v2/sources/blender_fur_cards.py)
  and [finishing implementation](../outputs/muse-panda-dance-flight-v2/sources/blender_finish_generated.py).
  Current implementations have the same relevant portable-coat behavior.

Audited GLB SHA256:
`693f8af27d9d18b5beeb344e6168f9c79471f82b9660780867b8d71956ff2c67`.
Exported-proof manifest SHA256:
`0ef1295db5d4892558367490bf40f07ebeaed3cb9cc84ce71be743eb7f855b25`.
The existing proof took 23.28 seconds, sampled 1,201,995,776 bytes peak direct-child
RSS and retained 72,545,104 bytes. Its process exited. These are baseline
measurements, not a forecast for a new coat or a browser FPS result.

## What is actually wrong

### 1. The portable coat barely covers the smooth undercoat

At ordinary display size the exported face, belly, ears and limbs read as smooth
material with scattered flecks. The back and side do not show a substantial
furry silhouette. Native hair is visibly different; its render uses a different
camera/lighting setup, so the images are not a controlled material A/B comparison.

Measurements from the actual GLB and atlas:

| Quantity | Observed value |
| --- | ---: |
| Base exported geometry | 51,137 vertices |
| Portable geometry | 24,181 vertices / 9,600 triangles |
| Ribbons | 2,400; each 6 geometric positions / 4 triangles |
| Total exported model | 75,318 vertices |
| Summed body triangle area, excluding companion/cards | 2.83583 m² |
| Summed ribbon triangle area | 0.08344 m², or 2.94% of body area |
| Atlas alpha coverage above the actual 0.18 cutoff | 27.09% |
| Card-area × atlas-coverage / body-area proxy | 0.80% |
| Ribbon extent along its inherited surface normal | 3.54–6.18 mm; median 4.83 mm |

The area proxy is **not measured visible coverage**: it ignores overlap,
projection, culling and the excluded emitter regions. It nevertheless explains
why adding a few more identical ribbons cannot produce a dense coat. Doubling
this representation still leaves the smooth surface dominant.

The current 5 mm-wide ribbon contains ten atlas fibers. At roughly 500 pixels
per 1.2 m full-body image, that is about two pixels across the entire ribbon,
not ten readable strands. Analytical box-downsampling of the saved tile changes
binary alpha occupancy from 27.09% at 256² to 51.98% at 16² and 50% at 2²; at 1²
every tile is above cutoff. This is not a GPU measurement. It demonstrates why
minification can turn many thin fibers into a solid speck rather than fur.

### 2. Native regional grooming is absent from the portable geometry

`groom_flow` changes native strand direction and length, including muzzle,
cheeks, crown and shoulder. `portable_coat` passes only the global direction and
fixed ribbon dimensions to `card_geometry`. The latter uses fixed lift/lean
coefficients of 0.35/0.8. It cannot reproduce the native regional volume or
direction. Increasing the native parent/child counts has no effect on the GLB.

### 3. Material export is present; this is not simply missing alpha or weights

Both fur materials really export `MASK`, cutoff 0.18, roughness 0.78,
metallic 0, UV0 and linear `COLOR_0`. All 9,600 card triangles face consistently
with their inherited outward normals. Normal lengths are unit within rounding;
maximum weight-sum error is about 1.2e-7. Cards are single-sided deliberately;
blindly enabling backfaces reintroduces the known inverted-normal lighting issue.

There are two remaining color concerns worth testing, not declaring proven bugs:

- The atlas is not a neutral white multiplier. Opaque atlas pixels average
  encoded RGB 0.896; the authored gray root/tint modulation darkens source-colored
  ribbons. Their sparse distribution makes that contrast conspicuous.
- Root color is interpolated from three **vertex samples**, not sampled from the
  actual emitter-triangle UV at the root. The sampler chooses each vertex's first
  UV loop. Texture boundaries, repaired contact UVs and seams can therefore
  produce a different color from the surface directly below a ribbon. Do not
  "fix" this by globally whitening the coat or suppressing legitimate markings.

The base body's sampled roughness median is 0.627 (5–95%: 0.467–0.678); metallic
median is zero. Raising coat-region roughness can reduce plastic-looking sheen,
but cannot supply missing volume or strands. Preserve intentional eye/nose shine.

### 4. Contact marks are a separate source-surface problem

The departing companion exposes a mottled, angular shoulder patch. Small dark/
bright raised-looking cheek marks remain near the old contact. Those areas also
remain limited in the prior native proof; they are not explained by sparse cards
alone. The source has authored closures, contact UV edits and retained topology
diagnostics; the normal-continuity pass changed shading, not those surfaces.

Do not hide these defects under a thicker coat and label the geometry repaired.
Exact per-pixel attribution to geometry, cap UV or shading still needs a controlled
fur-disabled/base-color/normal comparison. Existing source bytes stay unchanged.

## Generic repair direction

1. **Use the same regional field for native strands and portable roots.** Evaluate
   the existing generic `groom_flow` at each sampled root using actual normalized
   bounds. Apply its direction, comb amount and length scale; retain eye/nose,
   material and exact moving-part masks. No species coordinates in the backend.
2. **Build a multi-scale coat, not more identical narrow strips.** Use a short,
   source-colored surface pile with directional normal/roughness detail, plus
   visible geometric tufts at the silhouette and selected longer regions. Texture
   detail is supplementary; it is not a replacement for real portable fur geometry.
3. **Allocate ribbons by actual eligible area and requested regional density.**
   Use bounded seeded, spatially distributed roots instead of arbitrary fixed
   dark/light counts. Share per-root source UV sampling across material selection,
   color and skin inheritance. Never interpolate across unrelated UV islands.
4. **Design atlas fibers for their displayed footprint.** Wider resolved tufts,
   fewer fibers per tile and a neutral color multiplier are a better first test
   than ten subpixel fibers. Review near, ordinary and reduced display sizes;
   preserve sharp alpha silhouettes without additive bright speckles.
5. **Budget exported corners before construction.** The present 6-position
   ribbon can cost up to 12 exported vertices after tangent/normal splits.
   `floor((100000 - base_export_vertices - 4096) / 12)` allows 3,730 total ribbons
   here, not the generic 5,000 maximum. A different tuft topology must use its own
   conservative corner cost. Do not silently decimate the approved base model.

If this hybrid cannot give sufficient volume, the next representation change is
a separately authored low-resolution coat carrier with layered shells and tufts,
not copies of the full 51k-vertex mesh. Such a carrier needs fresh skin/projection,
alpha-layer and silhouette checks; it is not already implemented or verified.

## One bounded next experiment — not executed

Create a new derivative, e.g. `outputs/portable-fur-tuft-v1`; keep the approved
source, native groom, body geometry, rig, actions, textures and prior outputs.
Hypothesis: regional flow, resolved tufts, correct root colors and a matte
short-pile surface will read as fur while remaining below the fixed vertex limit.

- Add a new profile rather than changing `combed-v1` bytes. Start with at most
  3,600 six-position ribbons: worst-case total 94,337 vertices before any new
  surface-detail implementation. Proposed initial dimensions are 1.5–2× current
  width, native-region-derived length capped by existing bounds, and 2–4 resolved
  fibers per atlas tile. These are experiment parameters, not accepted defaults.
- Implement shared regional flow and root-UV color sampling first. Keep the
  source colors; remove unnecessary gray atlas tint in the new profile. Add
  short-pile normal/roughness detail only on observed furry surfaces, not eyes,
  scales or metal. Any texture derivative must retain its exact recipe and hashes.
- Build once under the existing 600-second / 2 GiB / 512 MiB bounds. No provider
  calls or paid retries. Recheck actual GLB counts, normals, weights and action
  evidence before rendering. Do not increase memory/vertex limits to obtain a pass.
- Run one exported-only preview: four 720px rest views and the existing eight
  `dance_flight` key poses under the same lighting/camera as the baseline. Use a
  fresh output folder and the existing bounded preview runner. Do not render a
  full video until these images show a real improvement.
- Compare ordinary-size silhouette, visible pile, smooth-surface exposure,
  flecks, eye highlights, markings, contact seams and motion attachment. Then
  inspect the browser; native render success does not prove browser parity.
  Record actual elapsed time, RSS, output/GLB sizes and browser behavior.

Reject the candidate if the coat remains smooth/plastic, gains grass-like clumps,
dark/bright flecks, shell bands, exposed masks or conceals an unrepaired contact.
Do not lower the quality requirement or call numerical coverage proof of success.
