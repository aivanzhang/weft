# Next portable-fur hypothesis: upright fringe fins

**Proposed, not implemented or accepted.** Keep the new matte undercoat and replace the existing random surface ribbons with small, static, skinned fins rooted along selected surface edges. Optimize the visible outline, not whole-body hair coverage. This is one representation change; no new model, shells, billboard, paid call or relaxed quality gate.

## Observed evidence

I viewed the rejected tuft preview's front and 1.36-second flight pose, and the adaptive-undercoat preview's front, side and matching flight pose. The tuft result has obvious dark dash/block patches, isolated crest spikes and a largely smooth underlying body. The latest undercoat result avoids that widespread confetti appearance but still has a smooth head, ears, arms and belly outline. The exposed cheek/shoulder marks remain a separate defect; fins must not conceal them and claim repair.

The current `combed-v1` ribbons lie mostly along the body: fixed lift/lean 0.35/0.8, 14 mm nominal length, 5 mm width, ten alpha fibers per tile, interpolated vertex root colors, global flow. The rejected `tuft-v1` added regional flow, exact root-UV color, larger cards and three broad fibers. Its preview still failed visually; its earlier full finish also exceeded the unchanged 2 GiB bound. Reusing its sampling and color helpers is reasonable; repeating its broad surface-card geometry is not.

Reuse the existing [coverage diagnosis](../../docs/portable-fur-diagnosis.md): the 0.80% area proxy is not visible coverage and was **not recomputed** here. More identical cards cannot establish density. Different native/exported lighting also prevents treating those views as a controlled softness comparison.

## The changed representation

**Working name: `edge-fringe-v1`.** A fin is a small 3D quad whose plane contains a short source-mesh edge segment and an outward groom direction. It stands approximately perpendicular to the body rather than lying across it. At a face-on surface it tends to appear edge-on; near the outline it exposes its tapered tips. This is a geometric hypothesis, not measured visibility yet.

- Build a deterministic, view-independent final mesh. A fixed multi-view design sample may prioritize likely outline edges, but fins never rotate toward the camera, appear/disappear by view or replace the body with a picture.
- Select eligible source edges using the existing fur mask, exact part ownership and regional flow. Keep eyes, nose, companion, material seams, authored closure faces and incoherent source-normal neighborhoods excluded and counted. Do not hard-code panda coordinates or bone names. A non-furry subject receives no fins.
- Score candidates against a bounded ring of design views and a small set of already-authored poses, then distribute roots with minimum spacing. Do not spend the finite budget uniformly across the whole body. Reject insufficient safe candidates instead of filling the count with overlapping fins.
- Each fin has two triangles. Its root endpoints sit on the actual edge; each tip inherits its corresponding endpoint's normalized four-weight skin, so the anchored edge follows deformation. Preserve the original base geometry, normal maps, rig and animation bytes.
- Use actual fin-plane normals and standard two-sided rendering, not the old body's inherited normal on a perpendicular plane. This changes the previous backface-lighting mechanism and requires its own front/back light test; it is not permission to enable backfaces on legacy cards.
- Reuse exact root-UV color sampling and the shared groom field. Limit outward extension to 5–9 mm for this 1.21 m example, with modest lateral lean. Parameterize dimensions from requested coat length/body scale, not species.
- Start with 6–10 mm root spans and **two coherent tapered tip groups** per tile, neutral-white RGB multiplied by source colors. One 1024² RGBA atlas, 4×4 tiles; keep `MASK` and the existing 0.18 cutoff. Do not lower alpha thresholds to obtain coverage. These are visible tip groups, not a claim of individual hair strands.

The driver is **upright orientation plus outline allocation**. A different atlas alone, larger surface ribbons, more native child hairs or a roughness change would not test this hypothesis.

## Fixed geometry, texture and display limits

| Quantity | Bound or observed value |
| --- | ---: |
| Latest actual base, excluding the two portable coat meshes | 51,367 vertices / 62,581 triangles |
| Current portable coat | 24,181 vertices / 9,600 triangles |
| Proposed maximum | 4,800 fin quads / 9,600 triangles |
| Conservative exported fin corners | 4,800 × 6 = 28,800 vertices |
| Data-only derivative ceiling, including 4,096 reserve | **84,263 vertices** |
| Integrated source/split conservative ceiling | **84,435 vertices**, unchanged current budget |
| Resulting triangle ceiling | **72,181**, below 100,000 |
| Atlas working size | 4 MiB RGBA8; 16 MiB if a native float image is allocated |
| Final GLB | Existing 64 MiB cap; current model 33,519,456 bytes |

The current preview bounds give 1.209953 m height. For planning only, an approximately 1.43 m orthographic framing gives the following **projected upper bounds for a face-on fin**; actual obliqueness, culling and occlusion reduce them:

| Display | 6–10 mm span | 5–9 mm height | Separation of two tip groups |
| --- | ---: | ---: | ---: |
| 720 px | 3.0–5.0 px | 2.5–4.5 px | 1.5–2.5 px |
| 512 px | 2.1–3.6 px | 1.8–3.2 px | 1.1–1.8 px |
| 256 px | 1.1–1.8 px | 0.9–1.6 px | 0.5–0.9 px |

Thus the target is a short fuzzy fringe at ordinary size, **not resolved fine strands at 256 px**, dense full-body native fur or Muse parity. A 256² texture tile cannot rescue a subpixel projected feature. Near-contact defects and white-body surface softness still need independent judgment.

## Cheap screen before any full hair rerender

1. **One analytic CPU screen**, at most 15 s / 256 MiB / 16 MiB output: a curved patch with a two-bone bend, constant and seam-split root colors, and known normals. Check fixed topology, endpoint attachment, winding, finite frames, exact color/weight ownership and all budgets. Project at 720/512/256 px from front/side/oblique/back; box-filter the two-tip masks at those footprints using the unchanged cutoff. Record collapse/merging and mark this as analytic, not GPU quality. Reject if 512 px projected tips systematically collapse or face-on patches remain broad visible blocks.
2. **One CPU existing-model derivative**, at most 30 s / 512 MiB / 96 MiB output: remove only the two old coat meshes from a new GLB derivative and add fins; preserve base/material/texture/rig/action data exactly. Use no native strands or `.blend` rerender. Save the recipe, seed, edge/root/source-UV/skin correspondence, actual exported counts, excluded-candidate reasons, hashes and rejected results. No new companion or cap work.
3. Only if both screens pass, request the root-owned **single bounded exported preview**, at most 120 s / 2 GiB / four threads / 128 MiB output: unchanged lighting/camera, front/side/three-quarter/back plus existing early/flight/landing poses, with matched current-baseline comparisons at 512 px and a reduced 256 px contact sheet. This is not a full native-groom rebuild. Existing previews took about 22–24 s and 1.21–1.27 GB sampled child RSS; those measurements are context, not a forecast or permission to raise limits.

Accept only a visibly softer outline without dash/confetti patches, spikes, dark inverted lighting, marking drift, gaps or motion detachment. Preserve the original deformation and skin gates. If the fins become a spiky fringe, disappear at the ordinary target size, or need surface-wide layering to look plausible, reject this hypothesis rather than expand it into shells or repeat the broad-tuften attempt.

## Handoff

Current work is only this read-only design review: **zero native, provider or AI-role calls**. No production code, approved asset or shared document changed. The current model remains `needs_visual_review` / partial component, not a completed high-quality fur solution. Root decides whether to implement the analytic screen. All reviewed inputs and images are hash-bound in `evidence.json`; originals stay in their existing folders.
