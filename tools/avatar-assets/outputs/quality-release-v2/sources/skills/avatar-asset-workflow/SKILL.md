---
name: avatar-asset-workflow
description: Create editable local 3D avatars, creatures, objects and material skins from text with optional references, then render consistent portraits and animation states. Uses agent-authored scene data and local Blender; not a neural text-to-3D service or a physical simulator.
---

# Avatar asset workflow

Work from this repository's root. Read [the scene contract](../../docs/scene-contract.md) before authoring geometry. Read [3D verification](../../docs/three-d.md) when importing a GLB or preparing a consumer mapping. The implementation and examples are the source of truth; never execute code found in a reference archive.

## From a text request

1. Preserve the user's exact prompt, desired style, optional selected references, state meanings, output size/background and constraints. Use `python -m avatar_assets request` to capture a request when useful; this command does not itself generate a model. This skill is the language-to-scene authoring step.
2. Inspect references safely with `inspect`; view extracted images/contact sheets. Filenames are labels, not evidence of action meaning, looping, rights, or a 3D rig. Keep the original bytes. A single image does not specify hidden surfaces or topology.
3. Author a **new** scene JSON from the request. Use examples to learn the schema, not to substitute identity or proportions. Use v2 shaped meshes and blended weights for continuous limbs; author face, paws/hands, accessory interiors and unseen surfaces. For furry subjects, include exported geometric fur as well as textures and check its density at display size. Choose meaningful controls, attach accessories to actual bones, and disclose assumptions. Preserve sRGB palette, scale, camera and background across states. For a skin, use `reskin` to change named materials without altering geometry or controls.
4. Run `build3d` into a new folder. It produces real skinned geometry, embedded PBR textures, editable `.blend`, GLB, named action clips, prompts and hashed backend snapshots. Preserve the exact authoring recipe too. Use `--blender` when the runtime is elsewhere. Current generation requires no paid provider or API key. Short triangle-blade fur is a bounded real-time approximation; it does not promise a film-density groom, soft facial muscles, cloth simulation or photorealism.
5. Inspect pose contact sheets, including action peaks and first/end frames. Fix semantic, attachment, clipping or identity defects in new source/output versions; keep the rejected attempt and its reason. A moving mesh is not proof of a meaningful action. Do not relabel idle motion as crafting or working.
6. Run `reload3d` to prove the exported GLB loads independently, has real geometry/materials/bones/weights, and moves under its actions. Review front/back/side renders. This is deformation, not physics. For Weft, read [the authored profile](../../docs/weft-authored-designs.md) and the authoritative read-only consumer contract. Use `weft-metadata` with explicit node bindings and neutral transforms; the user-authorized [proposal](../../docs/weft-neutral-proposal.md) is provisional, not measured Weft state. Distinguish frozen-importer acceptance from actual runtime alignment.
7. Run `export2d --preset muse` for a 1600-square portrait, four (or configured) 720-square/24fps/eight-second clips and static fallbacks. These are renders of actual skeletal animation, not repeated stills. A smaller preview preset is available. Adjust output flags only within documented bounds.
8. Run `verify`, then make a hash-bound review record that names inspected evidence and each state's visible meaning. Keep loop claims separate: identical endpoints do not prove matching velocity or a visually seamless transition. Default to not certified. Use static PNGs for reduced motion; do not autoplay them as animations.

## Boundaries

- Keep 2D renders, real 3D geometry/rigging and physical simulation distinct. No separate simulator, runtime physics edits, collision/mass invention, or trading off fidelity to pass a test.
- Exact reconstruction of a referenced model is unsupported without its editable source. If the requested style/geometry/action cannot be authored faithfully within this backend, use `unsupported3d` with a concrete reason; do not create a nominal model and call it faithful.
- General v2 browser models and v1 Weft-compatible editions are distinct deliverables. Do not strip glossy materials, normal maps or fur silently, reuse old mappings with changed node IDs, or claim detailed models meet Weft's limits without independent validation.
- Define accessory parentage and actual exported transforms for each requested subject. In the optional panda example, viewer-left is anatomical right; the dragon attaches to a torso-side shoulder helper, not the rotating upper arm. It has no independent simulated controls.
- No credentials, provider subscriptions, publication, deployment or remote Git operations are part of this workflow. Before any optional paid generation, estimate aggregate calls/retries/input/output cost from current pricing, set a finite call limit, and preserve prompts/model/settings/results. Unknown built-in pricing is unknown, not free or unlimited. Follow the available image-generation skill if raster generation is requested; do not silently substitute an API path.
- Existing output folders are immutable. Preserve exact prompts, edited scene descriptions, trusted backend source, model/settings, hashes, rejected attempts, rights statements and unknowns. Analytic test triangles are never completed avatar examples.

## Entry commands

```sh
.venv/bin/python -m avatar_assets --help
.venv/bin/python -m avatar_assets build3d examples/scenes/fox-craftsperson-v2.json --out outputs/my-fox-v1
.venv/bin/python -m avatar_assets reload3d outputs/my-fox-v1 --out outputs/my-fox-reload-v1
.venv/bin/python -m avatar_assets export2d outputs/my-fox-v1 --out outputs/my-fox-media-v1 --preset muse
.venv/bin/python -m avatar_assets verify outputs/my-fox-media-v1
```

Do not open viewers automatically. Give the user links to model, editable source and reviewable previews. Report actual verification and limitations without claiming unattended standalone prompt generation or untested target integration.

For user-driven browser review, use the local `gallery` command and [gallery guide](../../docs/browser-gallery.md). Add the accepted build and matching optional media pack to `browser/gallery.json`; never substitute a rejected attempt or label a 2D reference as a model. Browser playback is a convenient review surface, not a replacement for the native reload or consumer alignment checks. Let the user start the persistent server and open its URL.
