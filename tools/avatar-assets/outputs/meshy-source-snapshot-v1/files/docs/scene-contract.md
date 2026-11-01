# Scene authoring contract v1/v2

The text interface is the reusable agent skill. The deterministic CLI compiles authored JSON; it contains no language model and makes no network requests. A direct prompt without authored geometry returns `needs_agent_authoring`, not a fabricated success.

Use [fox](../examples/scenes/fox-craftsperson.json), [lamp](../examples/scenes/articulated-lamp.json) and [panda/dragon](../examples/scenes/panda-shoulder-dragon.json) as schema examples. Identity and anatomy are data, not renderer branches.

## Coordinates and geometry

- Scene units are meters. +Z is up; character front is -Y; +X is viewer-right in a direct front view. GLB export converts to right-handed +Y-up/+Z-front.
- Each parent-first bone has a unique `name`, nullable `parent`, and world-rest `head`/`tail`. Each visible part names one material and a fallback bone. Version1 uses rigid one-bone weights; version2 can author blended per-vertex weights. Neither implements runtime IK or physics.
- Shapes: `sphere` radius 1; `cube` side 2; `cylinder` depth 2/radius 1; `cone` base at -Z/tip at +Z/depth 2; `torus` major radius 1/minor radius .25. Per-part `location`, positive `scale` and XYZ Euler `rotation` (radians) place the shape before joining it to one skinned mesh.
- Material colors are **sRGB** `[r,g,b,1]`, converted to linear PBR color. Roughness and metallic are in `[0,1]`. Base surfaces are opaque; generated hair-card materials use explicit alpha masking. Reskinning changes only named materials.
- Closed primitive surfaces are authored, not reconstructed; assembled meshes are not promised watertight/manufacturing-ready. Version2 can replace intersecting limbs with continuous weighted meshes, but deformation still needs pose review.

## Detailed models: schema_version 2

Use `output_profile: "general"`. Version2 is explicitly rejected with `weft-v1`: glossy/normal-mapped/furred models are not silently flattened to that consumer's diffuse-only contract. Existing v1 Weft models remain available.

| Input | Meaning and bounds |
| --- | --- |
| `shape: "mesh"`, `vertices`, `faces` | Explicit local XYZ positions and triangles/quads; 30,000 control vertices/60,000 faces total. Finite coordinates, valid nonrepeated indices and nondegenerate faces required. |
| Mesh `uv` | Optional UV pair per vertex; finite, magnitude≤16. Otherwise a spherical projection is authored. |
| Mesh `weights` | Optional bone→weight dictionary per vertex; 1–4 known bones, positive weights, total1±0.00001. Missing weights use `bone` rigidly. |
| `segments`, `rings` | Primitive detail: 8–64 segments; 6–48 rings. Applies to relevant round primitives only; not explicit meshes. |
| `bevel`, `bevel_segments` | Rounded edges after object transforms: 0–0.3 meters and1–4 segments. |
| `fur` | `{count,length,width,seed}` retains legacy triangle strands. Optional `mode:"cards"` uses six-vertex curved ribbons with masked fiber atlases; optional nonzero `direction` supplies the comb vector. Up to10,000 elements/part,24,000 total and72,000 generated fur vertices; length0.001–0.04m, width0.0003–0.004m (cards≤0.04m), seed0–2³¹−1. Final exported-vertex limits still apply. |
| `native_groom` | Optional `{parents,children,length,radius,roughness,seed}` for an editable native-only groom on a copied, identically weighted base emitter. Parents16–10,000; children0–30; sum of `parents*(children+1)`≤240,000; length0.001–0.04m, physical root radius0.00001–0.001m, roughness0–0.01, integer seed. Optional paired `direction:[x,y,z]` and `comb` bend saved hair keys: finite vector components−1..1, nonzero norm, comb0..0.95. No dynamics or native-hair GLB export. |
| Material `surface` | `{pattern,resolution,strength,seed}`: `fur`, `scales`, `brushed`, or HD-only `iris`; square256/512/1024/2048, strength0–1, integer seed0–2³¹−1. All maps plus hair atlases together≤32 Mi pixels. Existing256/512 algorithms remain unchanged;1024/2048 synthesize fresh fine detail, not enlarged old images. |
| `studio` | Optional `{ground_z,world_strength,exposure}` adds an unexported matte floor and measured lighting controls. Optional `preset:"plush"` chooses stronger directional area lighting and AgX Medium High Contrast. Ground±5m, world0.05–2, exposure−2–2. |

Both legacy triangle strands and curved cards are **real skinned geometry**, distributed by surface area. They inherit interpolated weights (top four normalized influences). A card's atlas contains multiple fiber silhouettes; the card count is not a count of independently modeled or simulated hairs. Check density at the intended display size and close up, since narrow masked fibers can disappear at small sizes.

The optional native groom follows the same skeleton, but uses Blender's hair system and shader. Its reported fiber count is a configured upper bound, not a count recovered from a rendered image. Native rendering hides the cards for fully native-covered material groups; mixed coverage is rejected explicitly. Native render and GLB appearance therefore differ and must be reviewed separately. The saved `.blend` keeps editable native emitters and hidden browser-card geometry. Rebuild the GLB from the scene JSON to restore the export representation; do not expect a naive re-export of the native-render scene to recreate it. No native fur simulation or Muse-equivalence claim is implied.

Directional grooming projects the requested direction onto each root's tangent plane and progressively curves the strand, preserving its root and arc length. The local helper initializes editable particle keys, saves them into the `.blend`, and bounds each emitter to10,000 parents/150,000 keys/20s. The original six-field radial-groom behavior is unchanged. `scripts/author_combed_groom.py` can derive a new combed scene from any validated furred recipe; it never overwrites the source.

Surface patterns are deterministic local image synthesis, not provider-generated artwork. The backend preserves editable PNG BaseColor and tangent-normal textures under `textures/`, packs them into the Blender source, and embeds them into GLB with UV0/tangents. No external URL, CDN or texture fetch is needed. Surface texture alone is not declared geometric fur. Version2 uses AgX and three studio area lights for its rendered previews; browser lighting is independent.

The build report separates per-part triangle strands, cards and native-groom settings, and records blended weights, exported counts, texture parameters and hashes. Preserve every backend module copied into `sources/`; the exact authoring recipe belongs in the release source snapshot too. Native/basecolor textures, atlases and normal maps remain editable PNGs.

Before exporting v2 geometry, the backend removes microscopic zero-area bevel remnants and reprojects only collapsed/cancelling UV triangles into their geometric planes. Valid seams remain unchanged. The report records every repair and its thresholds. Blender computes the real tangent frames; the exporter does not invent arbitrary tangent vectors. Both `build3d` and `verify` independently require finite, normalized, mutually perpendicular normal/tangent frames and matching UV0 counts, preventing a model with invalid normal-map directions from being accepted.

## States

Each state has a safe `id`, plain-language `meaning`, boolean loop **intent**, duration 0.5–20 seconds (whole frame at 24fps) and 2–64 ordered keys spanning frame 0 through `duration*24`.

Each key maps bone names to optional `rotation` and `location`. These use **model-rest axes**, in radians/meters. The backend converts them through each bone's rest basis into Blender pose coordinates; do not preconvert the JSON to Blender's local +Y bone axis. Unspecified bones/transforms reset to rest at that key. General exports key all bone transforms; the `weft-v1` profile keys only controls used by that state to avoid unnecessary channels. Review transitions between each pair of keys. A root +Z offset is a jump. Parent motion carries child bones and props.

Keys interpolate linearly. Exported glTF rotations are sampled and played with the glTF transform convention. Position-continuous endpoints do not imply velocity-continuous loops. Keep loop output uncertified until the seam is reviewed.

## Attachments

`attachments` describes named `parent_bone`/`child_bone` pairs that must exist in the actual hierarchy, with `placement` and `authored_assumption`. The GLB contract derives actual exported local transforms and checks them against the binary. A dragon root can be a rigid child of the shoulder while its head/tail/wing descendant bones articulate. There is no separate dragon physics simulation.

## Limits and output specs

Experimental v2 materials can set `fur_atlas: "distance-v1"`; omitted/`fine-v1` preserves the original atlas pixels. Distance mode uses ten wider tapered locks per tile, alpha cutoff0.18 and one-sided cards so the renderer does not flip inherited outward normals. Atlas resolution and geometry budgets do not increase. It fixes minification visibility, **not visual acceptance**: overlapping coat layers can still show through each other. The test recipes/catalogs are not the default release. Exact backend snapshots distinguish the first two-sided experiment from the later one-sided refinement.

Version1 scene JSON:1MiB/64parts. Version2:8MiB/128parts. Both:32bones,16materials,12states and coordinates within±20m. No executable expressions, imported scripts, external textures or URLs in scene inputs. Explicit v2 topology is bounded data. Unsupported shapes/actions require additional authored capability, not silent approximation.

Both source and exported geometry are capped at100,000 vertices; GLB64MiB, JSON2MiB and decoded accessor values2million. These are general-profile limits, not Weft limits. A request can pass the control-mesh limit but fail after bevels/fur/UV splits; keep that failed attempt and reduce detail deliberately.

Build previews: 128–1024 pixels, 1–64 samples. `--portrait-only` skips pose contact renders during look development, not the exported clips or independent CPU skin/motion checks. Default job bound is300s,2 GiB sampled RSS,512 MiB output; overrides allow at most1800s. Renderer uses four CPU threads. Memory sampling is every250ms and is not an exact instantaneous allocation cap.

Full media: `muse` uses portrait1600/video720/24fps/8s; `preview` uses portrait640/video320/24fps/4s. Both default to eight video samples, matching the measured full-pack setting; portraits use at least 32 samples without the memory-heavy denoiser. Overrides: `--size` 128–1024 even, `--portrait` 128–2048, `--seconds` 1–20, `--samples` 1–64. Full renders retime actual skeletal clips to the chosen duration. The extra endpoint image is for seam measurement and is not encoded. H.264 is opaque; PNG fallback declarations use measured alpha, not file extension.

Optional `export2d --denoise-portrait` filters the finished portrait in a separate bounded CPU process using the OIDN library bundled with the pinned local Blender runtime. It preserves `portrait-raw.png`, exact alpha, source/library hashes and a separate processing receipt. It is display-referred, color-only denoising, not new geometry or AI-generated replacement imagery; inspect both images because fine fibers can soften. The post-stage is capped at90s/2 GiB/64 MiB and runs after Blender exits, avoiding their combined memory peak. Video and static fallback frames are unaffected.

Preserve scene JSON, exact prompt, backend Python source, source `.blend`, GLB, rig contract, native reload report, images/video, hashes and any rejected attempt. Rebuilding a scene promises stable content/rig semantics; Blender containers and PNG metadata may differ between runs. Do not claim bitwise reproducibility of all renderer outputs without measuring it.
