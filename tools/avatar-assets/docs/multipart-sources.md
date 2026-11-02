# Separate source parts for controllable avatars

## Status and ownership

This continues private checkpoint `3b2f1d3`. That turn made concrete progress:
mixed-material finishing and source-neutral rig metadata were tested and saved;
two actual fur experiments were correctly rejected. Neither turn completes the
hands-off quality/Studio goal. Existing assets and all failed attempts remain
unchanged. No new provider or pipeline AI-role call is part of this increment.

The next reusable component is `compose-sources`: a local, deterministic merger
for two or three already complete unrigged GLBs. It preserves separate source
ownership rather than cutting a fused companion away from a body. It does not
generate parts, reconstruct hidden surfaces, infer joints or create animation.
Implementation, fourteen focused tests, the real public CLI run, independent
array/material preservation checks and bounded native loading pass. This is not
an accepted multipart text-generation pipeline or a newly animated avatar.

Ownership: `agent_runner` implements `avatar_assets/compose_sources.py` and its
tests; `pipeline_contracts` independently reviews correctness and bounds;
`pipeline_review` prepares independent source/output checks. Root owns CLI wiring,
the real composition run, any bounded native preview, documentation and saving.
All implementation/review/native jobs are now terminal. Root owns the evidence
checkpoint; no paid work is running.

## Input contract

The JSON configuration explicitly provides `schema_version: 1`, a safe `id`, exact
`prompt`, `rights`, `space: "gltf_y_up_right_handed"`, `requested_rig_height_m`, and
two or three `components`. Each component names its own `id`, local `source`,
`source_sha256`, `source_meters_per_unit`, and `placement` containing translation
in metres, a unit quaternion in XYZW order and a positive uniform scale.

For every source point, the declared transformation is:

```text
p_out = translation + source_meters_per_unit × uniform_scale × R(quaternion) × p_source
```

There is no implicit centering, height inference, welding, simplification or
part substitution. Explicit source normals and existing tangent directions rotate;
UVs, triangle indices, image bytes and material properties remain source-owned.
Material names become `component_id__mINDEX`, with exact maps in the receipt.
Missing tangents remain missing: this is raw source preparation, not final
normal-map frame validation or a completed model with a skeleton.

The first profile requires one active identity-frame mesh instance per source.
Omitted identity transforms are valid glTF defaults. Nonidentity source frames,
skins, animations, joint/weight attributes, morphs, compressed features and
external/data URLs are unsupported. No unsupported feature may be silently
dropped. Constant-color and textured standard materials must retain their own
definitions; they do not need a shared image merely to compose geometry.

`requested_rig_height_m` does not change the assembly. The receipt predicts the
**additional** center/floor translation and uniform scale that the existing
native `create_rig` will apply later. Original assembly anchors therefore cannot
be treated as unchanged native world positions. Declared metres and unseen
geometry are authored assumptions, not physical measurements.

## Bounds and outputs

The new worker is limited to 60 seconds, 512 MiB sampled direct-child memory and
128 MiB aggregate output. Aggregate input GLBs and the final GLB each stay within
64 MiB; geometry stays within 100k vertices/100k triangles and two million decoded
scalars. Texture inspection permits 16 Mi-pixels per image and a newly explicit
96 Mi-pixel aggregate header budget, without full pixel decoding. This does not
raise the independent native renderer's existing 2 GiB limit or prove it will fit.
Later rig/export seams and coats still need their own remaining geometry budget.

The output keeps exact original inputs/configuration, a separately bound worker
configuration, trusted source snapshots, per-component reports, worker/resource
receipts and a versioned manifest. A successful raw composition contains one
identity-node mesh with source-owned primitives; it is explicitly unrigged.
Failures retain an unsupported result and evidence, never an accepted model path.
The generic `verify` command checks its hashes and raw geometry without inventing
a missing rig or animation.

## Saved-source development case

The primary inputs are:

- Robot: `outputs/workshop-robot-meshy-v1/stages/texture/downloads/model.glb`,
  SHA256 `5f00602dd727429277ae53c0e8940ba9162ea119e643ba554f081c00cc1b7e8c`.
- Standalone dragon: `outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb`,
  SHA256 `a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb`.

Together: 81,091 vertices, 87,821 triangles, 48,447,108 source bytes and six JPEG
maps totaling 75,497,472 pixels. Both are existing, separately generated whole
characters with no skins/clips, UV0 base-color/normal/metallic-roughness maps and
no tangents. The standalone dragon is **not** the requested panda shoulder dragon.
Reusing this pair is local development evidence, not fresh generation or identity
acceptance. Provider terms apply; commercial rights remain independently unverified.

`outputs/multipart-source-inventory-v2` preserves exact source/provenance receipts.
Its two harness failures are explicit: first, an omitted identity matrix was
incorrectly rejected; after that checker-only correction, an unnecessary
raw-versus-textured equality assertion failed. No interchangeability or wholly
passing inventory run is claimed. Use only the exact textured paths above.
Inventory manifest: `b3f7668b9edd822d2d22d18b378e87c3ff42890b14bbca76e28334975cff4420`.

`outputs/multipart-source-layout-v1/compose.json` authors a separated side-by-side
layout: robot height 1.2 m, dragon height 0.9 m and dragon yaw −20°. Exact uniform
conversions and translations come from source bounds and remain declared design
choices. Preparation PID 79435 exited in 0.262 seconds, RSS unsampled, with no
geometry mutation or provider/native call.

Executed public commands (use a new output folder for any subsequent run):

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
./run compose-sources outputs/multipart-source-layout-v1/compose.json --out outputs/robot-dragon-source-assembly-v1
./run verify outputs/robot-dragon-source-assembly-v1
```

The reusable saved-source recipe is `examples/compositions/robot-dragon-v1.json`;
its relative paths resolve to those same exact local sources. Both source hashes
were checked after creating it. It is a development fixture, not new generation.

The resulting `outputs/robot-dragon-source-assembly-v1/model.glb` is 50,393,860 bytes,
SHA256 `a2bdca0698d3ef777323889f8476ca80219bcde03c4e56a761c270fc97e44127`.
Its manifest is `a182cc11bbadb8a4559c00e7c50e6771a30765950fac9b7d7bb96bac35f1e9a7`.
Worker PID 81487 exited in 0.263 seconds; it finished before an RSS sample.
This is not evidence of zero memory use. The public verifier checked all 21
artifact hashes and actual raw geometry: PID 81772, 0.523 seconds, 176,586,752 bytes
sampled RSS. Evidence is `outputs/multipart-public-verification-v1`.

The frozen independent checker at
`outputs/multipart-assembly-independent-audit-preparation-v1` uses the input
equation and actual arrays, not producer success flags. Its one executed audit,
`outputs/multipart-assembly-independent-audit-v1`, passes all source vertex/
triangle/UV0/image and material/sampler checks. Maximum position error is
5.95925e-8 m; robot normals are exact, dragon normal component error is 2.98019e-8.
The unapplied native offset/scale prediction agrees exactly. PID 81725 exited in
0.262 seconds, RSS sampling missed the short run; process high-water RSS was
248,479,744 bytes. Audit manifest:
`3ea20e55cfc3d9c60ad55602cd250582237ce4db772adc969ca153e6ab48ffeb`.

The frozen source is `avatar_assets/compose_sources.py`, SHA256
`e44c1b4cff23d23644e241107f0b57a051b228ad6cce77aa3a4d7fa27e7f61f6`.
All 14 focused tests passed in 5.169 seconds. Fixtures cover transformed
multi-material geometry, constant materials, exact retained image/UV/index bytes,
determinism, invalid/oversized/rigged input, resource failure and changed evidence.
They are analytic triangles, not generated character-quality examples.
Test and release receipts: `outputs/compose-sources-tests-v1` and
`outputs/compose-sources-release-v1`.

## Native load and limited visual review

One explicit local call ran the existing EEVEE preview helper, with no engine
fallback or texture reduction:

```sh
.venv/bin/python -B outputs/multipart-native-preview-preparation-v1/execute.py --execute
```

This is an already executed immutable preparation, not a retry command. Native
PID 82014 exited in 3.103 seconds with 1,184,366,592 bytes sampled peak RSS, below
the unchanged 120-second/2-GiB/128-MiB limits. All four 720-square RGB views were
inspected: the blue/copper robot and green/cream dragon retain distinct materials,
silhouettes and visible details. Their side projection overlaps, as expected for
a side-by-side layout; this is not a perched companion or an attachment test.
No new fur, action, collision, reference-likeness or seamless-loop claim is made.

- Preview folder: `outputs/robot-dragon-native-preview-v1`.
- [Front](../outputs/robot-dragon-native-preview-v1/front.png) and
  [three-quarter](../outputs/robot-dragon-native-preview-v1/three-quarter.png).
- Editable imported scene: `source.blend`, SHA256
  `98ea6536e51c92cea7092cba5d17c41cc0be8b90bcee6ce429260409eb6a5db2`.
- Preview manifest:
  `64682af7552984ff39fd4be74c29778109adcc4fe4f8ee1f797655ba51dd6c34`.

Native inspection confirms one mesh, two material slots, 81,091 vertices,
87,821 faces and no skeleton or animation. The native scene is not an export to
replace the audited GLB; importer/exporter normal-frame fidelity is separate.
Hash-bound visual review and unchanged original/panda/dragon receipts are in
`outputs/multipart-integration-review-v1` (manifest
`f5fd593d54b2cfd2a7463e6551341607570b2c166b0e9e664a8c59855a48eab9`).

## Remaining whole-pipeline work

The current planner/provider path still creates one source. The Studio allowance
check still covers one generation job. Automatic multipart creation needs bound
per-component prompts/references/identities, aggregate geometry and cost estimates,
per-component ledger reservations/task checkpoints and user-visible progress.
This local merger does not bypass any of those gates or grant spending.

Mixed-image fur/color-driven finishing remains unsupported. Independent motion
requires a unified authored rig, part-specific weights, preserved custom-action
intent and final-media review. Reusing this assembly in a later AI run still uses
the separate AI-call allowance; it is not a free way around the exhausted study.
The pending fresh-example allowance is unchanged. Consistent Muse-quality fur,
fresh examples and an accepted real Studio delivery remain required.

## Next safe action

Do not rerun the unchanged assembly, native preview, failed cheek repair or coat
offset experiments. This source-preparation increment is verified. Continue with
the missing planner/finisher integration: bind component identity and action
ownership explicitly, preserve the recorded assembly-to-rig transform, and reject
multi-image fur until a material-aware color sampler is independently checked.
Those local changes do not require a provider call. A future actual multipart
generation needs its own aggregate estimate and ledger grant; the pending
105-credit/AI-call proposal remains three single-subject cases, not this pair.
No authorized balance or prior approval is silently expanded.

## Saved evidence and publication receipt

`evidence/multipart-sources-v1` preserves 147 origin records as 108 byte-identical
unique snapshots (1,943,151 bytes), including failures and the four inspected views.
Large models/scenes remain locally hash-bound, not copied into this compact bundle.
Bundle manifest: `6a2e2da38303232635186cd26ffe1361638b053dee063a79827d366bcc261ded`.
Packaging PID 82500 exited in 0.263 seconds; RSS was unsampled. Preparation and
job evidence are in `outputs/multipart-source-checkpoint-v1`.

The avatar-only commit/private-push result is recorded separately in
`.local/avatar-quality-publication-v14/receipt.json` after remote verification.
Its existence and confirmed status, not this pre-publication note, establish the
remote outcome. No deployment, public publication or consumer change is included.
