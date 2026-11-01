# Weft asset delivery v1

**Importer-accepted; actual Weft pose alignment remains provisional.** No Weft code was edited. The user authorized defining the visual neutral convention, not changing simulation physics.

## Immutable inputs

| Asset | Model | Mapping | Model SHA256 |
| --- | --- | --- | --- |
| Muse panda + shoulder dragon | `/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/weft-muse-panda-v1/model.glb` | `/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/weft-muse-panda-v1/mapping.json` | `b32e5cc08510035aa7c778dff33635ac8850af42536ac13daac9bfb92954f012` |
| Workshop robot | `/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/weft-humanoid-robot-v1/model.glb` | `/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/weft-humanoid-robot-v1/mapping.json` | `48cc6156fff0200b76c24c4ef785113da2389bdbb0924e121ba45e69ac7cf6cd` |

| Exported resource | Panda | Robot | Consumer limit |
| --- | ---: | ---: | ---: |
| Vertices | 11,036 | 8,338 | 20,000 |
| Triangles | 17,784 | 12,516 | 40,000 |
| Skin joints | 22 | 17 | 64 |
| Nodes | 24 | 19 | 128 |
| Material primitives | 6 | 6 | 16 |
| Total animation channels | 176 | 136 | 256 |

Both contain explicit NORMAL, JOINTS_0, WEIGHTS_0, inverse-bind matrices and real skeletal clips. Materials explicitly use metallic 0, roughness 1 and OPAQUE; no textures, URLs, morph targets or compression. These counts come from exported GLB accessors, not source primitive counts.

## Critical exported node IDs

| Body frame | Panda node/name | Robot node/name |
| --- | --- | --- |
| `mobile_root` | 23 / `AvatarRig` | 18 / `AvatarRig` |
| `torso` | 14 / `torso` | 9 / `torso` |
| `left_shoulder_body` | 13 / `upper_arm_left` | 8 / `upper_arm_left` |
| `right_shoulder_body` | 4 / `upper_arm_right` | 4 / `upper_arm_right` |
| `left_elbow_body` | 12 / `forearm_left` | 7 / `forearm_left` |
| `right_elbow_body` | 3 / `forearm_right` | 3 / `forearm_right` |
| `left_hip_body` | 20 / `thigh_left` | 15 / `thigh_left` |
| `right_hip_body` | 17 / `thigh_right` | 12 / `thigh_right` |
| `left_knee_body` | 19 / `shin_left` | 14 / `shin_left` |
| `right_knee_body` | 16 / `shin_right` | 11 / `shin_right` |

The mesh-containing `AvatarRig` is the mapped root. Source skeleton root bones 21 (panda) and 16 (robot) are unmapped rest descendants, not substitute containing roots.

Panda dragon root **9** is a direct child of shoulder helper **10**, under torso **14**. Upper arm **4** is a separate branch: raising it does not rotate the dragon off the shoulder. Dragon head **5**, tail **6**, outer wing **7**, inner wing **8** stay authored rest descendants in Weft. The placement is anatomical-right/viewer-left in a front view. Exact local attachment values are in the source build's `rig-contract.json`; every global rest matrix is in the bundle's `node-table.json`.

## Neutral, bind and assumptions

- Source GLB: right-handed, +Y up, +Z front, authored meters. Default node pose equals bind pose within recorded numerical tolerance.
- Proposed target: +Z up, -Y front, scale 1 (not calibrated); orientation xyzw `[0.7071067811865476, 0, 0, 0.7071067811865476]`.
- [Neutral positions and exact equation](weft-neutral-proposal.md): `G = O*S`, `C = inverse(F0)*G*B_world*inverse(G)`, consumer `J(q) = F_body(q)*C*G`.
- `mapping.json` has the actual per-node offsets, not guessed zero offsets. Root/torso/limb mappings cover the ten documented body frames. Live physics still has 19 joint angles plus root translation; no head, finger or dragon controls are recovered or added.
- Complete back/side surfaces, proportions, elbow/knee pivots and rigid-part weights are authored assumptions. Panda is a matte interpretation of the selected reference, not recovered fur/topology/anatomy. Rights and original licensing remain unverified.

## Evidence and next consumer action

`outputs/weft-import-proof-v1/results.json` records both successful calls to the frozen importer, its source hashes, generated seals and immutable source paths. Both native GLB reloads pass in `outputs/muse-panda-weft-reload-v1` and `outputs/humanoid-robot-weft-reload-v1`. The 21-image visual review is `outputs/weft-model-visual-review-v1/review.json`.

The bundle creation manifests retain `frozen_importer_verified: false` because they predate the separate proof; do not rewrite them. Pair them with the later acceptance report. Import acceptance does not verify target alignment, runtime motion, display, selected-appearance replay or native recording.

The Weft owner should use these exact model/mapping paths with its existing importer, inspect the actual neutral and articulated poses, and provide actual frame transforms if offsets need adjustment. Recompute a **new** mapping version using the documented equation; do not change physics or overwrite these bundles. Authored clips are provenance, not replacement simulation actions. This workspace adds no importer, renderer or recording implementation for Weft.
