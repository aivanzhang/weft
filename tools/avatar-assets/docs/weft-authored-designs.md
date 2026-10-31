# Weft-profile authored designs

Contract read: `/Users/ivan/kintsugi/weft/docs/avatars.md`, read-only. This workspace authors assets; Weft owns import, retargeting, display and recording. No Weft code, physics, action space, reward or policy input is changed here.

## Sources

| Scene | Design | Source counts |
| --- | --- | --- |
| `examples/scenes/muse-panda-weft-v1.json` | Muse-inspired panda, green anatomical-right shoulder dragon, golden craft sphere | 44 parts, 22 bones, 6 materials |
| `examples/scenes/humanoid-robot-weft-v1.json` | Original blue/cream humanoid robot, friendly visor, circuit workpiece and assembly probe | 49 parts, 17 bones, 6 materials |

Both declare `output_profile: "weft-v1"`. Every material explicitly has `metallic: 0`, `roughness: 1`, and alpha 1. These are diffuse paint colors, including the muted gold details—not metallic shaders. There are no external textures, image billboards, morph targets or inferred finger controls.

Both exported models passed native Blender reload/motion checks and the frozen Weft importer; evidence is in `outputs/weft-import-proof-v1/results.json`. Source counts are not exported-resource counts. Required ceilings remain 20,000 vertices, 40,000 triangles, 16 primitives, 128 nodes and 64 skin joints, plus the contract's byte/resource limits. Actual Weft pose alignment and display remain unverified.

## Ten-frame binding design

Both scenes use the same explicit names, recorded in `binding_design`. Anatomical left is world positive X and right is world negative X in the authored Blender Z-up, negative-Y-facing pose.

| Exported node | Weft simulated body frame | Deforming geometry |
| --- | --- | --- |
| `AvatarRig` | `mobile_root` | Containing node for both mesh and all skeleton bones |
| `torso` | `torso` | Torso, body panels and their rest descendants |
| `upper_arm_left` | `left_shoulder_body` | Anatomical left upper arm |
| `upper_arm_right` | `right_shoulder_body` | Anatomical right upper arm |
| `forearm_left` | `left_elbow_body` | Left elbow joint and forearm; hand follows |
| `forearm_right` | `right_elbow_body` | Right elbow joint and forearm; hand follows |
| `thigh_left` | `left_hip_body` | Left hip/thigh |
| `thigh_right` | `right_hip_body` | Right hip/thigh |
| `shin_left` | `left_knee_body` | Left knee/shin; foot follows |
| `shin_right` | `right_knee_body` | Right knee/shin; foot follows |

The panda now has separate thigh and shin surfaces and real knee bones; the former unsplit leg is not relabeled as a complete lower-limb rig. Each authored part carries a rigid weight of 1 to its named bone. Rounded elbow/knee surfaces cover the meeting points of the articulated segments. These remain toy-style rigid parts, not continuous organic skin.

Use the actual exported glTF node indices in `mapping.json`; scene-array positions are not node indices. Bind the actual root once, and each of the other nine named bones once. Do not bind the head, hands, feet, held tools or dragon helpers to invented controls. Unmapped descendants keep their authored local rest transform, as the contract specifies.

`binding_design` is a source-authoring aid, not a replacement metadata schema. The separate mapping-design JSON explicitly replaces its `root` bone entry with `AvatarRig`: the bone alone does not contain the mesh. Emitted `mapping.json` contains the frozen contract fields. Numerical offsets use the user-authorized [neutral proposal](weft-neutral-proposal.md), actual exported transforms and inverse-bind matrices. This is provisional alignment, not measured target coordinates or metric calibration.

## Panda identity and attachment

The selected reference is the supplied Muse WebP, SHA-256 `c072bc988b0116ba6ac02641467cde3cddf65ac76a98be5a19c0c54553c9f3b2`. The scene records the requested relative reference path and its role. The original archive and media are unchanged.

White face/belly, dark eye patches/ears/limbs/tail and the small green dragon preserve the visible identity. Dark-green horns and a light-green belly are authored from the reference palette. Fur, scales, unseen surfaces, precise limb lengths and rig structure are not recovered; this is an explicitly matte, stylized interpretation. The golden craft sphere is a declared new action prop.

`right_shoulder` is a helper under `torso`. `upper_arm_right` and `dragon_root` are separate children of that helper. Only `upper_arm_right` receives the simulated shoulder binding. Consequently the dragon stays on the torso's anatomical-right shoulder—world negative X, viewer-left in a front view—without rotating with the upper arm. The dragon's head, wings and tail remain rest descendants, not independently simulated joints.

## Authored preview states

All four states use only the ten mapped body bones. Head, hands, feet and dragon are never independently keyed in these profile scenes.

| State | Panda | Humanoid robot |
| --- | --- | --- |
| `idle` | Gentle body sway, sphere supported at waist | Attentive body rock, tools held low |
| `working` | Raises sphere toward face for inspection | Raises circuit workpiece for inspection |
| `making_something` | Alternating forearm strokes around the sphere | Short probe strokes toward the fixed circuit guide |
| `milestone_level_up` | Raises paws/sphere, small knee preparation and hop | Raises both tools, small knee preparation and hop |

The sphere and circuit traces do not change shape or appear dynamically. These are meaningful authored gestures, not fabrication or contact simulation. Clips are four seconds with matching endpoint poses; rendered seamlessness remains unverified. Weft retains these clips as provenance and instead displays the existing simulated pose—it must not play these actions as a replacement for physics.

## Rights and verification boundary

Each scene includes exact authoring prompts, explicit assumptions and a `rights` declaration. The panda reference's holder and license are unknown; user-supplied media and embedded metadata do not independently establish permission. The robot uses original local procedural shapes/materials without borrowed model, texture, font or reference assets; no separate asset license is invented.

Completed: real skinned builds, independent native reload and moving weighted vertices, front/back/side and state previews, actual node/attachment records, and ten-frame metadata acceptance by the frozen importer. See `outputs/weft-model-visual-review-v1/review.json` and both model/mapping bundles. Remaining for the Weft integration owner: check against actual frozen simulation poses and record target display/replay evidence. Source-schema or importer acceptance alone is not runtime alignment verification.
