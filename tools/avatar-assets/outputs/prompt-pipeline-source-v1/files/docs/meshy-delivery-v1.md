# Generated-avatar trial: deliverables and limits

The designated Meshy key was used for real generation. This trial produced two reviewable, editable, rigged models and a reusable workflow. **It did not achieve the requested Muse-quality finish.** The panda still has coat/dragon surface artifacts, a flatter muzzle and a different expression. Crafting and working lack recognizable task interaction. No new HD Weft mapping is accepted.

## View and inspect

From the Weft root:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v2.json
```

Open `http://127.0.0.1:8765/`; no Blender is needed. Compare the original reference, generated 3D, rendered clips and portraits. The default earlier collection is unchanged. Nothing is uploaded, and viewing makes no generation call.

Paths below are relative to `tools/avatar-assets/`. Do not overwrite them.

| Asset | Model, editable BLEND, design, proofs | Full rendered media |
| --- | --- | --- |
| Reference-conditioned panda/dragon | `outputs/muse-panda-meshy-refined-v6` | `outputs/muse-panda-meshy-media-v2` |
| Independent text-only robot | `outputs/workshop-robot-meshy-refined-v2` | `outputs/workshop-robot-meshy-media-v1` |

Each media pack has a1600px opaque portrait, four silent720px H.264 clips at24fps/four seconds, and exact first-frame PNG fallbacks. All96 decoded thumbnails differ in each clip. No loop is certified. Matching `*-portable-v1` packs omit only native frames and verbose render logs; their display asset bytes are identical. Originals and failed attempts remain local.

GLB identities:

- Panda: `a874d80ca58c0524d0b3a5a1a52d66700a4dc6c540c298936de24d108f001fd8`.
- Robot: `47576c7f64cf16fd2f6b704307fe76a3cd9aa7541ab50a530885449b2beea8bb`.

## Rig and representation

These are actual meshes with explicit normals, UV0, skin weights, inverse bind matrices, skeletons and animation curves. Both retain4K basecolor/normal textures and2K metallic-roughness maps. Panda has92,986 exported vertices/71,539 triangles/25 bones; robot45,386/51,320/24. Panda adds2,400 skinned cutout cards in GLB and299,000 configured native fibers in the editable BLEND. These are different representations, not a count of visible hairs or a quality score.

glTF is right-handed, +Y up, declared +Z forward, with world distances in meters. Honor the exported `Armature` scale of0.01 and the inverse bind matrices: local bone translations are inherited centimeter values, not standalone meter coordinates. The rest pose is the actual default node transforms, not an assured A/T-pose or recovered original anatomy. `exported-nodes.json` in each model folder lists every node ID, parent, local transform and skin. Do not replace this with guessed offsets or the earlier authored neutral proposal.

| Critical node | Panda ID | Robot ID |
| --- | ---: | ---: |
| Armature |25|24|
| Hips |24|23|
| Head |2|2|
| Spine / Spine01 / Spine02 |13 /14 /15|12 /13 /14|
| LeftArm / LeftForeArm |6 /5|6 /5|
| RightArm / RightForeArm |10 /9|10 /9|
| LeftUpLeg / LeftLeg |19 /18|18 /17|
| RightUpLeg / RightLeg |23 /22|22 /21|
| RightShoulder |12|11|
| CompanionAnchor |11|—|

The dragon occupies viewer-left/anatomical-right. Its4,682 selected base vertices are rigidly weighted to `CompanionAnchor`, parented to `RightShoulder`; the anchor originates at the upper-arm head. Color/spatial selection plus25mm expansion is an authored assumption, not semantic segmentation ground truth. It follows the shoulder in sampled restrained motion, has no independent dragon controls, and selection boundaries still need review. The immutable generic `rig-contract.json` has an empty attachment list; the actual attachment is documented in `exported-nodes.json`, `finish-report.json` and the contract audit. Its identity bone-control mapping is **not** a Weft joint mapping.

## Actions and rejected outputs

The four current clips are **locally authored** gestures. Idle is plausible; the small celebration is distinguishable. Crafting/working are weak pantomimes with no workpiece or separately articulated fingers. Passing movement checks is not semantic acceptance.

Panda's paid generated/retargeted motion clips are preserved but visually rejected. Incorrect provider skin weights assigned belly geometry to a hand. Spatial weight constraints and restrained replacement curves reduce the distortion; they do not recover anatomy. Robot's original rig contained a constant clip, which was never accepted as animation. Its current four actions are also local authoring.

Unchanged-fidelity HD Weft conversion is unsupported: these models exceed the frozen20k-vertex/40k-triangle/diffuse-only profile. Do not silently strip fur/materials or apply old node mappings. The old matte Weft bundles remain separate. No importer, physics, actions, rewards or recording implementation was modified here.

## Verification, costs and continuation

- 255 Python tests passed in9.427s, including hostile/oversized inputs, credit accounting, resume guards and analytic rig fixtures. Fixtures are not generated example assets.
- Both actual exported GLBs passed CPU skin/motion/attribute checks and independent native reloads. Full browser check passed13 grouped checks/all8 clips with zero external requests. Initial ready5.021s, warm model selection171–506ms on this host. The two-second software-WebGL sample is not sustained rendered FPS.
- Independent final audit checked full/portable hashes, parent binding, fresh video decodes, static fallbacks,366 byte-identical reused PNGs, canonical manifest serialization and the unchanged original ZIP in4.632s, process peak269,156,352B. This verifies existing bytes, not deterministic neural regeneration.
- 13 paid submissions settled at132 credits against the independent140-credit ceiling. Dollar purchase cost is unknown. Rejected paid motions are included; no refunds reopen the allowance. No other provider budget, account, subscription or deployment was used.
- Panda media hit its1200s limit; one67.726s continuation completed only the23 missing frames. All owned provider/native/browser jobs have ended. No persistent server is running. Current work is local, not committed/pushed or remotely backed up.

[Workflow commands](meshy-workflow.md) describe text/image configs, local finishing, bounds and resumption. [TODO](../TODO.md) retains task IDs, rejected attempts, ownership and next safe action. [Evidence index](../evidence/meshy-trial-v1.json) binds the selected artifacts and source snapshot. Exact prompts, settings, tool/model IDs and unknown rights are preserved. Original reference provenance is not independent license verification; unseen surfaces and rig structure are generated/authored assumptions.
