# Controllable 3D assets and verification

## Reference conversion result

The supplied panda WebP and videos have **no 3D mesh, materials, skeleton or skin weights**. Exact recovered reconstruction is unsupported. A newly authored 3D interpretation is possible, but must identify its sources and assumptions. A 2D preview is not a 3D deliverable. Hidden surfaces, anatomy, topology and joint placement are authored choices, not recovered reference facts.

`unsupported_3d(subject, reason)` returns a versioned JSON-compatible result with `status: "unsupported"` and explicit false geometry, rig, animation and Weft-runtime claims. No simulator or Weft checkout is changed.

## Supplied GLB contract

`avatar_assets.three_d.validate_glb(path, contract)` accepts a **local self-contained GLB 2.0** plus this JSON-compatible contract:

```json
{
  "schema_version": 1,
  "units": "meters",
  "axes": {"up": "+Y", "forward": "+Z", "handedness": "right"},
  "rest_pose": "Describe the authored default node pose; distinguish it from the bind pose",
  "bone_to_simulated_joint": {
    "named_bone": "actual_target_joint_id"
  },
  "target": {
    "name": "Name/version of the provided target joint contract",
    "joint_names": ["actual_target_joint_id"]
  },
  "authored_assumptions": ["List inferred anatomy, unseen surfaces and rig choices"],
  "attachments": []
}
```

Every skin joint must have a unique name and one explicit target-joint mapping. Builder-generated contracts target local authored bone controls, **not simulated Weft joints**, despite the legacy `bone_to_simulated_joint` field name. Other target lists must come from their intended consumer, not guessed Weft names. Matching a list does **not** verify axes, joint limits, control semantics, collision geometry or actual Weft loading. The result keeps `weft_runtime_verified: false`.

For the frozen ten-frame Weft profile, use separate `weft-metadata` output as described in [the authored profile](weft-authored-designs.md). Its `mapping.json` maps only real simulated frames; unbound descendants keep their local rest pose. `verify` checks that bundle's hashes only; the frozen Weft importer owns consumer validation. Do not feed its metadata schema into this general local-control validator.

This validator uses [Khronos glTF 2.0 storage conventions](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html). It loads binary positions, weights, joint indices, inverse-bind matrices and scene transforms. It checks affine transforms, hierarchy, material references, weight normalization and mapping. For included skeletal animation, it evaluates linear-blend skinning on bounded vertex samples at keyframe and intermediate times. First and last frames matching is not sufficient: a bend-and-return test confirms interior motion. Animation with no observed vertex movement is reported as unverified, not completed animation. Looping is never inferred.

`loading_verified` means **this local binary loader** loaded the supported data; it does not mean a graphical viewer or Weft ran. Materials and embedded textures are not rendered. Visual quality, identity and appearance still require separate review. CPU skinning is deformation verification, not a physical simulator.

## Dragon attachment

In the front-facing reference, **viewer-left is the panda's anatomical right**. Use an explicitly named anatomical-right shoulder bone after checking the authored model's orientation. Never infer the side from an ambiguous bone label.

For a rigid decorative dragon, create a direct child node of that bone and record an attachment entry:

```json
{
  "node": "dragon_attachment",
  "parent_bone": "authored_right_shoulder_bone",
  "mode": "rigid_child",
  "translation_m": [0.0, 0.0, 0.0],
  "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
  "placement": "Anatomical-right shoulder, viewer-left in the front reference; exact offset and rotation must be authored and reviewed"
}
```

These zero transforms illustrate the schema, **not a measured placement**. Set the real local transform in both the GLB and contract. The validator checks agreement and actual parentage. Declare whether the dragon is decorative; independent dragon articulation requires a real separate skeleton and explicit mapping, not this rigid-child profile. New collisions, joints, mass or simulation behavior are outside this workflow.

## Supported profile and limits

- File: 64 MiB; JSON: 2 MiB; 256 nodes; 16 skins; 128 joints per skin; 64 meshes; 128 total triangle primitives; 100,000 total instantiated vertices; 2,000,000 decoded accessor components. The larger component allowance accommodates valid detailed skins/indices without raising the vertex ceiling.
- One embedded buffer, PBR material references, four skin influences per vertex and inverse-bind matrices. The skeleton root is explicit or derived from the actual common joint ancestor when an exporter omits the optional field. Embedded PNG/JPEG image references are structurally checked, not decoded.
- Up to 16 animations, 128 channels each, 4,096 keys per channel, 120 seconds maximum, LINEAR/STEP skeletal translation/rotation/scale only. At most 128 probe vertices and 35 time samples per animation.
- No external/data URIs, compressed extensions, sparse accessors, morph targets, CUBICSPLINE interpolation or extra weight sets. Unsupported features raise `ValueError`; they are not silently discarded. Separate `.gltf` with sidecars is unsupported by this version; export a self-contained GLB instead.
- The binary validator itself uses no Blender, generation service, account, payment, upload, mesh repair or automatic conversion. The independent native check below deliberately uses Blender.

## Verification

```bash
python3 -m unittest discover -s tests -p test_three_d.py -v
```

The tests author an **analytic three-vertex triangle**, two bones and a moving bend-and-return channel in temporary directories. It is only a validation fixture, never one of the required generated character/object examples. Tests also cover unrigged meshes, static keyframes, mismapped joints, wrong attachment parents, malformed weights, non-finite coordinates, cycles, external URIs, oversized inputs and malformed binary ranges.

## Independent native reload

After exporting a real model, verify the GLB in a separate Blender process, not the source `.blend` session:

```bash
mkdir outputs/example-native-check
.tools/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --disable-autoexec --python-exit-code 1 --threads 4 \
  --python scripts/blender_verify.py -- \
  outputs/example/model.glb outputs/example-native-check
```

Use the workflow's bounded process runner for production invocation; the Blender script itself does not enforce an OS memory limit or interrupt a hung render. Output names must not already exist. The verifier never saves over the GLB or `.blend`.

The native importer reconstructs meshes, materials, armatures, skin weights and animation actions. The verifier preserves the importer's actual [action/slot assignments](https://docs.blender.org/api/4.5/bpy.types.AnimData.html), disables other animation layers, and evaluates each action at start, quarter, midpoint, three-quarter and endpoint plus four off-quarter samples. The extra samples avoid aliasing repetitive motion whose quarter points are all rest poses. Up to 128 vertex probes cover **every mesh/material subset**; inputs with more than 128 subsets fail instead of silently omitting some. This is bounded sampling, not proof of every vertex at every frame. Each action must move both pose bones and sampled geometry; a static/unrigged model fails. Importer-created custom bone-display meshes are excluded by their actual rig references, not their names.

`independent_reload.json` records actual counts, skeleton hierarchy, normalized imported weights, probe indices, action timings and measured displacement. Cycles CPU renders produce `independent-front.png`, `independent-back.png` and `independent-side.png` at 256×256, eight samples and four threads. Front/back/side labels use Blender Z-up/-Y-front after glTF import. The images show separate 3D viewpoints, but still require visual review for likeness and authored hidden-surface quality. Hashes confirm the source is unchanged. Weft compatibility and physical behavior remain unverified.

### Recorded native-check evidence

`outputs/fox-v2-reload-v3/independent_reload.json` passed on the real fox GLB: 6,923 imported vertices, 10,958 polygons, 20 bones, six materials and four moving actions. The bounded process took 0.765 seconds with sampled peak resident memory of 240,435,200 bytes. Front/back/side images show separate authored surfaces and depth. **This validates reload software, not final pose acceptance:** fox model v2 was separately rejected for intended-action axis errors and superseded by a corrected model.

Earlier preserved verification attempts document two corrected checks: v1 mistook Blender's generated bone-display Icosphere for source geometry; v2 sampled periodic action quarters that all coincided with rest. The current verifier excludes actual bone-display references and adds off-quarter samples. Blender's `--python-exit-code 1` is necessary because its default process status can be zero after a Python exception; always also require the JSON `status` to be `passed`.
