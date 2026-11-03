# Studio avatar → Weft ZIP

This local converter makes a **new lower-detail rendering asset** from an existing rigged Studio GLB. It does not generate another avatar, call an API, modify the original, or increase Weft's import limits.

Run from the Weft repository root, using the existing authoring environment and Blender installation:

```sh
PYTHONPATH=tools/avatar-assets tools/avatar-assets/.venv/bin/python \
  -m avatar_assets.weft_delivery \
  path/to/original/model.glb path/to/settings.json path/to/new-output \
  --source-sha256 EXACT_ORIGINAL_GLB_SHA256 --triangles 6000
```

The output directory must not exist. Import its **`weft-avatar.zip`** in the Weft sandbox's avatar panel. The ZIP contains only `model.glb` and `mapping.json`; normal Weft import remains authoritative. Conversion alone is not an import pass. Keep `delivery.json`, `native-report.json`, process receipts, and `manifest.json` with the original source for provenance.

`settings.json` requires explicit rights, visual scale, orientation and semantic bone names; the converter does not infer anatomy or rights:

```json
{
  "asset_id": "my-studio-avatar",
  "display_name": "My Studio avatar",
  "rights": {
    "holder": "Your declared rights holder",
    "license": "Your declared license",
    "source": "Original asset provenance",
    "declaration": "Your actual rights declaration"
  },
  "provenance": {"source_manifest_sha256": "YOUR_ORIGINAL_MANIFEST_SHA256"},
  "scale": 4.5,
  "orientation_xyzw": [0.7071067811865476, 0, 0, 0.7071067811865476],
  "body_roles": {
    "mobile_root": "WeftAvatarRoot",
    "torso": "chest",
    "left_shoulder_body": "arm_l_upper",
    "left_elbow_body": "arm_l_lower",
    "right_shoulder_body": "arm_r_upper",
    "right_elbow_body": "arm_r_lower",
    "left_hip_body": "leg_l_upper",
    "left_knee_body": "leg_l_lower",
    "right_hip_body": "leg_r_upper",
    "right_knee_body": "leg_r_lower"
  }
}
```

These example names, scale and Y-up-to-Z-up orientation match the recorded Studio panda, **not every asset**. Use explicit source names and the correct common orientation. Weft's left side is negative X. The v2 importer independently checks hierarchy, common bind pose and lateral orientation. Unsupported rigs fail rather than being guessed.

## Deliberate simplifications

- Connected meshes use rest-pose triangle collapse; many small disconnected fur cards use deterministic whole-card thinning. Surviving/interpolated weights remain; density and silhouette can change.
- Per-corner colors and base-color texture are resampled into bounded triangle atlases. This is not pixel-identical appearance.
- Normal, metallic/roughness, occlusion and emissive maps are omitted. Materials use a diffuse prior.
- Source clips remain in the original GLB but are not copied or executed. Weft's simulated joints drive the delivered avatar; it cannot change physics, contacts, rewards or task success.
- Only importer-generated bone-display helper objects are excluded; source mesh and skin-joint inventories must remain exact.

Native conversion is capped at 120 seconds, sampled direct-child RSS 1.5 GiB and 128 MiB output; packing has a separate 20-second deadline. Reserve at least 140 seconds plus metadata/readback overhead. No recurring resources or API spend. On macOS, Blender may require permission to initialize Metal outside the filesystem sandbox; use the same bounded local command, not an unbounded alternate pipeline. Failed attempts remain in their own output directories.
