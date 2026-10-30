# Local avatars

Avatars change the appearance of the existing shared-world human. They do not change its joints, collisions, actions, rewards, task gates or policy inputs. The original body remains available. Illustrative muscle bands are not measured muscle activation.

**Generic integration is checkpointed; full acceptance is unfinished.** The user redirected this session to the core tracks, so further avatar work is deferred here. Panda, robot and Muse assets are not included yet. The separate asset-authoring session owns those assets; Weft owns import, retargeting, display and recording. An analytic two-bone mesh is a test fixture, not a finished character. [Verification and remaining limits](handoffs/combined.md#frozen-generic-avatar-integration--not-full-acceptance).

## Import and launch

Import a factory package containing `avatar.json`:

```sh
.venv-lab/bin/python -m weft avatar-import --source /path/to/factory-package --out artifacts/my-avatar
.venv-lab/bin/python -m weft lab --resume artifacts/feasibility-fullspan-lab-01 --avatar artifacts/my-avatar
```

Or import a rigged model with explicit metadata; the importer computes resource hashes:

```sh
.venv-lab/bin/python -m weft avatar-import --model /path/to/character.glb --metadata /path/to/mapping.json --out artifacts/my-avatar
```

Repeat `--avatar` up to four times for a local selector. Open the complete printed URL yourself. No browser, server or download is started by an import. Import outputs must be new directories; old assets and completed `run/` evidence are never overwritten. The pinned `scripts/play_sandbox.py` launcher still uses the previously verified runtime and does not yet include these development features.

## Authoring contract

`mapping.json` contains exactly these fields:

```json
{
  "asset_id": "my-character",
  "display_name": "My character",
  "rights": {
    "holder": "Named creator or unknown",
    "license": "Applicable license or unknown",
    "source": "Original source or authored asset",
    "declaration": "Describe the user's right to use this asset; do not invent permission"
  },
  "provenance": {"tools": [], "prompt_history": [], "assumptions": []},
  "retargeting": {
    "profile": "weft-avatar-retarget-v1",
    "root_node": 0,
    "scale": 1,
    "orientation_xyzw": [0, 0, 0, 1],
    "bindings": [
      {"node": 0, "body": "mobile_root", "translation": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}
    ]
  }
}
```

This illustrates the schema, **not a correct mapping for every model**. `node` is the actual glTF node index. Bind each relevant bone to the appropriate frame; a node can have only one binding. The root must be mapped. Unmapped descendants keep their authored local rest transform; no independent head, hand or finger motion is inferred.

| Simulated frame | Role |
| --- | --- |
| `mobile_root` | Translated body root |
| `torso` | Torso rotation |
| `left_shoulder_body`, `right_shoulder_body` | Upper-arm rotation origins |
| `left_elbow_body`, `right_elbow_body` | Forearm rotation origins |
| `left_hip_body`, `right_hip_body` | Thigh rotation origins |
| `left_knee_body`, `right_knee_body` | Shin rotation origins |

The visual joint transform is `F_body × T(binding.translation) × R(binding.rotation_xyzw) × R(orientation_xyzw) × S(scale)`. Skin matrices are `J_joint × inverse_bind_matrix`. Matrices use column-major order and glTF's xyzw quaternions. Translation offsets are in the target frame's simulation units; scale maps authored model units to simulation units. This is an explicit visual alignment, not metric calibration or recovered anatomy. See the [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) for skinning and inverse-bind semantics.

A factory package adds `schema_version: 1`, `profile: "weft-avatar-asset-v1"`, `entrypoint` and `files` (relative path → SHA-256) to the same metadata in `avatar.json`. Retain original media, prompts, tool versions, rights declarations and assumptions. A Muse-like dragon attachment is newly authored geometry and rigging; the supplied reference ZIP does not contain a 3D model.

## Supported import profile

- Local GLB/glTF only, real skinned triangle meshes, explicit normals, one skin, `JOINTS_0`/`WEIGHTS_0`, optional UV0 base-color PNG/JPEG textures.
- Unlit materials or explicit metallic=0/roughness=1 diffuse-only materials; OPAQUE or MASK, with double-sided support. No silent conversion of unsupported materials.
- At most 32 input resources /32 MiB, 128 nodes, 64 skin joints, 20,000 vertices, 40,000 triangles and 16 primitives. At most eight textures, each ≤2048×2048, aggregate ≤8 MiPixels. Normalized render JSON ≤8 MiB, sealed package ≤64 MiB, processing ≤10 seconds. Lab catalog ≤4 avatars /128 MiB.
- Remote/data URLs, unsafe paths, symlinks, executable content, compressed/sparse accessors, morph targets, unsupported material channels/extensions and static image billboards are rejected. Authored animation clips are retained as provenance, not played instead of the simulated pose.

## Recording contract

The original six-file physical recording and action-replay checks remain unchanged. Avatar-enabled recordings add `appearance-events.jsonl`, `appearance.json`, original assets, normalized render data and frozen viewer code. Each appearance change names its preceding physical event index and exact event hash. Changing a look while paused inserts no action or simulated time; the final look at that state is used for its exported frame.

`avatar-replay.html` and `avatar-video.mp4` retain the selected appearance and settings. `appearance-manifest.json` is written only after export succeeds and binds the appearance files to the original physical manifest. The UI labels the original capsule/skeleton exports **Physics-body replay/video**; these are not substitutes for an unfinished avatar export. Missing/incompatible assets and unavailable GPU rendering remain explicit failures or labelled original-body fallbacks, never silent avatar success.

These exports reconstruct the **selected** appearance from saved states; they are not screen captures. If the live browser cannot display an avatar, it shows the original body with a notice. A capable replay/export device may still render the selected avatar. Per-frame client render failures are not recorded; this limit is also preserved in `appearance.json`.

The browser renderer uses a bounded worker and OffscreenCanvas/WebGL2. Native MP4 export uses the existing pinned PyOpenGL/MuJoCo offscreen context plus a bounded Node pose-math process; it performs no physics steps. No browser automation, new service or paid API is required. Exact browser GPU compatibility and sustained performance must be reported separately from mocked browser tests and native offscreen checks.
