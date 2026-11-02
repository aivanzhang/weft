# Reusable portable edge fringe

**Experimental; finishing execution is disabled after real-avatar failures.**
The generic geometry/color libraries and a small native fixture work, but actual
panda/fox exports fail stricter normal/weight preservation, and the panda exceeds
the existing memory limit. `finish_generated.finish` returns an explicit
unsupported error before creating an output or starting Blender. Existing
combed-card finishing and approved assets are unchanged. The new
[export-fidelity/staged-rendering experiment](export-fidelity-and-staging.md)
resolves the measured panda normal/weight and memory failures in a restricted
experimental path, but its actual GLB appearance is still rejected. The public
finisher is not enabled by those component results.

## Input

The experimental schema uses `portable_fringe` instead of `portable_fur`.
It can be validated and inspected, but must not be proposed by the AI author or
executed as a supported finish until the failures below are resolved.
The complete finisher still requires native `groom`, `groom_flow`, a shared
base-color image and the existing source/rig/action contract. The lower-level
adapter also accepts multiple selected source images; that does not establish
heterogeneous native-groom support.

```json
{
  "portable_fringe": {
    "profile": "edge-fringe-v1",
    "settings": {
      "seed": 7123,
      "min_fins": 1000,
      "max_fins": 2400,
      "length_m": 0.008,
      "min_spacing_m": 0.003,
      "min_edge_m": 0.0085,
      "max_edge_m": 0.020,
      "comb": 0.15
    },
    "design_views": [[0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]],
    "excluded_materials": []
  }
}
```

This snippet is not a complete asset request or a suitable recipe for every
subject. Choose counts, exclusions and directions from actual geometry. Views
use the applied Blender emitter frame and affect static allocation, not camera
billboarding. Excluded materials must exist; use observed closure/unsafe names.
Those exclusions apply only to fringe. Mixed surfaces still need measured native
groom exclusions or exact part masks to protect eyes, accessories and non-fur.

## Helper contracts and export limitation

In the helper arrays, safe indexed mesh edges anchor each fin. Endpoints inherit
original bone influences; tips reuse those endpoint weights. UV, material, part, normal,
coincident-edge, boundary and nonmanifold exclusions never weld the source or
invent missing topology. Raw repeated texture coordinates remain raw until
sampling: wrapping endpoints individually would change interpolation.
These are not yet final-model guarantees: the actual Blender export changes some
normals and small skin influences, even though source-native attributes remain
unchanged during the adapter call.

Opposed single-sided faces share authored source-outward shading normals. The
Blender adapter emits separate triangle corners because shared-index opposed
polygons produced invalid topology in the actual fixture.
Each fin gets a source-colored RGBA texture cell; color-coherent packing limits
cross-cell contamination. No old vertex tint is multiplied into the new texture.
Native strands remain a separate, editable representation. Fringe uses regional
direction but explicit uniform length/lean, not full native-groom parity.

The exporter can produce zero tangents on these fins because their source-outward
shading normals lie nearly perpendicular to the fin plane. Tangents are optional
for these base-color-only materials; they are needed by the normal-map path.
[glTF mesh specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes).
The finishing path removes only the unused `TANGENT` bindings from the exact
reported fringe materials. It rejects normal maps, material/primitive extensions,
morphs and ambiguous identities. It preserves every binary byte, other attribute,
node, skin and action; no replacement tangent is invented. Existing body/UV1
tangent processing and its limits remain unchanged.

## Limits and failures

- At most 4,800 fins, 100,000 vertices/triangles and 64 bones. Before native work,
  reserve `source_vertices + 12 * max_fins + 4096` and
  `source_triangles + 4 * max_fins`. Requests fail rather than silently shrink.
- Supported root spans are 8.5–20 mm; the silhouette domain is 20.8 mm. This is a
  supported geometry range, not automatic scale recovery or general long hair.
- Source textures are bounded encoded RGB(A)8, direct UVs, linear filtering and
  REPEAT addressing, at most 4096 per dimension and 16,777,216 pixels in aggregate.
  Native float sampling buffers are released before strand grooming.
- Packed cells are 64×32 pixels. Fixed nominal-tip color/alpha gates retain all
  failures; they do not certify every texel, mip level or lighting condition.
- Geometry/color builders have 10/20-second internal bounds; the native adapter
  has a 60-second bound inside the existing 600-second/2 GiB finisher guard.
  Measured process RSS does not cover all driver or descendant allocations.
- Before postprocessing, reserve two 64 MiB GLB derivatives plus 1 MiB for
  receipts inside the existing 512 MiB pack limit. Keep the raw export; never
  overwrite it or silently raise the cap.

## Outputs and verification

Experimental attempts retain `fringe-export-budget.json`, `fringe-report.json`,
`fringe-correspondence.npz`, the packed PNG, source snapshots and the existing
GLB, available native images and validation manifests. The failed panda attempts
did not finish their poses, editable scene or final validation; partial files
are not a completed pack. The fox retained its Blend/GLB but failed fidelity.
The correspondence includes the native-corner-to-helper-vertex map; repeated
corners are not additional hairs. Failed attempts remain immutable.
`model-native-export.glb` preserves Blender's raw output;
`model-fringe-export.glb` omits only admitted unused bindings, with its exact
allowlist/hash receipt in `fringe-export.json`. `model.glb` then carries the
existing body/undercoat tangent processing and final structural validation.

The 15 geometry, 16 atlas and eight strict-omission tests pass. A small actual
two-material/two-bone fixture also passes independent color, skin and five-pose
verification after unused-tangent omission. Those are component proofs, not
avatar acceptance. Earlier topology/checker failures remain preserved.

| Actual subject | Result |
| --- | --- |
| Panda, 3,600 fins | Both full finishes stopped at the 2 GiB guard. All 43,200 corners match position/UV uniquely, but 315 fail the unchanged normal limit and 414 fail the weight limit. Maximum normal drift: 0.796°; weight-component error: 0.000129. |
| Authored fox, 4,800 fins | Export and 9,600 nominal-tip color checks pass. The original fidelity gate still fails: 213 fringe corners exceed the normal limit (maximum 0.791°), alongside smaller body-normal round-trip differences. Full weighted-pose audit did not run after that failure. |

The packed-file atlas lifetime experiment kept texture pixels but did not solve
memory use: sampled peaks were 2,292,908,032 and 2,338,013,184 bytes. Freeing the
hidden image buffers produced no measured instantaneous RSS reduction. This is
not a successful optimization, and neither candidate is promoted.

The final unsupported-execution guard and surrounding contracts pass 59 tests.
Before enabling the profile, fix exported normal/weight fidelity without relaxing
the independent limits, complete native poses inside existing resource bounds,
then verify both real subjects and their actual exported appearance/actions.
Exact commands, hashes and failures are in the [handoff](../TODO.md).

The AI author is instructed not to choose this experimental alternative.
Existing combed cards remain the default. No provider retry, credit expansion,
dense-fur/Muse parity or fresh Studio acceptance is implied.
