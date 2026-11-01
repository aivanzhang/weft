# Authored scene examples

These editable JSON sources describe complete stylized 3D models, not image-to-mesh reconstruction or analytic test fixtures. The scene builder must render and validate the exported models before any visual-quality, rig-motion or seamless-loop claim is accepted.

| Source | Distinct design | Geometry / bones | State meaning |
| --- | --- | --- | --- |
| `examples/scenes/fox-craftsperson.json` | Red fox, cream face/chest, sea-green scarf, sweeping tail, sketchbook and pencil | 33 smooth-primitive parts / 20 bones | Listening idle; short sketching strokes; reading gaze; raised-tool hop |
| `examples/scenes/articulated-lamp.json` | Burgundy desk-lamp robot, paired hinged bars, brass shade, warm-colored bulb and work pad | 30 parts / 6 bones | Curious idle; precise tracing gesture; broad inspection sweep; upright stretch and double nod |
| `examples/scenes/panda-shoulder-dragon.json` | Rounded panda with a small green shoulder dragon and a golden craft sphere | 37 parts / 20 bones | Listening idle; sphere-shaping gesture; inspecting gaze; raised-paw celebration with dragon wing flaps |

## Shared conventions

- Coordinates are meters, Blender Z-up; front faces negative Y. Scene JSON supplies all materials, geometry, bone rest positions, rigid weights, action keyframes, camera and background.
- Keyframe rotations and translations use model rest axes: X right, Y rear, Z up. The builder must convert offsets through each bone's rest basis before writing Blender pose transforms. This avoids exposing Blender's local-bone +Y convention to prompt-authored scene data; a positive Z translation in these sources is an upward hop.
- Smooth spheres form the organic models; cylinders, rounded hinges, torus rims and a cone shade form the lamp. Each part has one named bone with full weight. This is editable articulated geometry, not an organic soft-skin deformation model.
- Each state is four seconds at 24 fps, authored at frame 0 through 96. First and endpoint poses match. A renderer should export frames 0–95, then inspect the final-to-first boundary; matching keyframes alone do not prove a seamless rendered loop.
- State motions are intended gestures. The fox does not create an ink texture, the lamp does not simulate illumination or fabrication, and the panda's sphere does not change shape. Those limits are recorded in the sources rather than represented as completed physical effects.
- Exact authoring prompts and hidden-surface/rig assumptions are preserved inside each JSON. The original panda archive remains independent and unchanged.

## Panda attachment

The panda's anatomical right shoulder is world negative X and viewer-left in a front view. `dragon_root` is a child of `right_shoulder`; the arm also descends from that shoulder, so raising an arm need not rotate the dragon as an upper-arm attachment would. The dragon has separate head, tail and wing controls. The shoulder placement, rear surfaces and rig are authored assumptions, not recovered facts from a 2D picture. The golden craft sphere is an explicitly new action prop.

## Verification still required

Load each source through the real scene builder, inspect front/side/back renders and all four motions, verify that held props remain attached, check clipping and framing, then load the GLB independently and test its bone mapping and motion. Preserve failed renders and revisions. Do not count source-schema checks as completed model, animation or target-runtime verification.
