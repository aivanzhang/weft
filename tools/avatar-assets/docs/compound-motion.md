# Compound avatar motion

The approved furry panda remains saved at checkpoint `4ade45a`. The follow-up
request is **the small shoulder dragon takes off while the panda dances**.
The standalone dragon is not a substitute. Current geometry work and failures
are recorded in [the handoff](../TODO.md); no combined animation is accepted yet.

## Reusable controls

`finish_generated` and the AI author support these data-only controls:

- An action can set `interpolation: "LINEAR"` to avoid curve overshoot.
- Each key can contain `translations_m: {"Bone": [x, y, z]}`. Components are
  bounded to ±2 meters, with at most64 named controls and32 keys per action.
  Translations require LINEAR interpolation. They use the bone's **rest-world
  axes** and follow its animated parent. They are not absolute world paths.
  An omitted translation at a key returns that control to zero.
- `camera_fit: "motion"` fits one fixed native camera to every authored key
  and midpoint, including a fur margin. It is a sampled envelope, not a proof
  against clipping between samples. Final playback still needs inspection.
- `part_rigs` can reweight up to four **already separated** material-defined
  parts. Each record has `name`, `materials`, `root_bone`, `layers`,
  `min_vertices` and `max_vertices`. The root and its descendants are the only
  allowed influences. Layers use the original full-mesh normalized bounds.
  A vertex shared with outside faces is rejected before any weights change.
  This requires `authored_rig` and excludes the old rigid `attachment` pass.

These controls preserve the previous rotation-only format. They do not add
physics, collision avoidance, ground contact, muscle control or dynamic fur.
Do not certify loops merely because endpoint poses match.

```json
{
  "id": "takeoff",
  "interpolation": "LINEAR",
  "keys": [
    {"second": 0, "rotations": {}},
    {"second": 2, "rotations": {"Wing": [0, 15, 0]},
     "translations_m": {"CompanionRoot": [-0.2, 0, 0.3]}},
    {"second": 4, "rotations": {}}
  ]
}
```

This is a schema example, not a generated avatar or a general wing rig. Actual
bone names, surfaces, wing pivots and masks must come from the selected model.
Do not translate a fused accessory: shared triangles will stretch the body.

## Evidence and limits

The analytic native fixture uses two weighted parts, a rotated/scaled armature
and independent parent/child translations. Its exported GLB reloads within
0.000000361m of sampled native positions. That verifies coordinate handling,
not panda appearance or flight meaning. The first two fixture comparisons
mistakenly included Blender's bone-display helper mesh; those attempts remain
preserved with their failures.

The generated panda has broad welded shoulder/head contact surfaces. Splitting
the parts exposes unseen geometry. Closures and their hidden colors are authored
assumptions, not recovered reference surfaces. The original mesh/texture bytes
and all approved outputs remain immutable. Local partition work records exact
selectors, topology changes, preserved triangles/UVs, cap geometry, source
palette choices and rejected strategies in new versioned output folders.

No provider calls or new paid rendering are required for this follow-up.
Do not change Weft's importer, physics, actions or rewards to accommodate it.
