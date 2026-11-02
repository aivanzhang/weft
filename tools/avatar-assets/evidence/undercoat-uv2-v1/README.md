# UV2 preservation evidence

Exact snapshots of 116 local artifacts, 2,775,651 bytes. The manifest binds every
origin path, snapshot, size and SHA256. Original outputs remain unchanged.

- 111 unique focused software checks pass; 11 legacy checks were repeated.
- A real Blender bake/export of analytic triangles preserves source UV1 maps
  while adding authored UV2. Its editable scene and native/corrected GLBs are
  included. These are **fixtures, not generated avatars**.
- The first native checker failed before baking because its pixel array used
  float64. The corrected checker uses float32; both attempts are retained.
- Cheek-reference and scar-selection diagnostics remain unsupported. Their
  referenced real panda model remains in its original local output folder;
  this bundle is not a backup of that large model or a successful geometry repair.
- No new provider generation or pipeline-role calls. No avatar/default promotion.

See [implementation and results](../../docs/undercoat-uv2-preservation.md) and
[current handoff](../../TODO.md). Sealed scripts refuse reuse of completed output
folders; a new experiment needs a new preparation and folder.
