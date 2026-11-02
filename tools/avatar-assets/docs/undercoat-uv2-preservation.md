# Preserve source textures while adding undercoat

Recorded November 2, 2026, 1:04 a.m. PST. This is a verified reusable finishing
component, not a new accepted avatar or completion of hands-off Studio quality.
No generation, paid pipeline-role call, server or browser ran for this increment.

## Change

The shoulder-color repair already uses UV1 (a second set of texture coordinates)
to preserve original normal and roughness/metallic maps. The old adaptive
undercoat rejected that source; blanket UV1 tangent correction would also have
reinterpreted those original normal maps.

- `scripts/blender_fur_undercoat.py` preserves one or two original UV layers and
  their hashes. It appends `AuthoredUndercoatUV1` or `AuthoredUndercoatUV2`, never
  overwrites a source layer, and restores active/render UV0. Three source layers
  or reserved-name conflicts remain unsupported.
- Selected material copies replace only their normal response. Existing linked
  roughness/metallic maps survive under `preserve-linked-response-v1`; configured
  constants apply only where no link exists. Reports record each socket's policy.
- `finish_generated.py` derives explicit authored targets from the native report.
  `generated_tangents.repair(..., authored_normal_targets=[...])` requires the exact
  clone name, UV1/UV2 index, selected face count, embedded PNG SHA256 and current
  `mikk-unshared-triangle-v-up-v2` bake policy. Material names must be unique.
  Multiple primitive bindings must sum to the exact reported face count.
- Only targeted tangent float4 ranges change. Untargeted source UV1/UV2 frames
  remain exact; invalid frames fail rather than receive an unrelated fallback.
  The legacy explicit UV1 API remains available for historical experiments.
- `three_d.validate_render_attributes` checks UV2 without raising the 100k-vertex,
  64 MiB-file, 1.1-million surface-scalar or 2-million global-scalar limits.
  This does not change Weft's separate diffuse UV0-only importer.

Correction targets have exactly these fields:

```json
{
  "material": "Example__short_pile_adaptive_v1",
  "normal_texcoord": 2,
  "selected_faces": 4,
  "texture_sha256": "<exact SHA256 of the baked PNG>",
  "frame_policy": "mikk-unshared-triangle-v-up-v2"
}
```

Four corrected primitives and 2,048 authored faces per target remain the maxima.
This is not an arbitrary UV remapper, hidden remesh or new dense-fur technique.

## Verification and immutable outputs

| Evidence | Result | Observed resources |
| --- | --- | --- |
| `outputs/undercoat-target-contract-tests-v1` | 74 tests pass | 0.522 s; 57,229,312 B sampled RSS |
| `outputs/undercoat-source-uv-tests-v1` | 26 adapter/planner tests pass | 3.903 s; 180,666,368 B sampled RSS |
| `outputs/authored-normal-target-tests-v1` | 11 new tests plus 11 already-passing legacy tests pass | 0.262 s; too short for RSS sample |
| `outputs/undercoat-uv2-native-v1` | Checker failure before baking: float64 image array rejected | 1.829 s; 189,022,208 B sampled RSS |
| `outputs/undercoat-uv2-native-v2` | Actual Blender analytic bake, preservation assertions and export pass | 1.565 s; 292,388,864 B sampled RSS |
| `outputs/undercoat-uv2-native-export-check-v1` | Exact texture/UV routes, targeted byte correction and GLB attribute check pass | 0.263 s; too short for RSS sample |

There are **111 unique focused software checks**, not 122 unique checks. The
second native attempt changes only the fixture pixel-array type to float32;
production code and verification thresholds are unchanged. Both attempts remain.
All owned processes exited. Native limit: one attempt per frozen preparation,
60 seconds, 1,536 MiB sampled child RSS, 32 MiB output. CPU export check: 30 seconds,
256 MiB, 8 MiB. Sampled direct-child memory is not whole-process-tree measurement.

Compact exact evidence: `evidence/undercoat-uv2-v1`, 116 snapshots / 2,775,651 B,
manifest SHA256 `e092fa988c703ba144fa9781aa0f9a0382fd4cd5467867d233f79a8b27a8871c`.
It includes the analytic scene/models, all failures and both geometry diagnostics;
the large referenced panda source models stay in their immutable local folders.
Packaging: `outputs/undercoat-uv2-checkpoint-v1`, 0.265 s; PID 59785 exited before
an RSS sample, bounded by 30 seconds / 256 MiB / 8 MiB.

Native fixture preparation: `outputs/undercoat-uv2-native-preparation-v2`,
SHA256 `c83af2b1c770fbbc57782b79c61916e8925f7dc6ef36a0553c659e4f2642ece4`.
Its exact command was:

```sh
.venv/bin/python outputs/undercoat-uv2-native-preparation-v2/execute.py
```

Do not rerun sealed preparations. Make a fresh preparation/output folder after
a changed hypothesis. Frozen checker, adapter, planner and runtime snapshots are
included. The editable analytic scene is `outputs/undercoat-uv2-native-v2/source.blend`.
Corrected analytic GLB: `outputs/undercoat-uv2-native-export-check-v1/model.glb`,
SHA256 `f811549aae8f165cbc3e8f43e4ae6ea190f1fa862c872d1eddf733c1c46c8555`.
Only **90 binary bytes**, all in the one authored UV2 tangent range, change.
Original color, normal and response PNG bytes and routing remain exact.

## Still incomplete

The real repaired panda source is still
`outputs/panda-shoulder-uv-repair-v2/model.glb`. This increment has not yet run
that source through full fur, rig, exported motion and visual acceptance. Its
new material names must participate explicitly in the intended masks; they are
not automatically inherited. Four selected material regions remain the bound.
Do not bypass the normal-transfer rejection or enable experimental fringe.

The separate cheek diagnosis proves the reference errors largely come from
folded source geometry, not just custom normals. Its bounded scar selection
remains unsupported; see the [contact diagnosis](contour-contact-diagnosis.md).
No approved model, gallery or default was replaced. Fresh text-driven Studio
acceptance still needs the pending allowance and successful real results.
