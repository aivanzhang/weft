# Reject damaging surface-normal transfers

Normal continuity changes surface lighting and fur directions, not geometry.
The former transfer could turn a target corner inward relative to its own face.
The new guard rejects that result before normal attributes or shading are written.
This prevents a known failure; it does not repair the panda's raised cheek patch.

## Behavior and saved evidence

`avatar_assets/normal_continuity.py` computes translation-relative Newell area
normals for the supported 3–64-corner polygons. Only transferred target corners
are checked. Each needs a positive face dot product; degenerate/cancelling target
faces also reject. Unchanged corners and intentionally authored fur normals are
outside this policy. No normal flipping, clamping or fallback occurs.

`NormalContinuityError.report` contains counts, input/selection hashes and at
most 32 face/corner/vertex witnesses. The native finisher writes
`normal-continuity-rejection.json`, then fails. This is not a completed model.

The pipeline pins both transfer source files when preparing a job. Before a
bounded repair receives rejection findings, the receipt must match its failed
stage's artifact hash and size, fit within 32 KiB and contain consistent bounded
counts/witnesses. Missing, changed, unbound or malformed evidence halts; it does
not buy another author call. Uncertain native process exit also stops repair.
Sealed historical deliveries remain read-only resumable with their saved sources.

## Verification

| Check | Result and preserved output |
| --- | --- |
| Planner and mutation boundaries | 13 tests pass, including per-corner rejection despite an outward mean, tangent normals, degenerate polygons, collinear first corners, scale, bounded witnesses and unchanged disabled operation. `outputs/normal-hemisphere-tests-v1`; 0.262 s bounded wall time, RSS unsampled. |
| Pipeline integration | 10 focused tests pass for new pins, legacy delivery, exact findings propagation, no downstream build/media/reviewer after rejection, unresolved exit and invalid evidence. `outputs/normal-rejection-pipeline-tests-v1`; 0.522 s, 63,733,760 B sampled RSS. |
| Actual-source CPU replay | Original v7 model/config, 51,055 vertices / 62,581 triangles. Rejects 172 of 1,095 transferred corners on 369 faces; minimum dot −0.9998156743452801, zero degenerate faces and zero adapter mutation attempts. All 32 witnesses reproduce independently. `outputs/normal-continuity-source-replay-v2`; 0.783 s, 221,675,520 B sampled RSS. |
| Actual Blender path | Real import and authored rig setup produce the same 172-corner rejection and save the exact exception report. A sentinel forbids downstream color decoding/fur/rendering. `outputs/normal-guard-native-v1`; 1.823 s, 429,211,648 B sampled RSS, exit 0 for the rejection check, zero rendered images. |

The CPU replay's first facade used NumPy scalars where Blender supplies Python
floats; the existing type guard rejected it before the new check. Preserve v1.
The v2 checker changes scalar representation only, not numerical values or gates.
No source model/config changed and no new paid generation or pipeline-role calls
were made. Sampled RSS covers the direct child, not total OS/driver memory.

Recorded commands, already executed:

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -B -m unittest tests.test_normal_continuity -v
.venv/bin/python -B -m unittest tests.test_normal_rejection_pipeline -v
# One-shot immutable native check; do not rerun into its existing output.
.venv/bin/python -B outputs/normal-guard-native-preparation-v1/execute.py
```

Exact manifests:

- CPU replay: `52cd9a7957a5958d62ac64d63100047d0dc7d8a6c984824a38d51d07c6908bf7`.
- Pipeline tests: `ccbd27e45da593642081205ff4dd1eeb17c41097446fd482b4ccf22f9e3c0578`.
- Native prepared inputs/code: `fd74de99b8f6aadc7ef3cec6d2138168bfc7879cde8e7641762475bdf8eef45d`.
- Native rejection receipt: `784e4c2e9ef8006cc3508c26b05bba7612f8a447b8ddcdb3d3695f4455519a4f`.

The old panda continuity recipe now correctly rejects on a new run. Approved
outputs are unchanged. A supported replacement must fix the diagnosed surface or
reference selection, not suppress the guard. Fresh Studio quality acceptance and
the pending generation allowance remain separate.
