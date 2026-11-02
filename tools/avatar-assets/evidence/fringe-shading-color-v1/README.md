# Exported renderer and fur color/shading evidence

**A corrected preview renderer and partial fur improvement, not an accepted
hands-off avatar or dense/Muse-equivalent coat.**

`manifest.json` maps exact local origins to deduplicated immutable payloads.
It retains the actual glTF culling control, production renderer source/tests,
analytic color-atlas helper, three new actual model builds, independent normal
and part-ownership audits, all32 matched EEVEE previews, and parent reviews.
Large GLB/Blend files remain local with hashes; this is not their remote backup.

Key origins:

- `outputs/fringe-culling-native-control-v1/findings.json`
- `outputs/export-preview-engine-tests-v1/test-receipt.json`
- `outputs/fringe-color-atlas-analytic-v1/RESULTS.md`
- `outputs/panda-fringe-color-atlas-v1/color-error-audit.json`
- `outputs/fringe-source-normal-audit-v1/audit.json`
- `outputs/fringe-four-way-review-v1/review.json`
- `outputs/fringe-part-ownership-review-v1/review.md`

Six new renderer tests and49 focused integration checks passed. A real fixture
shows the pinned Cycles path ignores tested single-sided glTF culling; bounded
EEVEE passes that control. Native hair/portrait rendering stays unchanged.

Matched actual previews show source-surface normals remove most white-body
speckling. The new atlas reduces nominal-tip errors above0.1 from194 to0 atLOD0,
but the combined coat still shows dark-region light flecks. Minification remains
under separate investigation and is not included in this completed bundle.
There is no fur assignment leak to the dragon. Contact/source defects persist;
no default, gallery or automatic fur-profile promotion is made.

No Meshy, Meta, image-generation or pipeline-role AI calls ran. Existing approved
assets, Weft's consumer/physics, original reference and spending limits remain
unchanged. Rights/provenance are inherited from pinned source manifests, not
independently verified. Exact commands/settings and bounds remain in preparations.
Completed append-only execution scripts must not be rerun in their old folders.
