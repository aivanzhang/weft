# Short-pile undercoat: component evidence

The independent agent review accepts a **promising optional surface-detail
component only**, not complete fur, Muse parity or a replacement for approved
assets. The black body/ears look softer and less glossy; white areas remain
smooth, the silhouette lacks dense fur, and existing cheek/shoulder separation
marks remain visible during flight. See [the exact review](review/independent-review.json).

This is a compact evidence bundle, **not a complete avatar pack or standalone
recreation**. Large GLB/Blender files stay at their original paths and hashes in
[the omitted inventory](omitted-original-artifacts.json). Original parent
manifests are byte-preserved snapshots; they describe the full original packs,
not the smaller directories beside them here.

## Retained evidence

- `model-evidence/`: exact generator/patch scripts, source geometry arrays,
  measured mask, authored UV1/tangents, actual normal PNGs, settings, plan,
  CPU receipt and model-preservation checks.
- `preview-evidence/`: exact renderer snapshots, settings, terminal job/log and
  full original preview manifest; large preview binaries are omitted.
- `reviewed-images/`: the three actual inspected candidate PNGs—front, side and
  flight at 1.36 seconds. They are unedited 720×720 RGB copies.
- `review/`: immutable comparison context and the independent component verdict.
- `tests/` and `test-receipt.json`: exact focused test sources and recorded
  results. Analytic fixtures are not appearance or generated-avatar evidence.
- `history/`: compact receipts for prior rejected attempts and UV attribution.
- `bundle-sources.json`: original path/hash mapping for every copied file.

The actual derivative has 75,548 vertices and 72,181 triangles. Its original
POSITION/NORMAL/UV0/base-color/skin/motion data and portable fibers remain exact.
Only the measured aliased surface receives authored UV1/tangents; per-region
normal textures and declared roughness are new. The result is **HD-only and not
Weft-compatible**. No live adapter/validator integration or automatic fur-quality
promotion occurred.

## Verify without rendering

From `tools/avatar-assets`:

```sh
.venv/bin/python outputs/panda-undercoat-uv1-evidence-v1/verify_evidence.py
```

The verifier reads only this bundle, checks the exact inventory and SHA256s, and
enforces its 10 MiB cap. It does not call a provider, render, or modify a file.
The compact bundle does not independently reload omitted models or rerun tests.

Scripts/configuration retain original absolute input paths and expected hashes.
Do not execute them against these sealed folders. Any future experiment needs a
new output directory, verified original binaries, a new bounded job receipt and
the shared worker slot. Keep existing adapters and the validator unchanged for
the next robot work. This component review does not authorize integration,
publication, new spending or changes to accepted assets.

New provider/AI calls: zero. CPU bake/patch: 2.864 s / 382,320,640 B sampled peak
RSS. Native preview: 22.254 s / 1,284,325,376 B. Both workers exited. These are
different jobs from the earlier failed full native finish; they do not establish
a fix for that memory failure.
