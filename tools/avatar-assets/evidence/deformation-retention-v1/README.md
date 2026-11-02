# Severe-witness retention — sealed regression proof

The reporting fix passes its numeric regression. The avatar remains **rejected**. This is not a new model, full finisher rebuild, rendered-quality pass or accepted Studio delivery.

The corrected per-pose and cross-pose selection retains three severe edge witnesses that the previous top-four truncation omitted. A fresh native import reproduced all eight original pose/topology checks: six poses retain three severe witnesses each, and all three remain in the cross-pose report. Acceptance thresholds, geometry, skin and animation were not relaxed or replaced.

The native process completed in **1.820 s**, with **325,517,312 B sampled direct-child RSS**, six memory samples and verified process exit. The wrapper result took 1.918 s. No rendering, AI or provider calls were made by this regression. Sampled memory does not measure every descendant or instantaneous machine-wide allocation.

## Contents and limits

`manifest.json` binds exact prepared inputs, fixed helper/finisher source, focused tests, native launcher and receipts, original/fresh numeric reports, and the full-suite log/source receipts. The fixed helper, finisher and diagnostic-test snapshots match the full-suite source hashes. The full suite passed **646 tests in 15.331 s**; its wrapper took 15.673 s and reports 140 sources unchanged. Analytic fixtures remain distinct from the actual imported robot samples.

The 24,462,804-byte `inputs/model.glb` remains at its original path and is referenced by SHA256. No GLB, Blender file, prior image or full frame archive is copied here. Native bytecode/cache files are likewise omitted and hash-bound; their exact source scripts are included. Every original file was verified unchanged before and after copying.

Earlier rejection imagery and weight-repair evidence remain in `evidence/robot-weight-repairs-v1`; those historical records correctly describe the bug as unresolved at their earlier checkpoint. This separate proof records its later correction without changing them.

## Verify without running a model

From this bundle directory, use standard Python:

```sh
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('manifest.json').read_text())
for item in m['files']:
    p=Path(item['saved_path'])
    assert p.stat().st_size==item['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print('Evidence hashes verified; the avatar remains rejected.')
PY
```

Do not repeat generation or treat this compact proof as a backup of omitted geometry.
