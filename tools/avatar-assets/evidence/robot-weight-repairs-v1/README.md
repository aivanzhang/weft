# Rejected robot weight repairs — sealed evidence

Both repair approaches remain rejected. This bundle is a compact review record, not an accepted avatar, automatic pipeline repair or complete model backup. All originals remain unchanged. `STAGED.json` and `README-STAGED.md` preserve the earlier pending checkpoint; this final manifest supersedes their temporal status.

## Actual outcomes

- The last permitted AI weight-repair call produced a wave-like arm pose but moved the torso35.53mm and visibly raised a blue chest-side wedge. Its original diagnostic is rejected.
- The separate opt-in hard-surface CPU projection preserved original core memberships, explicitly recorded55lost weak-fringe hand vertices and passed CPU stability checks. Those checks did not establish connected-surface quality.
- Independent native import measured93,680edges at8authored poses and found3severe edges in6poses. The reviewed raised-arm image still shows a conspicuous chest-side flap. The unchanged thresholds reject the derivative.
- The independent diagnostic exposed an existing production reporting defect: the old top-four witness truncation could omit all severe edges. The in-progress fix and any later validation are deliberately excluded. This bundle does not claim the production pipeline is corrected.

Native verification completed in16.307s, with1,152,794,624B sampled direct-child RSS and verified process exit. This is sampled memory, not machine-wide peak allocation. No full-clip visual acceptance, critic approval, accepted delivery, physics, Weft mapping or automatic promotion follows.

## Retained evidence

`manifest.json` maps original paths to local `saved_path` copies or exact `prior_bundle_reference` bytes in `evidence/studio-rigged-robot-v1`. Keep that prior immutable bundle beside this one. It preserves exact prompts/input images/schemas/answers/receipts, source scripts, numeric reports, core memberships and lost fringe indices. OmittedGLB/BLEND/full-frame originals are bound by path, size andSHA256. No copied source is executed by this evidence operation.

Selected failure images are under `files/outputs/`:

- `rigged-robot-weight-repair-probe-v1/model/renders/greeting/0003.png`
- `robot-stable-projection-native-v1/front.png`
- `robot-stable-projection-native-v1/renders/greeting/0003.png`

The13-call development envelope is exhausted:358,211input/13,885outputtokens; account dollars unknown. Snapshot/native diagnostics add no AI/Meshy/Meta calls. The fresh105-credit request is still pending, with no grant. Its revised50,000input-token per-call estimate is a planning assumption, not an enforced token cap.

The642-test full-suite/source receipts are retained. The later estimate-only change has21targetedchecks and a separately boundpipeline.py snapshot; it is not falsely identified as the earlier full-suite source. Those tests predate the in-progress severe-witness correction.

## Verify without generation

From `tools/avatar-assets`, use standard Python:

```sh
python3 - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd();bundle=root/'evidence/robot-weight-repairs-v1'
m=json.loads((bundle/'manifest.json').read_text())
for item in m['files']:
    if 'saved_path' in item:
        p=bundle/item['saved_path']
    else:
        ref=item['prior_bundle_reference']
        p=root/ref['bundle_path']/ref['saved_path']
    assert p.stat().st_size==item['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print('Copied and referenced evidence hashes verified.')
PY
```

Do not restart old jobs, reset the exhausted call ledger or promote a numerically passing but visibly broken model.
