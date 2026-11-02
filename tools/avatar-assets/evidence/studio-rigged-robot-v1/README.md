# Rejected rigged-robot Studio trial

This is immutable review evidence, not an accepted avatar, complete model pack or resumable job. Original outputs remain unchanged. No source GLB, Blender file or full frame archive is copied here; `manifest.json` binds each omitted original by path, byte size and SHA256.

The actual trial used the existing paid Meshy rig. Its final model passed numeric movement/stability checks, but the independent critic rejected a low wrist swivel as an unclear greeting. Four real AI calls were made in this trial. The actual UI recorded unsupported, and its owned browser/server exited. No delivery was accepted.

## Selected media

All paths are beneath `files/outputs/studio-b184afa00c2f4c2d853de705/`:

- Rejected four-second video: `media-portable-1/animations/greeting.mp4`.
- Static/reduced-motion fallback: `media-portable-1/fallbacks/greeting.png`.
- Portrait: `media-portable-1/portrait.png`.
- Two individual 720px poses: `final-evidence-1/isolated-native-greeting.png` and `isolated-exported-greeting.png`.
- Every actual supplied agent image is retained under `agents/`; prompts, answers, schemas, settings and process/usage receipts are unchanged.

The review images/MP4 are real rendered data, not analytic fixtures. The 629-test and 42-test logs are software contract evidence, not avatar acceptance. The 42-test source receipt is explicitly a post-check snapshot, not a prelaunch pin.

The separate `outputs/rigged-robot-weight-repair-probe-v1` diagnostic is deliberately excluded while its result is being sealed. Its call is outside this completed-trial snapshot; this bundle's twelve-call total covers only the four completed Studio jobs. No additional provider/model/native/browser call was made to assemble this bundle.

## Verify without generation

From this bundle directory, run standard Python:

```sh
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('manifest.json').read_text())
for item in m['files']:
    p=Path(item['saved_path'])
    assert p.stat().st_size==item['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print('Copied evidence hashes verified; omitted originals remain local.')
PY
```

Do not execute copied source as part of inspection, repeat Start on the unsupported job, reset budgets or call this compact evidence a backup of the omitted model sources.
