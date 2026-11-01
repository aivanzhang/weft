# Whole-body worker blocked-call diagnosis

## Status — October 31, 2026 (PDT)

One bounded **import-only** diagnosis completed under unchanged process/network
guards. Its actual blocked call came from MuJoCo's Darwin startup check. No
model, data, forward, fitter, physics, reset, media or inference call occurred.
No permission exception, preload, fake response or fit rerun was introduced.

| Evidence | What it establishes |
| --- | --- |
| `artifacts/surface-interval-fit-native-01` | Original authored fit completed inside its worker, but outer acceptance remains **failed** because it required zero denied attempts. That attempt's caller was not recorded. |
| `artifacts/surface-interval-fit-inspection-01` | Separate saved-output inspection, not a replacement passed acceptance or resimulation. |
| `artifacts/body-worker-import-diagnosis-01` | A new import-only attempt is identified by its captured code stack. Matching dependency versions do not retroactively prove the original caller. |

The original fit's saved audit remains
`.local/surface-interval-native-saved-audit-01.json`, SHA
`0dfdfda45f3504689062da154255557534fa05d2d7f8f8fd6940c6595b73cbdc`.
This diagnosis did not reread or relabel that study.

## Exact new result

- Diagnosis seal:
  `0511c39f86db708b9806e5d98398de3c7186529d2d2ec41c26e0a1b763633be9`.
- `diagnosis.json` SHA:
  `2a99bdac718842ed4997cdeafdc6263d04d188f496fa6a1ebae5573fd376f2c8`.
- Frozen runtime manifest:
  `.local/body-worker-import-runtime-01/manifest.json`, exact copied member
  `runtime-manifest.json`, SHA
  `18d88e6e1f09afe000de244282e9d11e0c6aa0ec789dbcd0bf3b936479242404`.
  Base `259cfc2b97f3c237a2ff80191abc5c590f9074c9`; 21 runtime files, only
  `whole_body_reference_worker.py` overridden for reviewed diagnostics.
- Worker SHA:
  `75ffd37356814c262976d547f24051b5fd68e5d4b1247feb839a1b93d684f2d8`.
- One blocked nested-process attempt; all eight work counters zero. The one
  six-frame diagnostic is 672 bytes, with zero omitted events.
- Child 0.182836458 seconds; supervisor 0.231152416 seconds; observed peak RSS
  67,977,216 bytes. These are descriptive observations, not a performance study.
- Bounds: one child, 12-second parent wall limit, 5/6-second child soft/hard CPU
  limits, 64-KiB receipt and per-stream logs. No paid calls or recurring cost.

The saved stack begins `subprocess.run` at line 555, called by
`mujoco.__init__` at line 43. The installed
`.venv-lab/lib/python3.14/site-packages/mujoco/__init__.py` contains the fixed
Darwin command `['/usr/sbin/sysctl', '-n', 'sysctl.proc_translated']` at that
location. Lines 47–48 catch `OSError`/`ValueError` and set `is_rosetta=False`;
the guard's `PermissionError` is an `OSError`. The attempted process was blocked.

Installed initializer: 8,236 bytes, SHA
`24000c0b3c195f2bdb28a65420c2004b019a9bb6f6737e4caa1de798ffc064c5`.
This is a post-run installed-file inspection, **not** an initializer-byte pin
captured before the original fit. Versions recorded in both studies are Python
3.14.7, MuJoCo 3.17.0, NumPy 2.5.4 and Gymnasium 1.4.0.

## Diagnostic contract and verification

`whole_body_reference_worker.py` records only bounded code identities:
module, filename basename, function and line. Limits are eight events, six frames
per event, 64 printable characters per symbol and 16 KiB total event bytes.
No arguments, locals, source lines or full paths are captured. No source-file or
OS lookup was added to allowed native-call paths.

Denial counts increment before diagnostics. Trace failure or overflow records
an omission and still raises the original `PermissionError`. Root reported ten
distinct focused checks passed; this review did not rerun them. The separate
camera-root worker's existing startup-probe exception was not copied here.

Independent saved-only audit:
`.local/body-worker-import-saved-audit-01.json`, SHA
`f880bf320b87a6e5e9d73ebc9dd77aa5e67231648333f30f476eb9bd2a41bf66`.
It verified sealed member hashes, all 21 runtime pins, counters, bounded trace,
static initializer command/fallback and unchanged blocking behavior. It did not
import the captured runtime, reproduce imports, run tests or execute native code.

Successful cache: 29 files / 392,196 bytes, 0.002009-second collection and
0.004291-second analysis. Two checker-only failures are preserved separately:
an assumed runtime-manifest filename and oversized inline tool output. Those
required repeating the small collection; the audit records approximately
1,168,352 total bytes read rather than claiming one-read reuse.

## Ownership and next safe action

Root owns worker code, tests, diagnosis and publication. This lane owns this note
and saved-only audit. No live handles, product edits or Git mutations remain in
this lane.

Retain the original failed acceptance and unknown original caller. Reuse these
diagnostic results; do not rerun the fit, weaken guards, or add a process exception
merely to make denied counts zero. Future unexpected denials can use the bounded
trace. This milestone establishes diagnostics only, not source reconstruction,
contact, anatomy or physical validity.
