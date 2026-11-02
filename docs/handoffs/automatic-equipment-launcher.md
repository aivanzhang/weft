# Launch the automatic-equipment example

## Ready local command

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example auto-equipment
```

Add `--console` for terminal interaction/recording, or `--check` for metadata
verification without starting a server or simulation. The user owns server
startup. The lab is `artifacts/equipment-frame-consensus-native-03/build/lab`;
its source videos and saved run are local, not fresh-clone contents.

This example has **manual and authored reference controls, no trained policy**.
Automatic refers to equipment-frame selection, not recovered physical truth.
Source geometry assumptions, root assistance and unknown contact accuracy remain
explicit. The old `--example equipment` still opens its unchanged v5 example.

## Implementation and boundaries

Owned source: `scripts/play_sandbox.py`, `tests/test_play_sandbox.py`.
The new preset trusts only Git commit
`e5c9676d59f9b7d184802d11a757e0db99a10f7b`. Its separate caps are 299 runtime
files, 8 MiB total, 512 KiB per file and 400 lab manifest members. Existing
presets and limits are unchanged; no arbitrary revision or development fallback.

The untrained-only metadata branch rejects attached policy files and policy/
learning claims, checks the source-manifest pin/member count, scene/task/reset
and run-manifest joins, 17 exact simulation-source pins and installed versions.
It reads only the run manifest, lab contract and lab configuration from the lab;
it does not read source/media/session payloads or replace full cold admission.
Runtime preparation still uses the existing no-follow/exclusive Git cache path.

## Verification — November 1, 2026, 4:39 PM PST

- **48 focused guarded cases pass:** 20 new automatic-example cases and 28
  existing launcher cases. Includes resealed incompatible policy/runtime/source/
  config claims, scoped bounds, preserved recordings and both interface routes.
  Elapsed 0.450640 seconds; child high-water RSS 87,982,080 B. All native calls
  were denied; zero attempts. The known import-time subprocess probe was blocked.
- Independent read-only review closed on the source/test pins below. No passing
  check was repeated for review.
- Actual command with `--console --check` passed in **0.349158 seconds**,
  44,351,488 B child high-water RSS. It prepared the previously absent trusted
  cache: **299 files / 6,533,297 bytes**, largest member 405,810 B. Installed
  Python/MuJoCo/NumPy/Gymnasium versions matched the saved lab contract.
- The manifest, contract and configuration hashes stayed exact. PID95287 is
  reaped, exit 0. No server, browser, physics, source proof, recording or media
  export ran. This is startup metadata verification, not new-scene interaction
  acceptance, a UI latency guarantee or full-task success. External cost: $0.

| Identity | SHA256 |
| --- | --- |
| Launcher | `916d3dded2a716aa6ae19786cd549c336cfdf76b6b3ace9f869ef4a486fda634` |
| Tests | `afbd0a0f3bafe18371489bd4b41d07bbcd7292fc08978b5af1ec4af9218b5d00` |
| Lab run seal | `1d4a32d609cfbb9647827116ddcb2a44f13067bbb307387f7aef18783481c900` |
| Lab contract | `24c1dcc0e68c3a8e8f9fb85965543d4c0619feccd96e8d0a1d648c6e9a803e28` |
| Lab configuration | `5b86e590302e2a410ee10992606b7dbdde76d59f036db5db8b8ecafe06a39db4` |

Plan, XML, pure guard/process receipts and actual command logs are preserved in
`.local/auto-equipment-launcher-01/`. `release-01.json` records their exact pins.
Root owns publication and guide/TODO/combined updates; preserve parallel
correlated-region, controller-study and independent avatar work.
