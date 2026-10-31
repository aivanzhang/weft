# Camera-based shared-world lab integration

Updated: October 31, 2026, 3:16 PM PDT. Owner: root. Implementation, combined
native acceptance and final saved/UI audits passed; the product checkpoint is
published and the exact private remote hash is verified.
The combined goal remains active.

## Scope and ownership

Use the existing `shared-exercise-lab`, `lab-console`, lab UI and recording path
for actual `shared-world-v6` scenes. Do not reprofile the body or erase camera
metadata. The camera task preserves original source clocks and uses unchanged
body-tracking arithmetic through an explicitly nonlegacy metric context.
Ground/equipment contact, feet, learning and avatar qualification remain absent.

- Root: common lab/recording/receipt/capability dispatch, camera capability and
  saved receipt helpers, integration tests, frozen native acceptance and Git.
- Support-learning lane: camera task fix, strict state/recording adapters and
  analytical tests. No native job in that lane.
- Support-UI lane: existing live/offline UI and camera-specific scripted checks.
- Exercise-reconstruction lane: partial-foot evidence is frozen; now reviewing
  root's backend integration read-only. Separate avatar work remains untouched.

## Current changes

`shared_exercise_lab.py` admits a distinct camera task, retains the complete
sealed candidate and exposes exact camera/ground artifacts. It rejects learning
attachments before expensive preparation and rejects a displayed coupling that
differs from the simulated coupling. Root assistance stays an explicit build-time
choice. Sessions use one real camera task engine, not a legacy wrapper engine.

`camera_shared_capabilities.py` exposes 19 body controls or 22 with explicit root
assistance, a 156-value observation, exact task/source/context identities and no
equipment/learning claims. `camera_shared_receipt.py` binds those identities to
saved runtime files without importing physics or parsing recorded actions.
The common recorder retains its 128 MiB journal limit; camera events have the
same explicit 512 KiB ceiling as detailed feet events.

The shared snapshot validator now prepares the immutable camera-coupling hash
once. It still checks each snapshot's actual coupling identity and coordinates;
no physical or scorer rule changed. The protected scorer remains unchanged.

## Verification so far

- Seven new camera capability checks and nine initial receipt checks passed.
  Seven additional receipt tamper cases initially had a wrong test expectation:
  the established exporter intentionally preserves its base export even when
  an optional evidence receipt is unavailable. Only that assertion changed;
  all seven passed on the second run. No product validation was loosened.
  Receipts: `.local/camera-integration-metadata-01.xml` and `-02.xml`.
- Thirteen new lab checks and 22 affected legacy feet-lab checks pass. The
  independent reviewer found an incomplete avatar guard before session exposure;
  v6 is now rejected there, closes its environment and cannot fail later after a
  recording reset. Receipts: `.local/camera-lab-plumbing-01.xml`,
  `.local/camera-lab-avatar-02.xml`, `.local/camera-lab-regressions-01.xml`.
- All 92 affected existing receipt checks passed in the new context:
  `.local/camera-receipt-regressions-01.xml`. No native physics or media runs.
- Camera task integer-action fix passed independent review. Extreme signed
  integers reject before state mutation; protected scoring bytes unchanged.
  Its owner retains the failing reproduction and changed-check receipt.
- The camera task covers 45 distinct analytical cases; the state/recording
  adapter covers 18. Initial fixture-only failures remain preserved, and only
  affected cases were rerun. [Task](camera-shared-task.md) and
  [recording](camera-shared-recording.md) retain exact commands and pins.
- Seven camera UI groups and 18 affected legacy groups/scenarios passed. Six
  additional actual saved-payload checks pass (live, receipt and offline replay
  for both modes), including every one of the 242 offline events per case.
  Receipt `.local/camera-shared-ui-actual-payload-01.json`, SHA
  `9aff45b1aa06ff6b573f0dc92a5ee0bd5fcda7a316c9f68dc122f942c1162a6c`.
  No product changes followed the native freeze. [UI evidence](camera-shared-ui.md).
- Root independently verified the frozen partial-foot source result: all 11
  sealed members plus manifest, 5,006,215 bytes, 240 rows and unchanged original
  statuses/coordinates. Original gaps: 282 / six negative; separate partial
  ankle/toe gaps: 227 / five negative. Different denominators are not a contact
  improvement. Receipt `.local/partial-foot-evidence-jog-saved-audit-01.json`,
  SHA `6674b091a251bb5a91b0896bdbc337f66ab6d6b4fa745e63f01a8757386db832`.

## Acceptance protocol and bounds

After owners froze and analytical checks passed, export committed `d1b461f` with
only reviewed integration overrides; exclude dirty avatar files. Reuse the
already sealed `artifacts/camera-shared-admission-jog-01/candidate` (manifest
`29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`).
Do not repeat reconstruction, source import, pose inference or the geometry audit.

Prepare roots-off/on labs through the ordinary CLI. Record a short manual action,
reset, then authored control over the unchanged complete interval or an explicit
early failure. Export and exact-replay the same actions, inspect the saved receipt
and resume. Preserve every failure and source/runtime pin. Native controls are
bounded to 1,000 including replay; native substeps to 9,000; wall time 60 seconds,
CPU 45 seconds and output 192 MiB / 2,000 files. Observe peak RSS rather than
claiming a hard memory cap. Source video is copied but not decoded; recording
video/GIF encoding is allowed. No network, source models, training or paid calls.
The exact job plan and hashes preceded launch. No live handle remains.
The parent also checks aggregate output at most once per second; the worker
checks preparation/export/case boundaries and reserves terminal-result space.
These are cooperative limits, not a filesystem quota.

### Frozen combined acceptance

Launched October 31, 2026, 3:09 PM PDT and completed at 3:10 PM PDT. Parent handle
`84739` is terminal. All three owners released code; independent review found
no remaining blocker. Runtime is committed `d1b461f` plus exactly 14 pinned
integration files, with no unrelated avatar edits.

- Plan `.local/camera-shared-play-plan-01.json`: `612e4c047fcdddd786e0c9ecf37c743cfadd504e4ea30a859fa3fb09fef76deb`.
- Frozen runtime manifest: `80a18d8881ae33e7ff5e02ff0609df0a525cb157ddbfc2358b853f2d3c6676aa`.
- Worker: `352f4c87ed4b95f3c7ae84f746b822ef224d7a9ac97ed66495371d7a26475b0a`.
- Output: `artifacts/camera-shared-play-01`. Do not restart, change owned
  product files, or inspect partially written results as completed evidence.
- Native source input stays the original saved candidate; no new source/model
  work or paid calls. The protected scorer still hashes to `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.

### Native result

Both source-bound labs completed CLI preparation → console manual control →
reset → authored control → video/GIF/offline replay export → exact action replay
→ saved receipt inspection → resume. Each recorded 240 actions (two manual,
238 authored), one reset and 242 events. All 480 actions replay exactly. User
record-stop closes the journal; separate task cancellation is idempotent, changes
no physics state/time and never rewrites a completed outcome.

| Original interval | Roots off | Explicit roots on |
| --- | --- | --- |
| Completed controls / accepted substeps | 238 / 2,142 | 238 / 2,142 |
| Simulated duration | 7.941267 s | 7.941267 s |
| Authored task success | No | Yes |
| Failed criteria | Mean and peak root tracking | None |
| Root error mean / peak | 5.884286 / 10.001390 model units | 0.021975 / 0.056608 model units |
| Angular error mean / peak | 0.336009 / 0.562267 rad | 0.145736 / 0.386686 rad |

These compare explicitly different control authority, not two learned policies.
Source scale remains a supplied camera prior; neither trial validates physical
ground support, body anatomy, contact or unassisted human balance.

Total: **17.484148 s** parent /17.007824 s worker, **960** live+replay controls,
**8,640** native substeps, 1,216 forwards/240 resets, **512,344,064 B** observed
child peak RSS. Two generated-video integrity decoders ran; no source-video
decoder, inference, training, provider call or network. All aggregate bounds
passed: **658 files /144,869,914 B**. No native rerun is needed.

- Study seal: `cd53ccda6f9f180e3afb07223f2252ec85ec6d67c7a7cb46ecd829e849c68bc3`.
- Result: `b3d09a75ffc7b9d3ace7717dcc1c5de5081f085e7b3541b13272399882c918b3`.
- Process receipt `.local/camera-shared-play-process-01.json`: `96e94d0982229c969181c350a3b03bef86a292c38ff746f1806b13c53cd8d0b0`.
- Roots-off run seal: `2d37cd2c1c89a5e50fe159836bdc9f1385d72b1c7e09a6cd628b7b807d32af64`.
- Roots-on run seal: `12ba0a99514e06f6e8ac8a63895a16c3fc739ec30596c236eed0ddf9e8b6391b`.

Ready console: `.venv-lab/bin/python -m weft lab-console --resume artifacts/camera-shared-play-01/roots-on/lab`.
Replace `roots-on` with `roots-off` for the unassisted-force comparison. These are
local saved examples, not files guaranteed in a fresh clone. Next product work
is useful unfamiliar-video ground/equipment recovery and meaningful learning;
do not redefine this playable source case as completion of the combined goal.

Git: the 31 reviewed product/test/document paths are published at
`cb5f8766ea6a675964716e54c6512f316c12c664`, with the exact private remote verified
on October 31, 2026, 3:16 PM PDT. Owned paths and index were clean, stash empty,
and all 1,249 prior unrelated avatar status entries remained preserved.
Receipt `.local/camera-lab-publication-01.json`, SHA
`f60b6ba8a4058e46bcb5474e8cf49c9a6ac774aba9a6210d6ee56a371b0e73fa`.
This small follow-up documentation checkpoint records that publication; no
product file changed afterward. Remote main was refreshed before publication
and needed no merge. No server, browser, deployment or paid call occurred.


## Independent saved-result audit — reconstruction lane

Completed October 31, 2026, 3:14 PM PDT. All 68 checks passed in
0.209753 seconds. One streaming hash pass covered all 658 new files /
144,869,914 bytes, including media bytes without decoding. Total reads including
process, plan and frozen-runtime metadata: 145,022,083 bytes, below the
192 MiB / 2,000-file / 30-second bound. Nine complete new/copied-candidate seals
match; no missing or extra output member was found. No native, adapter, model,
network or paid call ran, and no original source was reread.

- Command: `python3 -B .local/audit_camera_shared_play_01.py`.
- Audit receipt: `.local/camera-shared-play-saved-audit-01.json`, SHA
  `75cf98ee3660af0a44c356af337d600a4b9778863dbd9f649a277606b31ba363`.
- Audit helper SHA:
  `f4797e1ebf1c3c367e1da2ae32ce3b6828a4a490b2e181c74533cd7ced7ed04e`.
- Acceptance seal:
  `cd53ccda6f9f180e3afb07223f2252ec85ec6d67c7a7cb46ecd829e849c68bc3`.
- Process receipt:
  `96e94d0982229c969181c350a3b03bef86a292c38ff746f1806b13c53cd8d0b0`.

Both modes preserve one identical actual v6 scene, the original body bytes and
1/30-second body period, 1001/30000-second control period, nine substeps and the
full original 238-control interval. Each retains 240 source slots / 480 view
slots, 112 body poses and 128 unavailable body slots. All 18 selected historical
members match the inherited 242-member manifest; this does not freshly verify
its other historical members. Missing secondary media/rights remain unavailable.

Each journal has 242 events: start, two manual actions, one seeded reset and
238 authored actions. The saved exact-replay receipt checks all 240 actions and
242 events. Every journal state retains all eight camera identities, correct
clock/reward arithmetic and explicit root authority. The 34-file native closure
and full packaged implementation match the frozen runtime; protected scorer
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`
is unchanged. Resumed cancellation is separate from the completed recording.

Roots-off correctly fails root tracking: RMS 5.884285724825535 and peak
10.001390021509929, with zero root-force actions. Roots-on passes the unchanged
authored tracking gates: RMS 0.021974679231666266 and peak
0.05660835190979082, with explicit root assistance on all 238 authored actions.
Both complete the interval. This is successful software interaction/recording,
not unassisted balance, recovered support/contact or physical accuracy.

No audit failure or unexpected result. All jobs are terminal; no live handle.
Artifacts were not changed. This lane made no Git mutation and releases this
handoff section to root for publication; no additional run is needed.
