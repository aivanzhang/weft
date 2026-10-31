# Camera task controls and saved replay

## Status and ownership

Observed 2026-10-31, 3:12:57 PM PDT. Implemented, analytically checked and **verified against both saved native 19/22-control payloads**. Parent `/root` owns the combined native acceptance and release. This lane owns no running jobs, server, browser, provider calls or Git operations.

Released product/test files remain frozen at the versions used by the parent’s successful native runtime:

| File | SHA-256 |
| --- | --- |
| `src/weft/assets/lab.js` | `748387510b5ac77694a4b29c1cec6eedfa8f7783d968c91a34faacfc884eb925` |
| `src/weft/assets/shared-world-replay.html` | `57334bcc72a2ce5f2ca06d8947dd11114ee0cc84baace10588e8af09d2c20c20` |
| `tests/camera_shared_ui_smoke.mjs` | `125b28da569cf34bc57d0051196f7250c397075f9f12abd3037cdd46a9529770` |

No changes to task/scorer/gates, backend, capability producer, receipt producer, avatar assets, source evidence or old exports. Existing unrelated work was preserved. Git status/publication remains parent-owned; this note does not claim a clean worktree or remote checkpoint.

## Contract and implemented behavior

- Existing `control_profile='shared_exercise'`; actual `shared-world-v6`, `camera-shared-task-v1`, `camera-shared-state-v1` and `camera-shared-metrics-v1`. No legacy reprofile or task substitution.
- Manual, coast and authored body feedback use 19 joints, optionally 3 explicit root-force channels. No object forces, learned policies, foot controls or avatars. Stale policy/appearance metadata cannot enable them.
- Task/capability identity joins retain actual scene/task, coupling, source, body, source clock, metric context, original scorer, adapter and combined evaluator pins. Live outer state pins remain authoritative; inner engine snapshots have `profile` and coupling identity, **not** invented scene/task hashes.
- Source video, kinematic reference and simulated geometry stay distinct. Source time is the saved original offset plus simulation elapsed. The task's `start_offset_s=0` is relative to its reference, not the absolute original source time. The control/source period and unchanged body-model period are shown separately; the source-clock hash is an identity, never a timestamp.
- Malformed identity/profile/control/metric/clock shapes pause task controls and recording. Failure displays distinguish last valid geometry from later observed task time. Unknown initial mean errors remain unavailable, not zero.
- Only existing body tracking/floor gates apply. Equipment/contact rows are empty and hidden; an empty simulated inventory does not prove absence in the video. Supplied camera scale, ground, original body shape and authored mass are not measured anatomy or physical validation. Feet and ground/equipment contact remain unqualified.
- Geometry checks bound each displayed frame to 32 segments, 38 illustrative muscles and 256 contact records, with finite vector shapes. They protect presentation work, not replace native state validation.
- Saved receipt protocol is `camera-shared-full-state-replay-v1`, with all eight camera pins plus existing scene/task identities; policy and learned-check counts must be null. The view reads a saved receipt and does not rerun replay.
- Offline export reads explicit `camera_identity`, actual camera task, actual engine snapshots, camera metrics and engine-boundary status. It checks metadata relationships and pauses malformed playback. It does not claim per-frame cryptographic verification; the raw journal and saved replay receipt retain full authoritative identities. JSON parsing stays in the existing worker; selected-frame diagnostic checks are cached across view turns.

## Verification and preserved failures

Receipt: `.local/camera-shared-ui-checks-01.json`, SHA-256 `672a7f739b2a727dd316067a4c25042c88c01a1ec01cabcd54a8d142dcb6b8f1`. It preserves exact commands, exit codes, log hashes and limits.

Seven distinct camera groups now have passing evidence: explicit controls; malformed identities/metrics; source/failure clocks; serialized recording/reset; receipt pins; offline replay; repeated updates. These use hand-authored analytical fixtures, mocked DOM/Worker/Canvas and mocked requests—no native imports, media, network or browser. The reused harness's `actual_saved` flag only prevents feet-fixture augmentation; the new fixtures explicitly declare `evidence_kind='authored_analytical_fixture'` and are not saved native evidence.

- `.local/camera-shared-ui-analytic-01.log`: 6 pass, 1 test-only recording expectation failure. The test expected recording to precede an already queued reset; corrected to the existing serialized lifecycle.
- `-analytic-02.log`: corrected recording case plus 3 affected groups passed.
- `-source-offset-red-03.log`: 2 deliberate failures after correcting the fixture to a zero task-relative offset with nonzero source start. This exposed the UI's incorrect equality between the two offsets; product guard corrected.
- `-analytic-04.log`: all 7 groups passed after the source-clock fix.
- `-bounds-05.log`: final geometry limits and unchanged-text checks, 2 affected groups passed; other passing groups reused.
- Existing regression: 7 feet groups using `.local/shared-exercise-feet-ui-fixtures-02.json`, 7 person-only scenarios, and 4 selected common exercise/shared-world groups passed. Exact commands/logs are in the receipt; no unchanged full suite was rerun.

Final scripted measurement, 200 repeated updates:

| Analytical shape | Live | Offline view turns | Watched unchanged text writes |
| --- | ---: | ---: | ---: |
| Tiny body fixture | 17.11 ms | 4.16 ms | 0 |
| 32 segments / 38 muscles / 256 contacts | 149.01 ms | 108.34 ms | 0 |

These are mocked-script observations, **not browser responsiveness, render quality or native latency claims**. Existing shared-world framing is preserved; no browser visual acceptance occurred.

## Saved native payload acceptance

The parent reported its combined native run terminal and passed. No new native work was run here. Six additional checks passed: live initial/final payload, saved receipt view and exact offline bundle for both `roots-off` and `roots-on` under `artifacts/camera-shared-play-01`.

- Actual summaries enabled exactly 19/22 controls with real v6 engine profiles and no invented inner scene hashes. Learned/avatar controls stayed disabled. Source/body periods and unavailable initial mean errors displayed correctly.
- Both native trials completed the 7.9412666667-second interval with 2,142 valid/observed substeps. **Roots off did not pass** (root mean/peak limits failed); **roots on passed the simulated criteria with explicit assistance**. The UI preserves that distinction and does not imply physical validation or unassisted balance.
- Current offline script inspected every saved event, leaving the original exports unchanged. Exact saved bundle identity matched the actual capability identity. Both real receipt views passed the camera-specific pins/protocol/null learned-count checks; this view did not rerun replay.
- Check phase read 14,127,304 bytes from the captured summary cache, two HTML exports and two receipt views; elapsed 0.282334 seconds against explicit 32 MiB/20-second bounds. No media bytes were read or decoded. This is saved-data/script evidence, not browser visual or responsiveness acceptance.

Receipt: `.local/camera-shared-ui-actual-payload-01.json`, SHA-256 `9aff45b1aa06ff6b573f0dc92a5ee0bd5fcda7a316c9f68dc122f942c1162a6c`.

Command: `node .local/camera-shared-ui-actual-01.mjs > .local/camera-shared-ui-actual-01.log 2>&1` (exit 0). The check helper SHA-256 is `1615ade2bb25aff47179ba69e02e077f770f8d8c5da5f9e799e3d7c752067418`; cached summary input SHA-256 is `18841cb5a753d0cd31aaa8fd8742b1fbadf9a3181cbfce171cd8ce3e63989911`. Original summary file-byte hashes are preserved in the receipt. No product/test changes followed native freeze.

## Next safe action

- [x] Parent’s single frozen native run completed; no edits raced it.
- [x] Actual 19/22-control saved summaries, receipt views and complete offline event bundles passed presentation checks without source/model/physics reruns.
- [ ] Parent completes publication and updates the combined handoff. This UI milestone alone does not establish generalized reconstruction, recovered feet/contact or independent physical accuracy.

The new test module exports `cameraFixture`, `liveFixture` and `replayFixture`. The analytical suite is `node tests/camera_shared_ui_smoke.mjs`; optional trailing function names select affected groups. Backend/state/recording owners retain their files and acceptance obligations. Reuse this passing evidence unless relevant inputs change; do not repeat the native run for a presentation-only change.
