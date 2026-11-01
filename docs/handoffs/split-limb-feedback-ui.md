# Split arm/leg feedback: saved interface verification

## Result

The actual initial, completed-trial and fresh-resume summaries pass the existing
scripted interface checks. The frozen interface shows 22 controls, **Separate
arm/leg feedback**, failed training limits and 0/3 independent evaluation starts
for every controller. The completed live task remains **not passed**, separate
from the successful saved replay: 266 actions, one reset, 268 events and 257
verified learned actions. An impossible learned-action count is rejected.
A captured policy-step request has the expected shape but is not executed.

The lab is `artifacts/split-limb-feedback-play-05/lab`; its session is
`session-1dcf521d278e`. The whole attempt05 remains failed and has no outer success
seal. This check joins its exact saved captures and completed recording with the
separately successful `artifacts/split-limb-feedback-resume-01` proof. It never
synthesizes a replacement successful attempt05 result.

## Evidence and bounds

- Harness: `.local/split-limb-feedback-saved-ui-01/check-03.mjs`, SHA
  `051a57d2ae2a249ba7aa3ff7dadd9684021001a3e8adcef31ba7470a825f4ce0`.
- Plan: `execution-plan-03.json`, SHA
  `734f790e60e9e359a9ab35652e1669e2a8fc01abe3da428766f6b6f8dde5ae4e`.
- Passing receipt: `receipt-03.json`, SHA
  `720d472c4404895b54cafc95da972bbafcd1539de0c1f7b5b244e73071c06a42`.
- Actual check: **21 reads / 3,150,500 bytes / 0.421626 seconds**, exit 0.
  End-of-process RSS was 170,590,208 bytes, not a peak or browser measurement.
- Bounds: 10 seconds, 32 reads, 8 MiB selected bytes, 128 MiB Node old-space.
  No source/journal/media reads, native simulation, replay, server or browser.
  Original source, complete run and six recording members are verified by the
  separate unchanged lab/receipt checks described in the [play note](split-limb-feedback-play.md).

## Preserved checker failure and rendering limit

Check02 failed its broad zero-write assertion: **400 assignments**, not zero.
It had also watched two legacy generic labels that the v5-specific test did not
cover. Their existing callsites unconditionally assign text. Check03 changes only
the checker: it measures each field and keeps the limitation visible. The
product, runtime, task and scorer remain unchanged.

Across **200 identical updates**, the five policy fields and two replay/count
fields receive **zero assignments**; joint and score lists retain their identity.
The generic task-outcome and recording-evidence-status labels each receive
**200 assignments**. The 200 updates take 369.360334 ms in a scripted DOM; this
is not browser layout, responsiveness or sustained resource verification. No
claim of globally zero writes or an implemented performance fix is made.

The failed `receipt-02.json` (SHA
`01848537285e2d627adc00341293a19e8e078da7650690ed5452589a444b3b14`)
remains intact: 21 reads / 3,149,344 bytes / 0.393293 seconds. Its lost in-memory
cache required one bounded reread after the different checker hypothesis. Both
checks together read **6,299,844 bytes in 42 operations**; no completed native
work or source processing was repeated. The original unexecuted harness remains.

The review agent stopped after preparing check03; root reviewed its exact delta,
froze plan03 and completed the check. Appearance is excluded from this scripted
harness. Browser layout and avatars are not verified by this result. The full
video-to-world goal, unfamiliar-video reliability and physical accuracy remain open.
