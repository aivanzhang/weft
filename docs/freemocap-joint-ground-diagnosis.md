# Joint and ground diagnosis

**No new exercise body is admitted.** These checks narrow the causes of missing joints and reject unsupported floor estimates. The existing [playable squat sandbox](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/heavy-pose-model.md) stays unchanged.

The complete two-camera source, original 36,564 joint slots and previous failures remain in the [human-motion evidence](freemocap-body-evidence.md). Cam3 was undecoded during this diagnostic. The subsequent [frozen comparison](freemocap-fixed-lengths.md) opens it; it is now exposed development evidence, not final evaluation data.

## Inspect the source errors

- [Frame 416: standing, inconsistent left hip](../artifacts/freemocap-hip-inspection-01/original-hips-0416.png).
- [Frame 966: motion blur and overlapping arm/hip](../artifacts/freemocap-hip-inspection-01/original-hips-0966.png).
- [Frame 966 after crop inference, Cam1](../artifacts/freemocap-hip-crops-01/view-1/crop-proposal-0966.png) and [Cam2](../artifacts/freemocap-hip-crops-01/view-2/crop-proposal-0966.png).

The first two panels retain the original image estimates. Red means left, cyan means right, and an X marks inconsistent camera rays. Labels are model estimates, not measured joint centers. The crop images show separate proposals; they do not replace the originals.

## Left/right swaps and clock shifts are not general repairs

The [association diagnostic](../artifacts/freemocap-joint-association-01/report.json) retains all **3,324 sampled bilateral-pair slots**: 554 moments × six joint pairs. It reuses original geometry and adds 5,538 crossed-association ray pairs where all four image observations exist.

For hips, 38 moments pass the ray checks only after swapping Cam2 labels, but **170 pass only with the original labels**. Another 161 support both alternatives, 154 support neither, and 31 lack complete observations. Smaller ray error cannot establish anatomical side; no swap is applied. Frame 416 is selected by the largest error reduction among active cross-only cases, not by visual preference.

The [torso-clock diagnostic](../artifacts/freemocap-torso-clock-01/report.json) compares seven offsets from −6 to +6 original 30 fps frames on the same 1,296 joint pairs per offset. The cohort is present at every tested offset; no easier subset is selected.

| Cam2 offset, source frames | Torso reprojection RMS | Supported joint pairs / 1,296 |
| ---: | ---: | ---: |
| −6 | 14.93 px | 1,058 |
| −4 | 10.89 px | 1,102 |
| −2 | 7.44 px | 1,161 |
| 0 | **6.72 px** | **1,203** |
| +2 | 9.57 px | 1,139 |
| +4 | 13.58 px | 1,101 |
| +6 | 17.53 px | 1,064 |

The three fixed temporal blocks prefer +2, 0 and 0 respectively. No consistent nonzero offset improves the complete cohort. Original timestamps stay unchanged; this does not prove physical synchronization.

## Wider scene evidence still does not establish a floor

Room-edge tests use source frame 550, supplied camera parameters and fixed door-frame regions. They infer a direction only; even a valid direction would not supply floor height.

| Hypothesis | Observed result | Decision |
| --- | --- | --- |
| Separate vertical edges in both views | Per-view directions disagree **6.94°**, limit 2° | Reject |
| Merge occlusion-split fragments before fitting | Disagreement **13.09°**; support checks also fail | Reject |
| Match original carpet texture | 161 / 307 features, **zero mutual matches** | No plane fit |
| Normalize carpet perspective before matching | 0 / 46 features, **zero matches** | No plane fit |
| Bound detection to the carpet region before spending the feature quota | 13 / 46 features, **zero matches** | No plane fit |

Inspect the [grouped room edges](../artifacts/freemocap-room-vertical-02/grouped-edge-inspection.png). All detected fragments, links, rejected groups and axis hypotheses are retained. Group rejection is included in support denominators; removing a bad group cannot manufacture a pass.

The normalized search uses the previously **rejected red-cover plane only as an appearance-search prior**. It never substitutes that plane's depths for triangulation. Both floor masks are transformed into a common 34,630-pixel region; the descriptor ratio remains .7 in both matching directions. No old floor or camera check changes.

### Regional feature correction

[OpenCV's upstream SIFT implementation](https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp) limits features before applying the mask. A synthetic case on installed OpenCV 4.14.0 reproduces regional starvation by strong features elsewhere. The upstream branch is a reference, not a verified source identity for the installed binary.

`regional_features.py` bounds the original-pixel crop to 262,144 pixels, detects masked candidates, then selects at most 1,200 before computing descriptors. It neither blacks out source pixels nor silently downsamples. Image coordinates map back to the full source. Masking constrains feature centers, not every pixel contributing to a descriptor.

The real common-carpet crop uses 178,610 pixels including context. It recovers 13 Cam1 features but still no mutual matches. This is an implementation correction, **not a recovered floor**. The rejected plane and all earlier failures remain unchanged.

## Six crop proposals: mixed gains, no repair

[The crop test](../artifacts/freemocap-hip-crops-01/report.json) freezes three cases before inference: the cross-only hip case, an already exposed standing control, and the worst active raw hip error. Each view gets one crop using the existing .25 padding and original temporal anchors. The pinned Full IMAGE model runs six times in two bounded local CPU workers; no whole-video inference or video decoding is repeated.

| Source frame | Hip reprojection RMS, original → crop | Supported major joints, original → crop | Remaining problem |
| ---: | ---: | ---: | --- |
| 416 | 9.16 → 10.18 px | 9 → 9 / 12 | Gains left ankle, loses left knee; torso fails; no nearby supported root anchors |
| 550 | 6.11 → 6.08 px | 12 → 12 / 12 | Already supported control; not an independent accuracy gain |
| 966 | 24.95 → 13.01 px | 4 → 9 / 12 | Torso still fails; Cam2 crop has low confidence; later root anchor unavailable |

Both difficult cases remain unsupported under the existing rules. No failed sample is removed, no root is interpolated, and no crop replaces the original sequence. The six raw proposals, exact crop pixels, mapped coordinates and 99 selected joint slots remain inspectable. Runtime is 10.216 s including worker/model startup and export.

## Verification and cost

- Nineteen parallel-axis, twelve line-fragment and fifteen regional-feature tests pass: **46 checks**, with the earlier passing suites reused rather than rerun unchanged.
- [Final verification](../.local/freemocap-joint-ground-final-verification-01.json) checks nine manifests / **103 files**, all 118 current runtime hashes, six exact crop inputs, 594 mapped coordinates and 176 independent OpenCV reprojections. Maximum reprojection-accounting discrepancy is 1.71e−13 pixels. The original 115 extraction-time runtime files are unchanged; three modules are added.
- [Regional profile](../.local/regional-feature-profile-01.json): twelve repeats per case, real-crop p95 19.65 / 17.03 ms; maximum-area noisy fixture p95 42.92 ms. [Axis/fragment profile](../.local/room-axis-profile-01.json): five repeats, maximum 128-plane p95 8.74 ms and 128-fragment p95 6.57 ms. These measure CPU functions, not interface responsiveness or physical accuracy.
- Four new source frames are decoded for inspection, and six new crop inferences run. No third-camera access, new API call, upload, server, deployment, commit or push. All study workers ended. The Meta ledger remains **eight settled calls, $0.186152 of $5**, no unsettled reservation.

## Next work

1. The proposed fixed-length test is now complete and [rejected](freemocap-fixed-lengths.md). The [direct multi-view follow-up](freemocap-three-view.md) also fails to establish an accuracy upgrade. Seek stronger source-landmark evidence rather than treating fitted image agreement as anatomy; never fill missing support silently.
2. Obtain independently supported scene geometry for floor orientation and height. Raw, normalized and regional carpet searches have no matches; repeating those unchanged is not a next step. Preserve the failed red-cover and room-edge checks.
3. Qualify a complete body, contacts and changed-action behavior before presenting a new physically accurate environment. Cam3 has completed its frozen comparison and is no longer a fresh holdout. Ball/general-video failures and broader physical accuracy remain separate unfinished requirements.
