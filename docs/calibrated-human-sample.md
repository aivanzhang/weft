# Three-camera human-motion sample: calibration still unqualified

**The public three-camera human-motion sample is available at 6 and 30 fps. Both supplied calibrations fail the unchanged cross-camera consistency check. No body or learned policy has been built from it.**

**Follow-up:** a separate [two-view omitted-corner diagnostic and complete motion inspection](freemocap-body-evidence.md) now distinguish the failed monocular board placement from much closer stereo agreement. The new stereo diagnostic passes; the original check stays failed. Whole-body joint coverage and scene-derived floor placement remain insufficient for body admission.

Inspect the [two construction views](../artifacts/freemocap-test-inspection-02/construction-contact-sheet.png), [source provenance](../data/exercise/freemocap-test.source.json), [board-check results](../artifacts/freemocap-board-check-01/report.json) and [retained diagnostic alternatives](../artifacts/freemocap-board-diagnostics-01/report.json). Cam3 was undecoded at this initial stage. Its full-rate equivalent has since completed the [frozen comparison](freemocap-fixed-lengths.md) and is now exposed development evidence.

## Source and reserved view

The [official FreeMoCap test-data release](https://github.com/freemocap/skellysamples/releases/tag/test_data_v06_09_25) supplies three synchronized videos, camera parameters and board metadata. Its 37,864,781-byte ZIP matches the provider SHA-256 `543a38c86e591476975a8c3d8dd05c5243777a769c696e116c6aaea43f5170b1`.

The split was recorded before decoding pixels:

- **Cam1 + Cam2:** construction evidence. Both fully decode to 222 portrait frames, 720×1280, at 6 fps, with identical encoded timestamps.
- **Cam3:** reserved at initial inspection. Its full-rate counterpart was later opened under a frozen comparison protocol. The downsampled file is not a fresh holdout of that same recording.

The two inspected views show a board-calibration sequence, standing movement, a knee lift and the person leaving. This is a human-motion demonstration, not a labeled exercise routine. Preserve the full input, including inactive/absent-person frames; do not invent a continuous body or silently discard them.

FreeMoCap describes the test sequence as downsampled. Its encoded 37-second duration is retained; equal media timestamps do not independently prove physical synchronization. The [testing documentation](https://docs.freemocap.org/build/testing/) and current test fixture say no full-sample download source is known, while the public `sample_data_v06_12_25` release supplies a 166,958,457-byte ZIP. The larger archive was acquired in the follow-up below; the earlier discovery reports preserve what was known before that acquisition.

The repository carries an AGPL-3.0 notice, saved with the source. The seven-member archive contains no separate data-license notice. This work is local/private inspection, with no publication, account, installation, agreement acceptance or video upload.

## Calibration check

The supplied camera transforms use camera 0 as the origin, **not a gravity-aligned ground frame**. Translations are interpreted in millimetres and converted to metres for checking. Camera intrinsics include a negative radial-distortion coefficient; other supplied distortion terms are zero. No camera, scale, time offset or floor is fitted.

The [pinned provider board definition](https://raw.githubusercontent.com/freemocap/skellytracker/6f78af48b7a3c1809d110120a7cbd1d7f6120392/skellytracker/core/detectors/keypoint_detectors/charuco/charuco_board_definition.py) specifies 7×5 squares, 58 mm spacing, the `DICT_4X4_250` marker dictionary and a .8 marker-length ratio. Source code is retained as text, not installed or executed. Corner detection uses the already-installed OpenCV 4.14 implementation and its [official ChArUco method](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html). The discovery report's `source_commit` field contains a Git tree identifier, not a verified commit; the fetched file's hash remains pinned, and this labeling error is preserved rather than rewriting the sealed report.

The method and limits are saved before detecting corners on frames 0–99 of both construction views. All 200 view/frame slots remain, including missing detections. A board pose is fitted only to Cam1 pixels; its projected corners are compared with Cam2. Triangulated adjacent corner distances provide a separate metric check.

| Frozen check | Result | Requirement |
| --- | ---: | ---: |
| Poses with at least eight shared corners | 4 | ≥10 |
| Cam1 fitted-corner error, p95 | .926 px | ≤2 px |
| Cam2 projected-corner error, p95 | 32.59 px | ≤3 px |
| Cam2 maximum error | 33.30 px | ≤10 px |
| Triangulated 58 mm edge error, p95 | 1.84 mm | ≤5 mm |
| Positive bounded stereo depths | Pass | All compared corners |
| Overall | **Fail** | All requirements |

The four compared poses are frames 33, 34, 53 and 54. The other 96 prefix frames stay unavailable for this paired check. The result does not establish which combination of corner estimates, timing, scale, board shape or supplied camera parameters causes the mismatch.

A separate diagnostic reuses the saved detections without decoding video or detecting corners again. Both planar-pose solutions are retained, with no selection or parameter change. The alternate solutions have worse cross-camera errors; they do not repair the failure. Stereo reprojection RMSE ranges .108–2.587 px across these view/pose pairs, and board-plane deviations are roughly .4–1.0 mm. These narrower consistency results **do not override the failed frozen check**.

Three analytic OpenCV projection-convention checks recover known board placements to below 1e−9 m and cross-camera error below 3e−7 px. They check coordinate/API usage, not physical calibration or the entire study script. Evidence: `.local/freemocap-projection-convention-01.json`.

## Higher-frame-rate follow-up

A [timing diagnostic](../artifacts/freemocap-clock-diagnostic-01/report.json) retains the original 49 shared corners while testing nine fixed offsets from −3 to +3 small-source frames. Every nonzero offset loses some of those corners. None qualifies as a complete-cohort correction; no video is retimed. Half-frame alternatives interpolate only between observed endpoints with the same corner ID. Their lower errors on smaller subsets are not a passing result.

The [full public release](https://github.com/freemocap/skellysamples/releases/tag/sample_data_v06_12_25) was downloaded once: exactly 166,958,457 bytes, matching provider SHA-256 `c23d6a24f9ed8cca74bccf171128543d661b47e5780df5b446d60063ac4acf4f`. Cam1 and Cam2 each decode completely to 1,108 frames at 30 fps, 720×1280. Cam3 was still undecoded at this acquisition stage; its subsequent frozen comparison is documented above. This is denser sampling of the same recording, **not another independent video or subject**.

One bounded pass compares both complete construction streams with the small source and detects board corners on full-rate frames 0–499. It retains 1,000 detection slots and all five tested photometric phases, without human-pose inference. On the fixed 220-frame common cohort per view, full frame `5 × test_frame` has the lowest image error on 211/220 and 203/220 frames. This supports the expected downsampling correspondence; it does not prove synchronization between cameras. No timing adjustment is applied. Decoding/comparison/detection takes 4.250 s with two workers, excluding acquisition.

Both provider calibration files are checked against those **same saved full-rate detections**, using the original frozen method and limits. No new detections, parameter fitting or threshold changes are involved.

| Frozen check | Full-sample cameras | Test-sample cameras | Requirement |
| --- | ---: | ---: | ---: |
| Supported board poses | 18 | 18 | ≥10 |
| Cam1 fitted-corner error, p95 | .951 px | .931 px | ≤2 px |
| Cam2 projected-corner error, p95 | 35.46 px | 36.98 px | ≤3 px |
| Cam2 maximum error | 41.61 px | 43.45 px | ≤10 px |
| Triangulated 58 mm edge error, p95 | 2.55 mm | 2.45 mm | ≤5 mm |
| Positive bounded stereo depths | Pass | Pass | All compared corners |
| Overall | **Fail** | **Fail** | All requirements |

All 500 paired prefix slots remain for each condition: 18 compared poses / 234 cross-camera corners and 482 unavailable poses. Adjacent supported poses are correlated, not independent camera setups. The comparison takes .012 s, excluding the earlier decode/detection pass. More temporal detail resolves the minimum-pose-count shortfall but **does not resolve camera consistency**. Neither calibration is promoted, and existing exercise labs/defaults are unchanged.

## Evidence and next work

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| [Public metadata](../artifacts/freemocap-source-discovery-02/report.json) | 9 | `8e8393bf2d168c259e985ba2a2ade69f338aa08405bacf2e17db951278c12f2f` |
| [Pinned source and pre-pixel split](../artifacts/freemocap-test-source-01/report.json) | 12 | `524b46c95210866316ee42be1d2676387e6c1032628e1b85762ad3c9aecb3594` |
| [Aspect-preserving inspection](../artifacts/freemocap-test-inspection-02/report.json) | 14 | `973ed6e474235fbacab2394a4c96c453c9cc36fba941171739fdbd9118a0fad5` |
| [Board check](../artifacts/freemocap-board-check-01/report.json) | 31 | `d95cd1993836ee1b1a2af11970f0c8d426e2961c126b75e678009d72ff26c803` |
| [Pose diagnostics](../artifacts/freemocap-board-diagnostics-01/report.json) | 3 | `5b199f6da2ea72c015f25dedb3eb958dead0734ceb712eeaed10675b664a6322` |
| [Fixed-cohort timing diagnostic](../artifacts/freemocap-clock-diagnostic-01/report.json) | 4 | `fc625fa6587acf1660644f63bb4236cdf1efcbcc63a5c64c67189215d4fff446` |
| [Full-rate source](../artifacts/freemocap-full-source-01/report.json) | 12 | `515391a41d6d06a74d607a717e04af40f3c3d9ecd815aa854f9bf5557e36d4ea` |
| [Complete full-rate construction evidence](../artifacts/freemocap-full-evidence-01/report.json) | 24 | `4437e63bd85610258bbc255e7ad278759f562bbf84f7501e2d62d7733999b0eb` |
| [Both supplied-camera comparisons](../artifacts/freemocap-full-board-check-01/report.json) | 5 | `74a6e0e9a17f029ee6bc33050f219db9db3fffd99dabd01f5c77b571f67f4a48` |

The initial sandbox-DNS failure and stretched contact-sheet export remain saved. The corrected sheet reuses all ten original PNGs and complete decoder results; it does not repeat the successful video inspection. Board checking takes .928 s with two bounded workers, separate from acquisition and inspection. No new human-pose inference, Meta call, body build, policy training, browser or server ran.

The [follow-up](freemocap-body-evidence.md) completes the board-placement sensitivity diagnostic and two-view motion extraction without changing these results. The subsequent [frozen Cam3 comparison](freemocap-fixed-lengths.md) passes its stereo-board check but rejects the fixed-length joint correction. The [three-view study](freemocap-three-view.md) uses this now-exposed view as construction evidence. Missing/conflicting landmarks and independently supported gravity/ground remain unresolved. Existing [exercise sandboxes](selective-crop-exercises.md) and the [Heavy squat walkthrough](heavy-pose-model.md) are unchanged. This source does not establish anatomical accuracy, actual muscles, contact forces or physical changed-action predictions.
