# Exercise pose identity investigation

## Result

The extractor exposes two explicit modes: `video` (the backward-compatible default, with temporal tracking) and `image` (independent sampled frames). It does **not** automatically retry, choose between results, merge, select, or discard competing returned poses. The existing rule rejecting any multi-pose frame remains unchanged.

Explicit IMAGE inference removes the jumping-jack clip's multi-pose failure in the measured development run but regresses the single-leg-squat clip. VIDEO retains the passing single-leg result. Lunge remains rejected in either mode. Neither mode establishes person-count accuracy, accurate anatomy, successful body fitting, or physical transfer.

## Source inspection before changes

All 42 frames previously flagged as `multiple_people` were visually reviewed from decoded source images: two lunge frames and 40 jumping-jack/burpee frames. Each shows one visible exerciser. The frame-level labels, source checksums, and image checksums are in [ambiguous-person-count-v1.json](../data/exercise-annotations/ambiguous-person-count-v1.json). These are development labels, not a final holdout or landmark ground truth.

Evidence is preserved in `artifacts/pose-identity-review-01/`: raw frames, raw contact sheets, overlaid competing skeletons, and `registry.json`. Original full-pipeline failures remain in `artifacts/video-diversity-e2e-01/`.

The extra skeletons include inverted and partial-body estimates during floor transitions. They are not consistently close duplicates. Broad overlap-based suppression or choosing the most confident skeleton would conceal uncertainty and could discard a real second person; neither rule was implemented.

## Inference change and limits

[MediaPipe's documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) describes tracking in VIDEO mode as a way to avoid triggering the model on every frame. Independent IMAGE inference removes that temporal state. The observed difference supports a temporal-tracking contribution to these failures; it does not prove that every extra hypothesis came from a stale region.

The extractor API is `inspect_exercise(..., pose_mode="video")`; callers must explicitly select `pose_mode="image"` to use independent inference. Invalid modes fail before model or output access. Mode choice is an authored development setting, not an accuracy-driven automatic selector.

The model, two-pose maximum, detection/presence thresholds, frame sampling, quality thresholds, and raw competing-pose retention are unchanged. Evidence records resolved `running_mode`, `temporal_tracking`, and `identity_status`. The legacy `multiple_people` status remains for compatibility, but its error text explicitly allows competing hypotheses rather than claiming verified people.

One returned pose is **not** proof that only one person is visible. Missing people remain a known model limitation. Source video must meet the single-person input requirement; automatic person-count validation is not established.

## Direct shipped-extractor verification

`artifacts/pose-image-mode-verified-01/` contains an immutable extractor source copy, its checksum, per-case evidence, poses, joint estimates, previews, and results. These runs used the first modified extractor (IMAGE-only), not a monkeypatched detector. They did not run the full lab pipeline. That source copy predates the explicit-mode API; it is historical evidence, not verification of the final configuration wiring.

| Source | Original VIDEO result | New IMAGE result | Usable samples | Unresolved multi-pose samples |
| --- | --- | --- | --- | --- |
| Lunge/twist | Rejected | Rejected | 290/308 (94.2%) | 5, previously 2 |
| Jumping jacks/burpees | Rejected | Pose gate passed | 564/678 (83.2%) | 0, previously 40 |
| Kettlebell push-up | Pose gate passed | Pose gate passed | 127/129 (98.4%) | 0, unchanged |
| Single-leg squat | Pose gate passed | Rejected | 115/118 (97.5%) | 3, previously 0 |

Jumping-jack evidence still includes 71 low-confidence frames, 37 missing poses, and six invalid geometries. None were filled in as observed poses. Lunge's five ambiguous frames still fail the gate despite high overall visibility.

The later full-pipeline regression in `artifacts/exercise-body-e2e-02/single-leg-squat/` found competing IMAGE hypotheses at frames 14, 16, and 18; all three decoded source frames were visually inspected and show one visible exerciser. Baseline VIDEO inferred 118/118 usable poses without competing hypotheses. This regression is why VIDEO remains the default rather than silently replacing it with IMAGE. Both run artifacts remain preserved.

Observed direct-run durations were 6.27 seconds for lunge, 9.76 seconds for jumping jacks, and 2.74 seconds for push-up on this machine. These single measurements are not a controlled performance comparison or full-pipeline latency claim.

The earlier monkeypatched probe is preserved separately in `artifacts/pose-image-mode-study-01/`; it is not substituted for shipped-code verification.

## Two-person negative controls

Five same-scale, side-by-side source-crop montages were visually reviewed and labelled as containing two separate visible bodies. See [two-person-controls-v1.json](../data/exercise-annotations/two-person-controls-v1.json); images and raw outputs are in `artifacts/pose-identity-review-01/two-person-controls/`. These are synthetic composites, not natural multi-person video validation.

Both IMAGE and the previous VIDEO detector produce the same outcomes: three `multiple_people` rejections, one low-confidence abstention, and **one incorrect single-pose result** (jumping-jack frame 616 montage). VIDEO was repeated for five frames per montage and retained these outcomes. The missed second person is an existing limitation, not repaired by this change. Earlier smaller-person montages all returned no pose and were insufficient person-count controls; their evidence was retained.

Unit regressions additionally enforce that separated, overlapping, identical, and lower-confidence competing estimates are all preserved and rejected. Detector interruption preserves completed samples and closes the detector. Mode tests verify timestamped VIDEO dispatch by default, explicit IMAGE dispatch, resolved provenance, and invalid-mode rejection. The focused exercise-vision suite passes 21 tests.

## Remaining work

- Obtain natural multi-person recordings with independently labelled count, occlusion, and identity changes; current negative controls are limited synthetic cases.
- Resolve lunge hypotheses with independently grounded evidence rather than looser acceptance thresholds.
- Verify downstream body fitting, interaction, and recording separately. A passed pose gate does not establish those results.
- Keep pose estimates, manually inspected person-count labels, and measured physical/anatomical truth distinct.
