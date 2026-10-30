# Original-point flow: accurate control, worse real coverage

Verified October 29, 2026, **6:08 PM PDT**. This experimental method does **not** replace the default tracker or produce a qualified 3D world. It improves the known-warp control but loses original points earlier on the real recordings. That failure is retained, not hidden by reseeding or relaxed gates.

## Inspect the result

Open the local [study index](../artifacts/point-flow-diversity-01/index.html). Each case includes an original/current-image video, six previews, complete per-frame point records, and every attempted shape window. These generated artifacts are local-only, not included in a clone.

- Study: `artifacts/point-flow-diversity-01`.
- Manifest: `d66790112a92da30173664a68be1145457a9e151bbc44265b7cef8821c8ac57b`, **84 hashed files** plus the root manifest.
- Audit: `.local/point-flow-diversity-verification-01.json`.
- Inputs: [surface features](surface-feature-diversity.md) and [descriptor histories](feature-track-diversity.md), unchanged. Five views depict four already exposed recordings; none is fresh evaluation evidence.

## What changed

`OpticalPointHistories` in `src/weft/point_flow.py` follows only the original, distinct saved SIFT seed locations. Local optical flow proposes an adjacent location. Reverse flow, local appearance and affine alignment to an **unchanged original image patch** check that proposal. Rejected points end permanently; returning objects cannot revive their IDs.

The bounded contract retains at most 128 seeds for 500 frames, with source images at most 4,194,304 pixels. A skipped frame or time gap above 0.2 seconds ends active histories. Every declared seed and rejection remains in the journal.

- Local forward flow: 21-pixel window, three pyramid levels, 30 iterations.
- Reverse check: known prior point, full-resolution local refinement, maximum one-pixel return error.
- Local appearance: 9-pixel patch, correlation at least 0.9, standard deviation at least four.
- Original anchor: 31-pixel patch, fixed inner 21-pixel template, affine masked ECC alignment and independently recomputed correlation at least 0.9.
- Mask support: 16-pixel inset. Original patches are not clipped to whichever pixels happen to match later.
- Maximum original-anchor correction three pixels; accepted affine singular values 0.5–2 and axis ratio at most three. Histories cannot collapse within 0.75 pixels of one another.

No detection, pose/segmentation inference, source clock, scene schema, independent scorer, physics, dependency or ordinary-lab default changed. Appearance agreement does not establish physical point identity, rigid shape, source-edit continuity, human joints or anatomy.

The implementation uses the installed OpenCV 4.14.0 `findTransformECCWithMask` API. [OpenCV tracking documentation](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html) and [upstream ECC implementation](https://github.com/opencv/opencv/blob/4.x/modules/video/src/ecc.cpp) informed the mask/coordinate checks; upstream code is not asserted to identify the installed binary.

## Real-video outcome

The study retains **684 source frames, 1,090 object/frame slots, 221 original seeds and 27,988 seed/frame rows**. Only 71 seeds initialize with full patch/mask support. There are 986 subsequent accepted point pairs. All real objects have zero surviving original seeds by their final frame.

Counts below exclude initialization; the comparator is the prior descriptor method's opening-reference coverage, not ground-truth accuracy.

| Source / object | Later frames | Any flow seed | At least six flow seeds | Prior method: any opening anchor |
| --- | ---: | ---: | ---: | ---: |
| Side ramp / green box | 36 | 26 | 25 | 31 |
| Front ramp / wood | 35 | 2 | 0 | 11 |
| Front ramp / green | 35 | 21 | 0 | 35 |
| Toy van | 353 | 43 | 2 | 221 |
| Squat / person | 67 | 7 | 3 | 67 |
| Each of three balls | 140 | 0 | 0 | 0 |

Full original patches and mask support reject many seeds. Motion blur, changing visible appearance and mask support end the rest. No seeds exist for the small textureless balls. Later-frame agreement between the two methods has 698 comparisons, p95 0.676 pixels and maximum 1.721 pixels; agreement is **not** independent physical accuracy.

The unchanged [relative-shape estimator](track-shape-diversity.md) receives fixed, nonoverlapping 8- and 32-frame windows, preserving every suffix:

| Window length | Incomplete tracks | Insufficient depth motion | Too few frames | Qualified shape |
| --- | ---: | ---: | ---: | ---: |
| 8 frames | 129 | 3 | 8 | 0 |
| 32 frames | 36 | 0 | 3 | 0 |

All **179 windows** are retained. No new world is generated or promoted. The 32-frame experiment does not recover missing measurements by lengthening the window.

## Known-warp control

Both conditions reuse the prior study's original texture, mask, seed locations and 75 known transforms: 2.5-degree increments through 185 degrees. This is a planar software warp, not a measured physical object.

| Condition | Checked point errors | Final active seeds | Error p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Continuous | 4,176 | 55 | 0.03497 px | 0.05099 px |
| Missing masks at frames 31–32 | 1,756 | 0 | 0.03506 px | 0.04917 px |

All 5,932 errors are independently recomputed. Missing masks permanently end the histories even after the object reappears. These controls pass the frozen coverage/error gates; their visible point cohorts differ from the previous descriptor experiment, so they are not a matched real-video accuracy comparison.

## Verification and retained failures

Seventeen tests pass in `.local/point-flow-tests-05.xml` (0.14 seconds): known translation and 40-degree rotation, mask loss, no revival, source gaps, texture/appearance failures, identical-looking replacement limitations, bounds and mutation isolation. The unchanged rotation gate requires at least four surviving seeds and every published point below one pixel error.

Earlier reports `point-flow-tests-01.xml` through `-04.xml` remain. Coarse reverse searches jumped to unrelated patches; seeded full-resolution inverse checks restored coverage but translation-only chaining drifted 1.723 pixels. Unmasked original-patch alignment rejected valid rotations. Fixed inner-template masking solved that numerical failure without loosening the tests.

The independent audit checks all **84 new + 91 feature + 71 history files**, seven used source snapshots, source identities/clocks, all original lifetimes and all shape windows. It decodes all 684 source and 684 exported frames; recomputes 986 local/original-patch correlations, 5,932 analytic errors and saved shape projections without rerunning either tracker or the estimator. Maximum correlation-accounting difference is 3.28e−7; projection difference is 1.14e−13 pixels.

Two verifier implementation failures are preserved in `.local/point-flow-verifier-attempt-01.log` and `-02.log`: JSON loses the shape of an empty `(0, 2)` point array, and float32 half-pixel addition differs from float64 by at most 0.0000593 pixels. The verifier now checks the recorded arithmetic and independently requires the unrounded return distance to remain at most **one pixel**. Saved evidence and tracking thresholds are unchanged. Attempt 03 passes in 1.486 seconds.

Original/current side-box and squat previews were inspected. The vanished exercise points are shown as absent, not drawn as continuing trajectories. This is artifact inspection, not browser/device QA.

## Work and cost bounds

The study took **6.126 seconds**, two workers, with 60 seconds and 96 MiB per real case / 384 MiB total bounds. Output before the final report/index/manifest was 31,492,859 bytes. There were no new SIFT detections, model calls, downloads, simulations or paid requests.

Tracking includes mask loading and initialization, not export or UI response time:

| Case | Calls | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Side ramp | 37 | 23.017 ms | 27.230 ms | 28.207 ms |
| Front ramp | 72 | 1.822 ms | 9.279 ms | 12.468 ms |
| Toy van | 354 | 1.420 ms | 7.726 ms | 14.291 ms |
| Squat | 68 | 1.113 ms | 8.430 ms | 16.794 ms |
| Three balls | 423 | 0.271 ms | 1.390 ms | 4.211 ms |

No sustained UI-performance claim follows from these timings. No A job, server or trainer remains from this experiment.

## Reproduce only after changing the hypothesis

```sh
.venv-lab/bin/python -m pytest tests/test_point_flow.py
.venv-lab/bin/python scripts/studies/check_point_flow_diversity.py --out artifacts/point-flow-diversity-new
.venv-lab/bin/python scripts/studies/verify_point_flow_diversity.py --study artifacts/point-flow-diversity-new --out .local/point-flow-verification-new.json
```

Do not rerun the completed experiment unchanged or relax source gates to manufacture a candidate. Keep the current tracker. Next integration work needs source-qualified person/object identities, explicit coordinate/clock alignment and complete missing observations for B's shared-world constructor; physical qualification still needs stronger independent geometry and changed-action evidence.
