# Optional Full pose model and a new jogging sandbox

The optional Full pose model corrects the previously inverted jogging frame and produces a new, recordable manual body. **It is not a universal improvement:** visibility coverage regresses on some inputs, and the new body still floats above the supplied ground. Lite remains the default.

The subsequent [two-view-root candidate](two-view-root.md) improves this body's position while preserving its monocular shape. It has its own playable lab and recording; the original single-view results below remain unchanged.

## Try the saved candidate

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/full-pose-ground-jog-lab-01
```

Open the printed local URL. Inspect the source poses, use manual joint/release/push/support controls, and record with **Record & start** / **Stop & save**. This candidate has no learned policy. Muscle bands remain illustrative motor-command proxies, not measured muscle activation.

- [Source/body panels](../artifacts/full-pose-jog-interaction-01/index.html), including [the corrected frame 200](../artifacts/full-pose-jog-interaction-01/reference-0200.png).
- [Recorded action replay](../artifacts/full-pose-jog-interaction-01/session/replay.html) and [video](../artifacts/full-pose-jog-interaction-01/session/video.mp4).
- [Paired comparison results](../artifacts/full-pose-jog-paired-01/report.json).

This is **one-camera pose estimation plus explicitly supplied camera/ground calibration**. It is not the partial two-view skeleton converted into a body. The two-view experiment remains separate; no hidden joint was relabeled observed.

## What caused the inverted pose

The error already exists in Lite's raw image coordinates. At source frame 200, predicted shoulders are below the hips and predicted ankles are near the chest, despite high confidence. Frame 202 returns to ordinary ordering. [Labeled source-image diagnosis](../artifacts/jog-image-pose-diagnosis-01/raw-image-labels.png) and all 8,640 three-view/joint slots are retained. No manual joint correction, timing fit or simulator adjustment fixes this example.

Full estimates the frame-200 torso upright. Its native body vertical direction changes from −0.761 to +0.980. The separate two-view Full reconstruction also recovers an upright torso (+0.987), where the Lite pair rejected both shoulder rays. These are source-specific improvements, not a rule that all valid exercise poses must be upright.

## Explicit model choice

[Google's official pose guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) publishes Lite, Full and Heavy bundles. This earlier study added **Full float16/1**, pinned to 9,398,198 bytes and SHA-256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`. The later [optional Heavy comparison and squat sandbox](heavy-pose-model.md) are separate; Lite remains the default. Identity comes from bytes, not the filename. The [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) lists Apache 2.0 and describes sensitivity to orientation, scale and occlusion; accurate metric depth is outside its stated scope.

```sh
.venv-lab/bin/python -m weft fetch-pose-model --variant full
.venv-lab/bin/python -m weft lab \
  --video /path/to/exercise.mp4 \
  --profile exercise --body-model mobile \
  --pose-model .local/models/pose_landmarker_full.task \
  --out artifacts/my-full-pose-lab
```

Use a fresh output directory. The ordinary single-video lab still has its existing camera, missing-pose and physical-model limits; supplying Full does not add camera calibration or recover loads. `--pose-mode image` is a separate explicit choice, used for the jogging study. No inference video is uploaded by this local model path.

`fetch-pose-model` without `--variant` and labs without `--pose-model` still use Lite. Unknown variants, wrong hashes/sizes, conflicting existing models, oversized downloads and concurrent destination creation are rejected without replacing the existing file. No automatic fallback or per-frame model selection is introduced.

## Complete-input comparison

One predeclared Full candidate was tested on four complete inputs representing three movements: the two construction views of jogging, a bodyweight squat and a kettlebell push-up. Modes, sampling times and identity settings match the saved Lite baselines. The third jogging camera stays outside inference/construction. Two bounded CPU workers completed in **27.836 seconds**; no camera/reference joints, manual labels or floor scores entered inference.

Counts below are twelve-major-joint image observations with confidence ≥0.5 inside the image, not anatomical accuracy:

| Input | Decoded frames / samples | Lite available | Full available | Whole-body visibility/geometry gate |
| --- | ---: | ---: | ---: | --- |
| Jogging, drone1 | 240 / 120 | 1,273 | 1,241 | Pass → pass; same eight absent samples |
| Jogging, drone2 | 240 / 120 | 1,159 | 1,159 | Pass → **fail**; usable whole-body samples 104→75 |
| Bodyweight squat | 68 / 34 | 374 | 387 | Pass → pass |
| Kettlebell push-up | 257 / 129 | 1,217 | 1,358 | Pass → pass, but usable whole-body samples 127→116 |

Push-up metadata still declares 259 frames; the independently audited 257 decoded frames remain the comparison domain. More confident individual joints do not imply more usable complete poses. The four inputs do not add demographic diversity or independent physical validation.

Full's median detector calls were 12.06–13.53 ms on these runs. Historical Lite calls were 11.48–12.47 ms. Different run times and concurrent workloads prevent treating this as a controlled speed comparison. Complete inference stages took 10.42–11.80 s, including preprocessing, decoding and preview creation. No sustained browser/RSS claim is made.

## Fixed-slot jogging results

The candidate is constructed before reference comparisons; no alignment or threshold is fitted. Provider joints/cameras are reconstructed, not independent laboratory truth. Paired results use the same slots for both models:

| Comparison | Paired slots | Lite | Full |
| --- | ---: | ---: | ---: |
| Raw single-view absolute 3D RMSE | 1,344 | 0.4274 m | 0.3229 m |
| Native body absolute 3D RMSE | 1,344 | 0.4299 m | 0.3220 m |
| Native body maximum 3D error | 1,344 | 2.4457 m | 0.7578 m |
| Two-view absolute 3D RMSE | 967 | 0.08389 m | 0.08238 m |
| Two-view third-camera projection RMSE | 902 | 23.7709 px | **23.9036 px** |

The two-view Full candidate loses 66 previously available major-joint slots and gains 33, for 1,000 estimates versus Lite's 1,033. The separate projection comparison loses 33 and gains 33 slots. Slightly worse paired third-camera error and reduced coverage prevent claiming a blanket two-view improvement. Original-file cross-camera timing remains an explicit assumption, as in [the original two-view study](multiview-pose.md).

## Manual body and recording verification

The new camera-aware body retains 112 estimated poses, 120 unsampled frames and eight unavailable samples across all 240 source frames. It passes the unchanged 112-source/239-grid penetration checks, but **all 112 poses still have positive ground gaps**: minimum 0.1336 m, median 0.2954 m, maximum 0.4676 m. These are native collision-surface gaps, not verified airborne/contact phases. The original Lite candidate remains saved.

The source body is built from frozen Full evidence without repeating inference or training. `camera-body` now accepts a sealed `task-proposal.json` when no source `policy.json` exists. The selected manual joint is explicit; no policy weights or performance are fabricated/transferred. Invalid control joints fail before a lab is published.

The actual recording contains **39 actions, one reset and 41 states/video frames**. Every state, reward and termination flag replays exactly. Across 120 in-process step/state/JSON calls, median latency is 0.789 ms and p95 0.803 ms, maximum payload 35,182 bytes; this excludes HTTP, browser painting and sustained idle behavior. Source/body panels were visually inspected without browser interaction or starting a server.

**74 scoped checks pass:** 34 model/inference tests and 40 camera-body/ground tests, including untrained-source recording and invalid joint rejection. Earlier two-ray checks remain unchanged and were not rerun. Evidence is in `.local/full-pose-model-tests.xml` and `.local/full-pose-body-tests.xml`.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Raw image-label diagnosis | 5 | `6a43642cbd426732d91b4808d12824e3c17c320f7ff492485f569d007cc73b7f` |
| Official model/docs acquisition | 6 | `32951d51ed21388f02429b1664e06b4d8ff715ac44a49a1c233cced423bd5075` |
| Four-input Full comparison | 151 | `af85c63769391a474e1e8950d8b68c6d385f7e4fece970da7975c9bd0341daba` |
| Full two-view comparison | 31 | `f8d302915b5f17c0fcab0797493fdcacb71b803af1466baf3bc433e52a1f9cff` |
| Untrained body preparation | 125 | `bdf0d1bb241c6803f850fb841cb1e9a32728a0a1bb2d19f7b66dc5e11cad26f2` |
| New manual lab run | 235 | `6db1d6b246a7d8a03d6479d214fc7187fdbf0a0220a2db48dda8c614fe44b90b` |
| Exact interaction/recording | 22 | `ddbeda251a680d74d52d3bd539085952efcb83bef9c7d32bf1e3b289fcafcd4a` |
| Fixed-slot Lite/Full comparison | 4 | `546fad0443cfe906f93174180b9caae3561a6c38a6ce442746eca59695092442` |

Recording session: `session-17521f14379e`, seal `b8329fbb7506a6b78f08015549f30a1dc5d8b43fb345e6467140fa885a4b66d2`. Reuse its existing export with `check_camera_body_lab.py --recording` rather than creating duplicate verification runs.

At **October 28, 2026, 11:59 PM PDT**, Meta spending remains six settled requests, **$0.166152 of $5**, with $4.833848 remaining and no unsettled reservation. This work downloaded 10,939,566 bytes of free public model/documentation data and made no paid call. There is no deployment, new training, automatic default change, browser/device validation or physical promotion.

Remaining work includes supported cross-view timing, partial-joint/body integration, scale/root-height and foot contact, broader real ball/general-scene reconstruction, and independent changed-action validation. Better pose estimates and exact recording do not complete those requirements.
