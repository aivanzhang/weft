# Toy-van full-interval development check

Status: completed October 28, 2026, at 3:45:18 PM PDT. The full-interval pipeline rejected this source as unsupported; no simulated scene or learned policy was produced.

The study uses the complete 354-frame, six-second original toy-van video by OrangeLark64, licensed CC BY-SA 4.0. See [source provenance](general-object-source.md). Source SHA-256: `caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8`.

## Frozen inputs

`artifacts/general-object-e2e-01/inputs/manifest.json` seals seven files. Its SHA-256 is `f17b72ce2de141f28d777b03568ebe4b09447fb9210d8d97ad4104780ee43013`. Preparation completed October 28, 2026, at 3:33:44 PM PDT, before tracking.

The original source/configuration/manual annotations remain unchanged. The derived configuration changes only `world.target[1]` from `0.7` to `0.3`: the authored target must fit the image-width-normalized scene height, `0.5625`. This is not an inferred physical property. Frames 0–353, initialization at frame 0, the original ROI, and default independent tracking remain unchanged. The pipeline's normal analysis downsampling is allowed; the source file is neither trimmed nor resized.

Five frozen approximate body-envelope centers retain ±16 source-pixel uncertainty. Frame 140 remains null because the original crossfade blends two temporal views. Null means unknown, not physical absence. Frame 0 is authored initialization and receives no tracker-accuracy credit. All six slots are reported, including missing detections; errors are conditional on accepted detections, never filled with zero for missing tracks.

## Bounded execution

Executed once after source-edit coordination (do not reuse this sealed output directory):

```sh
.venv-lab/bin/python scripts/studies/check_toy_van.py run --out artifacts/general-object-e2e-01
```

The script verifies frozen inputs, snapshots implementation files, and invokes `weft lab --profile appearance --episodes 100 --eval-episodes 20 --seed 1729 --no-serve`. It uses one local worker, the existing 360-second worker bound, and a 420-second outer deadline. No paid API or server is involved. The output directory cannot be reused for another execution. The final manifest uses artifact-relative paths.

Original crossfades, hand occlusion, 3D rotation, and scale changes stay in the study. There is no dedicated crossfade or hand-manipulation detector; an unrelated rejection must not be described as one. This is a development check, not physical validation or permission to change defaults or promote a model.

## Actual result

All 354 source frames decoded; interval 0–353 remained intact. The CLI exited 1 after 2.95 seconds with `UnsupportedVideo`: “Moving camera or widespread background motion; appearance tracks do not establish fixed-camera physics.” The study wrapper exited 0 because it successfully preserved and graded that failure; this is not pipeline success. Training and evaluation were not reached.

The saved trace contains 144 tracked frames, one initialization, two ambiguous frames, nine not-detected frames, and 198 lost frames. Accepted coverage excluding initialization is 144/353 (40.8%). Permanent loss begins at frame 156. The camera/background-motion check rejects the source before the coverage gate; the trace also falls below its 80% coverage requirement. The first motion warning is frame 60. This does not separately establish camera motion versus widespread scene motion.

| Frozen frame | Result | Source-pixel error |
| --- | --- | --- |
| 0 | Authored initialization; not accuracy credit | 0, unscored |
| 70 | Tracked | 2; within ±16 uncertainty |
| 140 | Tracker accepted a center despite ambiguous source crossfade | Unknown; null retained |
| 210 | Lost, although source object is readable | Not scored |
| 280 | Lost, although source object is readable | Not scored |
| 353 | Lost, although source object is readable | Not scored |

Only one of four non-initialization center slots was tracked. The conditional RMSE of 2 pixels describes that single accepted observation, not full-clip accuracy. The accepted center at the crossfade exposes an unresolved limitation; it is not evidence that the source edit was detected.

All implementation hashes and seven input hashes remained unchanged. Final artifact verification covers 93 files; manifest SHA-256: `5a31bca8df13b6f7559dce2c02b347de1bd167d94d3c06fb04ab3b6cf360b432`. The trace SHA-256 is `064b1c72e43945ac8714d309c9e5a29758cb6dd57fc330870cf41de6dd4378f9`. Inspect `report.json`, `lab/failure.json`, and `lab/run/appearance/appearance-tracks.json` inside `artifacts/general-object-e2e-01`. No thresholds, labels, defaults, or promotions changed.
