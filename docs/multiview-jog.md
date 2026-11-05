# Jogging video: sandbox and 3D reference comparison

The complete jogging clip produces a playable mobile-body sandbox in explicit IMAGE mode. Its recording replays exactly, but missing poses, 3D disagreement and floor intersections prevent an accurate-reconstruction claim. This is a separate development case, not a new pass in the original 15-video suite.

## Open the results

- [Source and pose overlays](../artifacts/multiview-jog-comparison-01/index.html): cyan is the provider's projected 3D reference; orange is Weft's raw image pose. Both are estimates.
- [Recorded sandbox interaction](../artifacts/multiview-jog-interaction-02/session/replay.html): rotate, scrub and inspect the saved simulated states; this is playback, not live control.
- [Recorded video](../artifacts/multiview-jog-interaction-02/session/video.mp4).

For live control and a new recording, start the existing lab yourself:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/multiview-jog-image-01
```

Open the printed local URL, choose the manual sandbox, then **Record & start** and **Stop & save**. Resuming does not repeat pose inference or training. The default learner controls one selected joint; it has not learned to jog. Muscle bands remain illustrative.

## Source and reference quality

Credit: **Autelica (autelica.com)**. The [jog-turn sample](https://autelica.com/sample/) is [CC BY 4.0](https://autelica.com/sample/files/LICENSE.txt); this permission does not cover the separately licensed hands or pilot datasets.

One original drone video and its small reference files were downloaded: 11,454,809 bytes total, no paid service. The video is unchanged: 240 sequentially decoded frames, 1920×1080, 30000/1001 fps, 8.008 seconds. Exact source times are retained, including the unsampled final frame. [Source identity and attribution](../data/exercise/autelica-jogturn.source.json).

The provider estimates camera poses, synchronization and 3D joints from three moving cameras. Scale uses stated subject height; the ground is aligned using stance feet. These are reconstructed references, **not independently measured motion capture, calibrated anatomy or force/contact measurements**. Only drone1 video entered Weft inference in this original study. Provider joints and camera parameters entered its separate comparison, not body generation, training or repair. A later [camera-aware candidate](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-aware-body.md) explicitly uses camera parameters as additional construction input; the subsequent [two-view experiment](multiview-pose.md) adds drone2 image evidence and reserves drone3 for comparison. Both keep provider body joints outside fitting.

## Results across the original interval

Each mode sampled the same 120 even-numbered source frames; all 240 source frames remain in the comparison. No ambiguous hypothesis was selected, missing pose filled or suffix removed.

| Result | Default VIDEO mode | Explicit IMAGE mode |
| --- | --- | --- |
| Pipeline | Rejected: competing poses at frames 184 and 220 | Executable in 15.86 s |
| Sampled poses | 65 tracked, 53 partial, 2 competing | 63 tracked, 49 partial, 8 absent |
| Compared joint slots / all 2,880 | 1,339 | 1,273 |
| Other joint slots | 77 low confidence, 24 competing, 1,440 unsampled | 71 low confidence, 96 unavailable, 1,440 unsampled |
| Conditional mean image distance | 15.28 px | 15.41 px |
| Conditional mean relative 3D distance | 123.47 mm | 124.21 mm |
| Conditional 3D p95 / maximum | 268.34 / 1,053.82 mm | 273.47 / 1,053.82 mm |

The available denominators differ, so these means do not establish an accuracy improvement from changing modes. IMAGE mode avoids this rejection but misses eight sampled poses. The default body generator assumes a fixed camera; this moving-camera clip does not validate that assumption.

The comparison covers twelve shoulder/elbow/wrist/hip/knee/ankle joints. It transforms the reference into camera coordinates, then subtracts each pose's own hip midpoint. It fits **no scale, rotation, reflection, timing offset or identity reassignment**. Absolute root translation is therefore not evaluated. Joint definitions and both estimators' errors affect the distances.

As a basic consistency check, projecting the provider's own 3D joints into its three sets of 2D detections gives mean distances of 6.16, 5.60 and 6.14 px. This is an in-sample check, not a reproduction of the provider's held-out validation.

## Simulation and recording

The single-joint learner succeeds in 50/50 matched trials; random succeeds in 43/50 and coast in 39/50. This is an easy authored shoulder target, not exercise learning or physical transfer.

The independent reference-floor audit fails: **5/112 source poses** and **5/239 control-grid poses** exceed the unchanged 0.025 model-unit limit. Maximum overlap is 0.041491 model units, involving knee geometry. These two counts can overlap in time. No floor shift, pose lifting or threshold change was applied; whole-exercise training is not justified by this result.

The interaction journal contains twelve actions and thirteen states. Replaying the actions reproduces every state, reward and termination flag exactly, with the original 128-file run unchanged. The first study's final assertion incorrectly compared copied directory paths. Its failed evidence remains saved; the corrected study reuses the same export without retraining, another interaction probe or another model request. The first probe's unsaved action-effect scalar remains unavailable.

## Reproduce and inspect

The read-only comparison is `src/weft/camera_reference.py`; 18 analytic tests cover camera transforms, distortion, invalid geometry, missing/competing poses and preservation of unfitted scale errors. Their implementation hashes match the saved study snapshot. No full test-suite rerun is claimed.

```sh
# New output directories are required; these commands do not call a model.
.venv-lab/bin/python scripts/studies/check_multiview_pose.py --out artifacts/jog-comparison-new
.venv-lab/bin/python scripts/studies/check_jog_interaction.py \
  --recording artifacts/multiview-jog-image-01/sessions/session-d0e83074a6ea \
  --out artifacts/jog-recording-check-new
```

| Evidence | Verified files | Manifest SHA-256 |
| --- | ---: | --- |
| [Source bundle](../artifacts/multiview-jog-source-01/report.json) | 11 | `49877fe7a1e45170cfa70c9a0846dece5368491a61ee385a567f1ba886d9395d` |
| [Comparison](../artifacts/multiview-jog-comparison-01/report.json) | 37 | `49c3668260c086279a229f57f2bae3f409ec2f3e08a8e82f4fecd16f3d11ccb4` |
| [Failed interaction study](../artifacts/multiview-jog-interaction-01/report.json) | 12 | `273e17fdcd0b1d1bacb7ce58c6d802f03a41241ce554c75b242dc80c84c4b57b` |
| [Recovered recording check](../artifacts/multiview-jog-interaction-02/report.json) | 12 | `5ceca1a8c1e1b2fe3c4123e78dd95446944f4de3927b63da15c7e2cd3421547d` |

The full source decode, both pre-run plans, failed baseline and runtime snapshots remain saved. No browser interaction, server, deployment or paid request occurred in this study. At **October 28, 2026, 9:44 PM PDT**, the shared ledger still has six settled calls: $0.166152 committed, $4.833848 remaining of $5.

Follow-up: the [camera-aware candidate](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-aware-body.md) is now implemented and recordable. On the same 112 available frames, travel error drops from 10.51 m to 0.25 m against the reconstructed reference, but floor failures increase from 5/112 to 17/112. Reference joints remain excluded from fitting, all missing frames remain explicit, and no learned policy or physical promotion is claimed. Independently measured calibration/contact evidence remains necessary for stronger physical claims.
