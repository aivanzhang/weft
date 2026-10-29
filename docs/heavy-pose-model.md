# Heavy pose model: playable squat comparison

The optional Heavy model improves the checked squat limb directions, but loses usable poses on other videos. **It is not the default.** The new squat sandbox supports manual controls and exact recording/replay; it still has floor intersections and does not reconstruct measured muscles or exercise biomechanics.

## Open and test it

From the repository root:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/heavy-squat-body-01/lab
```

Open the full local URL printed in the terminal. This resumes saved results without downloading, rerunning inference or retraining. Keep the terminal open; Ctrl+C stops the lab.

1. **Video:** play the original squat, inspect pose overlays and use **What got tracked?** / **Next tracking issue**. All 213 source frames remain available; 107 were analyzed, with 104 usable estimates and three low-confidence estimates.
2. **3D body:** select **Play captured movement**, scrub the timeline and rotate the view. This is inferred pose playback, not physics. Muscle bands are illustrative motor-command proxies, not measured muscle activation.
3. **Sandbox:** select **Move it yourself**, choose a joint and use **Bend** / **Straighten**. Mobile-body controls include virtual support, release and pushes. **Follow video movement** uses an authored controller; **Learned controller** controls just one joint.
4. **Record & start**, interact, then **Stop & save**. Open the exported interactive replay or video to inspect your trial.
5. **Checks & limits** and **Files:** inspect the retained estimates, floor checks, training results and model assumptions. Passing a recording check does not establish physical accuracy.

Saved evidence is available without starting the lab:

- Source/body comparisons at [frame 50](../artifacts/heavy-squat-body-01/comparison-0050.png) and [frame 100](../artifacts/heavy-squat-body-01/comparison-0100.png).
- Verified [interactive replay](../artifacts/heavy-squat-body-01/lab/sessions/session-264ef83bbcb8/replay.html) and [recording video](../artifacts/heavy-squat-body-01/lab/sessions/session-264ef83bbcb8/video.mp4).
- [Five-video comparison](../artifacts/heavy-exercise-comparison-01/report.json) and [squat body/control report](../artifacts/heavy-squat-body-01/report.json).

The previous [Full-model squat and push-up candidates](selective-crop-exercises.md) remain unchanged and separately playable.

A subsequent [image-position fitting experiment](body-pixel-refinement.md) improves image agreement but worsens the Heavy squat's deepest floor intersection. It is rejected for promotion; the playable body described here is unchanged.

## What changed across five complete videos

Both conditions use whole-image, raw IMAGE-mode estimates at exactly the same 1,340 sample times across 2,677 decoded source frames. There is no crop, per-frame model selection, changed threshold or removed failed sample. The comparison reuses frozen Full estimates and makes 1,340 new local Heavy calls.

| Video | Full usable / samples | Heavy usable / samples | Heavy body-construction prerequisites |
| --- | ---: | ---: | --- |
| Single-leg squat | 115 / 118 | 114 / 118 | Reject: competing hypotheses and movement gap |
| Squat | 107 / 107 | 104 / 107 | Pass; new body built and checked separately |
| Kettlebell push-up | 116 / 129 | 73 / 129 | Reject: insufficient visibility and movement gaps |
| Jumping-jacks/burpees | 581 / 678 | 453 / 678 | Reject: 33 long movement gaps, maximum 1.535 seconds |
| Lunge/twist | 297 / 308 | 303 / 308 | Reject: three competing-pose samples |

Only squat passes both prerequisites; that is permission to construct a diagnostic candidate, not a physical-accuracy pass. Heavy returns 1,047 usable estimates versus Full's 1,216. Partial usable poses and missing joints remain explicit.

Fixed, previously exposed development labels give a different view of quality:

- Torso-side direction: **29 → 39 / 39** observable checks, ten gains and no regressions. These labels cover four clips, excluding squat; 15 unobservable slots remain excluded from scoring.
- Limb direction: **55 → 76 / 84** observable checks, 23 gains and two regressions. These labels cover squat, push-up and jumping-jacks; 12 unobservable slots remain excluded. Both regressions lose a previously scored right upper arm at push-up frames 30 and 120; Heavy still lacks eight observable predictions.

These are approximate source-image annotations, not new holdout data or verified 3D anatomy. Improved labeled directions do not offset the lost complete-pose coverage automatically.

## Squat body, floor and controls

The unchanged body builder retains 104 poses, compared with the original Full candidate's 107. On all 16 fixed squat body-direction checks, passes improve **11 → 15**, with five gains and one regression: the right forearm at source frame 10. Mean direction error is 11.54° → 8.58°; maximum is 32.14° → 21.10°. These are native rendered-body directions, not camera-calibrated 3D measurements.

Floor failures decrease from **55 / 107 → 45 / 104 source poses** and **111 / 213 → 95 / 213 timeline checks**. Across 317 matched checks, 25 recover and two newly fail at 2.6 seconds; three old source checks are missing. Maximum intersection is 0.1789 → 0.1332 model units. The frozen floor report inherits the helper's `crop-derived` wording, but this study uses only whole-image estimates. The failure counts and comparisons are unchanged.

The learned task is **left hip twist**, trained for 100 episodes / 8,351 steps. In 20 matched simulation trials, the learner succeeds 20 times, random actions once and no-control zero times. The earlier Full candidate used a different joint, so its learning score is not a matched model comparison. This does not learn the whole squat or the barbell load.

An authored 19-axis reference controller covers the full 7.067-second timeline in 213 steps. Its joint tracking error is 0.1082 radians; root error is 0.02484 model units. Small dynamic floor intersection does not erase the static reference failures or validate contact.

The saved trial contains **12 actions and 13 states/video frames**, with no reset. Every state, reward and termination flag replays exactly. No original run or recording was replaced.

## Model choice and reproducibility

[Google's official guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) publishes Lite, Full and Heavy bundles. Weft now accepts all three by pinned bytes/hash; Lite remains the default. Heavy float16/1 is 30,664,242 bytes, SHA-256 `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`.

```sh
.venv-lab/bin/python -m weft fetch-pose-model --variant heavy
.venv-lab/bin/python -m weft lab \
  --video /absolute/path/to/exercise.mp4 \
  --profile exercise --body-model mobile \
  --pose-mode image --pose-identity raw \
  --pose-model .local/models/pose_landmarker_heavy.task \
  --out artifacts/my-heavy-exercise-01
```

Use a fresh output directory. This is an explicit experimental choice, with no automatic fallback. The model is already cached on this machine. Local inference does not upload the video. The [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) identifies orientation/scale/occlusion limitations and excludes accurate metric depth from its stated scope.

Thirty-seven scoped model/inference tests pass in `.local/heavy-pose-model-tests-01.xml`. Two bounded CPU workers completed the five-video comparison in 24.112 seconds; detector median latency was 21.05–23.21 ms. These are measured local run times, not a matched Full speed comparison or browser responsiveness result. Building, training, checking and recording squat took 3.559 seconds after inference. No development server or browser was started for these checks.

Public model/documentation acquisition downloaded 32,205,610 bytes. The first DNS failure is preserved separately; it downloaded zero bytes. This work makes **no paid request or video upload**, provisions no recurring service and leaves the $5 Meta budget unchanged. Local CPU, storage and electricity are not measured as zero cost.

| Immutable evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Official Heavy acquisition | 6 | `e06baae59716511523970abc68f2227e5441a548b6a33c124285ae541c33b5f4` |
| Five-video comparison | 182 | `c1ce5b757d52a11040c0552adfc18c8116ace88fc36b88a93cdaeffd05093dd7` |
| Squat body and recording | 168 | `ca68c67d931012a9eddca3fe9f879ff59d99f3464274b4cb12a06331d0b67fbb` |

The recording seal is `8c067d18aafecf9a08e9ad9b59b37adc32a18577ae22557e08b9f325ada23595`. Missing poses, body proportions, floor contact, actual muscle activation, full-exercise control and independent changed-action physical validation remain unresolved. No default model or acceptance gate changed.
