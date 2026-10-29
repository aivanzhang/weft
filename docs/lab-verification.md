# Local lab verification

Observed **October 28, 2026, 11:44 a.m. PDT**, with the later focused CLI lifecycle check recorded in `.local/lab-cli-lifecycle-01`. Python 3.14.7, macOS arm64, `.venv-lab`, pinned `requirements-lab.lock`. All computation was local CPU; no paid inference, cloud resource, public release, or submission.

## Verified paths

| Profile | Preserved run | Evidence | Learned / random / coast task successes |
| --- | --- | --- | --- |
| Exercise | `artifacts/exercise-lab-03` | Real licensed single-leg-squat video → CPU pose estimates → MuJoCo body → reference projection/servo → Q-learning → lab | 100/100 · 90/100 · 64/100 |
| Planar tabletop | `artifacts/tabletop-lab-01` | Clearly synthetic three-object video with contact → tracks → Pymunk world → fixed repair gates → Q-learning | 100/100 · 12/100 · 19/100 |
| Original rolling | `artifacts/rolling-lab-01` | Original pinned real video → existing restricted rolling pipeline → lab adapter | 100/100 · 12/100 · 12/100 |

Exercise/planar training used 1,500 episodes; rolling used 15,000. Each profile compared policies on its own fixed world/task and matched evaluation seeds. These are not cross-profile comparisons. The exercise task is an authored left-knee target, not imitation of the complete exercise; its strong random/coast baselines show that it is an easy task. No result proves real-world control or recovered human physiology.

### Real exercise evidence

- Source: RickyBennison, [Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original hash `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
- Official MediaPipe Pose Landmarker Lite `float16/1`, model hash `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`; CPU delegate, no remote inference. [Provider documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python).
- 235 decoded frames versus a container count estimate of 236, explicitly recorded as a one-frame discrepancy. 118 poses sampled at 15 fps; four fully tracked and 114 partial. Right shoulder/elbow confidence is reliable in only 4/118 sampled poses; uncertainty remains in saved evidence and is not independently validated.
- Twelve connected modeled segments, eight sagittal hinge joints, sixteen illustrative antagonist bands. Three inferred reference angles exceeded authored limits and were transparently clamped in `human.json`.
- Authored reference-servo tracking error: **0.2181 radians RMSE** against inferred/clamped joint angles. This is a model-control measurement, not an independent physical score.
- Pipeline-reported duration: **14.03 seconds**, excluding final implementation copying/hashing. Run artifacts: **31,590,572 bytes**. One observed run is not a general speed benchmark.
- A second real, cropped-foot FitnessScape squat clip was also processed under `.local/exercise-pose-cropped-01`: 100/107 usable sampled poses, seven low-confidence outcomes retained. Cropped feet do not establish ground-contact recovery; no such claim is made.

## Integrity and recording

| Run | Verified run files | Manifest SHA-256 |
| --- | --- | --- |
| Exercise | 68 | `367d1a7ed9da160b51f64134d0af9dcd955e06cd2818c074cfe6e30fcb3446a7` |
| Tabletop | 62 | `a98216cd8bd7cde66fd78fce5db53850e91ba6ae152d0514414319fe1d8815d6` |
| Rolling | 66 | `78cdfa9ac05ed02b1f57add4e89d956683694fff5f64ca89661bd085816d59d4` |

Actual authenticated loopback HTTP checks opened each saved run, controlled the authoritative simulator, recorded 30 actions, stopped/exported, fully decoded all 31 video frames, verified all six export files, and confirmed the original run manifest remained unchanged. Temporary servers shut down afterward. Exercise checks also changed the selected joint and restored the trained joint before recording.

- Exercise recording: `artifacts/exercise-lab-03/sessions/session-9d615a54f52a`.
- Tabletop recording: `artifacts/tabletop-lab-01/sessions/session-0ac1b47e7992`.
- Rolling recording: `artifacts/rolling-lab-01/sessions/session-e9dc7e41a3f6`.
- Detailed results: `.local/lab-end-to-end-verification.json`; reproducible bounded check script: `.local/verify_saved_labs.py`.

The journals preserve exact simulated states and actions, not synthetic substitutes for missing physical evidence. Integrity hashes detect modification but do not authenticate an author who can replace both files and their manifest.

## Checks and performance

- **157 distinct Python tests passed across the full and focused final runs.** The full run initially had one stale environment-allowlist test expectation; it was updated to include the task-local matplotlib cache. The affected final HTTP suite passes 26 tests, including two added terminal-status race regressions. Recording and remaining tests passed under the full/focused runs. Evidence: `.local/lab-verified-tests.xml`, `.local/lab-http-final-tests.xml`, `.local/lab-recording-final-tests.xml`.
- **Seven deterministic UI checks passed:** running/ready transitions and artifacts, controls queued behind pending steps, one playback chain after restart, stale artifact loads not undoing recording, human camera/reference controls, joint/recording conflicts, and correct held-key release mapping. `node tests/lab_ui_smoke.mjs` uses a mocked DOM/canvas/fetch—not a browser.
- Dependency consistency passed. Runtime and offline replay JavaScript syntax checked. Source, pose, and human-body previews visually inspected as images.
- Twelve-body planar physics/snapshots: 3,000 steps, median **0.104 ms**, p95 **0.110 ms**; `.local/scene-engine-performance.json`.
- Real-pose-derived human physics/snapshots: 300 steps, median **0.114 ms**, p95 **0.120 ms**; `.local/human-real-physics-fmcstg0n/performance.json`.
- Actual saved-run HTTP stepping while recording: 15 two-step requests/profile; median **1.55 ms exercise**, **0.46 ms tabletop**, **0.29 ms rolling**. These include local request handling, not browser rendering or internet latency.
- Synthetic 1,000-action journal/export check: **2.36 seconds** export plus full 1,001-frame decode/integrity verification; `.local/recording-check-2pvwrhbc/measurement.json`.

These bounded measurements do not establish sustained browser responsiveness, all-input performance, or hardware portability.

## Preserved failures and recovery

`artifacts/exercise-lab-01` failed its implementation-stability check because source changed during that run; it remains preserved. Subsequent completed runs used stable snapshots.

Final CLI verification found a shutdown race: the worker published `ready`, but cleanup sometimes saw the still-exiting process and relabeled the lab `cancelled`. The code now waits for a terminal worker and preserves its published result. Affected completed tabletop/exercise lifecycle metadata was corrected only after verifying complete manifests; each has a `status-recovery.json` containing the prior status, reason, and verification. No scored `run/` file was rewritten. The current lab server uses the fixed lifecycle code; preserved implementation snapshots accurately retain the code that generated each run.

## Still unverified

Browser playback/device interaction, sustained UI performance, independently annotated pose accuracy, arbitrary exercise/scene coverage, accurate anatomy, muscle force/activation, grounded human balance, full-exercise learning, general human physical repair, physical intervention transfer, live Meta integration, deployment, and public submission. See [usage and limits](local-lab.md) and [remaining gates](../TODO.md).

## Sandbox usability checks

The local UI now has guided entry points, separate manual/video-following/learned modes, direct small-move buttons, plain-language goal state, local pose playback, and a Record & start → Stop & save workflow. Technical values and body assumptions are expandable. No inference, training, physics, or saved experiment evidence changed for this usability pass.

- `node tests/lab_ui_smoke.mjs`: 14 deterministic checks passed, including serialized mutations, single playback chain, stale-status protection, controller/manual isolation, trained-joint setup, navigation cancellation, local replay cancellation, finished-trial restart, and recording/export links.
- `pytest tests/test_lab_http.py -k public_assets -q`: 1 passed; updated UI assets remain served through the guarded local server.
- JavaScript syntax, HTML nesting, unique IDs, and element references checked.
- Browser layout, browser media playback, and sustained rendering performance remain unverified; no browser interaction was authorized.
