# Whole-body sandbox and recording

The experimental whole-body mode learns feedback for **all 22 motors**: 19 joint rotations and three virtual body-translation forces. It uses an already-built mobile exercise model, so pose inference is not repeated. A packaged whole-body lab provides visual controls, recording and matched-controller results. Ordinary exercise labs retain their separate single-joint learner; the two modes are not silently mixed.

## Try it

Open the prepared visual sandbox:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/whole-body-visual-lab-01
```

Open the printed local URL. In **Sandbox**, choose manual, learned or video-following control. Manual mode lets you select any of the 22 motors and apply ±25% of its limit; every other motor is zero. Release sets all motors to zero. **Record & start** begins a fresh trial; **Stop & save** exports the video and rotatable replay. Motor selection may change during a recording because each event preserves the full 22-value action.

The visible floor-conflict warning is intentional: these are diagnostic simulations, not validated human reconstructions. Full-body tracking errors, floor penetration and virtual root forces remain separate from the illustrative muscle bands.

Use the already-completed single-leg run without retraining:

```sh
.venv-lab/bin/python -m weft body-sandbox \
  --run artifacts/whole-body-learning-02/single-leg-squat-seed-11 \
  --record artifacts/my-whole-body-recording-01
```

Open the [saved single-leg replay](../artifacts/whole-body-learning-02/sessions/single-leg-squat/replay.html) or [video](../artifacts/whole-body-learning-02/sessions/single-leg-squat/video.mp4) to inspect the existing trial.

Train from a verified mobile run, using a new output directory:

```sh
.venv-lab/bin/python -m weft imitate \
  --run artifacts/mobile-exercise-e2e-10/squat/run \
  --out artifacts/my-whole-body-01

.venv-lab/bin/python -m weft body-sandbox \
  --run artifacts/my-whole-body-01 \
  --record artifacts/my-whole-body-recording-01
```

The terminal lists motor names and the location of `current-frame.png`, which updates after each command. It prints tracking errors, floor penetration and the actual virtual root-force vector. Open the image yourself to inspect the current pose; some image viewers require reloading it.

To package a newly trained policy for visual use, without changing either input:

```sh
.venv-lab/bin/python -m weft body-lab \
  --source-lab artifacts/mobile-exercise-e2e-10/squat \
  --learning artifacts/my-whole-body-01 \
  --out artifacts/my-whole-body-lab-01
.venv-lab/bin/python -m weft lab --resume artifacts/my-whole-body-lab-01
```

The source and learning runs must use exactly the same body model. Packaging copies only sealed artifacts, retains the legacy joint-learning files under `legacy-joint/`, preserves provenance and runs the bounded reference audit once before serving. It does not retrain, call an API or start a server automatically.

| Command | Meaning |
| --- | --- |
| `learned 6` | Run six steps using the learned controller. |
| `follow 6` | Run six steps using the separate hand-written reference controller. |
| `release 6` | Set every motor to zero for six steps; gravity and contacts remain. |
| `motor root_x .1 6` | Apply 10% of the root-x force limit; every other motor is zero. |
| `motor left_knee .1` | Apply 10% of that joint's torque limit for one step. |
| `reset` | Restart the same seeded trial and record the reset. |
| `status` | Print current independent metrics. |
| `quit` | Save the complete video, GIF and rotatable offline HTML replay. |

Each step lasts 1/30 second. Commands accept 1–6 steps. An episode is at most 180 steps, normally 60; shorter references reduce that horizon. A finished episode requires `reset`. Recordings stay outside the immutable learning directory.

`replay.html` permits rotation and scrubbing of saved states; it does not change the simulation. `events.jsonl` preserves exact actions, resulting states, forces, rewards and termination flags. On numerical failure, the failed attempt is retained in `recording.json`, and the valid prefix is exported without inventing a final body pose.

## What is learned

A bounded cross-entropy policy search samples 12 feedback weights: three features for each of four motor groups (torso, proximal limbs, distal limbs and root). The policy sees position error, velocity error and the simulator's computed gravity/dynamics bias forces. Nominal feedback gains are authored feature scalings. Search begins with zero weights, not a supplied successful controller; it selects candidates using simulated training return only.

The environment does **not** add a hidden joint servo or gravity compensation. Every controller supplies the same normalized 22-value action vector once per 30-Hz step, held across eight physics ticks. The zero policy means zero motor forces. The hand-written baseline uses unit feedback weights and the same action timing—not the older, faster substep servo.

This is structured, reference-conditioned control: the desired motion comes from the video and is available to the policy. It is not recovered human intent, a general movement policy, physiological muscle control or unassisted balance. Virtual root motors can support or move the body in ways a real human cannot. Muscle bands remain illustrative command-intensity displays.

## Independent checks and work bounds

Before new training and visual packaging, `whole_body_feasibility.py` checks every source pose plus a 30-Hz reference grid against the unchanged authored floor. It never advances physics, lifts a pose or refits the floor. The bounded report keeps every sample and conflict. Training remains explicitly diagnostic and retains all windows and failed trials; a preflight failure is not converted into a passing reconstruction.

Training uses three fixed windows near the first 60% of the reference. Evaluation uses three later windows with different reset seeds. The split aligns to source samples and leaves a conditioning gap: even the final training velocity feature cannot consume an evaluation-side pose. These are already-exposed development videos, not final holdouts; the body model itself is shared.

Success requires the entire horizon, angular RMSE at most 0.35 radians, root-position RMSE at most 0.15 model units, maximum floor penetration at most 0.025 model units, and no dynamics failure. These thresholds assess the authored simulation against its inferred reference, not physical human accuracy. Reward is a separate nonnegative tracking score minus bounded motor effort; incomplete episodes retain the full-horizon denominator. Nonfinite dynamics produce unavailable error metrics, not a false perfect score.

Defaults: eight generations × 16 candidates × three training trials, at most 60 steps each; four matched evaluation controllers × three trials. One job has a 120-second limit, with completed generations and interrupted trial details retained. No external API calls or cloud resources are used.

Artifacts include the human model, settings and source hashes, implementation copies, every candidate and trial result, the selected policy, all four evaluation results and representative state traces. Policy bytes are frozen before evaluation and verified afterward. No best seed is selected using evaluation results.

## Verified development results

`artifacts/whole-body-learning-02` completed all nine jobs in 40.385 seconds using two workers: three real-source body models × training seeds 11, 29 and 47. Each controller receives three later-window trials per seed; the table combines those nine trials, not nine independent videos.

| Source | Learned passes | Zero passes | Random passes | Hand-written passes | Learned / hand-written mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-leg squat | 9/9 | 0/9 | 0/9 | 9/9 | 0.727 / 0.939 |
| Squat | 3/9 | 0/9 | 0/9 | 3/9 | 0.672 / 0.871 |
| Kettlebell push-up | 9/9 | 0/9 | 0/9 | 9/9 | 0.592 / 0.881 |

The learned controller beats zero/random here but does **not** outperform the hand-written controller. All learned trials completed their horizon. Six squat trials fail the unchanged floor gate; their peak penetration reaches 0.216 model units, also present with other controllers. This is a retained failure, not a passing environment or a reason to loosen the gate.

[Reset-only diagnosis](whole-body-floor.md) establishes that all six failing squat starts already put their shin capsules 0.108–0.216 model units below the floor before any motor action. The inferred trajectory and first-pose floor alignment disagree; cropped feet, uncalibrated scale and omitted equipment prevent identifying the real physical cause. No reset was lifted, window discarded or threshold changed. A portable ten-file verified diagnostic bundle is `artifacts/whole-body-floor-verified-01`, seal `e4e1ff825ff3551c9427bd5ef2ade429d752cc5ead838a9b1712b21c24015175`; it preserves the original repository-relative manifest and its verified inputs without rewriting them.

Predeclared seed 11 was recorded for each source, followed by manual root force, release and authored following. Each recording contains 78 actions, one reset and 80 fully decoded video frames. All 22 command values, forces and root-force fields are cross-validated. The complete study has 223 verified files; manifest SHA-256 `5d4c78b6b6b0003ac059ffe72c44a73a584be90cbcd878608666dcc8a59b8165`.

All **680 Python tests pass** in `.local/whole-body-final-tests.xml` (20.417 seconds, no failures/errors/skips). Saved-state contact sheets were visually inspected and all three exported replay scripts pass JavaScript syntax checks: `artifacts/whole-body-visual-review-01`, seven-file seal `cd5c5e077ddc2baa358bf812c089af2f0b01c8a3841460e25517a819c5082bb8`. These checks do not establish browser/device responsiveness; no browser interaction occurred. Existing browser UI code was unchanged, so its previously passing 17 UI tests were not repeated.

`artifacts/whole-body-learning-01` preserves a preparation failure: an absolute script path caused a same-file copy rejection before training. No source was overwritten and no candidate ran. The corrected path and all implementation/tests were frozen before study 02.

## Remaining scope

Whole-body visual controls now pass 24 deterministic UI checks (`.local/whole-body-visual-ui.log`); actual browser/device interaction remains unperformed. The Python suite passes 708 tests (`.local/whole-body-visual-tests.xml`, 18.920 seconds); a subsequent preparation-event display fix passes its focused test. The visual HTTP study records 78 actions + one reset / 80 decoded frames and reproduces every state/reward/termination exactly by replaying the saved actions. Its six-step batch p95 is 7.39 ms in one short local sequence, not a browser-performance claim. `artifacts/whole-body-visual-http-01`: 129-file seal `37087bfc4afbc104aa3a6f3c14084e148c15893d609d30fb7956c7fb48c65e00`.

The complete-reference preflight finds conflicts in all three models: single-leg maximum 0.083 model units (8/118 source poses), squat 0.254 (40/100), push-up 0.066 (25/127). These kinematic conflicts do not contradict selected dynamic tracking-window successes: controllers can deviate from an infeasible target, and the sampled intervals differ. `artifacts/whole-body-feasibility-01` retains 1,048 checks, zero physics steps, 18-file seal `a073ca57b38102c63112227a59bd7f1d6ce7c789145eebdfc49f56d637fc48de`.

Longer-horizon reliability, grounded reference repair, calibrated physical loads, equipment interaction, wrist/ankle mechanics, anatomical muscles and independent real-world intervention checks remain open. A controller tracking a plausible inferred pose does not validate that pose or its physics.
