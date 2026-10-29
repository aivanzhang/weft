# Ankle and foot sandbox

The opt-in candidate adds two capsule-shaped feet and six ankle rotation controls to an existing mobile body. You can move it, release its motors, inspect contacts and record every action. **It is not a corrected human model:** the tested candidates still overlap the assumed floor. Default exercise models and all source runs remain unchanged.

## See the result

Open the [full-body squat replay](../artifacts/foot-body-view-01/fullbody-squat/replay.html) or [single-leg squat replay](../artifacts/foot-body-view-01/single-leg-squat/replay.html) locally. Use **Close-up of feet**, camera yaw, the frame slider and muscle toggle. These files replay saved states; they do not accept new physical actions. The [foot close-up](../artifacts/foot-body-view-01/fullbody-squat-foot-closeup.png) also shows the capsule geometry without starting anything.

## Control and record a new trial

From the repository root:

```sh
.venv-lab/bin/python -m weft feet-sandbox \
  --body artifacts/foot-reference-01/fullbody-squat/inputs/human.json \
  --feet artifacts/foot-reference-01/fullbody-squat/foot-reference.json \
  --record artifacts/my-feet-01
```

Use a new recording directory. The command prints the source-pose floor audit before accepting controls. It starts no server and makes no API call.

| Command | Effect |
|---|---|
| `hold 6` | Apply authored feedback toward the starting pose for six steps |
| `motor left_ankle_y 0.15 2` | Apply 15% of that motor's limit for two steps; all other motors are zero |
| `release 6` | Zero all 28 motor commands for six steps |
| `reset` | Restore the same starting pose and record the reset |
| `status` | Show pose error, floor overlap and task status |
| `quit` | Save the journal, video, GIF and offline interactive replay |

Each step is 1/30 second, with eight physics substeps. Commands accept 1–6 steps. Each episode is bounded to 180 steps, then requires reset. Ankle motor names are `left_ankle_z`, `left_ankle_y`, `left_ankle_x` and the corresponding `right_` names; these are generic rotation axes, not recovered anatomical degrees of freedom.

`current-frame.png` updates after each command. After quitting, open `replay.html` in the new recording directory. `events.jsonl` preserves all actions and states; `candidate-audit.json` preserves missing poses and floor failures. Body and foot input copies accompany the recording. The CLI refuses to place recordings inside either source artifact directory.

The single-leg clip also supports this workflow: replace `fullbody-squat` with `single-leg-squat`. The cropped squat lacks usable fixed foot shapes. The push-up lacks one foot at its original starting pose. Neither is silently initialized from a later, easier frame.

## Build from your own video

Use a fixed-camera exercise clip with the whole body and both feet visible. From the repository root, create a mobile exercise run, then a separate foot candidate:

```sh
.venv-lab/bin/python -m weft lab \
  --video /absolute/path/exercise.mp4 --profile exercise --body-model mobile \
  --out artifacts/my-exercise-01 --no-serve

.venv-lab/bin/python -m weft feet-build \
  --run artifacts/my-exercise-01/run --out artifacts/my-foot-candidate-01

.venv-lab/bin/python -m weft feet-sandbox \
  --body artifacts/my-foot-candidate-01/human.json \
  --feet artifacts/my-foot-candidate-01/foot-reference.json \
  --record artifacts/my-foot-recording-01
```

Each output directory must be new. The first stage must complete successfully; a rejected video does not become a foot environment. The [local setup guide](local-lab.md) covers dependencies and the pinned pose model. To try a supplied clip, use `data/exercise/bodyweight-squats-danielflefil-500.mp4`; its [source and attribution](exercise-fullbody-source.md) are saved.

`feet-build` verifies the source run and saves copied inputs, original/candidate foot references, floor diagnostics, implementation snapshot and integrity manifest. It reuses the saved poses without another video decode, model inference or API request. Missing feet remain unavailable; `report.json` states whether interaction can start.

To test the optional [15° image-direction correction](foot-image-refinement.md), add `--image-directions` to `feet-build` and choose another new output directory. It preserves body, root and floor placement and never promotes itself to a verified model. [See the tested build-and-record result](../artifacts/feet-build-cli-01/session/replay.html).

## What is inferred and what is authored

The [foot-reference builder](foot-reference.md) supplies fixed ankle/heel/index triangles and per-frame orientations. Native MuJoCo feet attach to the existing ankles using those triangles. Every existing body joint, root reference, segment and floor stays unchanged. Available source projections match the saved triangle endpoints within 2.50e-16 model units; original joint positions change by exactly zero. This verifies implementation, not anatomical accuracy.

Authored additions are a .018-model-unit capsule radius, .5 kg mass per foot, ±12 Nm ankle motors, passive joint damping and broad ±2π rotation-chart limits. The scale is uncalibrated. Capsule contact is not a measured sole or load. Self-collision stays disabled, as in the existing mobile body. Anatomical axial roll and sole orientation remain unknown.

`FootEnv` exposes Gymnasium `reset`/`step`, 28 normalized actions (25 rotation motors plus three virtual root-force motors), and 85 observation values: positions, velocities, fixed target positions and elapsed task fraction. No hidden controller runs during `step`. `hold_action()` explicitly computes the command that the caller then passes to `step` and records. No learned policy was trained for this candidate. Fifty displayed muscle bands are command illustrations, not measured activation or force.

The task is **hold the initial pose**, not imitate missing video motion. Source-pose audits never interpolate foot gaps. The original body-only audit still checks every source pose and its 30 Hz grid. Candidate foot checks cover exact observed body/foot timestamps only; unavailable slots remain null, and continuous/grid foot feasibility is not claimed.

## Four-source verification

All 388 raw pose frames remain in the frozen foot inputs. The existing body contains 379 accepted source poses; 245 have both foot references available and 134 remain unscored for candidate feet.

| Source | Original body floor failures | Candidate foot checks | Candidate floor failures | New interactive start |
|---|---:|---:|---:|---|
| Single-leg squat | 8 / 118 | 115 / 118 | 82 / 115 | Available |
| Cropped squat | 40 / 100 | 0 / 100 | Unavailable | Unavailable |
| Kettlebell push-up | 25 / 127 | 107 / 127 | 25 / 107 | Unavailable |
| Full-body squat | 13 / 34 | 23 / 34 | 23 / 23 | Available |

The unchanged penetration limit is .025 model units. Maximum checked candidate penetration is .1870 for single-leg, .0665 for push-up and .1770 for full-body squat. Different availability denominators are explicit; this is not a matched claim of physical improvement.

Both playable clips completed 180-step authored-hold trials without instability, but both failed the floor criterion. Each separate manual-command recording contains 14 actions, one reset and 16 fully exported video frames. The close-up re-exports preserve identical journal bytes and states.

There are **240 distinct affected passing Python checks**: 29 candidate checks and 211 existing body/recording checks. A fixture-directory setup failure was corrected and retained in the original test log. The final replay change passed its 105-check affected subset, and its emitted JavaScript passed `node --check`. No browser interaction was performed.

On the two 180-step local trials, physics/control steps had median .242–.247 ms and p95 .271–.275 ms. Thirty frame renders plus PNG writes per clip had median 5.22–5.24 ms and p95 5.28–5.58 ms. These are bounded local measurements, not browser latency or sustained CPU/memory guarantees.

The [source audit and trials](../artifacts/foot-body-01/report.json) took 2.05 seconds, used no API calls and remain unpromoted. The 135-file study seal is `fa32365bf7199742d65ae833576a417d69ac1293d92295ecfdc45a23f0feb334`. The [camera-only replay export](../artifacts/foot-body-view-01/report.json) has a 20-file seal `5278b5cbe3972f59d10cb002ee5fe0cf6eb3377979ef6ec5b0591fbdfd1698bf`.

Reproduce only after relevant inputs change, using new output directories:

```sh
.venv-lab/bin/python scripts/studies/check_foot_body.py --out artifacts/foot-body-repeat
.venv-lab/bin/python scripts/studies/export_foot_views.py --out artifacts/foot-body-view-repeat
```

The subsequent [image-direction experiment](foot-image-refinement.md) resolves some single-leg floor failures but leaves all models unpromoted. Next: establish independent camera/floor/foot geometry evidence. Adding feet is not evidence that balance, contact forces or the complete exercise have been recovered.
