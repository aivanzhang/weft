# Weft local lab

Supply a local video, inspect each processing stage, interact with the generated environment, and record your actions. The CLI starts a loopback-only lab and prints its browser link. No video is uploaded, no account is needed, and no paid model or cloud call occurs.

## Try the saved exercise run

From the repository root:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/exercise-lab-03
```

Open the complete URL printed by the command. Its temporary token stays on your computer. The terminal must remain running; Ctrl+C stops the lab. `--resume` inspects saved results without rerunning inference or training.

In **Video**, **What got tracked?** separates partial or unusable pose estimates from frames that were not analyzed. **Next tracking issue** seeks the original video without changing the simulation. Failed exercise runs remain inspectable, with simulation controls disabled. See the [current playable candidates and rejected-clip inspection command](selective-crop-exercises.md#inspect-tracking-failures-in-the-sandbox). Pose coverage is not proof of anatomical or physical accuracy.

The [new Heavy-model squat guide](heavy-pose-model.md#open-and-test-it) has a separate launch command, five-step walkthrough and saved replay. Its checked limb directions improve, but floor conflicts remain; the default model and earlier labs are unchanged.

The supplied video is [Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm) by RickyBennison, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Its source, unchanged bytes, license, hash, and limitations are preserved in [data/exercise](../data/exercise/README.md). A second FitnessScape squat clip is included as a more cropped input. Neither clip is a final holdout.

## Process your own exercise video

```sh
.venv-lab/bin/python -m weft lab \
  --video /absolute/path/to/exercise.mp4 \
  --profile exercise \
  --out artifacts/my-exercise-01
```

Use a **new output directory** for every run. Prefer a short, fixed-camera clip with one person, the whole body visible and good lighting. The default body has eight simplified hinges. Add `--body-model spatial` for 19 rotation axes, including lateral limbs and torso rotation. Neither model reconstructs wrist/ankle mechanics, ground contacts or full-body balance.

Add `--body-model mobile` for the experimental moving-pelvis model with an assumed floor, virtual support, release/push controls and contact-aware recording. [Open the verified example and inspect its limitations](mobile-human.md). This mode rejects unsupported root-position gaps instead of silently returning an anchored body.

The separate [whole-body sandbox](whole-body-learning.md) learns and records explicit commands for all 22 motors, with no hidden stabilizer. It reuses a completed mobile run and supports visual or terminal controls. Open the prepared visual lab with `weft lab --resume artifacts/whole-body-visual-lab-01`. Ordinary exercise labs retain the single-joint controller described below; whole-body packages identify their full-body task and floor-feasibility limits explicitly.

The opt-in [ankle/foot sandbox](foot-sandbox.md) adds capsule feet and six ankle controls, with terminal interaction and close-up offline replays. Its 28-motor task holds the starting pose; it does not learn or replay the exercise. Both playable source candidates still fail floor-overlap checks, so this is a diagnostic option, not the default body.

Limits: 100 MB, 60 seconds, 1,800 decoded frames, maximum 4096×2160 input; local pose inference samples at no more than 15 fps and 900 frames. Low-confidence joints and missing poses remain visible in the evidence. Multiple people, insufficient usable motion, and unsupported inputs fail explicitly. Do not interpret an estimated occluded limb as a verified observation.

### What the stages show

| Stage | Inspect |
| --- | --- |
| Video | Original video, annotated frames, sampled landmarks, confidence and source/model provenance |
| 3D body | Rotatable body and captured-pose playback; expand details for inferred proportions and authored assumptions |
| Sandbox | Manual movement, video-following controller, learned one-joint controller and trial recording |
| Learning results | Training history and matched learned/random/no-control trials |
| Checks & limits | Fitting records and limits. Automatic physical human-model repair is **not implemented** |
| Files | JSON, model XML, previews, source snapshots and integrity manifest |

**Three different things are deliberately separate:**

- **Inferred pose replay:** source-derived joint angles projected into the simplified body. It is not a physical rollout.
- **Reference servo:** an authored feedback controller drives the simulated joints toward those angles. It is not a learned human policy.
- **Learned policy:** actual Q-learning for one authored joint-target task. New runs choose the joint with the largest reliable, model-representable motion; the original saved run used the left knee. It does not learn the whole exercise. The sandbox selects the trained joint automatically before starting it. End any recording on a different joint first.

The body is a stylized articulated model, not a scanned anatomical mesh. Muscle bands (16 for the default body, 38 for spatial/mobile) illustrate modeled motor-command intensity; they are **not measured muscle activation, physiological force, or biomechanical validation**. Monocular depth/body proportions are estimates. Hinge/spatial models anchor the pelvis; mobile adds image-derived translation and an assumed floor. Stabilization and limb masses are authored assumptions.

## Interact and record

The **Start here** shortcuts open the captured movement, manual sandbox, or a fresh learned-controller demo. In **3D body**, use **Play captured movement** and the video-position slider to inspect the inferred poses. This playback does not run physics.

1. Open **Sandbox** and choose **Move it yourself**, **Follow video movement**, or **Learned controller**. Each mode explains what drives it; choosing a mode pauses movement.
2. In manual mode, choose a joint and tap **Bend** or **Straighten** for a small move. Or click **Start** and hold a direction. Focus the scene to use arrow keys / WASD; Space slows movement. Fine controls expose a single simulator step and raw values.
3. Read the current angle and target below the scene. Rotate the view or toggle illustrative muscles without changing physics.
4. Click **Record & start** to reset and start a fresh trial in the selected mode. Use **Pause** freely; the recording stays open.
5. Click **Stop & save**. When export finishes, download **Interactive replay** and open the HTML file locally to rotate/scrub the saved states, or download the video, GIF, or action data.

The simulation pauses on tab changes, window blur, or page hiding. **Start again** resets a finished trial; **Start over** resets without playing. Joint changes require manual mode and no active recording. Manual inputs are disabled in controller modes. Multiple resets may be captured within one recording. The local pose replay also pauses when leaving its tab/window.

Completed run artifacts stay under `run/`; interactive recordings go into new directories under `sessions/`. Every action, resulting state, reward, reset, and termination is preserved. Limits are 18,000 total journal events, 1,200 wall seconds, or 128 MiB, whichever comes first. A limit stops recording explicitly before the next action. Export retains all states in the video; only the preview GIF is sampled.

```sh
# Verify the immutable pipeline artifacts.
.venv-lab/bin/python -m weft verify artifacts/my-exercise-01/run

# Verify a completed recording.
.venv-lab/bin/python -m weft replay artifacts/my-exercise-01/sessions/session-EXAMPLE

# Export a closed journal left by a normal Ctrl+C shutdown.
.venv-lab/bin/python -m weft replay artifacts/my-exercise-01/sessions/session-EXAMPLE --export

# Non-destructively recover complete journal events from an interrupted recording into a NEW directory.
.venv-lab/bin/python -m weft replay artifacts/my-exercise-01/sessions/session-INTERRUPTED \
  --recover-incomplete \
  --out artifacts/my-recovered-session-01
```

An abrupt process kill may leave an unclosed `recording.json`, a truncated trailing JSONL fragment, or a partial video export. `--recover-incomplete --out <new-dir>` copies only validated complete events into `<new-dir>`, records explicit `recovery` provenance and source SHA-256 digests in `<new-dir>/recording.json`, exports `video.mp4` / `replay.html` / `preview.gif` / `manifest.json`, and leaves the original interrupted directory byte-for-byte untouched and unsealed. Saved MP4 uses the locally available MPEG-4 codec; the self-contained HTML/GIF replay is available when a browser does not support that codec.

## Tabletop and rolling profiles

For objects without bright colors, use the explicit-region `appearance` profile
and its required tracking/world configuration. See [appearance tracking](general-tracking.md)
for a runnable grayscale example and its source-interval limitations.

Exercise inference defaults to `--pose-mode video`, which reuses tracked regions.
`--pose-mode image` reruns detection on each sampled frame; it is an explicit
alternative, not an automatic repair. It removed false extra poses on the jumping-jack
clip but introduced them on single-leg squat. Both modes preserve and reject competing
pose hypotheses. See [mode tradeoffs](pose-identity.md). New exercise models infer a
fixed mean torso orientation in hinge mode; spatial/mobile modes add dynamic torso rotation.
Only mobile adds authored floor contacts. None establishes real human balance.

```sh
# Create a three-object SYNTHETIC software fixture, not a real physical video.
.venv-lab/bin/python -m weft demo --out artifacts/tabletop-source-new
.venv-lab/bin/python -m weft lab \
  --video artifacts/tabletop-source-new/video.mp4 \
  --config artifacts/tabletop-source-new/config.json \
  --profile planar --out artifacts/tabletop-lab-new

# Supply your own fixed-camera colored-object video.
.venv-lab/bin/python -m weft lab --video /path/to/tabletop.mp4 \
  --profile planar --out artifacts/my-tabletop-01

# Preserve the original restricted rolling-ball profile.
.venv-lab/bin/python -m weft lab --video data/rolling-ball.mp4 \
  --profile rolling --episodes 15000 --out artifacts/my-rolling-01
```

The planar profile uses Pymunk shared rigid-body physics, distinct saturated-color tracking, circles and static boxes, authored walls/controls, and an authored parking goal. Explicit HSV bands, roles, and target can be supplied using [the example configuration](../examples/planar-scene.json). Such configuration is a manual input, not automatic scene understanding. Same-color ambiguity, late-entering objects, moving cameras, and non-rigid objects are outside the supported profile. The current automatic tracker is not a general segmentation model.

The appearance profile additionally supports [explicit sections of longer videos](source-intervals.md). It preserves the full source and original frame numbers, tracks only the requested interval and labels pre-initialization frames as untracked. Exercise/planar/rolling whole-video limits are unchanged.

## 3D rigid-object scenes (`spatial-rigid-v1`)

Verified 3D rigid-object runs can be opened directly in the same local lab without modifying their sealed inputs or claiming physical validation:

```sh
# Optional: materialize a standalone analytic 3D fixture on a fresh checkout (no API calls).
.venv-lab/bin/python -m weft spatial-fixture --out artifacts/my-3d-fixture-01

# Package a verified 3D run into a new inspectable lab directory without starting a server.
.venv-lab/bin/python -m weft spatial-lab \
  --run artifacts/my-3d-fixture-01 \
  --out artifacts/my-3d-lab-01

# Or import and launch the loopback lab in one step (add --no-serve for terminal-only packaging).
.venv-lab/bin/python -m weft lab \
  --scene-run artifacts/my-3d-fixture-01 \
  --out artifacts/my-3d-lab-02

# Reopen an already-packaged 3D lab later.
.venv-lab/bin/python -m weft lab --resume artifacts/my-3d-lab-01
```

In **3D scene** and **Sandbox**, synchronized front (`x / height`), top (`x / depth`), and rotatable `3D` views show the same MuJoCo rigid-body state. Rotating yaw, zooming (`1×–3×`), or enabling **Follow moving objects** affects only your view and never steps physics. All eight actions (`coast`, `left`, `right`, `forward`, `back`, `up`, `down`, `brake`) are exposed through buttons and keyboard (`A/D` or Left/Right for `x`, `W/S` or Up/Down for depth `y`, `Q/E` or PageUp/PageDown for height `z`, `Space` to slow down). Live values inspect per-object 3D position, quaternion orientation, linear velocity, world spin, virtual force, substeps, and active contact forces. Untrained 3D runs keep **Learned controller** disabled rather than fabricating a policy; recordings save under `sessions/` and export to `video.mp4`, `preview.gif`, and `replay.html`. See [3D rigid-object sandbox](spatial-rigid.md) for contract limits.

### Multi-object controls and capability inspection

For multi-object 2D (`planar-rigid-v1/v2`) and 3D (`spatial-rigid-v1`) scenes with two or more dynamic bodies, **Sandbox** exposes **Object to control** (`/api/actor`) so you can switch which dynamic body receives manual forces without resetting live physics or altering `run/scene.json`. Static obstacles/supports cannot be selected, switching locks while a recording is active, and **Learned controller** automatically restores the trained object before stepping.

Every ready lab and sealed run also exposes a validated `environment-capabilities-v1` contract (in **Generated world / 3D scene / 3D body → Environment & RL contract**, `/api/status`, and the CLI):

```sh
.venv-lab/bin/python -m weft capabilities \
  --run artifacts/my-3d-lab-01 \
  --out artifacts/my-3d-capabilities.json
```

The capability summary explicitly declares representation family, physics backend, coordinate frame/units, controllable actors, action/observation spaces, authored task/reward/termination/reset rules, available controllers, `physical_validation: "unverified"`, and unsupported capabilities.

## Terminal-only processing

Add `--no-serve` to run all processing with stage/progress messages in the terminal. Open the saved result later with `lab --resume`. Optional `--episodes` and `--eval-episodes` bound local learning (defaults: 1,500 and 100). A free local port is selected unless `--port` is specified.

## Reproduce the environment

The full lab uses a separate `.venv-lab`; the original `.venv` is preserved. Versions are pinned for the tested Python 3.14/macOS arm64 environment. Other platforms have not been verified against this lock.

```sh
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -r requirements-lab.lock
.venv-lab/bin/python -m pip install --no-deps -e .
.venv-lab/bin/python -m weft fetch-pose-model
.venv-lab/bin/python -m weft fetch-exercise-sample
```

Downloads are explicit. Once dependencies, the pinned 5.8 MB pose model, and a video are present, processing is offline. OpenCV contrib supplies `cv2` for both the lab and MediaPipe; do not install overlapping OpenCV distributions into this environment.

## Execution and cost bounds

- Local CPU only: $0 external API/cloud charges; no recurring service is provisioned. Local electricity/storage are not measured as zero cost.
- Initial full dependency downloads are roughly 150–200 MB; installed environment and retained runs occupy more disk. Model: 5.8 MB. Included exercise sources: under 1 MB combined. Values are approximate except pinned model/source byte counts.
- One pipeline subprocess per lab; 300 CPU-second limit, 360-second wall limit, 512 MiB per output-file limit. Pose inference has a 180-second budget; training/evaluation has a 120-second budget. No hard whole-process memory isolation is claimed.
- The worker receives a small environment allowlist, not inherited service credentials. Only repository-owned templates/code execute; no generated arbitrary Python is evaluated.
- The server binds only `127.0.0.1`, requires a per-launch token for data/control, checks Host/Origin, limits concurrent HTTP handlers, and serves only that lab's artifacts. Recording export runs off the request thread. No telemetry or remote viewer assets.
- Source scans and video decoding remain bounded. UI requests are serialized; status polling stops after completion and pauses when hidden.

## Verification limits

The implementation is tested through Python integration tests, actual loopback HTTP calls, recorded-video full decode, and deterministic JavaScript DOM/canvas mocks. Source/pose/body preview images were inspected. Browser playback, real-device interaction, and sustained browser rendering performance remain **unverified** because no browser interaction was requested. See [local lab verification](lab-verification.md) for actual runs and measurements.

## Diverse-video development

Run the [pinned suite](../benchmarks/README.md) to test reconstruction, learning, interaction and recording together. [Current results](diversity-verification.md) retain three real-video failures and body-model limitations. Meta interaction advice is separate and opt-in; see [bounded access setup](meta-advisor.md).

## Spatial exercise model

Use `--body-model spatial` for the experimental 19-axis body, or `--body-model hinge` for the existing 8-axis model. The viewer exposes every axis and keeps raised limbs inside fixed view bounds. [Ready-to-open example, measured fidelity and remaining failures](spatial-human.md). Neither model establishes physical balance or measured muscle forces.

### Optional left/right correction

Add `--pose-identity temporal` to correct only short, strongly supported left/right swaps using nearby poses. The default `raw` mode leaves detector assignments unchanged. Both preserve original detections in `pose-landmarks.json`; `identity-audit.json` records corrections and abstentions. This cannot verify initial anatomical identity or resolve multiple-person ambiguity. [Evidence and ready-to-open example](temporal-identity.md).
