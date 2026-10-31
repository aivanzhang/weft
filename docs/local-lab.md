# Weft local lab

Supply a local video, inspect each processing stage, interact with the generated environment, and record your actions. The CLI starts a loopback-only lab and prints its browser link. The default local workflow uploads no video and makes no paid call. Optional Meta tracking is a separate, explicitly budgeted path.

For bodies and rigid objects in one simulator, use the [coupled 3D sandbox guide](shared-world.md). The shared scene binds body evidence and equipment masks to one source, with passive equipment proxies and verified recordings. Depth, materials and hand support remain unverified; it is not a recovered physical exercise.

## Recommended exercise sandbox

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example exercise
```

Open the printed local link. Choose **Sandbox → Learned controller → Record & start**. You can pause, step, switch to manual joint controls, reset, then **Stop & save** to export your recording. You start and manage the server; the launcher never opens a browser.

For terminal controls without a server, add `--console`. Nothing moves until you enter `step`; `help` lists controls and `record start` begins a seeded recording.

This single-leg squat example has a 3D body, illustrative muscles, 22 joint/root controls and the full 234-step/7.8-second reference. The newly trained controller and authored reference controller each pass **3/3 changed-start trials**; coast and random actions pass 0/3. Learned reward is 0.9206 versus 0.9167 authored, but learned peak joint error is slightly worse (0.667 versus 0.649 rad). Root assistance is explicit. These are simulator results on one exposed clip, not accurate anatomy, unassisted balance or arbitrary-video reconstruction.

The preset pins trusted runtime **`4ad01536dbbfe121b30b82192f96500d33f7108a`** and `artifacts/person-clock-workflow-01/lab`. It does not retrain or modify the older examples below. Its metadata-only `--console --check` passed in 0.195 seconds: 198 runtime files/3,849,085 bytes, 26 policy/18 lab pins, no server or physics. Saved source evidence is reused; the new training/evaluation/lab run took 30.7 seconds with no API charge. The [verified recording replay](../artifacts/person-clock-workflow-01/lab/sessions/session-b0beff1e2d13/replay.html) and [video](../artifacts/person-clock-workflow-01/lab/sessions/session-b0beff1e2d13/video.mp4) preserve 242 actions, a reset and 244 frames. The full 234-step post-reset trial passes and replays exactly. [Evidence and limits](handoffs/combined.md#reference-velocity-clock-correction--local-verification).

## Try the complete video-to-lab example

From the repository root:

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example video
```

Open the complete printed local link and keep the terminal running. Use **Sandbox** for manual or controller-driven movement, then **Record & start → Stop & save**. This example was built from the original kettlebell video through pose recovery, shared-world construction, bounded training/evaluation and one final lab in **29.6 seconds**. It still uses explicit equipment/camera assumptions; it is not automatic arbitrary-video reconstruction. Training found no improvement and task success remains **0/3**.

Open the [step-by-step report](../artifacts/native-crop-inspection-01/index.html) or the [verified recording replay](../artifacts/native-crop-build-02/learning_workflow/lab/sessions/session-658e7650b073/replay.html) independently. Recording acceptance covers 262 actions, one reset, all 254 post-reset steps and 264 exported video frames, with exact replay. These checks do not establish physical accuracy or browser visual quality.

This preset pins trusted Git runtime **`d0cdbe87d9f89b002e13c969ca1ac1549ae20107`** and the v2 lab at `artifacts/native-crop-build-02/learning_workflow/lab`. It checks exactly **21 policy/17 lab source pins** and installed versions; recordings stay in that lab's `sessions/`. The actual `--example video --check` passed in **0.178 seconds** with **190 files /3,602,474 bytes**, `.local/playable-video-launcher-verification-02.json`. It starts no server or physics; complete source/training/evaluation admission runs when opening the lab.

The no-flag shortcut below retains the original sandbox and its separate frozen runtime. `--lab` can select another saved lab only if it matches the chosen preset's exact profile and source pins; neither preset executes artifact-supplied Python or silently replaces a modified cache.

## Try the person-only 3D example

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example person
```

This real single-leg squat example uses one 3D body, ground, 22 controls, and the complete **234-step /7.8-second** reference interval, without dummy equipment. Open the printed local link; **Sandbox → a control mode → Start** runs it. The user starts/manages the server. [Inspect the automatic step-by-step report](../artifacts/single-leg-person-workflow-01/report/index.html).

The video→body→world→training/evaluation→lab/report workflow completed in **23.7 seconds**, with 235 original source slots and 118 pose references preserved. The selected controller equals the authored baseline and passes **0/3** task-success trials: peak joint-tracking error is about **0.958 radians**, above the unchanged **0.70** limit. A playable environment is not a successful learned exercise policy or physical reconstruction. Source footage and all these seeds are exposed development evidence.

Use **Record & start → Stop & save** to capture your own actions. The [verified saved replay](../artifacts/single-leg-person-workflow-01/build/learning_workflow/lab/sessions/session-f45b857fbde8/replay.html) and [video](../artifacts/single-leg-person-workflow-01/build/learning_workflow/lab/sessions/session-f45b857fbde8/video.mp4) contain **242 actions, one reset and244 frames**; all 234 post-reset steps replay exactly. The interaction/export/replay check took **8.36 seconds**. Exact simulator replay does not establish physical accuracy or browser visual quality.

Choose **Explain this trial** under a saved recording for body-error values, allowed limits and the first recorded crossing window. In this example, peak angle error crosses its limit between0.633–0.667s after reset (event29, replay state30/244). The running average crosses later but recovers; the peak still fails. Source-video, reference and live clocks remain separate.

The shortcut uses trusted Git runtime **`e1c54236ebf625264e2cb335cab65ba53a907ac1`**, including the saved body-failure explanation and person-only recording captions. The metadata-only `--example person --check` verified the unchanged22policy/18lab source pins and192exported files in **0.165seconds**; it did not start a server, run physics or establish browser performance. Full lab/source/policy admission still runs on opening. The original and `--example video` runtimes are unchanged; the prior person cache is preserved.

## Interact and record in the terminal

Use the same saved world, task and controllers without a server or browser:

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example feedback --console
```

The shortcut pins trusted runtime **`6003c96533a6f9cb610a9b162a498ebecd02a3ec`** and `artifacts/person-feedback-workflow-01/lab`, so development changes do not replace this example's behavior. It does not rebuild, train, or start movement automatically. Omit `--console` to use the 3D browser lab instead. Existing default/video/person examples remain unchanged; their older runtimes do not have terminal mode.

The metadata-only `--example feedback --console --check` passed in **0.169seconds**:198trusted runtime files/3,845,323bytes and exact26policy/18lab pins. It starts no server or physics; complete source/task/policy admission still runs when opening the lab. This verifies runtime compatibility, not browser performance. For another compatible saved lab, use `.venv-lab/bin/python -m weft lab-console --resume /path/to/lab` directly.

Enter these commands in the terminal:

```text
controls
record start
mode learned
step 6
motor torso_twist 0.25
step
reset
mode reference
step 6
record stop
files
quit
```

`controls` lists the actual available motors, actions and controllers. `motor` selects one motor and leaves others at zero; selecting a control does not move anything until `step`. Use `mode coast` to release control. `record start` resets the trial; later resets are recorded. `files` prints paths to the video, interactive replay and action data as export completes. Quitting or pressing Ctrl+C at the prompt saves an active recording and finishes its export. Native failures preserve the journal and print the appropriate verification or non-destructive recovery command.

The real exercise CLI check took **2.75 seconds**:11manual/controller actions, one reset,13recorded events, six learned decisions and exact action/state replay. The completed lab's861sealed files and prior recording stayed unchanged. This short interface check does not replace the full-task evaluation, which remains0/3success. The terminal shares the lab's existing recording/export bounds; it does not introduce a hard native-memory or export timeout.

## Build from your person-only exercise video

From the repository root:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --video /path/to/exercise.mp4 --person-only \
  --root-assistance --train --serve
```

This uses the existing pipeline, then automatically saves a step-by-step report. On success, `--serve` starts the local lab and prints its link; it never opens a browser. Choose **Sandbox → a control mode → Start**, then **Record & start → Stop & save** to export your trial. Without `--serve`, it prints the command to start the lab later. A rejected input still gets a report when its failure is sealed, but no sandbox starts.

Use `--console` instead of `--serve` to enter the terminal controls after a successful build and report. The two options are mutually exclusive; failed builds never start either interface. With neither option, the launcher prints both reopening commands.

- `--person-only` declares one selected person, ground-only support, and no task-relevant equipment or obstacles. It does not detect their absence.
- `--root-assistance` enables three virtual body-position controls; this is not recovered human balance. Omit it to disable that assistance.
- `--train` runs eight search episodes and compares the result with baselines on three separate starts each. Seeds are fixed, exposed development seeds; successful packaging is not proof of improved learning. Omit it to build an untrained lab.
- Optional `--policy-family feedback-residual` learns small joint corrections from current position/velocity errors. It requires `--train`, keeps the same eight-episode budget and leaves root assistance authored. Without it, the existing phase-based search remains the default. See [controller differences](shared-exercise.md#train-and-compare-a-frozen-policy).
- The default is the existing local **Lite / VIDEO / raw** pose recipe. No download or model fallback occurs. Optional `--pose-recovery selective-crop` explicitly selects the local **Full / IMAGE / raw** recipe.
- Each attempt gets a new `artifacts/video-…/` directory containing `build/`, `report/index.html`, bounded logs and workflow receipts. Inputs and completed builds are never overwritten. Use `--out` to choose a new directory.

To explain a saved failure **without reprocessing the video**, point to its inner build directory:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --inspect-build artifacts/video-EXAMPLE/build
```

The report shows the stopped stage, exact source artifacts, saved quality checks and affected original frames when available. Competing pose estimates do not necessarily mean multiple people; nominal video time and saved decoder time remain separate. Existing lab/runtime compatibility checks still apply on reopening after code changes; the fixed example launchers retain their own frozen runtimes.

Use [`shared-exercise-build --person-only`](equipment-proposals.md#person-only-video-without-equipment-files) for direct access with an explicit local pose model. It uses the same pipeline and assumptions as the launcher, not a separate product. Equipment absence is declared, not detector-verified; pose, geometry or task failures remain visible.

### Reuse an extracted body

If video analysis already produced a successful `exercise-body` package, reuse
it to build a new world, train, or inspect the results without analyzing the
video again:

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --human-run artifacts/single-leg-person-workflow-01/build/body_source \
  --person-only --root-assistance --console
```

This accepts a sealed construction-only body package, not a saved lab or rejected
pose estimates. It keeps the original source, poses, missing frames and model
identity. The existing constructors verify the complete package; the launcher
also checks that the new build consumed that exact saved body.

Add `--train` to run the same bounded learner, or use `--serve` instead of
`--console` for the 3D lab. Without training, learned control is unavailable.
For equipment scenes, replace `--person-only` with the existing `--evidence-run`
and `--config` inputs. Model, pose-recovery and source-note options apply only
to raw video and cannot replace saved provenance. Each build uses a new output
directory; reuse does not overwrite the original or repair its uncertainties.

The real saved-body build/report check completed in **2.22 seconds**, preserving
all 235 source slots, 118 poses and the 234-step task, with zero new inference,
training or simulated actions. Its [report](../artifacts/saved-body-sandbox-01/report/index.html)
and untrained lab are separate from the recommended trained exercise example.
The saved-output audit is `.local/saved-body-sandbox-verification-02.json`;
57 focused launcher checks and independent code review also pass.

### Check suspicious motion using saved images

Optional `shared-exercise-pixel-check` compares local image-patch motion with the original estimated joint motion. It accepts sealed pose evidence and a sealed extraction containing two to four original PNG frames; it never decodes the video again, runs a pose model, or changes the body. This is an advanced check on selected frames, not a whole-video accuracy test.

```sh
.venv-lab/bin/python -m weft shared-exercise-frames \
  --pose-run artifacts/single-leg-person-workflow-01/build/body_source \
  --frames 16 18 20 22 \
  --out artifacts/my-source-frames-01

.venv-lab/bin/python -m weft shared-exercise-pixel-check \
  --pose-run artifacts/single-leg-person-workflow-01/build/body_source \
  --extraction-run artifacts/my-source-frames-01 \
  --out artifacts/my-pixel-check-01

.venv-lab/bin/python -m weft shared-exercise-inspect \
  --run artifacts/single-leg-person-workflow-01/build \
  --pixel-support artifacts/my-pixel-check-01 \
  --out artifacts/my-pixel-report-01
```

The separate report card distinguishes local patch disagreement, no detected disagreement, and unavailable evidence. Agreement does not prove that a patch belongs to the correct joint or that the 3D pose is accurate. Occlusion, repeated texture, large motion, unsupported clocks and time limits stay visible; original pose-quality decisions and task results remain unchanged. Every output needs a new directory. The producer's five-second cooperative limit includes packaging; a partial result is not a completed check. The normal report command remains unchanged when `--pixel-support` is omitted.

`shared-exercise-frames` supplies the missing extraction step. Choose two to four increasing original frame IDs retained in raw pose evidence. It verifies the sealed source bytes, decodes one prefix without seeking, and preserves full-resolution pixels and native timestamps. Unknown or different clocks reject; frames are never resized or renumbered. Limits:64MiB source,1,800reads,4MiB/12output files and30seconds plus up to2seconds cleanup; native decoding runs in one supervised child. Memory is not hard-capped. The real four-frame extraction took0.427seconds/23reads, saved1,363,271bytes and reproduced all original RGB hashes/clocks. One PNG has different compression bytes, not different pixels. No inference, simulation or repeated patch matching ran.

The checked local [example report](../artifacts/single-leg-pixel-inspection-01/index.html) retains36joint-pair checks:5conditional disagreements,2without disagreement and29unavailable. Diagnostic and report together took0.603seconds with no new video decoding, inference or simulation. This is a selected-frame warning, not a repaired reference or accuracy score.

Other extraction producers must supply `source-frame-pixels-v1`: a schema1manifest and `report.json`, both with `status: pixels_extracted`. The report binds `source_sha256`, declares no resampling/automatic repair/API/inference/physics, and lists2–4ordered PNG rows containing `frame`, nominal `time_s`, `decoder_timestamp_s`, `file`, PNG `sha256`, `decoded_rgb_sha256`, `source_sha256`, `width` and `height`. Its manifest must cover the report and exact PNG bytes. Extraction provenance is supplied by that producer, not independently re-established by the pixel checker. The earlier `single-leg-source-pixels-v1` receipt is accepted through an explicit time-field compatibility adapter.

For a person interacting with equipment, the same launcher accepts `--evidence-run`, `--config` and optional `--source-mapping` instead of `--person-only`. Equipment geometry/camera assumptions remain explicit; it does not silently remove objects. Local build/report processing makes no paid API call and inherits no provider-key environment variables. Native calls run in a bounded child process; native memory is not hard-capped. The user owns any server started with `--serve`.

## Original full-reference sandbox

The five inspection shortcuts expose **Original → Tracks → Geometry → Sandbox → Recordings**, including saved initial-geometry panels and compact provenance. The [human-only exercise sandbox](shared-exercise.md) checks body tracking and equipment stability throughout a declared interval, with manual/coast/reference controls, an optional validated policy and exact recorded action replay. Prepare it explicitly with `shared-exercise-lab`; existing labs retain their task and virtual object controls. To combine a saved mobile-body lab with equipment evidence, use [one-command `shared-exercise-build --human-run … --full-reference`](equipment-proposals.md). It prepares the body directly, then creates one continuous task and one lab; no separate shared-world adaptation or intermediate lab is needed. Camera distance, body scale, box shape and unknown depth remain explicit assumptions.

To try the current full-reference human-plus-equipment run, start the lab yourself from the repository root:

```sh
.venv-lab/bin/python scripts/play_sandbox.py
```

1. Open the complete printed `127.0.0.1` link, including its temporary token. Keep the terminal running; Ctrl+C stops the lab.
2. Choose **Sandbox → Searched joint corrections → Start**. Selecting a controller alone leaves movement paused.
3. To act yourself, choose **Move it yourself**, select **Human motor to move**, then tap **Motor +0.25 / −0.25** or start and hold. Equipment stays passive.
4. Choose **Record & start** for a fresh recorded trial, then **Stop & save**. Download **Interactive replay**, **Video**, **Preview** or **Action data** under Your recordings; **Explain this trial** shows saved equipment failures.
5. Inspect **Video** for the original and source-observation downloads, or **Coupled 3D world** for rotatable views and estimated-pose replay. Source playback and simulation clocks are independent.

The launcher uses trusted committed runtime **`6396be84018e6cb6525614f07a664cf0bb673b34`**, not mutable development code or Python copied from a supplied artifact. It checks the 24 policy and 17 lab source pins plus installed dependency versions. Its bounded local cache under `.local/playable-runtime/` is verified, never silently overwritten or replaced with development code. Recordings remain in `artifacts/feasibility-fullspan-lab-01/sessions/`. Keep the existing `.venv-lab` and saved artifacts; no new key, install, inference or training is required on this checked workspace.

Optional `.venv-lab/bin/python scripts/play_sandbox.py --check` verifies the runtime and sealed lab metadata without starting a server or physics. The actual check passed in **0.183 s** with **167 files /3,108,210 bytes**, receipt `.local/playable-launcher-verification-01.json`; this is one startup check, not sustained browser performance or a new full lab admission. Complete source/training/evaluation admission still runs when the sandbox opens. Direct `.venv-lab/bin/python -m weft lab --resume artifacts/feasibility-fullspan-lab-01` remains supported while the checkout matches its pinned runtime; use the launcher as development advances.

The task runs **254 continuous actions / 8.47 seconds**, covering every retained body reference from 0.0667 to 8.5333 seconds. It does not invent the missing opening poses. All 257 source-frame slots and 121 body poses remain inspectable; native video timestamps and the estimated pose clock remain distinct.

**Source vs simulation** shows the retained reference, selected task interval and actual live/last-valid clock. Source playback is independent, not automatically synchronized. The full-reference preparation receipt explains the changed interval and preserved source files. Missing geometry panels are not presented as new fit evidence; the [offline shape comparison](../artifacts/silhouette-shared-world-03/shape-comparison.html) remains separate.

The 19 joint controls plus three explicit virtual root controls operate the human; equipment remains passive. Choose manual movement, coast, authored following or **Searched joint corrections**. The current v3 policy prioritizes the task limits, but it and the prior reward-selected policy still fail equipment stability on **all three fresh test starts**. Both finish the full interval; worst equipment displacement falls **0.4484→0.2496 model units**, still above **0.03**. The selected candidate came from saved training; no new refinement candidate beat it. The earlier two-second policy's 3/3 result does not apply here. Training choice, independent evaluation and live outcome remain separate.

Use **Record & start → Stop & save** to export the action journal, video and exact replay, including resets and controller identity. See the [full-reference task guide](shared-exercise.md#full-reference-episodes) for preparation, bounded training and the current acceptance evidence.

For a completed recording, choose **Explain this trial**. The equipment-stability card shows the first reported failing limit and its time window, with a separate episode for each reset. Expand **Evidence and limits** for reference-frame gaps, exact journal identity and all metric crossings. This background read does not advance physics, decode video or change the current trial. Saved contacts identify collision shapes present at the recorded boundary, not proven causes between boundaries. Long or incompatible recordings return an explicit unavailable result instead of a partial timeline presented as complete.

The current saved example is `session-04b7c0ac12c4`: 262 actions, one reset and 264 journal events, with all 254 post-reset steps and exact replay verified. Its final episode first reports speed/spin failure at **1.266667 s**, after a safe report at **1.233333 s**. That bracket is a saved simulation clock, not a native-video timestamp or exact contact-onset measurement.

Older labs and recordings are unchanged. Their exact runtime pins intentionally reject changed physics/learner source: use their archived implementation to rerun them, or prepare a new explicitly evaluated lab. The new runtime reproduced all 68 recorded actions and 70 snapshots/rewards from the previous two-second example exactly; that numerical check does not silently migrate its old policy or transfer its success to the longer task.

A shared exercise build can now be inspected even when it stops before creating a lab. Use [`shared-exercise-inspect`](equipment-proposals.md#inspect-a-completed-or-stopped-build) to make a local HTML report with the original video, pose previews, completed/stopped stages and preserved failure; no server or rerun is needed.

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

For a human-and-equipment world, use [`exercise-body`](equipment-proposals.md#prepare-body-evidence-without-a-lab) to prepare the body directly, without creating/training this body-only lab first. It accepts a raw video plus a local pose model, or reuses sealed pose evidence without inference. Then pass its output to `shared-exercise-build --human-run`, together with sealed equipment observations and explicit camera/equipment configuration. Existing mobile labs remain accepted. Only the final admitted `lab/` is playable; body-source evidence is construction-only. Equipment tracking/configuration are still required separately.

To avoid separate body/training commands, use the [combined video-to-trained-lab command](equipment-proposals.md#video-to-one-trained-lab). It checks inputs before inference, shows every stage, retains the same task through training and comparison, and creates one final lab with recording controls. Equipment evidence and camera/shape assumptions are still explicit prerequisites.

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

Use [`weft qualify`](runtime-qualification.md) to check a sealed environment's complete interval, exact seeded action replay and changed-control response without starting a server. It uses one protocol across object/body/shared-world action spaces and saves a separate result directory. Operational checks, original task outcomes and physical accuracy remain distinct; no learned policy is trained or evaluated by this command.

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
