# Weft local lab

Supply a local video, inspect each processing stage, interact with the generated environment, and record your actions. The CLI starts a loopback-only lab and prints its browser link. The default local workflow uploads no video and makes no paid call. Optional Meta tracking is a separate, explicitly budgeted path.

For bodies and rigid objects in one simulator, use the [coupled 3D sandbox guide](shared-world.md). The shared scene binds body evidence and equipment masks to one source, with passive equipment proxies and verified recordings. Depth, materials and hand support remain unverified; it is not a recovered physical exercise.

## Start here

- **Play with a ready exercise:** [launch the exercise sandbox](#recommended-exercise-sandbox).
- **Try your own video:** [build a person-only 3D environment](#build-from-your-person-only-exercise-video).
- **Understand a result:** [inspect saved evidence](#inspect-the-labs-saved-evidence) and [check a recording](#check-a-recordings-replay-evidence).

The ground, equipment and second-camera sections below are optional advanced
workflows, not prerequisites for trying the sandbox.

## Collect missing ground or equipment inputs

Use `surface-inputs` when you have saved video observations but need help preparing
support annotations. It creates an offline worksheet, shows what is missing and
keeps the original files unchanged. This is an assisted preparation tool; it does
not infer ground, prove contact or replace the playable sandbox.

**1. Create the worksheet.** Supply the saved observations. Add `--camera` and
`--body` only when you have the corresponding original files.

```sh
.venv-lab/bin/python -m weft surface-inputs \
  --observation path/to/observations.json \
  --camera path/to/camera.json --body path/to/human.json \
  --out artifacts/my-surface-choices-01
```

Open the printed report path. Choose the observed entity, name the image point
(for example, `left_ankle`) and name its possible support surface. These are
explicit choices, not detected contact. Download the selection JSON.

**2. Inspect the frame-by-frame draft.** Use your downloaded selection:

```sh
.venv-lab/bin/python -m weft surface-inputs \
  --resume artifacts/my-surface-choices-01 \
  --selection path/to/exercise-surface-selection.json \
  --out artifacts/my-surface-draft-01
```

The new report lists missing fields and every original frame. Download
`collection-request.json` and edit a copy. Its annotation templates preserve the
frame and source identity. Keep unsupported pixel locations, surface heights,
body mappings and uncertainty unknown; confidence is not a measured error bound.

**3. Check the supplied inputs.** Save the result in another new directory:

```sh
.venv-lab/bin/python -m weft surface-inputs \
  --resume artifacts/my-surface-draft-01 \
  --request path/to/edited-collection-request.json \
  --out artifacts/my-surface-ready-01
```

The report either explains the remaining unknowns or links `surface-config.json`.
That config is an input to the separate fitting commands below, not permission
to install a body. Each completed worksheet remains usable even when its inputs
are incomplete. No video decoding, model inference, simulation or paid call runs
in these three steps. [Exact input contract and limits](handoffs/exercise-surface-workflow.md).

Inspect the [completed example worksheet](../artifacts/exercise-surface-cli-acceptance-03/qualified/inspection/index.html)
or its [initial selection screen](../artifacts/exercise-surface-cli-acceptance-03/discovery/inspection/index.html).
This tiny authored fixture checks the workflow, not accuracy on an unfamiliar video.

## Inspect known ground or support points

When you have saved observations, supplied camera calibration and explicit surface
annotations, `surface-anchors` creates an offline, step-by-step report:

```sh
.venv-lab/bin/python -m weft surface-anchors \
  --observation path/to/observations.json --source-sha256 ORIGINAL_VIDEO_SHA256 \
  --camera path/to/camera.json --surfaces path/to/surfaces.json \
  --annotations path/to/annotations.json --out artifacts/my-surface-points-01
```

Open the printed `index.html` path. It shows preserved inputs, assumptions,
conditional 3D points and missing-frame reasons. The output directory must be new;
its parent must already exist. The complete timeline stays in `anchors.json`.

This is an optional calibration tool, **not automatic exercise reconstruction**.
It does not guess the camera, fit a body, install ground or prove contact. Missing
inputs produce an unavailable diagnostic. Authored inputs stay authored; contact,
load and grasp stay unknown. [Input schemas and a complete authoring example](handoffs/source-surface-anchors.md#public-api-and-concrete-authoring-example).

The [tiny authored example report](../artifacts/source-surface-cli-report-02/index.html)
retains one conditional point and four unavailable slots. Its source is a synthetic
descriptor, not a video or physical-validation result. Existing video sandbox and
recording workflows below remain separate from this diagnostic.

## Inspect saved wrist and equipment evidence

For a **completed** equipment-image study, create an offline report without
reprocessing the video or running the simulator:

```sh
.venv-lab/bin/python -m weft equipment-evidence-report \
  --run path/to/completed-equipment-study \
  --out artifacts/my-equipment-report-01
```

Open the printed `index.html`. Choose a page, original frame, wrist and object.
The report shows image overlap or separation, authored uncertainty, original
timestamps and why unavailable comparisons could not be made. Every frame and
pair remains in the raw downloads. The output directory must be new.

Image overlap is **not proof of touch**; image separation is not proof of physical
non-contact. Contact, support, grasp and load remain unknown. The command verifies
the selected saved metadata bytes, not all original video/mask evidence, and does
not turn a failed study into a successful result. A missing completed manifest
means the study must be resolved before this report can be generated.
[Implementation, checks and current source status](handoffs/equipment-silhouette-inspection.md).

The [completed kettlebell example](../artifacts/pose-equipment-inspection-01/index.html)
retains all 259 source frames and 1,036 wrist/object pairs. It shows 64 conditional
image overlaps, 146 separations and 826 unavailable comparisons. Only the explicit
construction selection was checked for partition consistency; excluded labels
remain unreconciled. This is an exposed development clip, not an unfamiliar-video
or physical-contact validation. The report opens without rerunning processing.

## Check possible wrist and equipment motion matches

For a completed pose/equipment study, compare the original motion-based
candidates with its saved image silhouettes:

```sh
.venv-lab/bin/python -m weft equipment-support \
  --run path/to/completed-equipment-study \
  --fixed-camera --out artifacts/my-equipment-support-01
```

`--fixed-camera` explicitly assumes the camera did not move; it is not measured
calibration. Open the printed `index.html`, then choose a frame, wrist and object.
Each candidate shows its original time window and whether image evidence kept
it, excluded it, or left it unknown. Missing points and later frames remain visible.

The [completed kettlebell comparison](../artifacts/equipment-support-01/index.html)
rejected both original motion-based candidates. It retains all 259 source
frames, 129 supplied poses and 516 pose/pair slots. No equipment support or
physical contact was established. Inspect frames 20 or 22, the right wrist and
tracked object `sam-2` to see the rejected windows. The command reuses saved diagnostics; it does
not decode the video, run pose inference, change the sandbox, or install a body.
The output directory must be new. [Exact evidence and limits](handoffs/equipment-support-workflow.md).

### Include wrist and finger points

If the completed study retains its original pose/audit files and decoded binary
masks, add `--hand-points`:

```sh
.venv-lab/bin/python -m weft equipment-support \
  --run artifacts/pose-equipment-silhouette-02 \
  --fixed-camera --hand-points --out artifacts/my-hand-comparison-01
```

The report shows wrist, pinky, index and thumb separately. A wrist can lie outside
an object's outline while a finger overlaps it. All four points must have usable
evidence; otherwise the hand result stays unknown. Overlap does **not** prove
touching, holding or carrying weight. The pixel ranges are authored assumptions,
not measured accuracy. No new pose model, video/PNG decoding or simulation runs.
Missing required cached masks stop the command instead of inventing evidence.
[Exact source custody, checks and limits](handoffs/equipment-hand-workflow.md).

You can open the [prepared hand comparison](../artifacts/equipment-hand-support-02/index.html)
without running anything. Inspect frames **20** or **22**, the **right hand** and
object **sam-2**. Both original seven-sample motion windows remain possible:
estimated finger points overlap the object mask even though the wrist-only
check excluded them. This does not establish physical contact or change the
sandbox's equipment physics.

## Fit a body using uncertain support ranges (experimental)

For an existing saved body, you can supply camera calibration, annotated image
points, a range of possible ground/support heights and an explicit body-to-world
mapping. The fitter penalizes positions outside those ranges; it does not pull
points toward an invented midpoint or move the floor.

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run path/to/sealed-exercise-body --objective surface-intervals \
  --surface-config path/to/surface-config.json --out artifacts/my-surface-fit-01
```

Open the printed report to compare the original and proposed body, inspect
supplied ranges and missing evidence, and see rejected corrections. The body is
**not installed automatically**. Existing image, floor and joint checks still
apply; unavailable inputs remain unknown. The numeric weight is an authored
tradeoff, not a physical measurement.

This is an optional assisted-construction path, not automatic contact recovery
from an unfamiliar video. Moving equipment, unsupported camera distortion and
missing calibration remain unsupported. [Exact config and bounds](handoffs/surface-interval-workflow.md#exact-configuration).

[Inspect the authored native test](../artifacts/surface-interval-fit-inspection-01/index.html).
The fitter completed; its outer checker failed on one blocked subprocess attempt.
The linked saved-only inspection preserves that failure and does not rerun or
promote the fit. It is not a new-video accuracy result.

### Use an existing camera-based body

For a sealed `two-view-root-body-v1` body, select the camera-aware mode:

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run path/to/sealed-camera-body --objective camera-surface-intervals \
  --surface-config path/to/surface-config.json --camera-image-scale 0.01 \
  --out artifacts/my-camera-surface-fit-01
```

The config must contain the **exact original camera bytes**. The example scale
is an explicit fitting choice, not calibration or a pixel-to-metre conversion;
allowed values are `0.000001`–`1`. Original camera, root motion, floor, first pose
and source times stay fixed. Supported positive radial distortion is preserved.

The report shows image error, supplied surface ranges, missing evidence and each
accepted or rejected correction. `proposal/angle-proposal.json` is an inspectable
angle overlay, **not a replacement body**. It cannot yet be installed in the lab
or used by `body-reference-trial`. Missing support height does not discard usable
image evidence, but it cannot establish support or contact. Moving equipment and
unknown body-to-world mappings remain unsupported.

[Implementation, verification and remaining work](handoffs/camera-surface-workflow.md).
[Inspect the completed authored native example](../artifacts/camera-surface-fit-native-02/inspection/index.html).
This example verifies the workflow, not accuracy on an unfamiliar exercise video.
The existing playable labs below are unchanged by this proposal workflow.

## Add feet to an existing world (experimental)

`shared-feet-build` adds two simulated feet and six ankle controls to an existing
person, equipment or static-support world. It preserves the original exercise
task, source clip, body, ground and scoring. Supply a foot-reference file bound to
that exact body; this command does not extract feet from a new video.

```sh
.venv-lab/bin/python -m weft shared-feet-build \
  --run path/to/sealed-world --foot-reference path/to/foot-reference.json \
  --task path/to/original-exercise-task.json --out artifacts/my-feet-candidate
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-feet-candidate --out artifacts/my-feet-lab
.venv-lab/bin/python -m weft lab-console --resume artifacts/my-feet-lab
```

Each output directory must be new. Omit `--task` only when the source package
already contains its explicit exercise task. Successful candidate packaging is
not successful lab admission: incompatible initial geometry still rejects.

In an admitted lab, try:

```text
controls
record start
motor left_ankle_z 0.1
step
reset
mode reference
step 6
status
record stop
files
record evidence
```

The original 19-joint body score stays unchanged. Ankle error and foot contacts
are separate, unscored diagnostics; the reference controller holds the starting
ankle angles, not later video foot poses. Root assistance stays exactly as the
original task declared. Learned policies, avatar skins, foot-support credit and
ground-load tasks are unavailable for this experimental extension.

Both authored static-support integration cases (root assistance off/on) passed
candidate → lab → console → export → exact replay → saved receipt → resume.
The real saved single-leg candidate **did not pass admission**: its right foot
starts inside the unchanged floor. It remains preserved at
`artifacts/coupled-feet-lab-acceptance-01`; no controls or replay ran on that clip.
Use the [recommended exercise sandbox](#recommended-exercise-sandbox) for the
existing playable video example. [Evidence and remaining reconstruction gap](handoffs/combined.md#lab-integration-preparation--october-31-2026-154-pm-pdt).

## Import a camera-based body (experimental)

For an existing sealed two-view body with supplied ground:

```sh
.venv-lab/bin/python -m weft camera-shared-build \
  --run path/to/sealed-camera-body/run --out artifacts/my-camera-world
```

This preserves the body, both camera/evidence documents and original video timing.
It writes `scene.json`, `camera-coupling.json`, `source-verification.json` and exact
input copies. `manifest.json` records whether construction succeeded. Each output
must be new. This importer does not accept arbitrary raw video. To make its
camera-based candidate interactive, use the existing lab:

```sh
.venv-lab/bin/python -m weft shared-exercise-lab \
  --run artifacts/my-camera-world --out artifacts/my-camera-lab
.venv-lab/bin/python -m weft lab-console --resume artifacts/my-camera-lab
```

Default controls are 19 body motors, with no virtual root forces. Add
`--root-assistance` when building the lab to explicitly enable three extra forces
that help the body follow its recorded path. This is assistance, not learned or
unassisted balance. In the console, use `controls`, `record start`, `mode reference`,
`step 6`, `reset 29`, `record stop` and `quit`. To inspect the graphical lab,
run `lab --resume` instead of `lab-console --resume`, then open the printed local
address yourself.

The saved jogging candidate is already available at
`artifacts/camera-shared-admission-jog-01/candidate`; do not rebuild it merely to
inspect it. Its body geometry passes the source/grid check but floats above the
ground. Foot attachment remains unavailable because the original starting right
heel is inconsistent across cameras. [Evidence and limits](handoffs/camera-shared-runtime.md).

The saved assisted lab uses its original frozen runtime. Current source correctly
rejects its old runtime pins; do not bypass that check or rebuild the saved run:

```sh
PYTHONPATH="$PWD/.local/camera-shared-policy-play-runtime-01/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/camera-shared-policy-play-01/lab
```

It retains the original 238-step / 7.94-second interval. Try `controls`,
`mode learned`, `record start`, `step 6`, `reset 29`, `record stop`, then
`record evidence`. Manual, coast and programmed reference controls also work.
The experimental trained controller adjusts joints; body-position assistance
stays programmed. The source, task and original policy have not been rewritten.
The frozen files and their saved pins were verified; this exact resume command
has not been newly executed. [Runtime and version requirements](handoffs/camera-policy-runtime-resume.md).

The complete console check recorded 246 actions, including 240 learned-controller
decisions, and replayed every action and full state exactly. Preparation, play,
export and resume took 13.4 seconds locally. Open the saved
[interactive replay](../artifacts/camera-shared-policy-play-01/lab/sessions/session-d730c17b1e7b/replay.html)
or replace `lab-console` with `lab` in the same frozen-runtime command for the graphical sandbox.
No browser was automated; graphical behavior has mock-based checks, not live-browser QA.

The earlier roots-on/off recordings remain unchanged under their archived runtime:
240 actions each, all 480 actions replayed exactly. Assisted tracking passed;
roots-off failed root tracking. [Historical recording evidence](handoffs/camera-shared-lab.md).
Feet, ground/equipment contact, unassisted balance and avatar skins remain
unqualified for this camera profile.

### Train and inspect camera-body corrections

Training is now available as a separate research artifact. It preserves the exact
saved task and explicit root assistance; it cannot override them with a flag.
Use new output directories:

```sh
.venv-lab/bin/python -m weft shared-exercise-train \
  --run artifacts/camera-shared-learning-01/lab/run --policy-family camera-feedback \
  --out artifacts/my-camera-training --seed 1729 --generations 2 --population 8
.venv-lab/bin/python -m weft shared-exercise-policy-evaluate \
  --learning artifacts/my-camera-training --out artifacts/my-camera-evaluation \
  --seed 924017 --trials 3 --jitter-scale .10 --max-seconds 60
```

Inspect `policy.json`, `training.json`, per-action `diagnostics/`, and
`evaluation.json`. Evaluation freezes the selected controller before comparing
it with coast, random and authored controls on the same separate starting states.
Previously used seeds are not fresh evidence merely because they are defaults.

The saved study selected nonzero corrections and increased average reward from
0.826211 to 0.827368 on three separate starts. Both learned and authored controls
passed all three trials; average joint error fell slightly, but peak joint error
and average root error rose. This is a small, mixed simulated-control result,
not learned balance or unfamiliar-video validation. The prepared lab above attaches
that exact policy and retains one evaluation archive with its embedded training
evidence. Its recording verifies learned decisions as well as full simulated
state. Ground/equipment contact and unfamiliar-video reconstruction remain
unfinished. [Results, bounds and exact evidence](handoffs/camera-shared-learning.md).

## Inspect static-support loads

Open the prepared support example:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/static-support-play-acceptance-01/present/lab
```

Choose authored reference control, then **Record & start → Stop & save**.
Inspect the declared body segment, load direction, minimum load, current simulated
load, contact coverage and gaps. Loads are fractions of simulated body weight.
Unknown values remain unavailable, rather than appearing as zero.

Use `lab-console` instead of `lab` for terminal controls:

```text
controls
record start
mode reference
step 6
status
record stop
files
```

The two-second analytical example passes with the support in place. The matched
[`lowered/lab`](../artifacts/static-support-play-acceptance-01/lowered/lab)
changes only support height by −0.6 world units: it finishes but fails root
tracking and contact requirements. Qualifying contact covers 91.78% of the
required window, below 95%; its longest gap is 0.121 seconds, above 0.10.
Both trials have exact recorded-action replay. The lowered case retains a
32.64-body-weight simulated impact; it is not a measured human force.

There are 19 explicit joint/torso controls and no translational root forces in
this example. The authored body template does not establish free-floating balance,
anatomy or biomechanics. [Support-policy training, comparison and recording](static-support.md#train-compare-and-record-a-support-policy)
are available; the verified search selected the unchanged authored baseline,
not an improved controller. Floor-force requirements and grasp are unavailable.
Avatar recording for this new support profile is not yet
qualified; launch it without avatar options. Existing learned object and person-only examples remain
separate supported modes in the same lab.

On a fresh clone, create the analytical fixture and lab without a server:

```sh
.venv-lab/bin/python -m weft shared-support-fixture --out artifacts/my-support-source
.venv-lab/bin/python -m weft shared-exercise-lab --run artifacts/my-support-source --out artifacts/my-support-lab
```

For a video or existing video-derived body, use the [guided support build](static-support.md#build-inspect-and-play-in-one-workflow) to construct, optionally train, inspect and open one final lab.
Support shapes and load requirements remain explicit authored assumptions.

## Play an object video without training

Add `--no-train` to a new **planar** or **appearance** video build. Tracking,
reconstruction checks and intervention checks still run; policy training and
policy evaluation do not. The resulting lab supports manual controls, reset,
recording and export. Learned controls remain disabled.

This verified local example tracks the declared first visible pass of an
existing grayscale ball clip, frames 39–60—not the entire 157-frame video:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/manual-video-acceptance-01/lab
```

Use `lab-console --resume` instead for terminal controls without a server.
To build a new run yourself, choose a new output directory:

```sh
.venv-lab/bin/python -m weft lab \
  --video data/general/rolling-ball-grayscale.avi \
  --profile appearance \
  --config data/general/annotations/rolling-grayscale.scene.json \
  --out artifacts/my-manual-ball-01 --no-train --no-serve
```

`--no-serve` builds without starting a server. `--no-train` cannot be combined
with training/evaluation episode counts or used to change an existing run.
The explicit setting is sealed; a missing policy in an older trained run is
still an error, not an automatic manual fallback.

The saved build took **0.92 seconds**. Six manual actions, a reset, eight exported
frames and an independent exact-action replay passed. The first acceptance
script used the wrong reset default; the corrected saved-only check used the
lab's explicit `jitter: false`, without reprocessing or exporting the video.
This is a 2D simulation with declared regions and assumptions, not reconstructed
3D physics. [Verification](../artifacts/manual-video-verification-01/report.json).

## Run the guided video-to-3D workflow

The existing builder now supports explicit object tracking, 3D assumptions,
optional training or saved-policy reuse, and a stage-by-stage artifact index.
Use the [complete command guide](spatial-scene-construction.md), or open the
verified local result without processing or training again:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/spatial-guided-play-acceptance-02/workflow/lab
```

Replace `lab` with `lab-console` for terminal controls. The
[saved workflow index](../artifacts/spatial-guided-play-acceptance-02/workflow/index.html)
links the source evidence, 3D assumptions and controller comparison. This example
uses an explicit segment of a ball video, authored zero gravity and no floor;
it is not recovered rolling physics. The learned policy passes 8/12 changed
starts, versus 12/12 for the separately tested fixed feedback controller.

The end-to-end check took 3.43 seconds, reusing the saved policy. Its recording
preserves 54 actions, one reset and 56 exported frames. One exact physics replay
passes; all 51 learned decisions also match the bound policy using the preceding
saved state. Three manual actions have no policy-decision claim. New spatial
recordings expose these separate checks under **Recording evidence**; legacy
recordings retain their original, narrower verification.

## Try the trained 3D object labs

Open either prepared lab without rebuilding or retraining:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/spatial-v2-play-01/authored
# Or use the source-linked, physically unverified ramp hypothesis:
.venv-lab/bin/python -m weft lab --resume artifacts/spatial-v2-play-01/ramp
```

Use manual controls to push an object, or select the learned controller and
start it. Change the view without changing physics. Record a trial, reset while
recording if desired, then stop/save and inspect **Recording evidence**. Actor
selection is locked during a recording. These examples use the preselected
seed-1729 policies, not the best policy chosen after evaluation.

For terminal-only interaction, replace `lab` with `lab-console` and try:

```text
controls
record start
action right
step 3
reset
mode learned
step 6
record stop
record evidence
files
quit
```

Both main examples passed their complete nominal simulated task. The integration
check took **2.39 seconds**:164 live actions +164 exact replays,24 existing packaging
probes, two recorded resets and three exports/169 frames, including a two-action
alternate-object recording. The alternate-object check is not a task success.
No retraining, source-video decode or API call occurred. The receipt reader added
no simulation or media work. [Learning results and scope](spatial-rigid.md#v2-learning-qualification).

## Recommended exercise sandbox

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example exercise
```

Open the printed local link. Choose **Sandbox → Learned controller → Record & start**. You can pause, step, switch to manual joint controls, reset, then **Stop & save** to export your recording. You start and manage the server; the launcher never opens a browser.

For terminal controls without a server, add `--console`. Nothing moves until you enter `step`; `help` lists controls and `record start` begins a seeded recording.

This single-leg squat example has a 3D body, illustrative muscles, 22 joint/root controls and the full 234-step/7.8-second reference. The newly trained controller and authored reference controller each pass **3/3 changed-start trials**; coast and random actions pass 0/3. Learned reward is 0.9206 versus 0.9167 authored, but learned peak joint error is slightly worse (0.667 versus 0.649 rad). Root assistance is explicit. These are simulator results on one exposed clip, not accurate anatomy, unassisted balance or arbitrary-video reconstruction.

The preset pins trusted snapshot **`225a53c8c0a9fb25f03b8473adfdf7f2754e037d`** and `artifacts/person-clock-workflow-01/lab`. Its updated evidence views and recording receipts preserve the original 26 policy/18 lab simulation-source pins, scene, task and policy. The actual metadata-only `--console --check` passes with 202 runtime files, no server or physics. It does not retrain or modify older examples. The original training/evaluation/lab run took 30.7 seconds with no API charge. The [verified full recording replay](../artifacts/person-clock-workflow-01/lab/sessions/session-b0beff1e2d13/replay.html) and [video](../artifacts/person-clock-workflow-01/lab/sessions/session-b0beff1e2d13/video.mp4) preserve 242 actions, a reset and 244 frames; the full 234-step post-reset trial passes and replays exactly. That older export predates durable receipts and is not backfilled. [Current evidence and limits](handoffs/combined.md#recording-receipt--verified).

### Inspect body-fitting proposals

The [image-first comparison](../artifacts/whole-body-reference-views-01/image-first/index.html)
and [floor-first comparison](../artifacts/whole-body-reference-views-01/floor-first/index.html)
show the latest saved body-fitting results without running another simulation.
Switch between retained source slots and control-grid checks, select a sample,
and inspect the original/candidate floor gaps and available joint-image errors.
Missing estimates remain unknown. Both proposals still fail the unchanged floor
limit and are **not installed in the working sandbox**.

Use `body-reference-report --run <sealed-study> --out <new-directory>` to build
another offline page. [Full command and verification](body-model-options.md#inspect-the-saved-proposals).

To inspect camera and body-position hypotheses while keeping the original joint
angles and body shape fixed, use the same proposal command with
`--objective camera-root --camera-focal-px 640`. The focal length is an explicit
authored assumption, not a recovered calibration. The new offline page preserves
missing observations and floor failures; it does not install a new body.
[Command, stage artifacts and limits](body-camera-root-refinement.md#try-it-on-an-extracted-body).

To propose a correction for **your own saved body**, use the normal CLI:

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run artifacts/my-exercise-body \
  --out artifacts/my-body-proposal-01
```

Open the printed `inspection/index.html` path yourself. It shows source samples,
control-grid checks, missing estimates, image errors, floor gaps and rejection
reasons. `proposal/` preserves the original inputs, separate candidate, frozen
code/settings, worker counts and logs. This command never changes the body,
policy or working sandbox. A completed command means the proposal and report
were saved, **not** that the candidate passed or was installed.

If fitting finishes but viewer generation fails, the CLI prints a separate
`body-reference-report` recovery command with the completed proposal's exact
manifest hash. Use a new output directory. This rebuilds only the viewer;
**do not rerun fitting** or overwrite the failed workflow.

The input must be a supported, sealed `exercise-body` construction output.
It reuses saved poses, without decoding video, running inference or making API
calls. The normal selective-crop recovery output is supported when its complete
proof is present. Weft checks the original video hash and every saved crop
decision; the report identifies targets from accepted crops. Both displayed
bodies use the same selected targets, so their errors are not a comparison of
the detector before and after cropping. Unknown producer versions, incomplete
recovery proofs, supplied cameras and already-corrected bodies stop explicitly.
Default fitting is image-first;
`--objective floor-first` selects the experimental alternative, not a fallback.
`--fit-seconds` can tighten the 30-second fit limit. See
[bounds and source verification](body-model-options.md#propose-a-correction-from-a-saved-body).

### Compare a fitted reference without changing the task

The saved camera-based jogging comparison is ready to inspect without rerunning:
[fit evidence](../artifacts/camera-reference-real-01/inspection-recovered-01/index.html)
and [simulation comparison](../artifacts/camera-reference-real-01/trial-03/index.html).
Both recordings cover all 238 steps and replay exactly. Fitted angles improve
agreement with detected image points, but fitted controls score slightly worse
on the original simulator task. Neither result verifies ground/equipment contact.
The interactive fitted sandbox is also ready. Use its verified frozen runtime so
ongoing reconstruction edits cannot silently change this saved environment:

```sh
PYTHONPATH="$PWD/.local/camera-reference-play-runtime-01/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/camera-reference-play-01/lab
```

Try `mode fitted`, `record start`, `step 6`, then `record stop` and `record evidence`.
Use `controls` to list motors, `motor left_elbow 0.1` for manual control,
`mode reference` for the original programmed controller, and `reset 29` to restart.
`record stop` saves an interactive 3D replay and video; `files` lists artifacts.
The verified mixed-control session recorded 246 actions, including 240 fitted
actions and one reset; every action/full state replayed exactly, and resume passed.
Preparation, console interaction, export, replay and resume took 11.93 seconds.
[Saved interactive replay](../artifacts/camera-reference-play-01/lab/sessions/session-06fb54b1d283/replay.html)
and [full verification details](handoffs/camera-reference-lab.md).

This is the known jogging development case with explicit body-position assistance,
not recovered balance, ground/equipment contact or unfamiliar-video accuracy.

After a source-fit proposal passes all of its checks, compare it as a **controller
alternative** on the original world:

```sh
.venv-lab/bin/python -m weft body-reference-trial \
  --proposal artifacts/my-body-proposal-01/proposal \
  --run artifacts/my-shared-world \
  --task artifacts/my-lab/run/exercise-task.json \
  --out artifacts/my-reference-trial-01
```

Use the original sealed world and its existing task; the command never creates
a new task from the proposed reference. Both programmed controllers use the same
seed and initial state. Only the fitted controller's joint-target correction
changes; physical geometry, equipment, root control, scoring targets and scoring
rules stay fixed. Neither controller is a newly trained policy.

Open the printed `index.html` path yourself. It shows task outcomes separately
from workflow completion, and links each available 3D replay, video, action/state
journal and exact replay check. The original world/task, proposal measurements,
controller recipe and source/runtime identities are retained alongside the result.
Failed tasks remain failed even when recording and export complete. Partial
delivery preserves its artifacts and returns a nonzero exit code.

This path requires an unchanged-first-pose proposal with complete source/grid
checks, the original nominal frame/FPS clock, and at most 900 control steps.
Unsupported clock/runtime or source bindings stop explicitly. `--seed` defaults
to 1729; both resets disable jitter. `--max-seconds` defaults to 60 and permits at
most 120 cooperative seconds, not a hard native-process deadline. At most two
trials, two exports and two exact replays run; no fitting, inference, training or
paid calls run. Optional `--proposal-manifest-sha256`,
`--scene-manifest-sha256` and `--task-sha256` pin known input identities.

This is a saved comparison, not a replacement sandbox or installed body. Continue
interacting and recording in your existing lab. A source-image improvement or
exact simulated replay does not establish accurate contacts, anatomy or physical
behavior on a new video.

The [verified local comparison](../artifacts/reference-controller-trial-cli-01/comparison/index.html)
finishes in 12.725 seconds with two complete 254-step trials, two 255-frame videos
and exact replay of both journals. Both controllers still fail equipment
stability; the fitted version slightly reduces displacement but worsens the
original-target tracking score. The working sandbox is unchanged.

### Try the optional two-camera fitted sandbox

The saved jogging example now fits both supplied camera views while keeping one
unchanged body, world, clock and task. A second camera is optional, not required
for the normal video workflow.

- [Inspect the fit, missing points and both cameras](../artifacts/camera-multiview-real-01/inspection-recovered-01/index.html).
- [Compare full simulation trials and recordings](../artifacts/camera-multiview-consumer-01/trial/index.html).
- [Replay the verified mixed-control session](../artifacts/camera-multiview-lab-recovery-01/lab/sessions/session-5548b0b5549c/replay.html).

Start the prepared sandbox yourself using its exact verified runtime:

```sh
PYTHONPATH="$PWD/.local/camera-multiview-lab-runtime-01/src" \
  .venv-lab/bin/python -B -P -m weft lab \
  --resume artifacts/camera-multiview-lab-recovery-01/lab
```

Open the printed local address. For terminal-only controls, replace `lab` with
`lab-console`, then try `controls`, `mode fitted`, `record start`, `step 6`,
`mode manual`, `reset 29`, `record stop`, and `record evidence`. Recordings are
new sessions; the completed source run stays unchanged.

Both 238-step comparison recordings replay exactly. The new sandbox also passed
manual/coast/programmed/fitted switching, reset, ten recorded actions, exact replay
and reopening in 8.20 seconds. It reuses the original task and recipe; no refitting
or training occurs when reopening. Terminal behavior and saved-page data were
verified; browser interaction was not exercised.

Image-point error improves in both views, but fitted controls score slightly
worse on the original task. Body-position assistance remains explicit; this is
not learned balance, verified ground/equipment contact, or unfamiliar-video
accuracy. [Inputs, settings, checks and preserved failures](handoffs/camera-multiview-workflow.md).

### Inspect the lab's saved evidence

Choose **Checks & limits → Evidence and gaps**, then expand a category. The same
14 categories used by the build report show saved measurements, failures, missing
measurements and exact local artifacts. Closed categories stay lightweight;
opening or refreshing them does not move the simulation or alter a recording.

In the current exercise terminal interface, use:

```text
evidence
evidence 12
```

The first command lists all categories; the second shows **Task and learning
utility**, including the saved learned/authored 3/3 results. If the summary is
still loading, enter the command again. Loading runs separately from controls.
Unknown completion or reward measurements remain unavailable, not zero. Saved
simulator success is not physical validation. **Failures recorded** can include
comparison controllers; inspect each controller's counts. Object scenes use the same summary;
learning results without a supported evidence mapping remain unavailable.

Current-source inspection also supports saved spatial-rigid controller evaluations.
**Task and learning utility** shows the original-start trials; **Robustness and
coverage** shows changed-start trials. Each keeps its own success denominator and
mean **episode return** (summed step rewards). Distances/jitter use relative
scene-width units, not measured metres. The two inspected saved studies retain
0/12 and 1/12 changed-start successes; these are not new training results. Older
pinned launcher snapshots retain their original views until explicitly updated.

Saved-payload checks cover this trained exercise, the untrained stronger-model
candidate and a rigid-object lab. Metadata/terminal snapshots took **0.315 seconds**
with zero native/media/API calls. The actual saved payloads also passed scripted
UI checks; no browser visual or sustained-use performance claim is implied.

### Check a recording's replay evidence

Use **Record & start**, interact, then **Stop & save**. Once export finishes,
choose **Recording evidence** beside that recording. It shows saved action/reset
counts, source identities and whether actions were replayed exactly. **Not
performed** is different from passing. Old exports without a receipt remain
unavailable; watching a video does not prove physics replay or physical accuracy.

New 3D rigid-object recordings pin their runtime when recording starts and verify
actions, seeded resets, selected actor, rewards, terminal flags and complete
states before export. This verifies the recorded actions—not a fresh evaluation
of the learned policy's decisions. Old recordings without this start-time
contract remain saved-state-only; already sealed exports are never backfilled.

In the terminal, use `record evidence` for the recording you just made, or
`record evidence session-d10a22aa4d5a` for the verified short example. If loading,
enter the command again. Reading evidence never moves the live body or reruns
the recording. Receipt and manifest links are available in the same panel.

The new short example verified six actions, one reset, eight frames and exact
replay in **2.75 seconds**, with one export and no extra replay for the receipt.
It tests recording/export/inspection, not complete exercise success. No API cost.

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

If a late failure leaves a complete lab, the launcher checks its saved identity
metadata and prints manual browser-lab and terminal resume commands. It still
exits with failure and never auto-starts that lab, even with `--serve` or
`--console`. Opening it performs the normal full artifact/runtime checks. Missing,
inconsistent or receipt-only outputs do not get a recovery command; the failed
build and its report remain unchanged.

### Read evidence and failures

Open the generated `report/index.html`, then choose **Evidence and gaps**.
All 14 evaluation categories show saved measurements, missing measurements,
failure examples and links to exact artifacts. Expand a category for its limits.
There is no overall accuracy score: an unavailable measurement is not zero error.

**Sandbox prepared** means construction produced a lab, not that its motion or
task passed. A saved geometry audit now shows floor conflicts prominently, with
separate source-pose and intermediate-pose counts and the unchanged task limit.
These compare estimated geometry with assumed ground; they do not measure anatomy.
Learning counts appear only when the saved world and task identities match, and
remain labeled simulator development results rather than physical validation.

Two saved examples demonstrate the report without starting a server:

- [Stronger pose-model candidate](../artifacts/heavy-video-inspection-02/index.html):
  prepared but untrained; 8/117 source poses and 14/235 intermediate poses fail
  the existing floor limit.
- [Earlier training run](../artifacts/person-scorecard-inspection-01/index.html):
  preserves its 0/3 learned-controller successes. This is **not** the newer
  [recommended exercise example](#recommended-exercise-sandbox), which passed 3/3.

These reports were generated together in **0.40 seconds**, using only saved
artifacts, with no decoding, inference, physics, training or API calls. They do
not replace or modify the original runs. New builds receive this report
automatically; `--inspect-build` adds it to an older build in a new directory.

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

For explicitly authored fixed supports, use the same launcher with a
[source-pinned support-prior file](static-support.md#build-inspect-and-play-in-one-workflow):

```sh
.venv-lab/bin/python scripts/build_sandbox.py \
  --human-run /path/to/construction-only-body \
  --static-supports /path/to/support-priors.json --console
```

The prior supplies support shapes, load windows, scale and root assistance.
Do not also pass `--person-only` or `--root-assistance`. The launcher saves the
step-by-step report and opens the console only after successful preparation.
Add `--train` if needed; without it, use manual or authored-reference controls.
These supports are assumptions, not automatically recovered contact.

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

To turn a sealed planar video scene into an explicit 3D hypothesis, use
[`spatial-scene`](spatial-scene-construction.md), then the existing
`spatial-lab` / `lab-console` workflow. Original evidence is preserved; depth,
materials and missing spin require explicit assumptions. This is not automatic
3D recovery.

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
