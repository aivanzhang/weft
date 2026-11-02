# Build and inspect an exercise scene

Use sealed **body-source evidence or a mobile exercise-v3 run/lab** and a sealed equipment tracking export to build **one shared world and a human-only lab**. Body-source preparation skips the unrelated body-only lab, control trials and single-joint training. Camera/shape assumptions remain explicit; this does not automatically recover arbitrary equipment from a video.

For an immediately playable example, use the [exercise sandbox](shared-exercise.md). For a person-only video, use [the equipment-free declaration below](#person-only-video-without-equipment-files). For a video with equipment, prepare its [equipment tracking evidence](shared-world.md), then use the raw-video route below or reuse a saved body. Keep the original video hash and any original/derivative mapping bundles. Saved-input assembly reuses inference results; neither route requests tracking automatically.

For explicitly authored **fixed supports**, use [`--static-supports` in the same build/train/inspect/lab workflow](static-support.md#build-inspect-and-play-in-one-workflow). This does not require equipment tracking or assert equipment absence; shapes and contact requirements remain declared simulator assumptions, not recovered facts.

## Person-only video without equipment files

For a video with one selected person, ground-only support, and no task-relevant equipment or obstacles, use `--person-only`. This is **your explicit scene declaration**, not automatic detection of equipment absence. Do not use it for weights, racks, benches or external support that affect the exercise. The same MuJoCo engine, learner, lab and recorder are used; no dummy objects are inserted.

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --person-only --video /absolute/path/to/bodyweight-exercise.mp4 \
  --pose-model /absolute/path/to/pose_landmarker_lite.task \
  --root-assistance --train \
  --training-generations 1 --training-population 4 \
  --evaluation-seed 60824391 \
  --out artifacts/my-person-exercise-01

.venv-lab/bin/python -m weft shared-exercise-inspect \
  --run artifacts/my-person-exercise-01 --out artifacts/my-person-inspection-01

.venv-lab/bin/python -m weft lab \
  --resume artifacts/my-person-exercise-01/learning_workflow/lab
```

Use fresh output directories; the sample evaluation seed is not guaranteed unused. If training is omitted, the final lab is `artifacts/my-person-exercise-01/lab`. `--root-assistance` adds three explicitly virtual root controls; without it, only 19 joint motors are available. Neither mode promises unassisted balance. Optional `--pose-recovery selective-crop` requires the pinned local Full model and the same fixed source-only recovery recipe described below.

To reuse inference, replace `--video/--pose-model` with `--human-run <sealed exercise-body source>`. Person-only mode accepts only a construction-only `exercise-body-source-v1` package, not a prior mixed world or lab. `--evidence-run`, `--config`, `--source-mapping` and `--run` conflict with the declaration and reject before inference/output. The default equipment path is unchanged.

The complete retained reference interval is mandatory; missing and unsampled slots stay missing. The new scene/task versions (`shared-world-v3` / `shared-exercise-task-v2`) use human control, no object target, and no equipment actions. Human tracking/floor gates are unchanged. Equipment/contact gates are **not applicable**, not passed; reward has no equipment-stability bonus. Only the newly pinned residual-v2 learner is supported on this scene. Earlier gain/feasibility policy profiles and the standalone `qualify` utility do not silently authorize the new variant.

Inspect `person-inventory.json`, `body_source/` (raw input), `candidate/inventory.json`, `candidate/reference-feasibility.json` and the same `learning_workflow/` outputs. Candidate construction alone is not lab admission or task success: the lab retains its initial-geometry gate, and reference-feasibility failures stay visible. Depth, scale, floor and body geometry remain model-derived/authored priors; muscles remain illustrative. The report checks included scene metadata before describing an equipment-free constructed world; a stopped preconstruction build shows only the requested scope.

**Verified scope:** `artifacts/person-only-native-03` exercises the complete saved-body→world→training/evaluation→one-lab→record/export/exact-replay path in5.24s:828 actions, all58 reference steps and68 recording frames. The original source is explicitly synthetic placeholder data, not a playable video or detector output. Selected and authored controllers each pass1/1 independent synthetic start; selection retained the neutral candidate, so no learned improvement is claimed.561 scoped Python checks and scripted UI checks pass; real person-only raw-video acceptance is the next step. Existing real-video examples and their0/3 failures remain unchanged.

## Video to one trained lab

The same builder can prepare a raw video and optionally train before creating its **one final lab**:

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --video /absolute/path/to/exercise.mp4 \
  --pose-model /absolute/path/to/pose_landmarker_lite.task \
  --evidence-run artifacts/my-tracking-evidence \
  --config /absolute/path/to/equipment-config.json \
  --full-reference --root-assistance \
  --train --evaluation-seed 52738961 \
  --out artifacts/my-trained-exercise-01

.venv-lab/bin/python -m weft lab --resume artifacts/my-trained-exercise-01/learning_workflow/lab
```

This still requires equipment observations and explicit camera/shape choices; it does not infer unknown equipment or download a model. Before local pose inference, preflight checks the original video hash, selected equipment/frame/partition, local model and any derivative proof. Withheld/missing observations and mismatched sources stop early. A selected equipment frame must also survive body inference; that later check remains unchanged. Input hashes are checked again before construction and finalization.

Raw pose defaults are `--pose-mode video --pose-identity raw`. Changes must be explicit. `--source-note` records attribution; omitting it leaves rights unverified. Saved inputs use `--human-run` or `--run` instead and reject pose overrides rather than repeating inference.

`--train` uses the existing joint-correction learner, selects a controller using **training results only**, then compares the frozen selection against coast, random and authored following on separate changed human starts. Equipment stays passive. All stages keep the exact same task, interval, physics and success limits. A completed search is not proof of improvement: an unchanged authored controller or a policy with zero successful trials remains clearly labeled. Evaluation is a development comparison, not an automatically fresh final holdout or physical validation.

Training defaults are two generations, eight candidates, seed 1729 and three evaluation starts with 0.10 human-start variation. Choose an explicit separate `--evaluation-seed`; the example is not guaranteed unused. Advanced overrides are `--training-seed`, `--training-generations`, `--training-population`, `--evaluation-trials` and `--evaluation-jitter`, all requiring `--train`. Preflight rejects a search exceeding 8,640 training actions or the existing evaluation limit; it never shortens the task. For example, a 254-action interval allows at most 8,128 training and 3,048 evaluation actions with these defaults.

The terminal prints each stage and the result's `launch_arguments` identify the final lab. A rejected/unsupported build exits nonzero and points to its preserved `result.json`; a ready lab with zero task successes still counts as a completed build. With training, inspect `learning_workflow/plan.json`, `training/`, `selected/` and `evaluation/` before opening `learning_workflow/lab/`. Without `--train`, the output remains `lab/`; no learning/evaluation runs. Original body/source packages remain construction-only. No command starts a server until you explicitly run `weft lab --resume`.

New allowances are explicit: raw-only builds allow 512 MiB, training-only 1 GiB and combined builds 1,152 MiB; these are maximums, not expected usage. Local work uses one worker per stage and no paid API. Default cooperative deadlines are 270 seconds for raw or training, 450 seconds for both; saved/no-training stays 90 seconds/384 MiB. Extended builds permit at most 480 seconds. Each helper retains its smaller stage limits, and the learning stage reserves five seconds for finalization. Native calls require an external process deadline for a hard stop. Failures preserve completed evidence and any already completed lab; they do not create a second lab or retry training.

### Explicit source-only pose recovery

For a fresh video, add `--pose-recovery selective-crop` and supply the pinned **Full** model instead of Lite. This uses one fixed IMAGE/raw recipe, not an automatic choice between models:

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --video /absolute/path/to/exercise.mp4 \
  --pose-model .local/models/pose_landmarker_full.task \
  --pose-recovery selective-crop \
  --source-note "Your source attribution and rights" \
  --evidence-run artifacts/my-tracking-evidence \
  --config /absolute/path/to/equipment-config.json \
  --full-reference --root-assistance \
  --train --evaluation-seed 52738961 \
  --out artifacts/my-recovered-exercise-01
```

The same build now exposes three source stages: **inspect original poses and image crops → verify pose replacements → construct the body**. Inspect `pose_proposals/`, `recovered_poses/` and `body_source/`; accepted and rejected replacements stay available. Missing or ambiguous poses remain explicit. Recovery does not use simulated collisions or rewards, invent missing poses, or guarantee body/world admission. Explicit VIDEO/temporal overrides and non-Full models reject before inference.

This option permits at most two sequential decodes, 900 full-frame and 64 crop inferences, and 48 annotation previews per pass. Every sampled pose and source clock remains; the preview cap is not a data cap. Too many eligible crops rejects instead of choosing a convenient subset. Proposal outputs plus the pinned original fit 128 MiB; the proposal stage pins but does not copy the original. Recovery then preserves its original-video copy and proof. No model download, upload or paid API occurs.

The three source packages reserve 384 MiB/1,536 files and 240 seconds together. Outer bounds become 768 MiB/3,136 files without training, or 1,408 MiB/5,632 files with training; default cooperative time is 450 or 600 seconds (maximum 660). Other routes retain their existing limits. Completed child hashes remain pinned through final packaging, and progress observers cannot change saved identities. The offline report below shows these stages separately, including retained pose evidence when later body construction stops.

To prepare proposals alone, use `exercise-pose-prepare --video <video> --pose-model <full.task> --out <new-directory>`, then the saved recovery command below. A proposal package is source evidence, not a playable environment.

**Local verified example:** `artifacts/native-crop-build-02` completes this native pipeline on the existing kettlebell source in **29.6 seconds**, preserving all 257 source slots and producing one 254-action/8.47-second lab. The bounded search finds no feasible improvement and all controllers score **0/3 task successes**. This verifies orchestration and a ready environment, not physical accuracy or arbitrary-video support. Open `artifacts/native-crop-inspection-01/index.html` for its stages, video, poses and results; [exact identities and limitations](handoffs/combined.md#native-video-through-one-final-lab--development-path-works) remain recorded.

To play that saved example while development continues, run `.venv-lab/bin/python scripts/play_sandbox.py --example video`. Its fixed trusted runtime keeps this v2 lab separate from the original no-flag v3 sandbox. [Controls and verification](local-lab.md#try-the-complete-video-to-lab-example).

## Inspect a completed or stopped build

Create a local report from an existing build; this does not rerun the video or start a server:

```sh
.venv-lab/bin/python -m weft shared-exercise-inspect \
  --run artifacts/my-trained-exercise-01 \
  --out artifacts/my-exercise-inspection-01
```

Open the returned `index.html` yourself. It explains which stages finished, where processing stopped and which stages never ran. Available original video, pose previews, body/equipment documents and assumptions are copied with their saved hashes. The input build stays unchanged. A generated report does **not** make a rejected world playable or prove task success.

For a partial build whose seal covers only its failure receipt, the report shows only that verified receipt; unfinished stage assets remain unverified and are not linked. The report verifies the metadata/media it consumes, not every unneeded member of the source package. It has no script, network dependency or live simulator.

Report preparation is bounded to 30 cooperative seconds and 128 MiB, with 64 MiB per copied file. Choose a new output directory outside the build. The regular lab still handles interaction, training controls and recording for admitted worlds.

## Reuse saved source-only pose recovery

A sealed selective-crop proposal package can now feed ordinary body preparation directly:

```sh
.venv-lab/bin/python -m weft exercise-pose-recover \
  --run artifacts/my-saved-crop-proposals \
  --video /absolute/path/to/original.webm \
  --out artifacts/my-recovered-poses-01

.venv-lab/bin/python -m weft exercise-body \
  --evidence-run artifacts/my-recovered-poses-01 \
  --out artifacts/my-recovered-body-01
```

Then supply that body to `shared-exercise-build --human-run` with the same explicit equipment/camera inputs. Recovery recomputes the existing fixed crop rules against the original image observations, preserves every accepted/rejected proposal and checks exact source clocks and model identities. It never selects using collisions, task scores or evaluation data. Ambiguous people, missing context and unsupported replacements remain unresolved.

**This command reuses saved proposals; it does not generate new crops from a fresh video.** The supported saved recipe is Full/IMAGE/raw. Ordinary raw-video defaults remain Lite/VIDEO/raw; there is no automatic model/mode contest or fallback. Recovery creates source evidence only. The body constructor still rejects failed pose quality, and construction-only bodies/seeds are not playable environments.

Limits: one worker, 30 cooperative seconds, 128 MiB and 512 output files, 64 MiB per file, and at most 64 crop proposals. Packages retain original video/raw poses, selected evidence, complete selection proof and frozen verifier sources. Inputs and completed outputs are never overwritten.

## Prepare body evidence without a lab

If the person is small in the full image or there are bystanders, add an explicit
fixed attention region to the raw-video command:

```sh
.venv-lab/bin/python -m weft exercise-body \
  --video /path/to/exercise.mp4 \
  --pose-model .local/models/pose_landmarker_full.task \
  --person-region 120 80 500 700 \
  --out artifacts/my-person-region-body
```

Replace the four numbers with **left, top, right, bottom** edges in the original
video's pixels. The region must stay inside the video and be at least 64×64 pixels;
right/bottom edges are exclusive. Keep the whole person and motion inside it.
This is manual attention, not automatic person selection or proof of identity.
Competing detections remain unresolved, and people outside the region are not
detected. Original video bytes, frame IDs, clocks and full-image coordinates stay
unchanged; raw region coordinates/confidence are also retained. Points outside
the input region receive zero effective visibility, never invented observations.

The resulting body uses the ordinary pipeline and can be passed to
`scripts/build_sandbox.py --human-run ...` with the appropriate person-only or
equipment inputs. A region does not calibrate scale, ground, contact or anatomy.
Saved evidence inherits its exact region; `--evidence-run` rejects overrides.

```sh
.venv-lab/bin/python -m weft exercise-body \
  --video /absolute/path/to/exercise.mp4 \
  --pose-model /absolute/path/to/pose_landmarker_lite.task \
  --out artifacts/my-body-01
```

Use the existing pinned local model, or obtain one with `weft fetch-pose-model`. The command reports landmark/body stages in the terminal and saves the source video, model identity, raw poses, missing/uncertain slots, annotations, canonical observations and mobile body. It does not train, run control trials, render simulated motion or create a task. Defaults remain `--pose-mode video --pose-identity raw`; alternative identity handling is explicit, not a silent fallback.

For a clip you explicitly treat as **one person**, `exercise-body --video` also
accepts `--max-pose-hypotheses 1`. This enables the model's single-person temporal
configuration, including smoothing in VIDEO mode, but can miss competing people.
The default remains two hypotheses; VIDEO mode alone does not prove tracked-region
reuse. The assumption is retained in raw poses, model metadata and saved-body
options. Quality, gap and floor checks do not change. This setting concerns people,
not equipment absence, and is separate from the `--person-only` scene declaration.
It is not a guarantee of identity, accurate joints or usable whole-body motion.

If pose evidence already exists, reuse it instead of repeating inference:

```sh
.venv-lab/bin/python -m weft exercise-body \
  --evidence-run artifacts/existing-mobile-lab \
  --out artifacts/my-body-02
```

This path inherits the sealed source/model/pose options and rejects overrides. The copied run configuration describes the original pose processing; `body-source.json` identifies the new mobile-body construction separately. It performs no video decoding or inference. Both paths reuse the current body builder's **one forward-geometry calculation** for the assumed floor; neither steps dynamics. Model-derived joints, root and floor remain assumptions, not measured anatomy or contacts.

`body-source.json` marks the result **construction-only**, with exact source/pose/model/body hashes; `source-clock.json` preserves sampled native timestamps separately from assumed frame-rate time. There is deliberately no `scene.json` or task. Open only the final assembled `lab/`, not this input package. A failed preparation retains its evidence and cannot become a ready lab.

Keep the body-source and complete assembly directories for the full evidence trail. The final `lab/run/` carries selected runtime/canonical evidence, not every raw preparation file; sampled native-clock bytes remain in the body-source and assembly's body-seed snapshot.

Limits: **64 MiB per input/output file, 128 MiB total package, 512 files**, and the existing 60-second/1,800-decoded-frame video limits. The 64 MiB video limit keeps the result compatible with the downstream builder. Default cooperative bounds are 180 seconds for raw video and 30 seconds for saved evidence; a stuck native call still requires a caller process deadline. Source annotation images are not simulation renders.

### Optional saved-mode recovery

If you already have original **single-person VIDEO** and **two-hypothesis IMAGE**
estimates from the same model, runtime and video, compare them without repeating
inference:

```sh
.venv-lab/bin/python -m weft exercise-pose-recover-modes \
  --primary-run artifacts/original-video-poses \
  --alternative-run artifacts/original-image-poses \
  --video data/exercise/example.webm \
  --out artifacts/mode-recovery-01
.venv-lab/bin/python -m weft exercise-body \
  --evidence-run artifacts/mode-recovery-01 \
  --out artifacts/recovered-body-01
```

Both inputs must be sealed, original raw-identity runs with complete matching
sample IDs and clocks. This explicit option looks for isolated limb-length and
image-position inconsistencies between original neighbors; it does not use floor
or task scores. It preserves all decisions, both original pose sets and the actual
model mode for each replacement. Inspect `mode-recovery/decisions.json` and
`pose-mode-recovery.json`; an empty selection is valid. Crop/identity-repair
chaining is unsupported. Body, quality and physics checks remain separate.

On the exposed kettlebell example, one replacement produces a new body that passes
382 sampled floor checks. The normal builder also re-estimates lengths and body
placement; this is not validated anatomy, stable control or equipment contact.
[Exact results and remaining limits](handoffs/pose-mode-recovery.md).

## One assembly command

Create a JSON configuration with these fields, using your own original video hash, observation IDs and original source-frame numbers:

```json
{
  "schema_version": 1,
  "profile": "source-equipment-proposal-config-v1",
  "source_sha256": "REPLACE_WITH_ORIGINAL_VIDEO_SHA256",
  "camera": {
    "mode": "fixed_pelvis_perspective_prior",
    "distance_model_units": 2.0
  },
  "geometry_prior": "floor_box_depth_equals_width",
  "equipment": [{"object_id": "YOUR_EQUIPMENT_ID", "source_frame": 12}],
  "target_object_id": "YOUR_EQUIPMENT_ID"
}
```

The distance above is an **authored example**, not a measured camera distance. This shape prior makes solid boxes resting on the existing floor, with unseen depth equal to width. It rejects incompatible projections, missing/withheld selected observations and initial collisions. It does not fill missing frames, move the floor or search for a different camera to force admission.

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --human-run artifacts/my-body-01 --body-scale 1.0 \
  --evidence-run artifacts/my-tracking-evidence \
  --config /absolute/path/to/equipment-config.json \
  --full-reference \
  --out artifacts/my-shared-exercise-01

.venv-lab/bin/python -m weft lab --resume artifacts/my-shared-exercise-01/lab
```

The first command prints stage progress and does not start a server. The second command is user-started. Add `--root-assistance` to assembly only if you want the three explicit virtual root controls; otherwise it uses 19 human joint controls. Equipment remains passive in both modes.

`--human-run` accepts the new body-source package, a legacy sealed mobile run, or its enclosing lab. `--body-scale` is an authored body-to-world scale, not a measured size; it defaults to `1.0` and must stay within `[0.5, 2.0]`. The assembly's body-seed stage preserves exact human/source bytes, source-frame slots and clocks, without physics or rendering. It is construction-only and cannot be opened as a lab. Unsupported body profiles, camera-reference bodies, incomplete source intervals and conflicting identities reject instead of falling back.

Existing shared-world inputs still use `--run artifacts/my-shared-body` instead of `--human-run`. The two options are mutually exclusive; `--body-scale` cannot change a prepared shared world. Neither route infers which equipment to use or silently selects a different source frame.

`--full-reference` builds one continuous episode from the first retained body pose to the last. It uses the existing `shared-world-v2` contract: at most 1,800 actions / 60 seconds, unchanged timestep and no chunk resets. Missing poses and source prefixes stay missing; the selected interval is not necessarily the entire original video. Unsupported endpoints reject instead of being rounded or retimed. Omit the flag to keep the original short-window behavior.

The full-reference choice is made before task/lab packaging, so it creates only one candidate and one lab. It does not attach a short-window policy, change success limits or require a temporary lab. Training runs only with the explicit `--train` option above. `candidate/reconstruction.json` records the original window and the selected interval; the final lab's `run/exercise-task.json` binds the task to that exact scene.

If tracking used a verified derivative, add `--source-mapping <bundle>`; repeat for distinct derivatives. Canonical root `scene-observations.json` and the older nested layout are supported without moving files. Conflicting sealed copies reject.

Each equipment frame must also be a retained body-reference frame. Equipment measurements must belong to the explicit construction partition; authored initialization boxes and raw appearance/generation/development/audit rows are not silently promoted. Missing or withheld observations remain unsupported, not filled from a nearby frame.

An optional `"frame_translation_xy": [0.2, 0.0]` explicitly translates the entire frame and invokes the existing bounded nominal dynamics check. It is not a body-only adjustment; it changes numerical bounds and must not be described as complete task equivalence. Omit it unless that coordinate migration is intended. The assembly command passes the resolved choice to construction; it does not invent or apply a second translation.

### Optional explicit equipment depth

To supply depth instead of assuming it equals width, replace these fields in the
complete configuration above:

```json
{
  "geometry_prior": "floor_box_explicit_depth_ratio",
  "equipment": [{
    "object_id": "YOUR_EQUIPMENT_ID",
    "source_frame": 12,
    "depth_prior": {
      "depth_to_width_ratio": 0.5,
      "provenance": "My supplied assumption: depth is half the width; not a verified measurement."
    }
  }]
}
```

This is a configuration fragment. Every selected object needs its own ratio and
explanation; the other fields remain required. `0.5` means half as deep as wide,
`2` means twice as deep. A ratio of `1` retains equal-depth numerical geometry,
but records that you explicitly supplied it. The guided setup page exports the
same fields for the ordinary builder.

Ratios must be finite numbers from `1/120` to `120`; explanations must contain
1–2,000 characters. These broad bounds do not guarantee an admissible box:
original size, image projection, floor and initial-collision checks still apply.
Supplied depth is kept in the proposal and object provenance, marked physically
unverified. It does not infer handles, contact, hidden shape or camera calibration.
Existing equal-depth configurations and defaults are unchanged.

The [saved ratio-one setup](../artifacts/equipment-depth-prior-native-02/configured/inspection/index.html)
passes the ordinary setup, proposal and native-construction path with unchanged
numerical geometry. This verifies the input path, not measured depth or full-task
success. [Implementation](handoffs/equipment-depth-prior-core.md) ·
[native verification and limits](handoffs/equipment-depth-prior-native.md).

## Inspect each stage

| Output | Meaning |
| --- | --- |
| `settings.json`, `stages.json`, `result.json` | Requested inputs, completed stages and outcome |
| `input-preflight.json`, `body_source/` | Raw input only: verified input identities and the construction-only body; raw observations remain distinct from assumptions |
| `body_seed/construction-seed.json` | Direct mobile input only: verified body/source snapshot and explicit construction-only scope; no intermediate lab |
| `proposal/input-config.json` | Exact user choices |
| `proposal/report.json` | Resolved assumptions, exact source bindings and admission/rejection reasons |
| `proposal/geometry-proposal.json` | Initially admitted proposal; absent on rejection |
| `proposal/geometry-candidate.rejected.json` | Preserved non-ready geometry, when a candidate could be formed |
| `candidate/` | Standard shared-world construction, complete source proof and existing reference checks |
| `lab/run/` | Immutable task, scene, evidence and runtime snapshot |
| `lab/sessions/` | New interaction recordings, separate from construction evidence |
| `learning_workflow/plan.json`, `training/`, `selected/`, `evaluation/` | Training opt-in only: frozen budget/task/seeds, all candidates, training-only choice and separate baseline comparison |
| `learning_workflow/lab/` | The sole final lab when training was requested; its recordings are outside all immutable build seals |

Every output directory must be new. Unsupported input stops before construction or lab creation. Later failures preserve completed stages. If a lab was already successfully prepared before an outer deadline or finalization failure, the failed assembly result explicitly retains `lab_status: ready` and its launch path; it never relabels that completed lab. A failure-only manifest identifies unsealed partial outputs rather than pretending they were verified.

The saved-input/no-training assembly deadline remains 90 seconds, checked around stages and during final hashing. Native calls and bounded source-verification calls already underway cannot be preempted. Each next stage reserves its complete output allowance before starting: optional body preparation 512 files/128 MiB, proposal 256 files/64 MiB, construction 512 files/128 MiB, lab 1,024 files/192 MiB, within the shared 1,600-file/384 MiB envelope. Body preparation also bounds its input to 512 files/128 MiB, each file to 64 MiB and its cooperative time to 30 seconds or the remaining assembly deadline, whichever is shorter. Oversize input is rejected, not trimmed. This default route runs no inference or training. The raw/training opt-ins have the separate allowances above; neither makes a provider call or deployment.

For separate stage control, use `shared-world-propose`, then `shared-world-reconstruct --full-reference`, then `shared-exercise-lab`. The proposal report includes the exact construction arguments. The reconstruction flag is optional and has the same interval semantics. Separate [human-only training/evaluation commands](shared-exercise.md) remain available; a ready geometry package is not a successful policy.

## Verified scope

**Saved recovery and offline inspection:** 174 current scoped checks pass (124 recovery/body/seed, 45 report, five CLI), with older failures retained in `.local/pose-recovery-inspection-tests-01.json`. `artifacts/pose-source-recovery-acceptance-01` preserves all **2,677 decoded slots /1,340 sampled observations /31 proposals /13 replacements** across five previously exposed clips. Squat and push-up match their saved body documents exactly and pass proof-preserving seed preparation. Single-leg squat and lunge remain quality-rejected; jumping-jacks/burpees is source-only here. The study takes **8.576 s**, uses two existing floor-forward calculations and **zero dynamics, decoding, inference, training or API calls**. This verifies reusable saved-evidence handling, not new-video or physical accuracy.

`artifacts/direct-exercise-inspection-01/index.html` is the actual raw-rejection report: **35 source references /38 output files /19,723,382 bytes**, generated in **0.350 s including process startup**. It retains the original rejected build unchanged and explains two completed stages, stopped equipment proposal and unrun downstream stages. Generation latency is not browser playback or visual QA; no browser was opened. [Exact studies and receipts](handoffs/combined.md#saved-pose-recovery-and-offline-inspection--verified).

**Raw input and learning:** 171 distinct scoped checks cover preflight, full-interval training/selection/evaluation, nested seals, failure preservation and CLI outcomes; `.local/exercise-build-checkpoint-01.json` records exact receipts and native-test version limits. An analytical fixture completes one trained lab with recording/export/exact replay; this is software coverage, not real-video accuracy.

The one real raw-video run, `artifacts/direct-exercise-build-01`, takes **6.874 s /$0** and preserves **257 source slots /129 inference samples /127 body references**. It then rejects an initial **left-knee/kettlebell intersection**. No training or new lab follows. Its 106 files (51,793,908 bytes) retain original video, pose/body evidence, assumptions and rejected geometry; [frozen plan and audit](handoffs/combined.md#raw-input-and-learning-integration--verified-with-rejection). This negative result is not an end-to-end success, and the existing playable lab is unchanged.

**Body preparation without a separate lab:** `artifacts/exercise-body-source-equivalence-01` passes on three saved exercise inputs: single-leg squat **235 source slots /118 body references**, squat **213 /100**, and kettlebell push-up **257 /127**. One uses the public `exercise-body --evidence-run` command, two the API. Bodies, root/floor/body-frame/spatial-pose documents match their originals; source videos, raw poses, model/options, identity audits and sampled native timestamps remain exact. Shared-world seed arithmetic also matches the old pure constructor, with source-binding/object provenance differences explicitly excluded from that comparison.

The single study takes **6.153 s including the process**, retains **393 files /114,222,777 bytes**, and seals `f6e126caa4aabefa3da3e421c0c30698a18f1ffe2dd5deb56290ff827f2664cd`. It performs three existing floor-forward calculations, **zero dynamics, decoding, inference, training, API calls or new labs**. Root `.local/exercise-body-source-root-verification-01.json` verifies all 392 covered files and saved body/source/clock custody without new geometry work. This is saved-evidence equivalence, not a new raw-video performance benchmark or improved reconstruction accuracy.

**135 distinct scoped checks** cover the new source path, CLI, bounded producer writes, old pose/body/seed behavior and study failure handling. The new end-to-end analytical fixture preserves 63 slots (29 usable, three sampled missing, 31 unsampled), builds one 58-action declared task/lab with unfamiliar passive foam-pad/training-weight equipment, and executes one legal human action. A real constructor rejection retains its prior evidence without publishing a lab. The tests do not qualify the entire 58-action task or measure physical accuracy; `.local/exercise-body-source-tests-summary-02.json` identifies the receipts. Raw-video orchestration is fixture-tested; real saved evidence is tested as above.

**Direct mobile input:** `artifacts/mobile-source-assembly-01` uses `--human-run` on the saved mobile lab, with existing equipment evidence/configuration and the verified derivative mapping. It completes in **3.977 s**, including **0.756 s** for body preparation, and seals **397 files /102,458,085 bytes including the manifest**: `04ef00208ceecd8585021db55bd27bff654c8cb025db4ea6b17c65a46b782635`. The final physical scene is identical to `full-reference-assembly-01`; task and human bytes match exactly. All **121 references /257 slots /254 actions** remain, and only the selected `sam-2`/`sam-3` equipment reaches the final world. Three reference actions and one reset pass in **0.554 s**, with no object forces or immutable-run changes. `.local/mobile-source-assembly-verification-01.json` verifies all 396 covered members. The new **untrained** lab is `artifacts/mobile-source-assembly-01/lab`; it does not replace the trained example or claim task success. No seed physics/model creation, video decode, inference or paid request occurs.

A separate analytical fixture reaches the same builder with unfamiliar foam-pad/training-weight IDs and missing body slots 0/1/31. It preserves all 63 source slots, exposes only the selected equipment in one v2 lab, and steps the body without equipment forces. The corrected end-to-end case passes `.local/mobile-source-pipeline-tests-02.xml`; this verifies reusable input handling, not independent video or physical accuracy.

**Full-reference assembly:** `artifacts/full-reference-assembly-01` runs the public command with `--full-reference` in **3.148 s** (3.407 s including CLI startup), producing **374 files / 88,686,504 bytes including its manifest**, seal `2b7a5954682099a6a06216362f24ad5df6b475cf265bf10b1b2d8f3534ad1965`. All physical settings and source documents match the earlier box assembly; only the scene profile and task interval change to 254 continuous actions. All 121 poses / 257 slots remain. Three reference-control actions and one reset work through the lab API; the sealed run stays unchanged. This new lab is **untrained**, not a replacement for the trained example above. Root verification: `.local/full-reference-assembly-verification-01.json`. No inference, fitting, paid calls, server or browser was used.

**One-command acceptance:** `artifacts/shared-exercise-assembly-01` completes the public assembler on the existing 257-frame kettlebell evidence in **3.225 s**, within its 90-second bound. Its 362 files total about 88.3 MB; root seal `4028bb4e36176252d4eb3de4f14b9c81839fc979192a17289583ba37b0d5c931`. The prepared lab is `artifacts/shared-exercise-assembly-01/lab`. Independent verification checks all 361 sealed members; the root manifest is the remaining file. No inference, fitting, training or API call was needed. The existing 31 assembly/bounds checks remain passing.

The earlier `artifacts/shared-equipment-proposal-integration-01` verifies the three stage commands separately. All physical scene settings match the previous candidate exactly; only equipment provenance differs. Proposal preparation took 0.805 s and lab preparation 0.900 s. Its 365 files total 88.15 MB. The overall run took 131.1 s including a source-write coordination wait, missing its separately declared 120-second wall bound; that overrun remains recorded. No stage was retried, and no inference, fitting or paid call was repeated.

The proposer still uses **`floor_box_depth_equals_width`**, and these source scenes remain solid boxes. [Compound collision/rendering](shared-world.md#authored-compound-objects) is separately tested with analytical shapes; it does not infer source handles or admit compound equipment through this configuration. Strict source-compound proposal admission is in progress.

These earlier integration results use a two-second task on an exposed development clip, not a new camera calibration or generalization result. The full-reference option removes that interval restriction; unknown depth, approximate shape, body–equipment overlap and missing hands/grasp remain. Broader-video automation and independent physical accuracy are still open.
