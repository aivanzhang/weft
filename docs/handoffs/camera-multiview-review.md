# Camera multiview consumer review

## Status and ownership

Implementation and review by `/root/support_learning` in the consumer lane.
Owned product paths: `camera_reference_inputs.py`, `camera_reference_controller.py`,
`shared_exercise_reference_trial.py` (routing only), and
`camera_reference_recording.py`, with four new multiview consumer test files.
The later measured budget correction also owns the narrow
`shared_exercise_reference_inputs.py` allowance dispatch.
The announced narrow `camera_reference_lab.py` and `camera_reference_receipt.py`
joins are also owned here. The recording adapter needed no code change: its
existing recipe factory dispatch and full-state replay handle the new explicit
recipe. Reconstruction
owns the optional second-view fitter; root owns input routing,
pipeline, worker, CLI and later native coordination; support_ui owns the offline
report. Analytical and mocked checks are complete below. No source studies,
fitting, native models, media, paid calls or Git work were performed here.

The new proposal has an **implemented and natively verified consumer route** on
the preserved jogging input: one complete matched trial, then a separate short
interactive lab recovery, recording, exact replay and resume. The fitted trial
scored worse than the authored baseline; this is delivery evidence, not a
control-quality or generalized-reconstruction improvement. See the terminal
results below. Analytical and mocked checks remain separately scoped.
Single-view profiles still reject undeclared multiview extensions. The path below reuses the existing fitter, matched-trial
loop, controller arithmetic, simulator, scorer and recording protocol; it does
not create a second optimizer or simulator.

## Agreed producer boundary

Coordinated with reconstruction before schema freeze:

- Objective: `camera_multiview_surface_interval_v1`.
- Method: `blocked-multiview-camera-body-surface-interval-reference-v1`.
- Distinct overlay: `camera-body-multiview-angle-proposal-v1`.
- Original primary `input_bindings` remain exact. A separate `additional_view`
  uses `camera-multiview-evidence-extension-v1` rather than adding undeclared
  fields to the old single-view profile.
- One original body, root track, shape, floor, angular clock and full reference
  interval remain authoritative. `human_candidate` stays null. Only angular
  targets differ; the overlay is never a replacement human specification.
- The same twelve absolute model endpoints are projected through two unchanged
  cameras. Each saved endpoint needs `(view_id, joint)`; joint names alone
  collide across the 24 image terms. No camera poses or image targets are
  interpolated onto the floor-check grid.
- Secondary world/clock alignment and person correspondence are explicit
  supplied assumptions, not inferred calibration, identity or contact. Initial
  scope is the existing identity same-world/clock declaration. Equal source or
  frame IDs are not a substitute for that binding.

The secondary extension must bind exact camera/evidence bytes and their canonical
identities, ordered observation input pins, alignment bytes/canonical identity,
correspondence canonical identity, primary/root-view identity and a versioned
qualification digest. A correspondence supplied as a dictionary has a canonical
hash; do not invent an independent raw-file hash by reserializing it. The exact
frozen config bytes can authenticate that original declaration.

Root froze aliases: reuse exact source snapshots `inputs/secondary-camera.json`
and `inputs/secondary-evidence.json`; new members are
`inputs/camera-secondary-config.json`, optional
`inputs/camera-secondary-alignment.json`, and
`inputs/camera-secondary-observation-00.json` through `07.json`. Require exact
members from the pinned descriptor,
not whichever files happen to be present. Retain missing/occluded/excluded and
decoded-tail slots for both original clocks; never query held-out coordinates.

`plan.secondary_view` uses `camera-secondary-view-fit-config-v1`, with
`config_sha256`, `primary_source_sha256`, `secondary_source_sha256`,
`camera_sha256`, `evidence_sha256`, ordered `observation_sha256s` (1–8),
nullable `world_alignment_sha256`, canonical `correspondence_sha256`,
`image_scale` and `weight`. Root's new `camera_multiview_inputs.py` exposes
`camera_multiview_member_names(descriptor)` and
`validate_camera_multiview_snapshot(files, descriptor, *, source_sha256=None)`.

## Gates and units

Keep the original fourteen gate names, limits and checks:

`complete_bounded_solve`, `first_reference_frozen`, `fixed_nonreference_fields`,
`source_times_and_count`, `angle_correction`, `correction_rate`, `joint_limits`,
`pointwise_floor_nonregression`, `floor_feasibility`, `endpoint_depth_change`,
`source_endpoint_image_nonregression`, `surface_interval_nonregression`,
`camera_projection_domain`, `camera_endpoint_depth_change`.

For the new profile, image/domain/camera-depth checks cover every applicable
endpoint in both views. Additionally require explicit per-view checks and a
complete, exactly two-view inventory. `report.multiview_images.views` retains
dynamic per-view checks; `report.camera_fit.multiview` retains frozen source
qualification. A declared view with no usable qualified target must remain
unavailable, not become a vacuous successful multiview qualification. Missing,
null, dropped or false view checks must prevent fresh controller admission even
if all original fourteen booleans are true. The report may still inspect rejected
or budget-exhausted proposals; inspection is not admission.

Each global image/domain/camera-depth gate must equal the conjunction of the
corresponding per-view booleans. This preserves the existing exact fourteen-key
failure accounting. An additional post-fit gate outside those three would need
explicit versioned status/failure accounting; it cannot be silently display-only.

`whole_body_reference_refinement.image_errors` already returns raw pixels in
camera mode. The existing report independently checks **each** known endpoint's
before/after error, not just a pooled mean. Preserve that rule for both views;
aggregate improvement cannot excuse one view or endpoint regressing. Preserve
invalid-known-endpoint counts and null errors, not a reduced successful subset.

Each view's `target_authored_loss_units` and projection/Jacobian use its own
authored image scale. Raw-pixel gates do not use a blanket primary scale. The
0.08 supplied-camera depth bound remains unscaled. Existing `point.weight`
means source confidence `min(visibility, presence)`; keep authored view weight
separate or explicitly derive the effective objective weight. Do not relabel
confidence multiplied by an authored factor as source confidence.

The qualification digest should cover the complete versioned view bindings,
both scales, secondary weight, per-view checks and denominators, excluding its
own digest field. This is a frozen qualification record, not measured truth.

## Exact consumer changes required

| Boundary | Current guard / required extension |
| --- | --- |
| `whole_body_reference_worker._plan`, `_result`, `run_worker` | Explicit new objective/method/overlay branch; exact secondary members and runtime closure; returned raw pins, scales, extension identity and qualification must match the frozen plan. Keep default exact file sets. |
| `whole_body_reference_report._model`, `_camera_inputs`, `_camera_targets`, report preparation | Current method, overlay keys, primary pins and twelve-point inventory are single-view. Add a strict multiview branch with exact secondary selected bytes, per-view arithmetic/slots/gates and lossless downloads. Never present it through a temporary fake single-view proposal. |
| `camera_reference_inputs._load_camera` | Currently requires the old method/objective and exact fourteen gates. Select the explicit new family, authenticate required secondary files/config and complete report validation, retain all selected members and bind the extension in identities. Keep primary raw scene/body/camera/evidence comparisons unchanged. |
| `camera_reference_controller._overlay`, `_report`, `_provider`, recipe factory | Current exact overlay/marker/key sets reject the new family. Add a distinct controller/recipe factory or discriminated branch using the same immutable `ReferenceController` numerical methods. Require all primary and per-view gates before producing a fresh provider. |
| `shared_exercise_reference_trial` | Reuse one matched loop and camera engine. If the selected-input receipt becomes versioned, update the exact family dispatch rather than testing only `camera-reference-inputs-v1`. Preserve all secondary aliases through `proposal_members`, plan and terminal recheck before models. |
| `camera_reference_recording` | Explicitly prepare the new recipe family, then reuse unchanged `reference_proposal` events, raw recipe identity, manual/authored transitions, reset semantics and full-state replay. No source fitting during replay. |
| `camera_reference_lab` | Reuse the selected trial proof and loader; require the exact new alias inventory and extension identity on prepare and resume before native admission. It already rejects missing/extra preserved proposal aliases. |
| `camera_reference_receipt` | It hardcodes `camera-source-fit-controller-recipe-v1`; add an explicit new recipe branch and join secondary evidence/qualification identities to the retained sealed trial. Saved inspection remains custody validation, not a fresh replay or source-fit qualification. |

Recommended new execution profiles:
`camera-multiview-source-fit-controller-v1` and
`camera-multiview-source-fit-controller-recipe-v1`. These names are a consumer
proposal, not released APIs. Keep existing v1 factories strict. The recipe should
retain the exact overlay, original scene/task/eight camera identities, raw report
and overlay pins, complete runtime map and compact view-qualification identities.
Do not copy full per-view target arrays into the recipe. Reconstructing a sealed
recipe remains `archived_execution_only_not_source_fit_admission`.

The outer recording/event protocol can remain unchanged because controls,
snapshots and task scoring do not change, provided its recipe dispatcher rejects
unknown families and validates the new one explicitly. Keep the public controls
19 body rotations, with the existing explicit optional three root channels;
no new force channels or hidden root changes. Improved image fitting is not
proof of improved simulated control or physical transfer.

## Runtime and size boundaries

All source/algorithm preparation stays off the action path. Cache the provider,
arrays, source identities and qualification once; no per-step file reads,
canonical hashing, camera projection or report parsing.

Old recipes pin the complete execution map, including controller and recording
modules. Default serialized output compatibility does **not** make an old recipe
compatible with changed execution bytes. Keep original frozen runtimes and
artifacts intact. Never repin an existing recipe, silently accept hash differences
or execute arbitrary archived source to manufacture compatibility.

Current limits that an extra view must respect or explicitly justify changing:

- Fitter/worker: existing fixed time/call limits; count both projections and
  reserve both views' final audits. Combined worker output is at most 16 MiB.
- Selected loader: at most 8 MiB per member, 64 MiB aggregate camera read budget
  including rechecks. The trial reserves a third read; it does not have 64 MiB
  of fresh unique input space. Legacy remains 32 MiB.
- Offline report: default 16 MiB selected-input/output budgets and existing
  camera-overflow-only alias/gzip storage. Compression is lossless storage, not
  permission to drop logical rows or bypass raw input limits.
- Controller: report at most 16 MiB, recipe at most 1 MiB, 900 references,
  1,800 original slots per admitted view, 128 bounded runtime entries.
- Lab selected proof: 32 MiB / 64 members; proposal aliases at most 32. Extra
  camera/evidence/observation copies count against these real limits.

Preflight the full selected inventory and projected output before fitting or
models. A larger view report can hit the 8 MiB loader member limit even if the
16 MiB worker result passes. Measure an authored serialization estimate before
claiming end-to-end capacity; reject oversize without clipping. No limit increase
or real-data size experiment is authorized by this review.

## Parallel implementation recommendation

Reconstruction can finish the fitter/extension and analytical checks while root
implements exact config/worker routing and support_ui adds report validation and
display. Once the descriptor and extension freeze, a separate consumer lane can
own source loader plus versioned controller/recipe and matched-loop routing;
root can coordinate recording/lab/receipt integration. Consumer implementation
is now assigned as listed above; shared scorer and numerical kernels remain
outside this lane. Root owns `camera_multiview_inputs.py`, pipeline, worker and
CLI. support_ui owns the report; avoid overlapping edits.

Small subsequent pure checks should cover mixed scales, a per-view regression
hidden by an improved mean, zero-ready secondary targets, missing/changed view
pins, excluded/tail preservation, unknown recipe rejection, exact selected alias
custody and cached action/replay preparation. Native matched trials and an
interactive acceptance remain root-owned future work after independent review
and frozen inputs/runtime. No new source, calibration or contact claim is made.

## Implemented consumer contracts and checks

New discriminators, with old profiles unchanged:

- Selected inputs: `camera-multiview-reference-inputs-v1`.
- Controller: `camera-multiview-source-fit-controller-v1`.
- Recipe: `camera-multiview-source-fit-controller-recipe-v1`.
- Trial family: `camera_multiview_angle_overlay`.

The recipe retains the exact angle overlay and compact `multiview_images` audit,
plus `additional_view_sha256` and `multiview_qualification_sha256`. The first is
the canonical hash of the complete `additional_view` extension, not its nested
binding hash. The second is the canonical hash of the dynamic per-view audit.
Both join the selected-input identity and sealed trial on lab attachment/resume.
Canonical hashes use Weft's default-spaced sorted JSON; raw file hashes preserve
supplied whitespace. No original policy, scene or task is relabelled.

The existing `camera-shared-reference-recording-v1` envelope and
`camera-shared-reference-full-state-replay-v1` protocol remain unchanged. The
explicit recipe family selects the new validation path. Recorded fitted actions
remain `reference_proposal`, never `learned`. Replay reconstructs the immutable
recipe under its exact runtime; it does not repeat source fitting. Saved receipt
inspection joins secondary camera/evidence raw hashes to both the independently
retained original source and trial aliases, and joins config/observation and
qualification pins. `secondary_world_alignment_sha256` binds the exact retained
explicit alignment bytes, and the retained trial family must be multiview.
It does not claim to recompute source geometry or replay.

Multiview runtime selection adds `camera_multiview_inputs.py`,
`camera_multiview_fit.py`, and `camera_multiview_report.py` to the existing camera
closure. The final selected-data readback recognizes the new camera family for
the original 64 MiB cap at that checkpoint; the measured correction below
supersedes only the multiview allowance. Legacy remains 32 MiB. The single-view
and legacy runtime selection branches remain unchanged.

Evidence currently comprises **36 distinct new checks and 10 affected existing
checks**, not native acceptance:

| Receipt | Scope |
| --- | --- |
| `.local/camera-multiview-controller-tests-01.xml` | 12 new + 2 old; exact recipe roundtrip, separate scales, per-view counts/clocks/gates, no pooled improvement, roots off/on, cached actions |
| `.local/camera-multiview-input-tests-01.xml` / `02.xml` | 2 old pass in 01; fixture's intentional unselected missing member caused 8 setup failures, preserved; corrected selected fixture then 8 new pass in 02 |
| `.local/camera-multiview-trial-tests-01.xml` | 5 new + 4 old; same matched loop/failure retention and three-read camera/legacy budgets |
| `.local/camera-multiview-runtime-tests-01.xml` | 1 new; actual bounded multiview code snapshot includes report/fit/input helper, reviewed numerical pins unchanged |
| `.local/camera-multiview-delivery-tests-01.xml` / `02.xml` / `03.xml` | 6 new + 2 old initially passed; two analytical-recording cases had test-only argument-shape mistakes, preserved in 01/02; corrected directory argument passes both in 03 |
| `.local/camera-multiview-custody-delta-tests-01.xml` | 2 new missing-alignment/family cases + 3 affected positives; final self-review closed those saved-only receipt joins without changing fresh admission or numerical code |

The delivery checks run real task/scorer/Recorder/replay code with analytical
dynamics only: five mixed actions, two fitted actions, one reset, seven events,
exact full-state replay with roots disabled/enabled. Lab prepare/resume uses real
selected-byte custody and a stubbed independent report validator; controller
preparation is mocked for that packaging check. The separate controller checks
exercise the actual unchanged numerical provider. Full new report validation is
owned by support_ui and must be independently frozen before native admission.

Saved producer integration: `.local/check_camera_multiview_consumer_01.py` read
only the manifest/body/overlay/report from `.local/camera-multiview-analytic-02`.
The producer seal is
`0527213eff0296380ada56b73b1e8d6c313e28d3a52344ee3cd19b4754921cc7`;
report pin `711233c4fa074a2b40d14bb224540a909182a24d84708d36d4749b2487231ccb`;
overlay pin `f8129287b3132c4cdbb0c10271f931cfbae1bfbc2c2f5071bd0e25beeaccbdd1`.
The actual new controller and archived recipe roundtrip passed in 0.032412 s,
reading four files / 131,593 bytes. Three original references, 36 known targets
per view and all 14 gates remain; compact recipe size is 14,249 bytes. Receipt:
`.local/camera-multiview-consumer-producer-01.json`. This is an authored analytical
producer result, not genuine source ancestry, native scene admission or a refit.

Next safe action: finish independent consumer/report review, freeze the complete
current runtime and preflight actual selected sizes. Root alone owns any later
bounded native proposal/trial/lab run. No live handle exists in this lane.

## Independent reporter review — closed

Read-only review of support_ui's implementation is clear at:

- `whole_body_reference_report.py`:
  `213a7b565a9558fb93842eef74a18b405bb8f6b87016e6758e95b342396576a6`.
- `camera_multiview_report.py`:
  `8ca3129cabcfa375de564fd06bd861c10cb3f4cf6cc8f236f810fccdfdacac89`.

Four findings were corrected by its owner: saved valid projections now obey the
absolute 0.2–100 depth domain; secondary radial coefficients stay in 0–1; every
secondary camera slot obeys the original frame/fps grid; and selected secondary
details show their own source identity/decoder/status while explicitly retaining
the primary body-reference identity. Targeted owner tests and default golden
results were reused, not rerun. The owner reports 30 distinct passing cases.

Exact profile/descriptor/raw/canonical joins, complete slot denominators,
construction-only payload access, each view's raw-pixel/depth arithmetic and
conjunction into the original fourteen gates were reviewed. The old-mode
serialized outputs and lazy optional-import boundary remain unchanged. No
unique-alias accounting or limit increase was introduced at that checkpoint:
input/output limits were 16 MiB. The measured input-only correction below
supersedes this boundary; output remains 16 MiB.

Receipt: `.local/camera-multiview-report-review-01.json`. This review did not run
tests, reports, fitting, models, media, source inference or a source study. It
does not establish source accuracy or native admission. Root owns the next
frozen end-to-end acceptance and all real-source execution.

## Measured multiview input budgets — frozen

Root authorized a narrow input-size correction after inspecting the exact
compact secondary package, seal
`fac738246d603a213e885e5898a40ac0499689ed7435f61ee7e60922e66003ca`.
Its observation file is 2,769,293 bytes; every parsed field and original slot is
retained under the new raw pin. No fit or source loader ran for this measurement.

The selected consumer inventory has **17,768,405 fixed bytes**, plus the new
report, angle overlay, plan, proposal manifest and any source-proof size delta.
Using the old compact report size (5,208,872 bytes) and old sizes for the other
three variable files gives 23,106,000 selected bytes. Three exact read passes
would consume 69,318,000 bytes, exceeding 64 MiB by 2,209,136 bytes. This is a
planning comparison, not the size of an actual two-view fit result.

The trial now reserves **80 MiB only for the authenticated multiview family**:
at most 27,962,026 initially selected bytes, with room for the loader-finish and
trial-finish hash rechecks. It still reads and hashes every required alias.
The extra allowance is 16 MiB of bounded local reads, not paid API work.
With an 8 MiB report, 1,805,013 bytes remain for the new overlay, plan, proposal
seal and any proof delta. The actual new outputs must still fit; this estimate
does not authorize clipping or skipping verification.

`load_reference_inputs` and the dedicated camera wrapper expose optional
`multiview_max_input_bytes=None`. Existing callers keep their prior limits.
The trial explicitly passes `2 * floor(80 MiB / 3)`; the loader selects this
allowance only after the sealed report names the new method and all fourteen
saved gates pass. Complete source/report validation still precedes models.
Single-camera inputs remain 64 MiB, noncamera inputs 32 MiB, and every selected
member remains at most 8 MiB. Source-proof (128 MiB), runtime (16 MiB), preserved
input-copy (32 MiB) and output limits are unchanged and separately accounted.
The existing lab attachment already reuses authenticated source bytes and its
bounded 32 MiB selected snapshot; no new caching mechanism was added.

Frozen consumer product hashes:

| File | SHA256 |
| --- | --- |
| `shared_exercise_reference_inputs.py` | `900891f8653731ec0a89fe574946a493a415cfcc45467e141162ed72cba7648e` |
| `camera_reference_inputs.py` | `9594115794d12371d7ca3bb0768cee09f8ba1f0bd22fc148531e9ffe59bf33a0` |
| `shared_exercise_reference_trial.py` | `fff3a512d4ead87e36a88bbf162e831a9a45a76dedcbf9d5b8ddf3a7c6d5e9bb` |

Nineteen targeted analytical/mocked checks passed in the 0.534-second process:
sealed-family selection, failed/mixed gates, invalid bounds before reads,
80 MiB boundary accounting, old 64/32 MiB limits, default callers and the same
matched trial loop. Native and network entrypoints were blocked; no physical
trial, source payload, report generation or fitter ran.
Receipt: `.local/camera-multiview-consumer-budget-tests-01.xml`, SHA256
`8f662a774781e247fe8a7c0248f4f13f79ace552a63c84f3735f3b634bb31cde`.
Measurement: `.local/camera-multiview-consumer-budget-measurement-01.json`, SHA256
`d484ca3e1bdeb23a041c57247881b53def2713335eef67dac05d98a6d244aa7d`;
five small metadata files / 23,716 bytes and five named data-file stats,
0.001023 seconds. The reproducible helper reads no source arrays.

The independent report-cap delta review is also clear. Frozen
`whole_body_reference_report.py` is
`5c06e9e44145664cb32017a1ffa4a32c2f3310a229ad607b4c30925e868a75c2`;
the helper hash above is unchanged. Automatic input bounds classify under the
old 16 MiB limit and select 24 MiB only for the explicit two-view method.
Explicit smaller caller caps remain binding; old profiles cannot receive the
increase. New-mode members remain 8 MiB, HTML 2 MiB and total output 16 MiB.
All logical alias bytes are charged normally. Owner tests were read/reused,
not rerun; no report recovery or source admission was performed here.

The fit/trial preparation below is historical; terminal results now follow it.
No live job is owned by this lane; local external spend is $0. No Git operation
or saved-artifact mutation occurred.

## Actual two-view consumer acceptance — preserved preparation

Root reports one completed source fit at
`artifacts/camera-multiview-real-01/fit-01/proposal`, seal
`27c1922017155f2160b805af3fdccf4d3620933944dfab56911d921e9b797adc`.
Its outer workflow failed only the offline HTML-size bound. Preserve that
failure; no refit is planned. The consumer uses the completed proposal's exact
`report.json` (6,924,187 bytes), angle overlay and selected original inputs.
It does not require the recovered HTML inspector or reinterpret its status.
All fourteen producer gates still must pass the actual selected-input loader.

The original world remains `artifacts/camera-shared-admission-jog-01/candidate`,
seal `29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`.
The exact roots-on task remains
`artifacts/camera-shared-policy-play-01/lab/run/exercise-task.json`, raw SHA256
`3fefe443f56153c0b5a5d50a7c1c318b1a930a01c93f70c298ca18314f98e558`.
Use only `.local/camera-multiview-real-runtime-01`, manifest
`5d22340e520e958c31d41392121103aa5f8f9a8ff116226219a2a7f6957a6de0`;
do not refresh it from the live tree or the unrelated later avatar commit.

Prepared local files:

- `.local/camera-multiview-consumer-plan-01.json`:
  `f22aec1e4b0c4474d7665122c10327487919a575e727b279d202db47687da047`.
- `.local/run_camera_multiview_consumer_01.py`:
  `3e7c84495ae1d30f7e8d97e228386b99fc8e9a24eb30889efc8bd75b4ca36f5d`.
- `.local/prepare_camera_multiview_consumer_01.py` generated the plan using
  only named metadata reads/stats and syntax parsing; no product imports or
  native work. Its receipt output reports 86,936 metadata/code bytes read.

Actual selected-file sizes total 24,829,763 bytes; all three reads require
74,489,289 bytes, leaving 9,396,791 bytes below 80 MiB. Exact hashes still must
pass during admission; file-size planning is not an admission result.

Root approved one existing matched authored/fitted comparison, seed 1729 and
no jitter, with the unchanged 238-step task. It will retain early terminal
failures and both recordings/replays. Then the same new recipe attaches to a
new lab: manual 2, coast 2, authored 2, fitted 2, reset 29, fitted 2; one mixed
recording/export/exact replay, saved receipt inspection and fresh resume.
There is no repeated full fitted interval in the interactive lab.

The one process is bounded at 972 live-plus-replayed controls, 8,748 substeps,
3 exports, 128 models, 16 data allocations, 2,500 forwards and 160 resets;
180 seconds worker wall time, 195 seconds caller wall time and 150 CPU seconds.
Outputs are capped at 384 MiB / 2,500 files, 64 MiB per file. The 768 MiB RSS
threshold is an observed acceptance condition, not a hard memory limit.
The caller must enforce log/output limits and verify timely exit; the worker's
outer seal explicitly remains `execution_completed_pending_caller_exit` until
that separate verdict. Inner trial/lab seals retain their own normal semantics.

Existing lab custody must hash/copy `source.mp4` (11,196,027 bytes) and
`background.jpg` (892,469 bytes); root explicitly confirmed this unchanged
custody is allowed. All source decoding, acquisition, inference and fitting
remain blocked. Only the three new `video.partial.mp4` exports may be opened
for verification. No API calls, training, server, browser or paid work.

Root owns read-only helper/plan review and any later single invocation. Do not
execute it from this lane or duplicate an active job. At preparation release,
`artifacts/camera-multiview-consumer-01` and its process receipt are absent.

## Terminal matched trial and lab recovery

On October 31, 2026, at 11:23 PM PDT, both root-owned invocations were terminal.
No native work was performed by this review lane. Do not repeat the fit or
matched comparison to recover the later packaging failure.

The first combined invocation completed the entire matched trial, then failed
at lab preparation with `Package input is missing or unsafe`. Its fit-only
runtime lacked `pyproject.toml`, `requirements.lock` and
`requirements-lab.lock`; the first missing path was the runtime-root
`pyproject.toml`. It had all required lab assets. The failure is preserved at
`artifacts/camera-multiview-consumer-01`, outer seal
`fca1bf9a379292e23056ac506d59a9ff99377b8c028d1e771be44d8f73397a1d`.
Two scene/task validation models ran after the trial, but no lab or mixed
session was admitted. A separate unreached harness mistake expected a returned
`status` key; the actual prepare API returns the sealed run identity.

Its completed `trial/` remains independently sealed:
`bbaa245a99eb96aad153acdc2e8a2184b9dc403440847700606829027f2231d2`.
Both controllers retained all 238 actions / 239 events, seed 1729, no jitter,
240 original source slots, 112 reference rows and the fractional source clock.
Their entire initial snapshots matched. Both full-state replays passed, and
all 238 fitted decisions were regenerated. Total live-plus-replayed work was
952 controls / 8,568 substeps / two exports. All 238 paired control vectors
differed; maximum absolute channel difference was 0.9528508549599293.

| Original task metric | Authored | Fitted |
| --- | ---: | ---: |
| Mean reward | 0.8262091657332693 | 0.8061517650825086 |
| Angular RMSE (radians; lower is better) | 0.145736325284855 | 0.1563043102489617 |
| Authored simulated task gates | passed | passed |

The fitted controller is **not a task-quality improvement**. Source image
fitting and simulated task tracking are different objectives; neither result
establishes physical contact, balance, anatomy or transfer.

Root extracted only the three missing metadata files from committed
`41cfee28`, then the preparation helper copied all 277 original runtime members
unchanged into `.local/camera-multiview-lab-runtime-01`. Its 280-member seal is
`21b2386878375702a55abc89c05d85bdeafe1ae10b1e521aa4865613e9daf088`.
No recipe or code was repinned. The lab-only plan is
`.local/camera-multiview-lab-plan-01.json`, SHA256
`dd35bbd3c6396f423e816276569abcd59517408402a9443580df91ebb0a519c0`;
helper `.local/run_camera_multiview_lab_recovery_01.py`, SHA256
`3da8179426280bc16e9857cf64aa2358c5834d026322c08def06c1bc96826430`.
It fixes the harness to check the published run/status and returned manifest
pin, with no trial/fitter call. Bounds: 90-second worker / 105-second caller,
75 CPU seconds, 224 MiB / 1,500 output files, 20 controls / 180 substeps /
one export. Setup ceilings remain conservative; actual validation used 120
resets independent of the short interaction length.

The separate recovery passed once in 8.201604 seconds, with 402 files /
78,861,238 bytes and observed peak RSS 451,690,496 bytes. Ready lab:
`artifacts/camera-multiview-lab-recovery-01/lab`; run seal
`9269eca5b5941a80f1c005925b08b9ecbd11d9a8909a223bf49066720453532e`.
Outer seal:
`e5694d64f57fcd1611edf059d45f661aa0d845f8ef35caecf4d6f528355db975`.
Session `session-5548b0b5549c` contains manual 2 / coast 2 / authored 2 /
fitted 4 actions, one reset from seed 19 to 29, and 12 total events. Its one
exact replay verified all ten actions and four fitted decisions; fresh resume
passed. It uses the unchanged recipe
`ba717d643952ff4cdc3802d363ae354b70b00f0ffa775542515cac444521191e`.
The short session is exploration, not another completed full task. There was
no source decode, inference, refit, training, network or paid call. The known
MuJoCo startup subprocess attempt remained denied; no permission was widened.

Independent saved-only receipts:

- `.local/camera-multiview-trial-saved-audit-03.json`, SHA256
  `195c4c92a5add6aa21f4146feab7de5744141b648a99f49edf9469c31a3d87b8`:
  23 selected files / 13,793,687 source bytes; each journal streamed once.
  Including cached analysis, 32,555,954 bytes stayed below the 32 MiB bound.
  Checker receipts 01/02 preserve incorrect assumptions about report-only
  replay status and the recipe envelope. Corrections used cached bytes only;
  no replay or journal reread. Default-spaced canonical hashes were used.
- `.local/camera-multiview-lab-saved-audit-01.json`, SHA256
  `295c0d9a6e37cc3068a28e0edadab6a1441c46691852d7fd34fbd5025cdcfd27`:
  21 selected files / 1,996,527 bytes, one 275,287-byte journal stream,
  0.030904 seconds active read/check. Verified original scene/task/recipe,
  additive runtime, mixed labels/reset clocks, receipt custody and resume.
  The parent's complete output hashes were reused; no media was decoded or
  redundantly rehashed.

Next safe action: use the prepared lab through the existing local launcher and
keep its frozen recipe/runtime intact. Root owns publication and any new
studies. The original failed combined acceptance and failed viewer workflow
remain failed; successful inner trial and separate lab recovery do not relabel
them. Generalized unfamiliar-video reconstruction and physical support
qualification remain open work.
