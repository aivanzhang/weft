# Optional second-view workflow integration

Updated October 31, 2026, 11:00 PM PDT. The combined goal remains active.
This is optional supplied-camera construction, not a second-video requirement
or proof of unfamiliar monocular reconstruction or physical contact.

## Ownership and current state

Root owns `camera_multiview_inputs.py`, the opt-in branches in
`whole_body_reference_pipeline.py`, `whole_body_reference_worker.py`, `cli.py`,
the new input/workflow tests, common documentation, and eventual frozen acceptance.
Reconstruction owns `camera_multiview_fit.py` and the narrow existing-fitter
integration; UI owns the saved report; the consumer agent owns exact loader,
controller, matched trial, recording/lab/receipt compatibility. See
[fitting](camera-multiview-fit.md) and [consumer review](camera-multiview-review.md).
No lane owns the protected scorer or unrelated avatar changes.

The preceding equipment comparison is published at
`41cfee28ceef55146fbbf6e29ef8248c475bd46d`, with exact private remote verification
in `.local/equipment-support-publication-01.json` (SHA
`c461b151ee7ddc39059ec99334e935cfc49cfed6e3c62e173ba3b7eaebeaedf8`).
Its two original co-motion candidates were both excluded, not admitted as contact.
All prior saved studies and the playable jogging lab remain unchanged.

## User entry point being connected

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run path/to/original-sealed-camera-body \
  --objective camera-multiview-surface-intervals \
  --surface-config path/to/primary-surface-config.json \
  --camera-image-scale 0.002 \
  --secondary-view-config path/to/secondary-config.json \
  --out artifacts/new-two-view-proposal
```

This route has a completed real-source fit; offline report delivery is recovering
from a size failure. Full trial/lab acceptance remains pending below.
It uses the existing bounded worker and saves a distinct
`camera-body-multiview-angle-proposal-v1` overlay, not a replacement body.
The original single-view command and recipe family stay strict. New recipes
must explicitly qualify both views before trial/lab/recording use.

## Exact input contract

The secondary config has only these fields:

```json
{
  "schema_version": 1,
  "profile": "camera-secondary-view-fit-config-v1",
  "primary_source_sha256": "<original primary video SHA256>",
  "secondary_source_sha256": "<original secondary video SHA256>",
  "observations": [{"path": "observations.json", "sha256": "<raw file SHA256>"}],
  "world_alignment": {"path": "alignment.json", "sha256": "<raw file SHA256>"},
  "correspondence": {
    "primary_entity_ref": {"document_sha256": "<raw observation SHA256>", "object_id": "person"},
    "secondary_entity_ref": {"document_sha256": "<raw observation SHA256>", "object_id": "person"},
    "status": "assumed",
    "method": "Explicit same-person correspondence",
    "uncertainty": "Not independently verified identity or anatomy"
  },
  "image_scale": 0.002,
  "weight": 1.0
}
```

Scales and weights are authored loss settings, not measured uncertainty.
The optional alignment may be `null`; absence remains unknown and the core
rejects fitting before geometry rather than falling back silently. The supported
mapping is an explicit identity world/clock declaration between the two exact
original cameras; equal frame IDs or frame rates do not establish synchronization.

Camera and evidence bytes come only from the original checked body snapshot:
`inputs/secondary-camera.json` and `inputs/secondary-evidence.json`.
They are reused, not copied under duplicate aliases. New files are
`inputs/camera-secondary-config.json`, optional
`inputs/camera-secondary-alignment.json`, and numbered
`inputs/camera-secondary-observation-00.json` through `07.json`.
Relative config members cannot escape their directory or traverse symlinks.

`plan.secondary_view` pins the config, both source identities, raw camera/evidence,
ordered observation hashes, optional raw alignment, canonical correspondence,
and explicit image scale/weight. `camera_multiview_member_names()` derives the
only allowed extra inventory. Snapshot validation and worker preimport validation
share the exact contract; a broader source package cannot supply an unselected
alignment. Original camera/evidence canonical joins are independently checked
against the body's original two-view root record by the fitter.

## Bounds and checks

- Config ≤64 KiB; alignment ≤16 KiB; each observation/pose/camera ≤8 MiB;
  one to eight observation documents and ≤16 MiB total secondary selected input.
- Existing overall input bound stays 32 MiB/64 files; worker output stays
  16 MiB combined, workflow output 64 MiB/96 files. Preparation intersects the
  overall deadline and gets at most five seconds. No blanket capacity increase.
- Existing native/QP/time/projection caps, first reference, body/root/floor,
  original source clocks, fourteen gates, and scorer remain unchanged.
- Worker output checks require exact additional-view pins and separate raw-pixel,
  projection-domain and depth gates. Pooled improvement cannot hide a failing
  view. Honest rejected proposals remain saved as rejected.

Root checks completed: **52 distinct new cases** — 28 input cases and 24
workflow/worker/CLI cases. The initial 27 input cases and 23 workflow cases passed;
after closing an unselected-alignment lookup, the two affected option cases and
one new regression passed. The new cold-runtime import and the existing import
check also pass; all 126 affected camera/surface/pipeline/worker legacy cases pass.
Receipt `.local/camera-multiview-root-checks-01.json`, SHA
`0cdd5db6616e1d5b11ae8b216cccd648db239cb8f48f83a8151765d058cd8fa0`,
records that checkpoint. A subsequent formatting-only change writes the new-mode
worker report as compact JSON; the affected output case passes. One saved
authored report is 163,753 bytes pretty versus 107,528 bytes compact, with the
complete parsed document unchanged. This is not a real-source performance claim.
Receipt `.local/camera-multiview-report-encoding-01.json`, SHA
`cb1edfcb2d54a4546a75761e283c50b822edb1b669cac85bc62aace9de9726da`.
No unchanged broad rerun. These are metadata/mock
checks, not real-source/native acceptance or browser QA. Reconstruction has its
separate analytical QP evidence; do not add these counts twice.

Commands:

```sh
.venv-lab/bin/python -B -m pytest -q tests/test_camera_multiview_inputs.py
.venv-lab/bin/python -B -m pytest -q tests/test_camera_multiview_workflow.py
.venv-lab/bin/python -B -m pytest -q \
  tests/test_camera_multiview_inputs.py::test_exact_snapshot_preserves_existing_camera_copies_and_optional_unknown_mapping \
  tests/test_camera_multiview_inputs.py::test_unselected_alignment_in_broader_source_package_is_never_consumed
```

## Earlier pre-fit next action — superseded below

Finish and independently review fitter/report/consumer integration; run affected
legacy checks and cold-runtime checks once. Then freeze exact source metadata,
settings, committed-base runtime plus reviewed overrides, execution limits and
new output directories before any saved-source fit/native trial. No scientific
job or source inference is running. Current cost is $0 provider/recurring spend;
no keys or shared Meta ledger were read. Do not repeat earlier completed fits,
equipment studies, trials or sandbox recordings.

Current changes are local/uncommitted. Main remains the only branch; preserve
the separate avatar changes and saved artifacts. Publish only the reviewed scope
after relevant acceptance, and record the actual remote result rather than a
planned hash. General unfamiliar-video ground/equipment/contact recovery remains
unfinished regardless of optional two-view software success.

## Actual saved-input preparation — October 31, 2026, 10:34 PM PDT

One reviewed metadata-only preparation passed; root handle `12531` exited zero.
It consumed ten exact saved input files/7,635,202 bytes, in a committed-base
Python runtime with only three reviewed preparation/core overrides. No source
video, native engine, fitter, inference, training or paid API was executed.
The caller measured 1.156368 seconds; child observed peak RSS 110,788,608 bytes.
This is measured local preparation, not reconstruction accuracy or a memory cap.

Output `artifacts/camera-multiview-real-01/secondary-inputs-01` has six files and
5,653,160 bytes. Its prepared-metadata seal is
`9e6f6c9b2e62c0e0ebb2612e40e01a05d4bf4777676136d7d696e707ab038867`.
The required timely terminal process receipt is
`.local/camera-multiview-source-process-01.json`, SHA
`0ee7cd1cfc20de88f7e200aa0b9fb4831b8b50419f9116ead846ac30b445c6e5`;
input-qualification receipt SHA
`0449e0b827799033b0f4a7feb0ef058ba04951afda1f3313fab3241a2e6c426b`.

All 240 source slots per view remain. Original 112 references select 112 of the
120 secondary pose rows; eight excluded rows and 120 unsampled slots remain
explicit. There are 1,241/704 known image targets and 103/640 missing targets
across primary/secondary views. The camera/world/clock/person mapping is still
an explicit assumption from the original packaged jogging data, not independent
synchronization or identity evidence. All 480 primary surface-channel slots
remain unavailable; no contact, floor, root or body geometry was changed.

Final plan03 SHA
`958805019520b8e3418c80649335509cad2f5502f0d896e90fcab56b1817a339`;
helper SHA `ba230af020797a8560139f4ff37eaff2a3caa0c33714c65342bbbe279ac6143e`;
supervisor SHA `5c745798056a98dbba45e394112d1b77ca16d82c192863039f46b5c42cb706b4`;
runtime manifest SHA `c40b75ac20141b5c5b4a97c2a21bd89b25a0abfad423969349715579202591fc`.
Earlier plans01/02 are preserved and were never executed. Preexecution review
closed preexisting-directory writes, manifest-byte reservation, and late terminal
I/O acceptance. Only caller-measured timely exit zero qualifies the prepared
metadata; a prepared seal alone does not.

Core is frozen with 30 new analytical cases; consumer integration has 36 new
and ten affected passes and a clear independent read-only review. UI is closing
the actual saved-depth and missing-secondary-pose boundaries before freeze.
Current capacity concern: the secondary observation document is 5,145,496 bytes,
with 1,949,660-byte pose and 128,196-byte camera. The existing report reads
duplicate primary aliases into its 16-MiB input allowance. Measure/deduplicate
exact aliases or use a separately pinned lossless encoding before a source fit;
do not launch a predictably oversized report or truncate evidence. No fit/native
job is active. The prior successful source preparations/fits are not to be rerun.

## Real two-view fit — October 31, 2026, 11:00 PM PDT

The single actual fit is terminal. Its **worker completed**, but the outer
workflow failed while building HTML larger than 2 MiB. Preserve both outcomes:

- Proposal: `artifacts/camera-multiview-real-01/fit-01/proposal`, seal
  `27c1922017155f2160b805af3fdccf4d3620933944dfab56911d921e9b797adc`.
- Failed outer seal:
  `7454d904787761c163a52328cb7df17f62d951efced7c4bf4572f8ed43f7806a`.
- Process receipt: `.local/camera-multiview-fit-process-01.json`, SHA
  `b250f0f31181900bdfa5d1842e26b7174f1dd508d1895e89a79eaa5481bb3173`;
  supervisor: `.local/camera-multiview-fit-supervision-01.json`.
- Reviewed execution plan: `.local/camera-multiview-fit-plan-02.json`, SHA
  `50b8305687964245df576be36e28ee8a55026ca9a6a246cff34a063776b3394c`.
- Runtime: `.local/camera-multiview-real-runtime-01`, seal
  `5d22340e520e958c31d41392121103aa5f8f9a8ff116226219a2a7f6957a6de0`.
  It contains 277 files/5,949,550 bytes from committed `41cfee2` plus fourteen
  reviewed overrides. Later avatar-only commits are intentionally excluded.

Worker time was 4.167200 seconds; caller finished unsuccessfully in 5.230636
seconds because report delivery failed. The worker used two model compilations,
four data allocations and 18,266 forward calls; all fourteen construction gates
pass. There were zero physics steps, resets, media decodes or inference calls.
Observed child peak RSS was 163,577,856 bytes, not a hard memory guarantee.
The known optional MuJoCo startup subprocess probe stayed blocked. No paid API
or recurring resource was used. Independent saved audit02 passes: 375 accepted
updates changed 104 of 112 references, with 666 QP calls and 6,860 projections.
Mean detected-point error is **12.792154 → 10.868088 pixels** for the primary
view (1,241 targets), and **27.036614 → 24.971869 pixels** for the secondary
(704 targets). All 351 saved floor checks have zero penetration; all 480 surface
channels remain unavailable, so contact is still unknown. This is exposed source
agreement, not task improvement, unfamiliar-video accuracy or physical truth.

Audit `.local/camera-multiview-real-fit-saved-audit-02.json`, SHA
`991993f9800efd5f7d90ef3ef7d1b513125d0d96d6c3e4be935b0e76f6be5b98`,
checked four saved files/7,049,736 bytes in 0.09585 seconds without fitting,
projection or native execution. Audit01 remains as a checker-only failure:
subtracting before/after error at a floating-point boundary differed from the
producer's exact `after <= before + 1e-8` comparison. The correction used the
unchanged producer comparison, not a looser gate. Both audit reads total
14,097,886 bytes, below their 24-MiB bound.

The CLI now distinguishes a sealed proposal from failed viewer delivery and
prints a shell-quoted, exact-pin report-only recovery command. One new test and
one affected legacy CLI case pass; independent review is clear. Receipt
`.local/camera-multiview-cli-recovery-checks-01.json`, SHA
`fcdf222c25c180f6c1252b88bd1fb9c36d9b456feb7ea3e9b0b8ee33d57eb18a`.
This raises root coverage to 53 distinct new cases; scientific runtimes/results
were not changed for a terminal message.

### Lossless encoding and measured limits

The fit consumed `secondary-inputs-02`, not rewritten preparation01. Its seal is
`fac738246d603a213e885e5898a40ac0499689ed7435f61ee7e60922e66003ca`.
Compact JSON reduced observation bytes from 5,145,496 to 2,769,293 without changing
any parsed value, array, canonical identity, 240 source slots or 112 selected
records. Only the config's observation raw hash and entity-document raw hash
changed. Original inputs and their preflight identity remain intact. Independent
saved audit `.local/camera-secondary-encoding-saved-audit-01.json`, SHA
`95dfbe8663d1c93fd1b133dece3eace6839eb41c9b0c8352326c74cdab9b7474`,
passed without source processing or refitting.

Only explicit multiview reports may read 24 MiB of selected input; legacy modes
retain 16 MiB. Explicit tighter limits are honored, each new-mode member remains
bounded at 8 MiB, and output/HTML stay 16 MiB/2 MiB. Consumer qualification retains
all three exact read passes under an 80-MiB multiview-only cap; single-view camera
and noncamera caps stay 64/32 MiB. The fixed retained inputs total 17,768,405 bytes
before report/overlay/plan/proof deltas, so this is a measured extra-view allowance,
not skipped validation. Nineteen affected consumer budget cases and independent
review pass; receipt `.local/camera-multiview-consumer-budget-checks-01.json`, SHA
`be0afe061f837da0602b3e74d87163449038b285081d2ce5f05822470d719db2`.

### Next safe action and ownership

**Do not repeat the completed fit.** UI owns a smallest new-mode-only layout fix
and saved-data sizing; root will execute the reviewed reporter in a new recovery
directory. Reconstruction independently audits saved per-view arithmetic and
custody without geometry or fitting. Consumer prepares one matched original-task
trial followed by a ten-action manual/coast/authored/fitted recording with reset,
export, exact replay and fresh resume. A recovered viewer is not a trial dependency;
the sealed completed proposal is. No source video, inference, new training or paid
call is required. All failed output directories remain immutable.

Current work remains uncommitted on main. Main/cached origin are at avatar-only
`050deb54933d052abcca1b786dabdd00ad5aaa53`, a descendant of the science base.
Preserve the separate dirty `avatar_replay.js` and `avatar_replay_smoke.mjs`; stash
is empty. Publish only reviewed two-view code/docs/tests after delivery checks.
The overall goal remains active, including unfamiliar-video reconstruction and
independently supported ground/equipment contact; this optional known jogging
case does not complete either.

### Bounded consumer execution prepared

Plan `.local/camera-multiview-consumer-plan-01.json`, SHA
`f22aec1e4b0c4474d7665122c10327487919a575e727b279d202db47687da047`,
and helper `.local/run_camera_multiview_consumer_01.py`, SHA
`3e7c84495ae1d30f7e8d97e228386b99fc8e9a24eb30889efc8bd75b4ca36f5d`,
are reviewed for one execution in the existing frozen real runtime. The plan
uses the original 238-step task, authored/fitted matched controls, three recording
exports including one ten-action interactive session, one reset and fresh resume.
Maximum work is 972 live/replay controls and 8,748 native substeps. Worker/caller
limits are 180/195 seconds, CPU 150 seconds, output 384 MiB/2,500 files, with a
768-MiB observed-RSS acceptance check rather than a hard RSS limit.

The exact selected metadata total is 24,829,763 bytes; three retained read passes
total 74,489,289 bytes, below 80 MiB. Lab custody also requires hashing/copying
the original sealed 11,196,027-byte source video and 892,469-byte background;
that existing-media custody is allowed, but source decoding stays blocked.
Only the three newly exported recording videos may be decoded for verification.
No fitting, inference, learning, paid call or source acquisition is permitted.

Root owns `.local/supervise_camera_multiview_consumer_01.py` and final timely
exit-zero acceptance. The worker's `execution_completed_pending_caller_exit`
status alone is not final delivery. Output must be new at
`artifacts/camera-multiview-consumer-01`; supervisor/worker receipts will be
`.local/camera-multiview-consumer-supervision-01.json` and
`.local/camera-multiview-consumer-process-01.json`. Their absence means execution
has not completed; never launch a duplicate if the root process is still live.

Root launched the single consumer acceptance under process handle `7664`.
Final supervision is hash `5c42a92f137f792ff3583fa23c6b5c102564232776bbc15baa86dd05a73218a0`;
review02 is `3583dc7b5128f2f4286a2d62d9b321bc9bc5e32d7f4a294213924dc52f86c0f7`.
Preexecution review replaced unbounded pipe capture with a combined 1-MiB
selector buffer and kill/reap on overflow or timeout. Aggregate disk size is a
post-stage/terminal acceptance check, not a hard live disk quota. Root owns the
handle, which is now terminal as recorded below. Do not start another full run.

### Consumer01 terminal: trial complete, lab attachment failed

Root process `7664` exited 2 after 15.041553 seconds. The worker completed both
238-step trials, two exports and both exact full-state replays, then stopped at
lab preparation with `ValueError: Package input is missing or unsafe`. No mixed
session was created. Preserve this partial delivery; **do not repeat the fit or
the matched trial**.

- Completed trial seal:
  `bbaa245a99eb96aad153acdc2e8a2184b9dc403440847700606829027f2231d2` at
  `artifacts/camera-multiview-consumer-01/trial`.
- Failed outer seal:
  `fca1bf9a379292e23056ac506d59a9ff99377b8c028d1e771be44d8f73397a1d`.
- Process receipt SHA:
  `1539caa2ab2bc109bd4040cdd8ebc7827e039444640117e64f39ddb2221adf60`.

The attempt used 952 live/replay controls, 8,568 substeps, 42 compiled models,
three data allocations, 960 forwards and four resets. All attempted native
calls returned; peak RSS was 392,609,792 bytes. There are 109 retained files,
55,878,075 bytes. Source/task/scorer and initial state remain identical.
Both controllers pass the original authored simulator criteria, but fitted
reward is **0.806152 versus 0.826209 authored**, and angular RMSE is worse:
**0.156304 versus 0.145736 rad**. Do not claim control-quality improvement.

Diagnosis found three missing project-metadata files in the fit-specific frozen
runtime: `pyproject.toml`, `requirements.lock`, and `requirements-lab.lock`.
All public rendering assets were present. Lab preparation reads the project
metadata after native source/task validation; the absent `pyproject.toml` caused
the stop. A separate unreached harness assertion expected `status: ready`, but
the actual API returns paths/manifests; recovery must check the sealed run and
lifecycle status instead. Neither issue requires changing product/scoring code.

The three exact files were acquired from a local archive of committed `41cfee2`,
not the current tree, at `.local/camera-multiview-lab-project-files-01`, seal
`0f70eca3b1192aa235c9c26e6226b39823a55d7f7c9df4a2e1c7a90545862da3`.
They total 1,723 bytes. Consumer owns independent trial-journal audit and a
**lab-only** recovery in a new directory/runtime with those additions; all 277
existing runtime members and the completed recipe must stay byte-identical.
Never repin the recipe or rerun completed controls to fix packaging.
Report recovery continues independently and will inline its bounded packed display
data, avoiding a new local-file origin policy. No native job remains live.

### Offline viewer recovery — completed October 31, 2026, 11:19 PM PDT

The single public saved-report recovery completed in **0.830279 seconds** caller
time (0.706331 seconds generation), with no fitting, native engine, source decode
or API call. `artifacts/camera-multiview-real-01/inspection-recovered-01` contains
26 files/16,614,274 bytes; manifest SHA
`64c6d9bbb55d869d193664924810cab84b83bf4c6c4ac1c58f8bf41b73755ade`.
Root verified all 25 sealed files, 21 logical source identities including exact
gzip/alias recovery, and 27 local links. Original completed proposal and failed
outer manifests remain unchanged.

The viewer preserves 120 source-pose rows, 239 control-grid rows and all 240
original slots in each camera. HTML is 1,891,052 bytes; the complete plain JSON
model is 2,997,652 bytes. Its 1,847,907-byte packed inline display copy has SHA
`6d8c653f1c0b3401e2fe32c6e288ca855d655a483fed9a1fb5f7e9492e728186`,
checked against the HTML and model metadata. It uses no external script/fetch or
server; full data remain in the plain model and exact source downloads. Peak
observed RSS was 160,317,440 bytes. Existing HTML/output limits are unchanged.

Process receipt `.local/camera-multiview-report-recovery-01.json`, SHA
`694aa8e556d69a437a0d7ce0d1006c9912deecfbcebcf2a1c0d0bf0fac8d0ea6`;
caller receipt `.local/camera-multiview-report-recovery-supervision-01.json`, SHA
`98c5d79e0c0b06c7cacf52a0a851a816dd55cddec686af97005aff86e030edc2`.
Plan02 SHA `d18240512c47c28d15837e471d5dbb88a34139bd1e46d0a6c2e46a5903a72013`;
runtime `.local/camera-multiview-report-runtime-01`, seal
`fca4bd76d4e21c772a21c276af76862c0dc40dc355b856af8a9b62106d102d9c`.
Only the two reviewed report modules differ from the prior 277-file fit runtime;
all 275 other pins remain exact. Unexecuted plan01 is preserved. UI owns one
bounded actual-payload DOM check, not another generator or browser run.

### Lab-only recovery prepared

Plan `.local/camera-multiview-lab-plan-01.json`, SHA
`dd35bbd3c6396f423e816276569abcd59517408402a9443580df91ebb0a519c0`,
uses the completed trial and exact recipe
`ba717d643952ff4cdc3802d363ae354b70b00f0ffa775542515cac444521191e`.
New runtime `.local/camera-multiview-lab-runtime-01`, seal
`21b2386878375702a55abc89c05d85bdeafe1ae10b1e521aa4865613e9daf088`,
keeps all 277 prior files and adds only the three exact committed project files.
Root independently checked that manifest delta. The corrected harness checks
the real API's sealed run/lifecycle status and returned manifest hash.

One execution may prepare `artifacts/camera-multiview-lab-recovery-01/lab`, record
ten new mixed actions and one reset, export once, replay those ten actions once,
read the saved receipt without native work, and resume. Limits: 20 total controls,
180 substeps, one export, 90/105 seconds worker/caller, 75 CPU seconds and
224 MiB/1,500 terminal output files. No new matched trial, fit, source decode,
training or paid work. Parent uses the already reviewed bounded-stream supervisor;
root owns the invocation and terminal verdict. Preserve consumer01 unchanged.

### Lab-only recovery — completed October 31, 2026, 11:23 PM PDT

Root handle `67884` exited zero. Caller verification passed in **8.201604 seconds**;
the worker finished in 8.088550 seconds. The new outer seal is
`e5694d64f57fcd1611edf059d45f661aa0d845f8ef35caecf4d6f528355db975`,
with 402 retained files/78,861,238 bytes. The prepared lab is
`artifacts/camera-multiview-lab-recovery-01/lab`, run seal
`9269eca5b5941a80f1c005925b08b9ecbd11d9a8909a223bf49066720453532e`.

Session `session-5548b0b5549c` contains two manual, two coast, two authored and
four fitted actions, plus one reset: ten actions/twelve events total. All ten
actions/full states replayed exactly and all four fitted decisions were verified.
Saved receipt inspection performed no native work, and fresh lab resume passed.
This deliberately short interaction is not a full-task success; the two complete
238-step task comparisons remain in the previous trial and were not repeated.

Actual work: 20 live/replay controls, 180 substeps, 41 model compilations, five
data allocations, 148 forwards, 120 validation resets and one video export/check.
Observed peak RSS was 451,690,496 bytes. The expected optional MuJoCo startup
process probe stayed blocked; there were no other denied/forbidden operations,
source decodes, fitting, training, paid calls or server/browser interactions.

Process receipt `.local/camera-multiview-lab-process-01.json`, SHA
`5df69014222dc6304295e19cf7b0cfb61c05d9654ab24c56e8b876441d4e65ad`;
caller receipt `.local/camera-multiview-lab-supervision-01.json`, SHA
`a0730a54b7794243c008accfb7ec9953b75e9b9411a2802dab70bb070d4f477d`.
Both earlier failed outer workflows remain preserved. The complete journey now
has source-fit evidence, a recovered viewer, full matched trials, a playable lab,
mixed controls and recording/replay; it does not establish physical accuracy or
unfamiliar-video/contact recovery.

### Independent saved checks

Actual viewer audit02 passed using only the three saved page/model/manifest
files: all 480 camera-slot visits, 224 fitted-view details and 2,688 packed/full
point records agree. Three thousand unchanged selections cause zero DOM writes;
109 point nodes remain live. Receipt
`.local/camera-multiview-recovered-ui-audit-02.json`, SHA
`ed0e1fca63756fee770c588bd42b2780b00b6bf61a0645eff81dec08481a5977`.
The checker01 failure confused 120 retained source rows with 112 fitted
references; it remains preserved. Both reads total 9,783,542 bytes within the
explicit 10-MiB aggregate bound. This is scripted DOM behavior, not browser QA.

Matched-trial audit03 also passed: identical initial snapshots, exact original
source/task/body/scorer/runtime/secondary identity, 238 actions/239 events per
controller, 2,142 substeps each, and all fitted decisions covered by saved replay
proof. All 238 paired action vectors differ. Receipt
`.local/camera-multiview-trial-saved-audit-03.json`, SHA
`195c4c92a5add6aa21f4146feab7de5744141b648a99f49edf9469c31a3d87b8`.
Journals were streamed once; cached checker-schema corrections did not repeat
physics or media work. Checker failures remain retained.

The short-lab audit also passes: 21 selected files/1,996,527 bytes, one
275,287-byte journal stream, 0.030904 seconds active checks. Recipe, task, scene,
all 277 original runtime members, three approved metadata additions, 12 events,
ten actions, four fitted decisions, reset seeds 19→29 and exact replay/resume
identities agree. Receipt `.local/camera-multiview-lab-saved-audit-01.json`, SHA
`295c0d9a6e37cc3068a28e0edadab6a1441c46691852d7fd34fbd5025cdcfd27`.
It reused the parent's whole-output inventory and performed no simulation/media
work. No verification job remains live.

### Publication and next ownership

The reviewed checkpoint contains 29 source/test/document paths. All fourteen
product source hashes match their reviewed releases; the protected scorer is
unchanged. Root owns scoped commit/push and records actual commit/private remote,
scope cleanliness, stash and excluded active work in
`.local/camera-multiview-publication-01.json`. If that receipt is absent, publication
is unfinished; do not infer it from this preparation note. No generated evidence,
runtime cache, key or unrelated avatar change belongs in the commit.

The combined goal remains active. Reconstruction now owns an independent
named wrist/pinky/index/thumb evidence helper, tests and narrow pose/silhouette
preparation changes. Root owns its later constraint/selector/CLI/UI integration.
This next lane preserves default wrist behavior and may only constrain original
equipment candidates; image overlap cannot become a contact/grasp/load claim.
No source/native pilot is running. Preserve its work separately from this frozen
two-view publication; do not wait for it to publish already completed work.
