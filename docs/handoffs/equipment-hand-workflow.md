# Optional named-hand equipment evidence

## Current result

The opt-in command and [completed offline report](../../artifacts/equipment-hand-support-02/index.html)
work end to end on the saved kettlebell clip. Both original right-wrist/sam-2
candidates at frames 20 and 22 are kept: each original seven-sample window has
seven qualified hand intersections, zero exclusions and zero unavailable groups.
No new candidate, contact, support, grasp or load is inferred. The earlier
wrist-only exclusions remain unchanged in their original artifacts.

All source and generator pins, original windows and complete source slots have
separate saved audits; details below. General unfamiliar-video reconstruction
and independent physical accuracy remain unfinished. The published checkpoint
outcome belongs in `.local/equipment-hand-publication-01.json`; absence means
publication is unfinished. Preserve unrelated avatar changes and all failures.

## Initial implementation checkpoint — October 31, 2026, 11:48 PM PDT

One Weft pipeline: this extends `equipment-support`, not the simulator or task
scorer. Public `--hand-points` uses retained wrist/pinky/index/thumb estimates and
cached binary equipment masks. Every selected hand requires all four qualified
points; one overlap is image consistency, not contact, grasp, support or load.
No filled hand shape, threshold tuning, new video/PNG decode or model call.

The prior two-camera fit/viewer/trial/playable lab was pushed as
`1c18243d5b90132afbf6f95145193c6220f23b79`; exact private remote verification is
`.local/camera-multiview-publication-01.json`, SHA
`4e55477f9c3544da6371a0123f92658c7a7c2aa138fb37883434b7d01df88e47`.
Its scope/index were clean and stash empty; separate avatar edits remain owned
by that session. No earlier fit, trial, media or source processing is repeated.

## Implementation and ownership

- Root: new `src/weft/equipment_hand_snapshot.py`, existing
  `equipment_support_workflow.py`, `cli.py`, new
  `tests/test_equipment_hand_snapshot.py` and affected support CLI tests.
- Reconstruction: [producer and exact source-point contract](equipment-hand-evidence.md).
- Consumer: [restrict-only selector and unavailable-point review fix](equipment-hand-constraint.md).
- UI: [complete offline named-point report](equipment-support-hand-report.md).

The existing saved-wrist loader and default outputs stay unchanged. Opt-in
preparation restores its verified snapshot, requalifies exact retained
context/raw/audit/source-manifest custody, and preserves the original human,
equipment, mask, clock and source-equivalence identities. Finger ranges inherit
their corresponding wrist's **authored**, not calibrated, uncertainty and the
original confidence threshold. Original construction selection remains fixed.

The binary callback joins the original mask manifest, saved raster-read identity,
outer copied-PNG hash and binary hash/dimensions. Missing required binaries fail
explicitly; they cannot become zero masks or trigger decoding. Selected bytes
remain within 32 MiB; raster work remains at most 512 loads/4 MiB. Each stage
reserves its existing cooperative deadline; the public workflow remains 15–30
seconds, 64-MiB/384-file output, with completed seal written last. These are not
hard process/RSS limits. The source pilot adds separate process limits.

Original wrist files remain exact under `source/`; new annotations, precursor
metadata, hand rows and explicit settings are under `hand/`. Source identities
are rechecked before publication. No body, floor, equipment physics or protected
`shared_exercise_task.py` scorer changes.

## Verification before the source pilot

Root: 42 distinct focused checks pass. The single real-component authored
workflow composition was repeated after the consumer's narrowly reviewed
coordinate-access ordering fix; it passes. Receipts are
`.local/equipment-hand-integration-tests-01.xml`, `-02.xml`, and
`.local/equipment-hand-root-checks-01.json`. No broad unchanged suite rerun.
Producer: 27 distinct passes; consumer: 19 new +34 affected passes; report: 18
distinct new/affected passes. These counts overlap default-golden coverage and
are not an aggregate physical-accuracy metric. All four components received
read-only independent review; no outstanding implementation blocker.

## Frozen source pilot — not executed at this checkpoint

Plan `.local/equipment-hand-acceptance-01/plan.json`, SHA
`f7940afdb5aae1db089f575c0356526087320aed2df91ce731f94bb17836fae8`.
Helper `.local/run_equipment_hand_acceptance_01.py`, SHA
`02dc5ff0f37b11cd5136b5d08a620d199b52bed1bc52f8d6814fac17b8efc281`.
Runtime `.local/equipment-hand-runtime-01`, seal
`3da8c788ed205f53950b0a5ec945ca2f1c805e936dd9eb6440a71ea17c7a309a`:
283 files/6,023,979 bytes from committed `1c18243` plus ten exact reviewed hand
overrides. No active avatar worktree content is used.

Source `artifacts/pose-equipment-silhouette-02`, completed seal
`455665c0aaa6d71e582801033c502bbf35d4bcb3046cbb28041aa2d39ff444b4`.
Only seal metadata/stat preflight read so far: 188 selected possible input files,
20,998,959 bytes, including 146 previously decoded binary masks/253,054 bytes.
Keep all 259 original slots/129 pose rows and the original 77-pose construction
prefix; the known exposed development clip is not an unfamiliar holdout.

One invocation, new output `artifacts/equipment-hand-support-01`; 30-second wall,
25-second CPU, 20-second cooperative workflow, 1-MiB combined captured streams,
16-MiB single-file write limit and 512-MiB observed-RSS acceptance. Output remains
64 MiB/384 files. Source-equivalence media may be hash-verified/copied, never
decoded. Network/subprocess/model/decoder/provider guards are active. Cost:
$0 external/API and $0 recurring; one local bounded CPU run. The shared $5 Meta
ledger and keys are untouched. No deployment, browser, submission or server.

Next safe action: independent plan/helper review, then root owns one invocation.
Preserve any failure and completed products. Assess qualified overlap or explicit
abstention, not whether the two previously excluded candidates become accepted.
Afterward audit saved outputs without rerunning the producer and publish only
reviewed owned paths. General unfamiliar-video reconstruction and independent
physical contact remain unfinished; the durable goal stays active.

## Pilot 01 terminal failure — October 31, 2026, 11:51 PM PDT

The reviewed public command ran once and failed after 2.453572 seconds at the
consumer's additional-cache guard, before any output directory/report/manifest
was published. Root handle `16033` is closed, exit 1. Preserve
`.local/equipment-hand-acceptance-01/{plan.json,preparation.json,stdout.json,stderr.txt,process.json}`
and frozen runtime 01. Observed child peak RSS was 192,823,296 bytes; no timeout,
capture overflow, model/decoder/native/API activity. The error is
`complete point/prefix cache exceeds bound; no slots dropped`.

The producer had completed but its in-memory output was not yet archived by the
workflow. This is a capacity failure, not evidence that fingers overlap or do
not overlap. No scientific result/count from this attempt is published.

Diagnosis from code: the hand consumer inherited the **wrist-only 2-MiB**
additional cache cap. It contains four named points per hand, not duplicate
baselines. The authored 720-point check did not cover this source's 1,232 selected
point slots. Consumer owns a narrow explicit hand-only 8-MiB allowance and real
capacity/overflow regression; wrist stays 2 MiB and final output stays 8 MiB.
UI owns matching explicit old/new hand-bound validation; no fields, source
selection, thresholds, candidate rules or scientific input changes.

A second source invocation requires those reviewed checks and a new frozen
runtime/output. Its helper saves the eight successful hand-preparation artifacts
and a separate stage seal **before** selection/report. A later failure can then
reuse the saved work. That stage seal cannot stand in for a completed workflow.
The combined final output/checkpoint remains bounded to 64 MiB/384 files.
No second run has started at this checkpoint.

## Pilot 02 passed — October 31, 2026, 11:56 PM PDT

The capacity regression reproduced the old limit with 1,232 authored points,
then passed with all records unchanged: 2,125,538 cache bytes/2,852,522 final bytes,
0.285606 seconds. Only the hand-local cache constant changed. The reporter keeps
historical explicit 2-MiB hand results inspectable and accepts the new 8-MiB
declaration, enforcing each declared bound; wrist stays 2 MiB. Consumer has 54
distinct passing checks, reporter 20. Root's one affected actual-component
composition passes again; `.local/equipment-hand-integration-capacity-check-01.json`,
SHA `c5a143e48d8cdf59da63a79703ae9065c709b0c91c8f4178ccc58d20cc84be89`.

Runtime 02 `.local/equipment-hand-runtime-02`, seal
`f4487bf3538aa4602b3884296238fcbec5071bd38453ae419bec09945dc11e32`,
retains 281 of 283 files exactly; only the two reviewed cache-contract files
change. Plan `.local/equipment-hand-acceptance-02/plan.json`, SHA
`e9c3e114e0de1f45b4d6a73fc67eb08b2c086b3483308ef3b9f71afd37ae0cd3`;
helper `.local/run_equipment_hand_acceptance_02.py`, SHA
`2c3cb5dc2f2fb5ac31cf08a90b3f1f8d830875470c0d872c434d43168c8b15d2`.
Independent review cleared the exact changed runtime/plan/helper before one
invocation. Original source pins/settings/thresholds and candidate rules stayed
unchanged. Root handle `72780` is terminal, exit 0.

Completed output `artifacts/equipment-hand-support-02`, seal
`a607f6df387c59ae4a833718d783d1a5fc9493d8f2b71847a82cdb2cd17e6c38`:
254 files/35,247,312 bytes. Separate preparation checkpoint
`.local/equipment-hand-acceptance-02/hand-checkpoint`, seal
`e50ce97aeab18982f8d40b9ec13b20cb2df7937c0a5214b43eca5b8bb9762d83`:
nine files/8,080,184 bytes. Combined output remains below the original
64-MiB/384-file process acceptance. Both are terminal; do not repeat preparation.

Process `.local/equipment-hand-acceptance-02/process.json`, SHA
`500629c0bc4140f5f288620884854d892000420862d53313d7c570492c51d91a`:
2.480205 seconds, 202,457,088-byte observed child peak RSS, no timeout/capture
failure; no decoder, model, physics or paid API call. Successful preparation
read 146 cached binaries/253,054 bytes, made 814 point comparisons, and retained
all 259 slots/129 poses/1,036 hand-pair slots/4,144 point slots. The producer has
89 conditional group intersections, 109 exclusions and 838 unavailable groups;
these are **not** original candidate-window decisions or physical contacts.

[Open the completed hand report](../../artifacts/equipment-hand-support-02/index.html).
Independent saved-output and UI audits are the next boundary; they may read
bounded saved artifacts, not rerun producer/selector/report/native work. Root
owns publication after those results. Session/thread
`01a18139-c198-70d3-8cfe-e8e79b56fa2a`; main remains the prior published `1c18243`
at this checkpoint. Preserve separate active avatar edits and empty stash.

## Independent saved audits and publication

Saved-output audit `.local/equipment-hand-support-saved-audit-02.json`, SHA
`52d4f8a6e39b126dfa262927ce38d55aaec880967b2c6a3556b58a180a046e00`,
passed: one 254-file output pass; 258 total metadata/output reads/35,335,088
bytes/0.211121 active seconds. All 188 source copies and 47 generator members
match their exact source/runtime pins, including retained source-proof members.
The recovered baseline exactly matches the prior wrist-only audit. Every original
window, frame, pose and pair survives. Both candidates at frames 20 and 22 have
7/7 qualified hand intersections; no candidate was created. Preserve the first
auditor-only failure: it counted unavailable producer points by status instead
of their recorded reason labels. Its correction used cached data, not an output
or source reread, and changed no product artifact.

Saved UI audit `.local/equipment-hand-support-saved-ui-audit-02.json`, SHA
`f16b467619507dd583c26986e5a34b2522dc722a41155c57336aa9734de5a0e7`,
passed: ten reads/5,413,436 bytes/0.239465 seconds, including the existing Node
harness. All 259 frames/1,036 frame-pair views, 308 construction hands/1,232 points
and 2,108 window links match raw data. Four thousand unchanged selections cause
zero DOM writes; 1,000 changed selections take 9.829 ms and create no nodes.
This is a scripted DOM check, not browser visual QA or a sustained-use guarantee.

All preparation/source/audit handles are closed. No additional source, selector,
report, physics or paid work is needed for this milestone. Publish only the 22
reviewed product/test/documentation paths in
`.local/equipment-hand-publication-scope-01.json`; the final plan repins documents
after these audit notes. `.local/equipment-hand-publication-01.json` records the
actual commit, private remote verification, clean owned scope/index and stash
state; absence means publication is unfinished. Do not discard active avatar
work to make the whole worktree appear clean.

Next product gap: make the same retained-source evidence reachable from an
ordinary new-video workflow without a bespoke study script. Inspect existing
producer entry points first, keep one pipeline, and require explicit assumptions
and bounded work. The current saved clip does not establish unfamiliar-video
or physical-contact reliability, so the complete goal remains active.
