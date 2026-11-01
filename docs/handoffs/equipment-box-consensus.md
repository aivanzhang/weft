# Construction-only equipment frame selection

## Status and ownership

Implemented; **169 distinct focused checks pass**, with zero native calls in
those checks. Root's real saved-source guided setup selected sam-2 frame28 and
sam-3 frame18 under the fixed policy. The first integrated build rejected the
duplicated receipt metadata at the unchanged scene-size limit. Attempt02 admitted
the proposal and candidate, then stopped at the parent lab file reservation.
The restrictive allowance fix passed 31 additional focused checks and review;
attempt03 now completes the public saved-input build and ordinary ready-lab
admission. Root's independent saved audit passed. No media, model,
source acquisition, API, paid work or Git operation ran in this lane. All checks are terminal. Root owns execution,
shared setup/report integration, the guide, TODO/combined notes and publication.

Reconstruction owns `equipment_box_consensus.py`, narrow changes to
`equipment_proposal_config.py`, `equipment_proposal.py`,
`shared_world_reconstruction.py`, `shared_world_pipeline.py` and
`exercise_scene_inputs.py`, plus the three new focused test files and this note.
The integration follow-up also owns `exercise_scene_pipeline.py`,
`shared_exercise_lab.py` and the focused lab-output-allowance tests.
Scratch, pre-edit source copies, receipts and release pins are under
`.local/equipment-box-consensus-01/`. Controller, task, scorer, body, floor and
source clocks are unchanged; independent avatar work remains untouched.

## Concrete gap and evidence boundary

The ordinary producer previously required one manually chosen frame per object.
It inverts that single inferred image box under explicit camera/floor/depth
priors. The conditional silhouette producer also requires that same pinned frame;
it does not infer camera, hidden depth or physical empty space.

The existing real guided setup has 118 eligible original frames per object,
121 body references and 259 original slots. Its inherited sam-2/sam-3 frame8
choice is manual. This came from one bounded availability-metadata read of
`artifacts/shared-exercise-config-real-01/configured/snapshot.json` (762,928 B),
not a new source-coordinate study. The new policy was fixed before reading or
running selection on those boxes. No frame or threshold was chosen from task
failures.

This removes the frame-choice step when the image observations agree. It does
**not** recover ground, camera distance, scale, unseen depth, object identity,
contact or physical stationarity. Those assumptions remain explicit. Existing
stationary-foot/hand candidates, pose-to-surface annotation copying, finite-support
interval clipping and nominal measured-target calibration were not duplicated.
Failed model-world ground tilt, body-pixel fitting and mask-only physical claims
remain rejected. Blocked source/agreement searches were not repeated.

## Public configuration and shared pure API

Manual equipment rows remain exactly `{object_id, source_frame}` plus
`depth_prior` only in the existing explicit-depth mode. The opt-in form replaces
the frame, never supplies both:

```json
{"object_id": "selected-model-object", "frame_selection": {"method": "construction_box_consensus_v1"}}
```

`CONSENSUS_METHOD` and `validate_frame_selection` are exported from
`equipment_proposal_config.py`. There are no caller-adjustable numerical policy
settings or automatic fallback. Requested configuration bytes remain unchanged.

```python
from weft.equipment_box_consensus import select_equipment_box_frames

receipts = select_equipment_box_frames(
    construction_evidence,
    source_binding=binding,
    body_references=references,
    object_ids=automatic_world_ids,
    deadline=caller_deadline,
)
```

Inputs are the exact bytes/binding returned by the existing
`build_shared_world_binding`; original observation custody and source-equivalence
proofs remain that caller's responsibility. Body rows are exactly
`{index, source_frame, evidence_id, time_s}`. The estimator checks complete
construction grids and exact body/source/evidence/clock joins without reopening
media or proofs. It processes all requested IDs in one bounded batch.

The fixed image policy uses edges normalized independently by image width/height:

- At least three eligible observations spanning at least 0.2 source seconds.
- Every edge's complete range is at most 0.02 image dimensions.
- Each edge's fitted end-to-end linear drift is at most 0.01 in absolute value.
- No sorted edge gap over 0.01 may divide two groups containing at least two
  observations each.
- The selected frame is the exact-rational L1 medoid: the observed box with
  least total normalized edge distance to all eligible boxes. Exact ties choose
  the earliest original frame. Sorted prefix sums avoid quadratic pair storage.

Every eligible outlier and partial observation participates in every applicable
gate and cost. No trimming, temporal filling, geometric-domain retry or task-based
selection occurs. Excluded/non-body/missing slots remain unavailable, not zero
coordinates. Apparent occlusion can cause abstention. Movement below these
authored image tolerances and correlated model errors can remain undetected.
Observed dispersion is not a calibrated error interval.

## Receipt, construction and raw-video integration

Each `equipment-box-consensus-selection-v1` receipt pins the exact filtered
evidence bytes, canonical source binding, ordered original body references and
observation document/entity. It preserves both source and body times; a derivative
decoder clock is not replaced with the body's nominal time. Original metadata
tail slots remain explicit. Decoder-timed v3 sources retain the existing strict
metadata-count equals decoded-count invariant; they cannot invent extra original
slots from a conflicting derivative count.

`SLOT_FIELDS` defines compact complete rows: source frame, observation ID, source
time, partition/admission/visibility, body-reference index/time, evidence IDs,
unavailable reason, eligible pixel bounds, signed normalized edge deviations and
L1 cost. Full statistics, fixed policy, unavailable reasons and a diagnostic
medoid remain even when `selected` is null. `validate_selection_receipt` checks
saved structure only; it does not rerun inference or authenticate source media.

The producer resolves an actual observed frame and retains the full receipt as
`equipment[].frame_selection_receipt`. The constructor independently recomputes
the complete receipt against its actual sources and compares canonical bytes,
including numeric types, before ordinary geometry admission. The same receipt
is retained in the proposal and source candidate's `frame-selection.json`.
The scene's `frame_selection_reference` carries its exact canonical receipt hash,
selected observation, source/body/binding identities, policy hash/tolerances,
coverage counts and unchanged unknown physical status, bounded to 4 KiB per object.
`receipt_location: source_candidate_package` means `candidate/frame-selection.json`;
after lab packaging it means `lab/run/source-run/frame-selection.json`, not a file
beside every copied `scene.json`. The packager checks each reference against the
complete independently checked proposal receipt before publication. Rejected selections retain their receipts without a
ready proposal. No optional selection file or helper dependency is added to an
all-manual output. The new numerical inference is explicitly distinguished from
zero model-inference calls.

Fresh-video input preflight cannot choose a frame before the body exists. Its
automatic branch validates named construction-visible inferred-box candidates
against the existing authored-ROI prohibition, records
`deferred_until_body_references` and selects nothing. The ordinary producer later
uses the actual body references. Manual preflight checks remain unchanged.
The helper is archived only where the opt-in execution depends on it.

## Bounds and verification

One estimator batch: up to 12 selected objects, 1,800 original slots, 900 body
references, 32 MiB aggregate inputs, 4 MiB complete cached receipts and five
cooperative seconds, shortened by the caller's deadline. No hard process-memory
claim is made. The **existing 500,000-byte proposal limit remains unchanged**;
large complete receipts reject rather than being trimmed or moved outside
admission. Raw-input and package limits are unchanged.

Latest distinct outcomes: 48 new core/preflight cases, 102 affected legacy cases
and one scratch-only frozen-code byte comparison. The latter compares an entire
manual proposal and constructed scene to exact pre-edit code, with only native
admission mocked. Manual configuration, proposal and scene payloads remain exact;
implementation archives correctly record changed source bytes.

`checks-01` passed 135 cases in 1.028107 seconds. `checks-02` passed 19 selected cases
in 1.541306 seconds, including four affected cases already covered in 01.
`checks-03` passed two clock cases in 0.260280 seconds. Its valid case replaces the
earlier unparameterized fixture; the invalid-count case is new. The total is
151 distinct logical cases at that checkpoint, not the raw union of historical test IDs.
All use the existing pure guard with 10 CPU seconds,
4 MiB per output file and denied network/child processes/native entrypoints.
The existing MuJoCo import-time process probe was denied, not executed. A pytest
JUnit property-format warning is retained in 02; no case failed.

One authored metadata workload retained two objects × 1,799 original slots and 899
body references in 0.181046 seconds, with 1,176,843 cached receipt bytes. It is a
metadata responsiveness observation, not body/native admission, unfamiliar-video
accuracy or physical uncertainty coverage. It does not override the smaller
ordinary proposal limit.

Receipts and exact selected commands are recorded in `release-02.json`;
`release-01.json` remains the superseded pre-review checkpoint. Tests
cover an independent quadratic medoid oracle, exact ties, noise, retained outliers,
drift/scaling, competing cohorts, missing/withheld/partial evidence, separate
source clocks, caller deadlines, byte bounds, receipt tampering, unchanged default
paths, source package copying and dependency closure. Native/model boundaries
are mocked, not bypassed in production.

Independent read-only review closed the core and the narrow v3 count correction.
Removing only that final count condition recovers the exact previously reviewed
helper. The original separate-clock fixture had inherited a nominal-source
metadata tail when converted to decoder timing; it now retains the original
tail separately and gives the derivative its valid exact count. No validator was
relaxed, policy tuned, passing suite repeated or source study run.

## Real saved setup and integrated acceptance

Root's actual public guided resume produced
`artifacts/equipment-frame-config-real-01`, seal
`12e60e6d10f44cac4e7b7a350288d56013b0599d93931b173d08ac3173cdde3e`.
It completed in 1.126666 seconds with 143,327,232 B peak RSS: one selector batch,
zero model/native/media/API calls. Both objects retained all 259 original slots,
118 eligible observations and the same 121 body references. Root verified the
190 sealed members and all 176 source copies once; audit
`.local/equipment-frame-config-real-01/saved-audit-01.json` is
`a5f658b1a05770361bb17d501d37cc09415201b72485dcc51c64d844f22e6a14`.
This is source-bound image consistency, not physical stationarity or lab admission.

The separate `.local/equipment-box-consensus-source-01/plan.json` remains
**unexecuted and superseded** by that actual public setup. Its small saved identity
comparison found equal body-reference and equipment-document identities, but
different complete construction/binding hashes from the older producer domain.
The comparison receipt is `93923af7305f215657f3ba4c12931541b53ec8fbacdf2ff04208bed76445cf6d`;
no redundant selector batch or source-media reread followed.

The first acceptance is under `.local/equipment-frame-consensus-native-01/`.
It invokes the actual saved-human `shared-exercise-build` once, using the ready
setup's exact human/equipment/config/mapping inputs, body scale1, explicit root
assistance and full reference. There is no separate setup, inspector, playback,
learning or rollout. The runtime reuses the qualified geometry-resume runtime
with only six reviewed core and two setup/report overlays: 297 files, 6,478,535 B,
seal `11648e66000b1fcc10b3c6b6bf4ab6499c0555fd39c8247a072a5e831823cc0e`.

Bounds: 40 CPU/cooperative seconds, 60 external seconds, 512 MiB observed RSS;
the existing product ceiling is 384 MiB/1,600 files, with a separate
4 MiB/16-file worker reserve and two 1 MiB logs. Expected native totals are
43 models, nine data allocations, 550 forwards and 505 resets; small ceilings
48/10/560/512 cover validation-only variation. Exactly two mandatory coordinate
migration checks may use 32 controls/512 substeps. No other dynamics or copies
are allowed. A ready result must retain exact producer→constructor→scene receipts,
body/source clocks, task gates and ordinary lab proof/admission. New geometry
need not equal the old box geometry; rejection is preserved without fallback.

Root executed frozen plan
`3da50d60885268d5d64ae01f6133b3aa10955476f097db6121b49613e293d927`
once. The preserved result is `rejected_initial_geometry`: the scene metadata
exceeded its existing 500,000-byte bound. It stopped before candidate/lab creation;
this is a representation failure, not evidence of physical infeasibility.
Output seal: `376c5f2732ad3692711c9cbac2b3a03e15e908b86527c416b8eece51fec7362a`.
Root's single saved audit is `.local/equipment-frame-consensus-native-01/saved-audit-01.json`,
`f0e69ed3da58753a785f7c42a514a45e97b1645dccadeafac75907df0f964760`;
it verifies all 66 sealed members, three inner seals and source/config/body/receipt joins.
Actual resources: 2.291985 seconds, 224,755,712 B peak RSS, 67 files/27,836,850 B;
13 models, two data allocations, 20 forwards, two resets, 256 substeps,
16 paired migration controls and two selector calls. The receipts matched the
guided preview except the explicitly separate construction/binding domains.

Saved-only sizing found a 475,793-byte baseline scene plus 161,670 compact bytes
of receipts. That baseline-plus-receipts size control reached 637,498 bytes; the
unpersisted failing scene's exact size was not claimed. The bounded analytical
representation regression then materialized the saved proposal with cached
receipts and mocked selector/native boundaries: 634,616 pre-initial-geometry
metadata bytes become 476,719, saving 157,897 bytes. Only receipt representation
changes; geometry, body and task values remain identical. This does not establish
native admission or initial contact clearance.

`checks-04` passed 30 focused cases in 1.215873 seconds, zero native attempts:
17 new compact-reference cases, one saved-size/equality case and 12 affected
existing cases, including frozen manual byte parity. Total current distinct
coverage is 169. Mutation tests cover receipt hashes, choices, policy/source
identities, numeric types and locations; the actual lab-copy packaging path is
tested with native boundaries mocked. Full unavailable and unselected slot bytes
remain hash-covered. The complete receipt/proposal and both existing 500,000-byte
scene/proposal limits are unchanged. All original release and attempt01 artifacts
remain intact.

Independent read-only review closed the three-source compact-reference delta
without findings; it read the existing focused receipt without repeating tests.
## Attempt02 candidate admission and parent allowance fix

Root executed attempt02 plan
`621a39317f6870ee6e261bddac8a7a3149482680e85264d658617d02e9fd6c03`
once: proposal and full-reference candidate are ready, but assembly failed before
lab preparation. Candidate seal:
`2bbe0335fe99d335b622ed996f53bbbdc0b99c095d7cf999fbe55fb2f073bb90`.
The actual compact scene is 479,848 bytes, including normal native-admission
metadata; this is distinct from the earlier analytical 476,719-byte measurement.
All proposal and complete receipt bytes match attempt01. Reference preflight
passed 121 source-pose plus 255 control-grid checks with zero dynamic steps; it
does not establish task success or physical accuracy.

Actual attempt02: 4.763526 seconds, 284,835,840 B peak RSS, 138 files/52,971,869 B;
35 models, seven data allocations, 427 forwards, 383 resets, 512 migration
substeps/32 controls and three selector calls. There was no lab-native work.
Root ran the saved audit once:
`.local/equipment-frame-consensus-native-02/saved-audit-01.json`,
`cd988daae291d0391d178479edca9e516803f0e0f9b9d12c6b5ccb14f15f24b1`.
It verified the failed build's 132 members, the candidate's 70 members, exact
receipt/proposal parity, compact-reference links and protected scene fields.

The failure is file-count reservation, not bytes or geometry: 131 existing files
+ 1,536 maximum lab files + eight reserved metadata files = 1,675, above the
unchanged 1,600-file parent limit. The ordinary final lab now receives the smaller
remaining allowance, 1,461 files here. `prepare_shared_exercise_lab` accepts an
optional restrictive `output_limits=(files, bytes)` and checks its complete
materialized output before creating the output directory. Each value must be a
positive integer no larger than the existing standalone maximum. Standalone
defaults, nonfinal stage reservations and saved-geometry reuse remain unchanged.
No evidence is removed or limit increased.

The 31-case guarded allowance check passed in 0.822318 seconds with zero native
attempts: 22 new boundary/handoff cases, one frozen standalone byte comparison
and eight affected existing cases. Exact complete-closure limits pass; one fewer
file or byte rejects before output creation. Parent evidence remains intact.
Root and the independent reviewer closed the two-source delta without findings.
Details are under `.local/exercise-lab-output-allowance-01/`.

## Completed integrated acceptance

Root executed the corrected full public build once, not a standalone lab plus
another build. No acceptance-only recovery path was introduced. Frozen plan:
`.local/equipment-frame-consensus-native-03/plan-frozen-01.json`,
`99fe6bc120578cd77327b8d0708854d5f21402f7648ebab9ca94590aac74dd7b`.
The executed supervisor command was:

```sh
.venv-lab/bin/python -B -P .local/equipment-frame-consensus-native-03/supervisor.py 99fe6bc120578cd77327b8d0708854d5f21402f7648ebab9ca94590aac74dd7b
```

That immutable output cannot be rerun in place. It consumes the same guided
setup, source/equipment/mapping bytes, scale1, explicit root assistance and
full-reference choice. Only the two reviewed output-allowance modules differ
from attempt02's runtime. No controller, avatar, task, geometry or clock change
was included.

Result: `artifacts/equipment-frame-consensus-native-03/build/lab` is ready, with
no learned policy. Exact seals:

- Terminal acceptance: `9e5268d00e4549e69d18a688ab8da0a8f4c852380eb93253e2bdc1d850fc61ad`.
- Public build: `3623cb0004e4c1dd19a8a7fdaa2f58e6cac23e563cfa78480c296f98d267f97c`.
- Immutable lab run: `1d4a32d609cfbb9647827116ddcb2a44f13067bbb307387f7aef18783481c900`.
- Root saved audit: `.local/equipment-frame-consensus-native-03/saved-audit-01.json`,
  `de9b8a172f8734e1590dcaccf616cc47f3eb468fa8442b2ecf0314dd76999c2a`.

Actual resources: 5.793494 seconds, 381,583,360 B child high-water RSS,
543 files/107,886,770 B. The lab uses 404 files/54,766,231 B within its restrictive
1,461-file/192-MiB allowance; the whole build uses 537 files/107,766,395 B within
1,600 files/384 MiB. Native work exactly matches the static expectation:
43 models, nine data allocations, 550 forwards, 505 resets, no copies, and
512 substeps/32 controls solely in the two mandatory coordinate migration checks.
The lab phase itself has zero dynamic steps. Three numerical selector calls and
six normal source-proof calls were retained; no model inference, media decoding,
training, API call, playback, recording or new task rollout ran.

Root verified all 542 terminal sealed members, the build's 533 sealed members
and lab's 400 sealed members once. Complete proposal/receipt bytes match attempt01;
ten protected output hashes match attempt02, including the exact candidate scene,
task, body, construction evidence and source binding. The receipt map survives
unchanged at `build/candidate/frame-selection.json` and
`build/lab/run/source-run/frame-selection.json`; compact scene references join
those complete receipts. All ten runtime overlays, the original gate hash,
121 source-time kinematic references and raw parent→lab allowance were checked.

This removes manual frame choice on this consistent saved construction source
and qualifies its ordinary world→lab integration. Camera, scale, floor, hidden
depth, solid shape and materials remain priors. It does not establish physical
stationarity, full-exercise success or unfamiliar-video accuracy. The complete
goal remains open. Both earlier failures, all receipts and runtimes remain
immutable; there are no live processes owned by this lane. Root owns the separate
zero-native offline-report qualification and publication.
