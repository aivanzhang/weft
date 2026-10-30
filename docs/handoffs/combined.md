# Combined Weft implementation handoff

## Latest continuation — October 29, 2026, 8:57 PM PDT

The previous goal-file turn was **progress**: the full combined objective was saved/opened. This continuation is **progress**: exact media equivalence and safe evidence custody now support a source-linked body/equipment sandbox. The full generalized video-to-world-to-RL goal remains active and incomplete. The temporary replacement prompt is a dated snapshot; use this handoff and `TODO.md` for current state.

### Implemented and independently checked

- `source_equivalence.py` checks all sealed files, recorded pixel/frame mappings, dimensions, native/decoder origins, durations and bounded producer records. It preserves original and derivative identities; it does not independently redecode media or authenticate a producer. It currently supports the complete YUV420P/lossless-libx264 preparation contract, not arbitrary transformations.
- `world_evidence.py` now rejects conflicting canonical-frame partitions across entities/documents; verified derivative aliases add the same protection. Empty `not_sampled`/`unsampled` slots remain withheld but are not treated as protected holdouts. Audit/unassigned reservations cannot be bypassed through another document. No old evidence is relabeled.
- `shared_world_evidence.py` verifies consistent original signatures across all mappings, strict mapping paths and combined 64 MiB/256-file limits with a 30-second preflight deadline. Immutable exact byte snapshots support packaging. Directory-FD traversal closes the reproduced parent-symlink race. A quadratic malformed-FFmpeg-log scan was replaced: a maximum 4,194,304-byte malformed log rejected in 32.074 ms locally; this is not a general performance guarantee.
- `shared_world_packaging.py`, lab and training changes require outer-manifest coverage for every observation, bundle manifest and sealed child; preserve complete independent policy evidence; copy with exclusive/no-follow handles; and verify destination sources before sealing. Nested manifests are no longer silently omitted. Snapshot source-equivalence and packaging implementations too.
- `shared_world_reconstruction.py` validates exact body bytes and every required construction reference, source box bounds, camera mapping, assumptions and initial separations. It creates dynamic box proxies with zero reference object forces and no learned-policy claim. A separate explicit XY-frame migration checks eight matched nominal steps; construction never reframes automatically.
- `shared_world_pipeline.py` and CLI `shared-world-reconstruct` connect sealed intermediate inputs to a normal reusable candidate/lab. Canonical display metadata comes from verified source identity/count, not arbitrary copied `source.json`. Geometry proposals remain bounded to 500 kB, full body/scene JSON input to 8 MiB, and runtime scene metadata to **500 kB compact UTF-8 JSON**. Formatting whitespace is not counted as scene data; oversized ASCII/Unicode provenance still rejects.

### Real-source result and limitations

- Reused the existing kettlebell body, SAM masks and mapped video. **No new paid call, inference or video decode.** Original/derivative SHA identities and all 257 source frames remain unchanged. Two observation documents plus the complete mapping total **16,463,975 pinned bytes**. Body slots retain 129 construction / 128 unsampled entries; equipment retains 771 construction slots with original raw pre-init/identity evidence. Binding missing counts do not independently classify pre-init phases.
- `artifacts/equipment-camera-bridge-01` rejects the original/rotation-only floor-box hypotheses. Existing projection RMSE 16.662 px; floor residual 12.36–14.36 px; initial working-box overlap 0.092002 model units. Do not reinterpret rejected-before-collision candidates as collision-free successes.
- `artifacts/equipment-camera-perspective-01` tests four authored fixed-anchor distances. Selected distance 1.5875915296 model units / focal 271.127582 px improves construction RMSE to **11.0857 px**, preserves all body/root/scale/floor/reference values, and keeps later box-edge drift ≤2 px. Median solid-box silhouette IoU remains **0.4707/0.5728**; there are no recovered handles or cavities. Working-bell wrist gap is **0.18178 model units**, not verified support/grasp. Proposal SHA: `af321ed546567eb7fdb7dd6c161987e7339c8697a6ba505d4214f1d332f48fa7`.
- The old shared-world framing excludes the idle box's left edge by 0.0749985 units. That failure is preserved. An explicit +0.2 X translation moves the whole frame, old objects and target together and extends the numerical upper X bound; no source-relative body/equipment/floor relocation. It changes the lower-side reset/escape envelope, so full task equivalence is **not** claimed. Eight nominal matched-control steps compare ten contacts: maximum state error **1.68e−13**, contact-geometry error **2.43e−14**, contact-force difference zero.
- New candidate: `artifacts/source-equipment-integration-04/candidate`; seal `9de5e09fdedc8105fa2eedac380e6e0dfc6b61c20dba6d2fb4f6ceb367049703`; scene hash `ac33757a648ea45769a04541d83248f15fec56209b75cc0424e48b69c7e2d638`. Initial/source-pose/control-grid checks pass their declared tolerances; maximum recorded penetration remains 0.000558844 model units. This is not physical calibration or independent exercise accuracy.
- New lab: `artifacts/source-equipment-integration-04/lab`. Human, `sam-2` and `sam-3` each have a verified recording: **12 steps, one reset, 14 events/video frames**, exact replay. Study seal **`3fa9c1e83b339c463d7c4650808a24ec8ea6da870ab30a2748632040509c942d`**, 180 verified files. Mutable lab lifecycle/new sessions are outside the study seal. See `results.json` for exact recording paths/hashes.
- **Both 60-step baselines fail.** Coast fails angular/root tracking. Authored reference passes the current body tracking gate but displaces the working box 0.052836 units and leaves speed 0.790428 units/s, failing object stability. Its 44 simulated human/target contact records do not establish grasp or source hand support. No training was run to hide this mismatch.
- Construction **1.016 s**; lab packaging **0.720 s**; 30 local three-step control+JSON batches p95 **3.923 ms**. These are stage/control timings, not browser/device QA or sustained resource measurements. Source projection and interactive preview PNGs were visually inspected with local image tools, not a browser.
- Preserved unsuccessful integration attempts: `source-equipment-integration-01` (pipeline body input incorrectly limited like a small proposal), `-02` (constructor had the same separate bound), `-03` (default JSON formatting crossed the 500 kB metadata check). The full 659,006-byte body record is retained; scene compact data is 475,792 bytes. No physical/scoring tolerance was loosened and no source frames were dropped. The two `*-size-diagnostic-unwritten` paths were never created.

### Verification and resume

Latest selected receipts cover **285 distinct passing cases**, not the whole repository or independent physical validation:

- `.local/source-equivalence-confinement-tests-01.xml`: 170 loader/equivalence/input-bound cases, including deadline, alias, partition, ordering and path-race checks.
- `.local/shared-world-construction-final-01.xml`: 53 input-bound/construction/migration cases; includes the two compact scene-size regressions.
- `.local/source-equivalence-integration-base-01.xml`: 122 scoped base checks; newer loader receipts supersede overlapping cases.
- `.local/shared-world-packaging-01.xml`, `-reader-02.xml`, `-legacy-01.xml`, `-analytic-01.xml`: complete custody, bounded readers, legacy source/no-equivalence, and analytic training/lab/recording checks. Overlap is deduplicated in the receipt.
- `.local/shared-world-pipeline-02.xml`: six sealed-pipeline cases, including metadata identity, ambiguous/oversized proposals and a full body record larger than the geometry-proposal limit.
- `.local/shared-world-equipment-contract-01.xml`: seven physics/task/actuation contract cases.

The check/study handles and all delegated workers are terminal. No development server, deployment or external submission/message was started. Meta ledger rechecked: **9 settled, $0.235952 committed, $4.764048 remaining, 0 unsettled**. Remote fetch showed `main` 0/0 ahead/behind before publication preparation; stash list was empty. Preserve all old/new artifacts and pending scoped changes. Final publication/verification receipts are `.local/source-equipment-publication-01.json` and `.local/source-equipment-checkpoint-01.json` when written by the integrator.

**Next safe actions:** resolve body/equipment depth/hand-support hypotheses and the reference controller's object-stability failure; freeze a separately versioned exercise task with human-only actuation and sustained body-part/contact/stability criteria; preserve old task/scorer definitions; carry reusable construction into the ordinary video workflow and improve stage visibility; test unfamiliar clips and fresh independent physical/learning outcomes. Current masks, proxies, exact replay and tests do not complete these requirements.

## Previous published checkpoint — October 29, 2026, 7:40 PM PDT

- Recorded: **October 29, 2026, 7:40 PM PDT**, America/Los_Angeles.
- Goal: [complete A+B objective](complete-goal.md); user installed it in thread `01a18139-c198-70d3-8cfe-e8e79b56fa2a`. Both former tracks now belong to this session; agy is closed. Earlier ownership/pause notes are historical.
- Previous goal turn: **progress** (complete objective saved, opened, installed and read). This continuation is **progress**: inherited runtime integrated, source bindings implemented, mobile evidence corrected, one bounded tracking call completed, full masks preserved and four coupled recordings independently replayed. Overall generalization/physical-accuracy goal is **not complete**.
- Checkout: `/Users/ivan/kintsugi/weft`, `main`; baseline `66490b9`. Remote fetch confirmed 0/0 ahead/behind before these edits. Preserve local studies, hidden budget/key configuration and reviewed stash backups. Publication status is recorded below when completed.

## Implemented and checked

1. **Mixed runtime:** inherited `shared_world.py`, `shared_world_lab.py`, `shared_world_learning.py`, replay asset and lab/CLI/capability/recording integrations. One model/data/clock, manual human/object controls, reference control, bounded learning, inspection and recording. Preserve pure-profile interfaces.
2. **Core review fixes:** separate kinematic inspection state; target-specific contact gating; rejected initial human/static intersections; strict evidence-document bindings; unsupported calibration rejected; explicit derived force units; no per-action MuJoCo recompilation during policy action selection. Core policy-call p95 measured 2.169→0.432 ms over 100 calls; function-only timing, not UI latency.
3. **Construction evidence:** `shared_world_evidence.py` plus 11 focused tests. Exact SHA-qualified entity references, complete source slots, construction-only counts, explicit authored additions and verified evidence documents at packaging/loading/training. Unrelated sources, missing/altered evidence and metadata-only source claims reject. Raw source clocks are retained, not silently aligned.
4. **Mobile observations:** `observations.py` now preserves v1's eight hinges/fixed support, exports v2's 19 hinges/translation anchoring and v3's 19 hinges + three root slides + authored virtual actuation. No false anchored/contact claim for mobile input. Unknown profiles abstain; conflicting metadata rejects. Kettlebell's 257 slots, 129 pose records, 128 unsampled slots and raw clocks stay intact; exported axes match saved XML.
5. **Usability/provenance:** shared human/object controls, policy availability, recording locks, reference-vs-simulation separation and three-pane replay have deterministic checks. Shared/rigid readouts no longer call uncalibrated force/length values newtons/metres; unknown object geometry is not called measured.

### Verification receipts

Before the final targeted policy/failure-clock follow-ups, **143 distinct Python cases have latest passing results** across these receipts (initial fixture failures are retained, not erased):

- `.local/mobile-observations-tests-01.xml`, `mobile-observations-model-parity-02.xml`, `mobile-observations-model-parity-03.xml`: 66 distinct latest passes; fixture corrections did not change runtime gates.
- `.local/kettlebell-preparation-tests-02.xml`: 11 passes.
- `.local/shared-runtime-core-01.xml`, `shared-runtime-integration-01.xml`, `shared-runtime-source-training-01.xml`: 11 distinct passes.
- `.local/shared-world-evidence-tests-01.xml`: 11 passes.
- `.local/combined-runtime-regressions-01.xml`: 12 passes.
- `.local/combined-http-tests-01.xml`: 32 passes, using bounded localhost test servers, not a development server.

UI: 43 lab checks plus spatial and shared replay suites. `.local/combined-world-ui-03.log` retains one stale SI-label expectation after the unit correction; `.local/combined-world-ui-04.log` passes the corrected targeted case. Mocked DOM/worker tests do not establish browser/device QA.

Final follow-ups: `.local/shared-runtime-partial-clock-01.xml` and `shared-runtime-failure-state-02.xml` verify ten targeted cases. Failure-only `partial_action` metadata records real completed substeps (example .0979167 s instead of nominal .1 s); normal snapshots stay unchanged, and historical failure traces are not rewritten. `.local/spatial-policy-version-verification-01.json` records 23 policy-version/training/CLI/lab checks: legacy v1 action math restored exactly, new training explicitly v2, 617 old recorded decisions unchanged without rerunning old physics/training.

Final latest-result union: **176 distinct Python cases pass**, plus 43 lab UI checks and both 3D replay smoke suites. `.local/combined-checkpoint-verification-01.json` records the exact report set and code/doc paths; the latest UI label correction passes in `combined-world-ui-04.log`. Syntax, new-document links, credential-pattern scan and whitespace checks pass. No original threshold or final-evidence partition was relaxed.

## Source preparation, paid call and decoded evidence

- Failed `artifacts/kettlebell-tracking-input-01` remains unchanged: the source native-clock expectation was wrong. Successful `...-02` preserves 257 native YUV pixel hashes and native PTS at 640×480. Original native PTS 3–8536 ms and decoder-relative 0–8533 ms have a measured 3 ms origin offset; no media timestamps were shifted. Durations differ on 86 frames and remain separately recorded.
- `...-02` seal: `2c3cf30c05aeddd1d0751ee054993f1c9875bf85beb0f6e7c959f411a3a9e49c`; derivative SHA `34e74d9ab854a474ba647395f232e61dfde5cfa166bb449fd3057084c6dd5416`; 6,023,720 bytes. Original SHA remains `4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`.
- `scripts/studies/track_kettlebell_shared_source.py --out artifacts/kettlebell-shared-tracking-01 --execute` made exactly one SAM request, three manually inspected frame-8 boxes, no retry. Application allowance $0.16, conservative estimate $0.1542; settled charge **$0.0498**. Native source rights/attribution preserved. The separately rejected jumping-jacks upload was not retried.
- Ledger verified after the request: **9 settled, $0.235952 committed, $4.764048 remaining, 0 unsettled**, one shared $5 project budget. No other paid call in this continuation.
- `scripts/studies/export_kettlebell_shared_evidence.py` produced `artifacts/kettlebell-shared-evidence-01`: **771 slots, 24 pre-init slots, 747 masks**, no later missing proposals; all PNGs/pixel counts checked, centroid accounting difference ≤5.68e−14 px. Root seal `10b8067dbaada2046a5508da45c0a6d07c3a7dfdd8002b6f10e669ac59f5be24`, 1,211 files.
- Observation document `evidence/scene-observations.json`: `d2004cddde40fa1013aa564bca71449040a81190bbd6198866c0fa8482960e13`. Masks `masks/mask-evidence.json`: `54e1b630dbe0608c3ab55d7c911c952f678e080ac54feba34f4c718577c39818`. Submitted derivative identity and `sam-1/2/3` IDs remain unchanged; original/native linkage is retained in each raw record and `inputs/preparation`.
- Selected panels 0/7/8/128/192/256 inspected. Separate person/bell proposals are visible, but identities remain conservative/unverified; overlap group expands to all three at frame 139. Working-bell confidence is lowest (minimum .5048646, median .8598492). No physical geometry/contact promotion follows.

## Coupled integration evidence and retained driver failures

- `artifacts/combined-world-integration-01`: candidates and labs constructed; study driver incorrectly requested nine steps while the interface permits six. Preserve seal `246d1fbe235ac90e9506c326fdb788428c47f1d57d4d3a6d561f8c9e323235f7`. Only the lab asset changed afterward; no claim this snapshot is the latest interface.
- `...-02` reuses those verified candidates and records both actors in both cases. All four recordings were completed and verified before a driver-only PNG error (`numpy.ndarray` has no `.save`). Preserve the entire 205-file study, seal `b270ebc71d9a652c94cf1f7c849f927d3823e5fa60f821a5db98430aa1cd1ec6`. Source/core were unchanged during this run.
- `artifacts/combined-world-inspection-03` finishes only unsaved inspection: verifies the four existing recordings, exports four PNGs and measures new core timings. No source inference, construction or recording repeated. Six-file seal `00a95f50a31210c550c0afc214ec8a180ee12b43062eda792d1c3fdc33be9abd`.
- Each recording: 12 steps, one reset, 14 events/video frames, exact state replay. Fresh 30-sample three-step+snapshot p95: authored 2.643 ms, source body 1.654 ms; original unsaved timings unavailable. This is not UI or sustained-use performance.
- Authored fixture has no reference conflicts. Source-linked body has **two source-pose and three control-grid static conflicts** with authored props, maximum penetration .000558844 model units. Preserve those failures; do not promote this as recovered equipment or valid physical exercise.
- Source body lab: `artifacts/combined-world-integration-02/source_body_authored_props-lab`, sealed run `eaeb898550e5161b9fe30efa6327573b33b4e7f2efb97da1e82b3ef4ddcd92c0`. Body observation SHA `eb776ba3676674430ec61c9d5bdf8a4fbab499854b826fbfb666187f964693b5`; complete 257 slots, 129 construction / 128 unsampled. The ball and pedestal remain authored.

## Remaining work and next safe action

- [x] Finish targeted legacy-policy compatibility and partial-action failure-clock fixes, preserving normal records, old action interpretation and independent success gates.
- [x] Implement a validated **original/derivative equivalence bridge** and complete packaging; see the latest continuation above. Original hashes and clocks remain distinct.
- [x] Construct an initial uncertainty-labeled person/equipment proxy candidate in one frame, with preserved rejection and explicit migration. This does not establish correct shape/depth, identity, support or grasp.
- [ ] Resolve human reference/static/floor conflicts without hiding poses, moving the floor silently or weakening protected gates. Preserve source fits versus actual changed-action outcomes.
- [ ] Improve and independently evaluate coupled learning across changed starts/actions; retain failed baselines. Whole-body virtual assistance is not unassisted balance; illustrative muscles are not EMG.
- [ ] Carry reusable construction through the ordinary video-to-lab flow and test unfamiliar clips; keep all stage artifacts and clear unsupported/error states. Measure end-to-end responsiveness and sustained behavior. No browser interaction unless requested.
- [ ] Obtain fresh independent validation for broader physical claims. Existing Cam3/repeat studies are exposed development evidence; current unit tests, authored contacts and source tracking do not complete generalization.

## Ownership, processes and publication

The tracking request, all source/export/integration studies and final core/policy test handles are terminal. No development server, deployment or external message/submission was started. Do not rerun the paid request, duplicate finished studies, modify sealed runs or reset the shared ledger.

Only tested code/docs/scripts belong in the Git checkpoint. Preserve local evidence and hidden configuration outside Git, review the complete diff, then verify publication and stash state. The goal remains active.

**Publication:** implementation checkpoint `24e754c2df0d9bb7af15f6bbe1350ea2b00224ef` was pushed to private `origin/main`; an independent `git ls-remote` matched exactly. The worktree and stash list were empty after that push. This documentation receipt may advance the branch afterward; use actual Git state and `.local/combined-publication-01.json` for the final receipt. No local videos, generated runs, credentials, budget ledger or model caches were published or deleted.
