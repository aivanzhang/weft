# Avatar assets handoff

## Session and ownership

- Workspace: `/Users/ivan/kintsugi/weft/tools/avatar-assets`, migrated into Weft by explicit user request. Preserve unrelated Weft work and its active goal.
- Thread: `01a18ebe-a17c-7cc0-9117-06ca476fb31c`; title: Integrate avatar assets. The historical core-tool milestone is complete; the user's current avatar-quality deliverable is not.
- Recorded: 2026-11-01 00:01 PDT.
- Core workflow, gallery and detailed-model workflow upgrade are verified. Muse-quality parity has not been achieved; local tooling success does not complete the user's visual target. Weft’s goal is untouched.
- User clarification preserved: complete controllable 3D from text is primary. Meshy provides automatic text/reference base generation; the new bounded AI pipeline also authors rigs/motions and performs sampled review/repairs. Human quality acceptance remains separate. Keep the portrait/four-state workflow too.
- User authorized defining a neutral pose. It is an authored proposal, not an observed Weft pose or permission to change its physics.
- Current Meshy work: **15 completed submissions / 167 credits reported and reserved**; account-specific dollar price unknown. Earlier local-only milestones used0 paid calls. No recurring service, deployment, new remote repository, account or borrowed credentials/budget. The user previously authorized a push to Weft’s existing private remote. A proposed four-call raster batch was not executed.
- Resume guard: preserve original ZIP and existing output folders. Recheck job handles below before resuming or terminating anything. No persistent background service was created; gallery test servers were temporary.

## Active follow-up: higher-quality generated avatars

User asked to save the current pipeline, then improve Meshy-based results. The
baseline is committed and remotely verified at `050deb5`; original outputs stay
unchanged. [Current refinement plan, ownership and bounds](docs/meshy-quality-upgrade-v1.md).
Selected `outputs/agent-dragon-polish-v1`: smoother geometry with unchanged4K
textures/identity/four motions; independent reload passed. New bounded smoothing,
material profiles, hair sheen and diagnostic controls are reusable; an early
triangle regression gate feeds measurements back to the repair agent.368 tests
pass. Neither panda trial nor dragon material trialv2 is accepted; failures and
source snapshots are retained. This is not Muse-quality parity.

HD dragon media completed: `outputs/agent-dragon-polish-media-v1-portable-v1`,
1600px portrait plus four720px/24fps/4s clips,96 distinct decoded frames each,
exact static fallbacks, opaque/no seamless-loop certification. Full frames stay
local. The new before/after browser catalog is `browser/gallery-polish-v1.json`;
actual playback, seeking, mobile and reduced-motion/error fallbacks passed.
All owned handles (60224 media,41941 native gate,99013 browser) ended; no server
is left running. No new paid calls; ledger remains167 credits with8 available.

Additional35-credit approval for a new panda base is pending. **Do not submit**
the preserved `outputs/muse-panda-smooth-base-v1` preparation: image-mode text
only guides texture; a clean geometry guide and new frozen plan are required.
Next safe action: review the catalog below. After approval, prepare one reviewed
clean-reference candidate; no paid retry or budget purchase. Keep current packs
immutable. The original Muse/fur-quality target remains open.

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-polish-v1.json
```

This pass is checkpointed separately from unrelated active Weft work. Final Git
revision and remote verification belong in the root-local receipt
`.local/avatar-quality-publication-v1/receipt.json`; an absent successful receipt
is not proof of a push. Selected model/source, portable media and review records
are tracked; rejected full binaries and complete video frames remain local.

## Git release checkpoint — October 31, 2026, 10:44 PM PDT

- User requested Git cleanup and a push to the existing private remote. Scope is `tools/avatar-assets/`: reusable code/skill, HD and generated candidates, completed dragon/robot agent deliveries, portable media, tests and provenance. No generated output or original reference is deleted or rewritten.
- Separate camera-multiview work outside this directory remains owned by the active Weft session. Do not stage, stash, reset or hide those changes to make the whole working tree appear clean. The avatar scope and index must be clean after publication; unrelated active edits can remain.
- Baseline `main` and private `origin/main` both verified at `41cfee28ceef55146fbbf6e29ef8248c475bd46d`; stash list empty. No LFS/account/subscription or paid generation is introduced. Credentials, ledgers, environments and full frame archives stay ignored.
- Existing evidence remains immutable: 323 passing tests, two real browser playback proofs and the source/delivery hash audits. Publication-only documentation changes do not rerun generation or render jobs. Historical uncommitted/unpushed notes below and in sealed evidence describe their recording time, not this release checkpoint.
- Read-only publication review verified all192 inventoried artifacts in the final dragon/robot model and portable-media packs, plus required resources for the current catalogs. Historical fur-comparison, Meshy-candidates and nonportable quality catalogs retain local-only dependencies; fur-comparison v2 fails current rig-facts validation. README labels these limits. No real credentials were found in2,463 selected text files; token-shaped permission-test fixtures remain synthetic.
- Exact commit, remote verification, selected paths and final scope/index status belong in the local publication receipt `.local/avatar-assets-publication-20261031-v1/receipt.json` at the Weft root. A receipt without successful remote verification is not proof of a push. Resolve the revision with `git log -1 -- tools/avatar-assets`; preserve ignored full runs for complete cached resumes.

## Current follow-up: prompt-driven AI pipeline — verified

The requested reusable text-prompt workflow is implemented and verified on two actual generated sources. This completes the bounded pipeline milestone, **not universal prompt success or the original Muse/fur-quality target**. No Weft importer, physics, actions, rewards, active goal or Meta ledger changed.

- [x] `generate` coordinates a data-only planner → Meshy generation or explicit source reuse → measured geometry inspection → rig/motion author → independent native reload/exported appearance → separate visual critic → bounded repair → real media/static fallbacks/portable gallery. No subject-specific recipe was manually supplied to the two successful agent runs. Robot style/state IDs came from its prompt alone.
- [x] Delivered dragon `outputs/agent-dragon-v3`: accepted attempt2,9bones/36,322vertices,1.6m authored height;1600px portrait and four720px/24fps/4s clips,96 distinct decoded frames each. States idle/look_around/wing_flap/milestone_level_up. GLB SHA `71484ae9183e78415096fc252f5c58d00eeecf8a6bca4a8d472b9ba60789de00`. First planner→delivery757.037716s; media534.994s/1,177,485,312B sampled direct-child RSS. Full frames remain local; complete compact media is in `media-portable`.
- [x] Delivered robot `outputs/agent-robot-v3`: accepted attempt1,11bones/45,386vertices,1.2m authored height;512px portrait and four256px/24fps/2s draft clips,48 decoded frames each, retiming full4s actions. Wave has47 unique decoded thumbnails; the other clips have48. States idle/look/wave/celebrate. GLB SHA `e2e1c4e73a83773673a9343d3ae0baeddea84dbace9ac61ac7ee203a9ec89d3e`. Execution-stage span195.415325s, excluding prepared-request wait; media55.215s/640,729,088B. Draft media is not an HD render; both source meshes retain4K maps.
- [x] Native reloads passed; model packs retain exact prompts, AI configurations, source/backend snapshots, GLB, editable BLEND, local-rig contract and exported bone IDs. Dragon mesh9/armature10/root8; robot mesh11/armature12/root10. Units meters,+Yup/+Zforward. These are standalone local controls, not Weft body-frame mappings or the panda shoulder attachment.
- [x] Actual browser proofs `browser-agent-dragon-v1` and `browser-agent-robot-v1` passed in11.760s/10.610s: all8GLB clips/videos, play/pause/seek, static/reduced-motion fallbacks, injected loading failures, mobile fit,0unexpected errors/0external requests, verified owned browser/server exit. New `browser/gallery-agent-v1.json` combines exact delivered paths with friendly titles; validated2assets. Prior galleries remain unchanged.
- [x] Completed-job cache resumes made0generation calls: dragon0.744259s tool observation; robot0.592917s with all5agent receipt hashes unchanged. Proof `outputs/agent-resume-proof-v1`. The core backend/input pins prevent silently reopening a job under changed code/settings; never retry uncertain calls or overwrite unsealed native work.
- [x] Full local suite323 tests passed in12.545s (`outputs/prompt-workflow-tests-v3.txt`). These analytic fixtures test gates, not avatar quality. Coordinate guides, bounded actual-key sampling and useful failed-state measurements were also exercised on real geometry. Skill validation passed. The final read-only audit `outputs/prompt-pipeline-audit-v1` passed0.148550s/27,803,648B: exact delivery/media/browser bindings, real fallback/source continuity, all28agent calls terminal, unchanged original archive SHA and unchanged Meshy ledger.

### Decisions, failures, costs and bounds

- **No new Meshy/Meta calls.** These tests reuse the previously generated unrigged dragon/robot GLBs. The existing Meshy adapter supports the new paid geometry branch, but fresh synthesis was not purchased again for this wrapper test. Canonical ledger remains15settled submissions/167credits reserved and reported; base140 plus the prior dragon-only35grant gives175aggregate. Only8base credits remain; a new35-credit job needs real authorization, not a higher CLI cap. No new account, subscription, purchase, hosted service or publication.
- Actual AI usage across all6jobs: **28calls,666,866input tokens/31,201output tokens**; cached-input16,640 and reasoning-output1,638 are separate reported fields, not extra totals. USD cost is unknown. Existing authenticated CLI0.166.0 ran ephemeral, tool-disabled, schema-constrained roles; exact underlying default model version is not exposed/pinned. Every prompt/image/answer/settings/usage/thread handle remains in `agents/`. Do not call the verification free.
- Default5calls/1repair; successful jobs used finite7-call/2repair envelopes and240s per-agent deadline. Local stages keep4CPU threads,2GiB sampled direct-child RSS, bounded logs/output/time; at most2heavy native jobs ran together. Agents cap1GiB sampled RSS and selected media/input/output bytes. Total output cap2GiB per job; no new stage after an invocation's1h watch window. Resume does not reset calls or repairs. Browser verification has its separate3GiB sampled owned-tree bound, including shared-page double counting.
- Preserved failures: dragon-v1 successfully answered but the client-warning parser rejected the receipt; narrow startup warnings are now recorded, unknown errors/tools still rejected. Dragon-v2 missed wing coverage; robot-v1 missed brief gesture extrema, robot-v2 still stretched its wrist. None became an accepted delivery. Improved normalized front/side coordinates, informative key poses, repair images and precise skin measurements—not looser acceptance—enabled later runs.
- `outputs/robot-wrist-diagnostic-v1` traced adjacent Root/Hand vertices across a weight-box gap: a4.217mm edge stretched to163.417mm. Generic `deformation_diagnostics.py` now supplies actual edge locations/influences through sampled-state reports. The accepted robot still has22.6335mm sampled edge extension near an arm joint; agent acceptance means no **major visible** defect in supplied views, not zero deformation, physics or all-frame collision proof.
- Original MuseZIP remains unchanged: `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Hidden surfaces, units/rest poses, bones and weights are generated/authored assumptions. Rights metadata is not independent license verification. No certified seamless loops, human approval, Muse parity, facial/finger controls, prop interaction, flight/contact physics or automatic Weft conversion.
- **No active owned jobs or persistent server.** Tools75064(dragon),28264(robot),14136/91702(browser),53989(diagnostic native),29645(tests) ended successfully; earlier unsupported jobs also terminated. Logs/receipts carry native PIDs and agent thread IDs. Never restart old provider tasks or duplicate these completed runs.

### Handoff and next safe action

[Commands/contract](docs/prompt-pipeline.md), [actual results/canonical IDs](docs/prompt-pipeline-results-v1.md), [reusable skill](skills/avatar-asset-workflow/SKILL.md), [release index](evidence/prompt-pipeline-v1.json). Source checkpoint: `outputs/prompt-pipeline-source-v1`; original full runs and rejected assets remain local. `.gitignore` allowlists selected complete models/compact media, requests/agent evidence and failed-attempt metadata; it excludes full video frame sequences and rejected binaries from the Git subset, not from local retention. Cached resume requires the complete local job, not a compact checkout. See the Git release checkpoint above for publication scope; unrelated Weft edits are untouched. Hashes alone are not a remote backup.

```sh
# From Weft: view the two completed examples; user starts/stops this server.
./tools/avatar-assets/run gallery --catalog browser/gallery-agent-v1.json
# Safe completed-run verification; does not generate again.
./tools/avatar-assets/run generate --resume outputs/agent-dragon-v3
./tools/avatar-assets/run generate --resume outputs/agent-robot-v3
# New prompt: prepare only, without model/provider calls. Use a new output name.
./tools/avatar-assets/run generate --prompt 'A friendly violet dragon with small wings and a long tail' --preset preview --max-credits 35 --prepare-only --out outputs/violet-dragon-next
# Actual execution uses the saved cap AND the canonical authorized ledger.
./tools/avatar-assets/run generate --resume outputs/violet-dragon-next --execute
```

Next safe action is user review of the combined gallery. Fresh paid geometry requires a new authorized job; no further request is needed to view or verify current assets. The original panda's Muse-equivalent face/fur target and any faithful Weft HD derivative remain explicitly unverified/unsupported, not silently completed by this pipeline milestone.

## Previous follow-up: standalone emerald dragon

The requested dragon with motions is complete for the selected standing gestures; earlier panda/robot Muse-quality limitations remain open. No Weft importer, physics, actions, rewards or Meta ledger changed.

- User explicitly approved **35 additional Meshy credits**, no paid retries/purchase. Exact approval: `examples/meshy/emerald-dragon-approval-v1.json`; plan SHA `d74a91542e849fa16bcbc8c5e04526d9260bcf9000a45cc02b34d70f15010800`. Append-only job-scoped grant retains base140; aggregate175, actual167 across15 settled submissions,0 unknown/overrun. No additional provider rig/motion call or recurring service. Dollar purchase price remains unknown.
- Meshy generated shape + textures only. Mesh task `01a1954b-d585-72d0-b0eb-26b636356569` cost25, seed536030566; texture `01a1954d-4400-77c2-b0bc-c3c7cd1e424f` cost10, seed3041861271. Generation tool26878 finished143.406s. Raw model SHA `a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb`, preserved at `outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb`.
- Generic local `authored_rig` finishing now supports explicit normalized Z-up bones, authored height, ordered spatial skin masks and four-influence normalized weights. Subject-specific configuration and visual acceptance remain assisted; this is not universal automatic anatomy detection. The dragon's19 bones and all4 motions are locally authored, not Meshy auto-rigging or Weft joint angles.
- Selected model: `outputs/emerald-dragon-finished-v3`, real36,322 vertices/36,501 triangles/19 joints, embedded4K basecolor/normal +2K metallic-roughness, real skin/normals/UV/tangents. GLB SHA `b87ee5f5fe0c3b9ab6b84d9a310104a75408ccc247d68c27fbb1d85685840d14`. Source/config/scripts/BLEND and sampled poses retained. Final native tool64508:13.941s /1,314,603,008B sampled peak RSS. Recipe `examples/meshy/emerald-dragon-finish-v3.json` reproduces the local workflow from preserved generated source, without a paid call.
- Four4s clips: idle, look_around, wing_flap, milestone_level_up. Exported CPU skinning at33 times/clip shows head12.13cm on look-around, wing19.85cm on flap and tail14.76cm on celebration; sampled feet unchanged within0.000001m. Native reload `outputs/emerald-dragon-reload-v1` passed1.803s /569,196,544B. This proves sampled deformation, not flight/contact physics or all possible joint ranges.
- Full media `outputs/emerald-dragon-media-v1` finished tool31552,527.181s /1,179,238,400B sampled RSS; no continuation needed.1600px opaque portrait, four720px/24fps/4s silentH.264 clips,96 distinct decoded frames each,97 retained sourcePNG/state including endpoint, exact first-frame static fallbacks. Optional OIDN-filtered portrait and raw original both retained. Endpoints match; no seamless-loop certification.
- Compact gallery media `outputs/emerald-dragon-media-v1-portable-v1`:40 artifacts /10,222,767B, byte-preserving subset. Full frames/logs remain in the original folder. New catalog `browser/gallery-meshy-v3.json` adds dragon first and preserves panda/robot/reference; default/previous catalogs unchanged. Per-asset bounded `initial_view` makes the long tail visible and resettable; legacy defaults remain unchanged.
- Actual final browser proof `outputs/browser-dragon-media-v2` passed14 grouped checks/all12 skeletal clips, playback/seek/reset/video/mobile/reduced-motion/error fallback and0 external requests/page errors. Initial ready4.923s; warm selection90–467ms, not measured sustained FPS. v1 passed but its tail-obscuring initial view was improved; both proofs/screenshots retained. Native/browser portrait and all four state contact sheets reviewed. Small bow/wing/tail celebration is accepted at sampled scope, not human approval or Muse equivalence.
- Delivery audit `outputs/emerald-dragon-delivery-verification-v1` passed in2.667s /201,850,880B process peak (bounded worker3.063s /201,048,064B sampled). Rechecked model/media/reload hashes, canonical manifests, parent binding, formats, source frames/fallbacks, unchanged original generatedGLB and original MuseZIP, and exact ledger totals. Tool68975 ended0. Analytic fixtures are not counted as generated examples.
- Tests:261 full-suite checks in8.329s (`outputs/dragon-workflow-tests-v1.txt`); after final metadata/camera changes,16 generated-workflow +24 gallery tests passed. Reusable skill validated. Initial wrong unittest invocation and system-Python missingPyYAML were corrected using discovery/existing isolated environment; no installation or budget change. Verification notes preserve these failures.
- Rejected outputs preserved: finished-v1 hides tail, v2 raises arms and pulls belly. v3 removes arm-raise curves and uses the full-tail camera. Arm bones exist but broad arm articulation, fingers, blinking, mouth controls and locomotion are not accepted features. Coarse/faceted close-up scales remain; resolution is not Muse equivalence. Authored rest height1.2m, GLB+Yup/+Zforward/meters, mesh19, armature20. Exact node IDs/world rest matrices and corrected local-rig contract: `outputs/emerald-dragon-review-v1`. This additive contract corrects inherited humanoid/provider narrative labels without mutating the model/media parent manifest; future exporter wording is fixed.
- Explicit unsupported Weft unchanged-fidelity result: `outputs/emerald-dragon-weft-unsupported-v1` (36k vertices/PBR vs20k/diffuse, no creature mapping). This is a standalone dragon, not a replacement shoulder attachment. Weft's consumer and prior compatible panda/robot bundles remain unchanged.
- **No active owned jobs or persistent server.** Paid tasks are terminal. Native inspection81219, finishes80017/82063/64508, media31552, reload56228, browser66828/20599 and audit68975 ended. Do not duplicate or restart these paid/native jobs.
- [Delivery and IDs](docs/dragon-delivery-v1.md), [reusable workflow](docs/meshy-workflow.md), [evidence index](evidence/dragon-delivery-v1.json), source checkpoint `outputs/dragon-source-snapshot-v1`. Selected generated source/final model/compact media/evidence are Git-allowlisted; rejected attempts/full frames remain local. Current changes are local/uncommitted/unpushed. Unrelated Weft edits and prior outputs remain intact; evidence hashes are not a remote backup.
- Next safe action: user can start the catalog below and review the dragon; no Blender/provider call needed. A new character uses a new prompt/plan and geometry-specific rig config; do not repeat the exhausted35-credit job or infer a new allowance. Further panda quality work remains separate and should use cheap local look-development first. No automatic purchase, deployment or publication.

Exact commands from Weft root:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v3.json
cd tools/avatar-assets
.venv/bin/python -m avatar_assets verify outputs/emerald-dragon-finished-v3
.venv/bin/python -m avatar_assets verify outputs/emerald-dragon-media-v1-portable-v1
# Only after changing the catalog, use a new verification folder:
node browser/tests/gallery.mjs outputs/browser-dragon-next browser/gallery-meshy-v3.json
# Rebuild locally only when changing the authored recipe; never overwrite current outputs:
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/emerald-dragon-finish-v3.json --out outputs/dragon-next
```

## Previous follow-up: authorized Meshy generation

Checkpoint: **2026-10-31 12:30 PDT**. Explicit key/use authorization is in this thread. The discount/key-verification session stopped after leaving adapter drafts; this session owns their extension and all generation/finishing below. Key stays owner-only in Weft `.local/credentials/meshy-api-key`; never copy it into source, outputs or asset requests.

### Completed and verified

- **13 paid submissions, all succeeded;132 credits reported/reserved of140 ceiling.** Starting balance3300 is not spending authorization; USD credit purchase price unknown. Panda92+robot40. No automatic paid retry, unknown submission, pricing overrun, new subscription, Meta-key/ledger use or remote publication. Canonical journal: Weft `.local/meshy/ledger.sqlite3`; never reset it. New higher-than-estimated charges increase the reserved floor, survive refunds and stop further paid work until pricing is reconciled.
- Generic `meshy plan/run/submit/status/fetch/account/ledger/budget` supports text or one hash-pinned image, named prompts/states and finite credit limits. Actual completed panda/robot jobs resumed in0.176s/0.025s with zero new requests/charges: `outputs/meshy-run-resume-proof-v1`. This is completion-cache latency, not generation latency.
- Actual reference-conditioned panda and independent text-generated robot retain raw GLBs/FBXs, exact prompts/config/reference bytes, sanitized receipts and stage tool versions. Source/model IDs below. Split-normal cleanup is bounded offline≤200k input vertices and still requires≤100k runtime output; original bytes remain unchanged.
- **Current reviewable 3D packs:** panda `outputs/muse-panda-meshy-refined-v6` (92,986 vertices/71,539 triangles/25 bones), robot `outputs/workshop-robot-meshy-refined-v2` (45,386/51,320/24). Both have four **locally authored**4s gestures, explicit normals/UVs/tangents/skins,4K basecolor+normal textures and2K metallic/roughness maps. Panda adds2,400 portable cutout cards and299,000 configured native fibers; these are distinct representations, not quality scores or simulated hair.
- Both CPU proofs and independent native reloads passed: `muse-panda-meshy-reload-v1`, `workshop-robot-meshy-reload-v1`. Actual browser proof `browser-meshy-refined-v1` loaded both GLBs, selected all8 clips and saw zero external requests/page errors. Timings6.842s/2.251s include screenshots/clip checks, not raw load/FPS. Latest catalog `browser/gallery-meshy-v2.json` includes matching compact media. Full browser proof `browser-meshy-media-v1` passed13 grouped checks/all8 clips, initial ready5.021s, warm selections171–506ms, zero external requests. Default HD gallery remains unchanged.
- **Visual target is unmet.** Closer panda shape is not Muse equivalence. Smaller eyes/flatter muzzle, coarse source fur/dragon texture and browser fur artifacts remain. Crafting/working are weak pantomimes without props/finger articulation; their task meanings are not accepted. Explicit unsupported outputs: `meshy-muse-fidelity-unsupported-v1`, `meshy-weft-hd-unsupported-v1`. No new Weft mapping/importer/physics changes. Review evidence in `meshy-integration-review-v1`; contract audit binds actual node IDs/model hashes and unchanged originalZIP SHA.
- Provider panda motions were genuinely generated and retargeted, then **rejected visually**: source weights assigned16,185 dominant vertices, including belly, to `RightHand`. Generic spatial weight corrections, expanded rigid `CompanionAnchor`, and restrained explicit local curves replace them; they are not relabeled accepted provider clips. Panda anchor node11→RightShoulder12;Hips24;Armature25. Robot Hips23/Armature24. Full transforms/skins in each pack's `exported-nodes.json`; root scale must be honored.
- Repeatable finishing configs: `examples/meshy/muse-panda-finish-v4.json`, `workshop-robot-finish-v1.json`. Commands/docs in `docs/meshy-workflow.md`; reusable skill updated/validated. Native reuse requires matching source/design/backend hashes and an exited successful native job; only changed postprocessing reruns.
- Final passing checks: **255 tests in9.427s** (`meshy-workflow-tests-final-v1.txt`), including14 Meshy accounting/input tests and4 continuation tests. Earlier failed Pillow exception-type check is preserved and fixed. Actual-model/media audit `meshy-delivery-verification-v1` independently verified full/portable hashes, parent binding, video decoding, fallback identity,366 reused PNGs, canonical manifest serialization and original ZIP in4.632s/269,156,352B process peak. Analytic fixture tests are not generated assets. Scoped `git diff --check` passed.

### Rejected attempts / measured bounds

All failed folders remain. Groomv1/v2 exposed parent centimeter scale;v3 exposed a non-bone density mask in a skin signature. Finished candidatev1 failed action-slot binding;v2 exceeded100k exported vertices from tangent splits;v3 found four zero tangents;v5 additionally found two nonorthogonal tangents. Finalv6 reuses v5's completed native pass and repairs exactly six bounded tangent entries, preserving the native export. Geometry/UV/weights/motion are unchanged by that repair.

Dense nativev4 reached2,197,028,864B sampled RSS and was stopped after17.883s by the unchanged2GiB guard. Reduced45-child pass completed in63.186s/2,044,051,456B; final postprocessing/native reuse took≈1s. Robot finish12.922s/1,242,333,184B. At most2 heavy Blender jobs/4threads each. Native jobs≤600s/2GiB/512MiB, no paid compute. The600k theoretical native profile is not a measured safe default;90 children failed on this asset.

### Completed media / next safe action

- Panda `outputs/muse-panda-meshy-media-v2` completed and verified429 artifacts. Originalv1 is a preserved wall-time failure:1200.074s,365 finished animation PNGs,2,079,604,736B sampled peak. Single continuation tool67882 exited0, native67.726s/1,707,753,472B; copied365 video PNGs plus portrait and rendered only23 missing frames. Aggregate native time1267.800s. No second continuation or increased memory limit.
- Robot `outputs/workshop-robot-meshy-media-v1` completed and verified419 artifacts. Native612.815s/1,199,587,328B; tool61091 exited0.
- Each pack has1600px opaque WebP/PNG portrait, four720px/24fps/4s silentH.264 clips (96 actual frames/96 unique decoded thumbnails each), exact first-frame PNG fallbacks, contact sheets, source frames and manifests. No certified loop. Panda keeps raw and isolated OIDN-filtered portraits; filtering does not fix source geometry or recreate fur.
- Byte-preserving compact packs: `muse-panda-meshy-media-v2-portable-v1` (~11.45MB) and `workshop-robot-meshy-media-v1-portable-v1` (~7.52MB). Source frames/logs are omitted from the copies, not deleted. Catalog `browser/gallery-meshy-v2.json`; browser tool76360 exited0. Independent final audit tool66238 exited0.
- **No active owned jobs or persistent servers.** Original provider tasks are all terminal. Earlier native reloads84367/13206 and browser79179 also ended. All handles are historical; do not restart them or duplicate paid tasks.
- [Delivery/critical node IDs](docs/meshy-delivery-v1.md), [workflow](docs/meshy-workflow.md), and [evidence index](evidence/meshy-trial-v1.json) provide exact artifacts, hashes, commands, costs and limits. Source snapshot: `outputs/meshy-source-snapshot-v1`. New work is local/uncommitted/unpushed; preserve unrelated Weft changes. Evidence hashes are not a remote backup.
- **Quality goal remains open.** Native and browser review still shows coarse coat/dragon detail, flattened muzzle, patch-transition artifacts and shiny browser fur speckles. Crafting/working are not accepted as meaningful tasks. The current outputs are reviewable candidates, not Muse-quality delivery or a Weft-ready HD conversion.
- Next safe quality work: use cheap still/pose drafts to isolate source-mesh/material defects from groom and skinning defects before another full media render. Refine face/coat/dragon surfaces and add visible workpiece interaction with subject-appropriate controls; validate new outputs independently. Keep current originals. Do not repeat the rejected provider clips, raise the140-credit ceiling, spend the account balance, or claim that more fibers/4K resolution alone meets the target.

Exact current view/verification commands (from Weft root):

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v2.json
cd tools/avatar-assets
.venv/bin/python -m avatar_assets verify outputs/muse-panda-meshy-refined-v6
.venv/bin/python -m avatar_assets verify outputs/workshop-robot-meshy-refined-v2
.venv/bin/python -m avatar_assets verify outputs/muse-panda-meshy-media-v2
.venv/bin/python -m avatar_assets verify outputs/workshop-robot-meshy-media-v1
# For a changed catalog only, use a fresh output:
node browser/tests/gallery.mjs outputs/browser-meshy-next browser/gallery-meshy-v2.json
```

### Task IDs (resume, never recreate)

Panda: mesh`01a19328-cb63-70cb-9bd2-38762eb49ee5`;rig`01a19332-019d-7010-9e9a-757e5cc7fb25`;motion idle`01a19335-e864-737b-92ec-64d683502bce`,craft`01a19335-f33a-76a7-910d-46de7fe47a76`,milestone`01a1933b-a4ef-7095-8c57-d8b3e224e00f`,working`01a1933b-b107-76ed-a02c-49451c8d729c`;retarget idle`01a19337-b106-7708-bec8-be7e1767053d`,craft`01a19337-bd76-74e4-8225-027b0c75133b`,milestone`01a1933f-b49e-7777-bc1f-9d64fbbc18b9`,working`01a1933f-b98d-72a9-a1ce-4d00add9167e`.
Robot:mesh`01a19328-d810-7246-8101-fb010c440a75`;texture`01a1932c-7881-71e8-8071-9a7c8bce89ae`;rig`01a1932f-00fe-7455-bb70-c6a3b335a554`.

## Previous follow-up: provider research and browser-fur experiments

User reaffirmed that the deliverable is the finished Muse-quality avatar, not a recommendation to operate Meshy. No provider promises the full specification, and no neural backend is connected. Re-inspected the original WebP against `muse-panda-hd-media-v1-portable-v1/portrait.png`: the current model still has a flatter muzzle, more protruding eyes, narrower belly, longer-looking limbs and much less convincing coat breakup. Raising resolution alone will not correct these differences. Retain the generic workflow, but judge each new avatar candidate visually and through actual motion; do not substitute fiber counts, file validity or a platform ranking for acceptance. This clarification changed documentation only; no new model, paid call or quality success is claimed.

User asked whether Meshy is the best platform and requested research. They also reinforced that delivery remains our job, not work to send them elsewhere to perform. [Research and recommendation](docs/generation-platform-research-2026-10-31.md): Hi3D v3.0 is the first **quality-focused base-model trial** candidate; Tripo P2 is a production/topology/creature-rig alternative; Meshy7.1 is an all-in-one alternative, not a proven overall winner. No provider was connected or called, and no neural backend was implemented. Published rankings do not establish Muse fur quality, rig deformation or Weft compatibility. Hi3D official prices conflict; preserve both and confirm the account-side price before any paid job. No new purchase/account/subscription is authorized.

- [x] Diagnosed browser fur minification: the original atlas has9.53% alpha-mask coverage at full resolution,0.049% at16px/tile, and0% at8px/tile. New opt-in `materials.*.fur_atlas: distance-v1` uses wider locks/cutoff0.18; the same8px/tile test retains61.6%. These are analytic alpha tests, not appearance scores.
- [x] Generic recipe derivation: `scripts/author_browser_fur.py`; both new recipes reproduce byte-for-byte. Geometry, actions and native groom are unchanged. Legacy atlas pixels retain their measured hash. New source/backend snapshots are in each build.
- [x] Actual new panda/fox builds and browser checks: final candidates `outputs/muse-panda-browser-fur-build-v2`, `outputs/fox-browser-fur-build-v3`; catalog `browser/gallery-fur-comparison-v3.json`; screenshots/16 real skeletal clip selections/local-only request proof in `outputs/browser-fur-comparison-v3`. The first wider two-sided experiment exposed flipped-normal shading; the one-sided version fixes that part but still shows layered-coat intersections. **Rejected for default-gallery promotion**, not a finished fidelity upgrade. Review: `outputs/browser-fur-review-v1/review.json`.
- [x] Attribute/embedded-texture/skin/motion proof passed for both candidates in `outputs/browser-fur-attribute-proof-v1` (1.9891s,252,493,824B process peak). Panda model SHA `fde2533d535fcafcf6eef9676c59270f921925ed93272de4a5101c65c7c3c2cc`; fox `dca7ff3f314451c30099a739918725d8a214d0e0faa103de633320bd2efb3057`. No new Weft mapping or native-media pack was attached.
- [x] Fixed a measured runner exit race: macOS can lose process-memory data just before waitpid reports exit. ESRCH now allows one250ms wait for a verified exit; a still-live child remains a monitor failure. No invented memory sample, relaxed resource limit or relabeled failure. Regression tests cover both cases.
- [x] 214 Python tests covered:206 passed in the initial sandbox run;8 HTTP tests could not bind loopback and then passed separately under native execution. Logs `outputs/browser-fur-tests-v1.txt` and `outputs/browser-fur-http-tests-v1.txt`. Initial failure evidence remains. `git diff --check` passed before documentation completion; final check is recorded with the evidence index.
- [x] Preserved failed `fox-browser-fur-build-v2` (exit0 but monitor failure), `browser-fur-comparison-v2` (catalog correctly rejected that failed pack), and rejected first wider-card builds. Candidate native builds took10.943–19.085s, at most1,516,486,656B sampled RSS,4CPU threads each and at most2 concurrent builds. Browser timings include screenshot/clip checks and are not load latency or rendered FPS measurements.
- [ ] Local fidelity work: test a generic exclusion mask on an analytic layered emitter before another character build; black undercoat crossing an ivory outer region is a current hypothesis. Then revisit face/body proportions, patch transitions and actual groom breakup. Do not claim the material change alone fixes Muse likeness.
- [ ] Unattended generation: select and integrate an authorized provider with a bounded/resumable job journal, immutable prompts/model versions, credit reservations, uncertain-submission protection and local validation. Existing `request` still returns `needs_agent_authoring`. No API key was requested in chat, read from Weft or borrowed from another project.

All task jobs finished:7601/15307(first builds),46997/8329(second builds),98702(fox monitored retry),33138/74146(browser passes),81589/61084(tests/attribute proof). No background server or provider job remains. Latest work is local, uncommitted/unpushed. Experimental outputs are preserved locally, not claimed remotely backed up. The default gallery and prior HD release remain unchanged. Total provider calls/cost0/$0.

Exact local commands (from `tools/avatar-assets`; use new output folders):

```sh
.venv/bin/python scripts/author_browser_fur.py examples/scenes/muse-panda-hd-v4.json --id muse-panda-browser-fur-v1 --out examples/scenes/muse-panda-browser-fur-v1.json --check
.venv/bin/python scripts/author_browser_fur.py examples/scenes/fox-hd-v3.json --id fox-browser-fur-v1 --out examples/scenes/fox-browser-fur-v1.json --check
.venv/bin/python -m avatar_assets gallery --catalog browser/gallery-fur-comparison-v3.json --port 8766
node browser/tests/fur-comparison.mjs outputs/browser-fur-comparison-next browser/gallery-fur-comparison-v3.json
```

The gallery command is user-run; do not start a persistent server automatically. Next safe action is the local layered-emitter experiment or provider adapter preparation, **not a paid generation, account signup or subscription purchase**. Source/evidence index: `evidence/browser-fur-followup-v1.json`.

## Previous follow-up: HD fur and surface upgrade — verified

The user prioritized **visual fidelity**: Muse-like fur/hair, sharp textures and refined face/dragon detail on real rigged 3D assets. This bounded upgrade is implemented and tested; **exact Muse quality remains unsupported**, not declared complete by texture or fiber counts. Runtime alignment is deferred. See [HD decisions/results](docs/hd-quality-pass-v1.md), [release index](evidence/hd-release-v1.json) and [sampled visual review](outputs/hd-visual-review-v1/review.json).

| Subject | Final recipe under `examples/scenes/` | Final model/source under `outputs/` | Matching full media under `outputs/` |
| --- | --- | --- | --- |
| Panda + dragon | `muse-panda-hd-v4.json` | `muse-panda-hd-build-v3` | `muse-panda-hd-media-v1` |
| Fox | `fox-hd-v3.json` | `fox-hd-build-v2` | `fox-hd-media-v1` |
| Robot | `workshop-robot-hd-v1.json` | `workshop-robot-hd-build-v2` | `workshop-robot-hd-media-v1` |
| Lamp | `lamp-hd-v1.json` | `lamp-hd-build-v2` | `lamp-hd-media-v1` |

- Implemented reusable scene options: native1024/2048 seeded surface maps, radial iris detail, skinned alpha-masked curved fur cards, optional editable combed hair, plush studio lighting and isolated portrait denoising. Old recipes/algorithms and outputs remain unchanged. Rebuild GLB from JSON; the native-render `.blend` hides its separate browser-card materials.
- Panda:7,688 browser cards/238,328 configured native fibers,31 emitters. Fox:5,000 cards/155,000 native fibers,20 emitters. These are separate representations, not independently counted rendered hairs. Mechanical models have appropriate metal surfaces, no animal fur. All models retain real skins and four skeletal clips.
- Final exported attributes/textures/skin proofs passed in `hd-final-attribute-proof-v1` and `hd-final-attribute-proof-v2`. All four final independent reloads passed in each subject's `*-hd-reload-v2`. Generic comb proof at Weft `.local/native-comb-probe-v1/attempt-3` verifies roots, arc lengths, bend and saved keys; it is analytic, not a completed character.
- **209 tests passed in8.191s** (`outputs/hd-quality-verification-v1/python-tests.txt`). All four final recipes reproduce exactly (`hd-recipe-reproducibility-v1`). The reusable skill validates. Whole GLB rebuilds are not bit-identical; use the accepted hashed files.
- Four1600px portraits,16 silent720px/24fps videos, exact first-frame static fallbacks. Panda/fox/lamp48frames/two seconds each; robot72frames/three seconds. Every decoded thumbnail is distinct. Actions are retimed in full, not truncated or repeated stills. All images/videos are opaque; no seamless loop is certified. `hd-media-verification-v1` verified all full/portable packs, parent binding, raw/filter provenance and unchanged ZIP in0.986s/42,647,552B process peak.
- Default gallery uses matching `*-hd-media-v1-portable-v1` packs; full native frames/logs remain local. Frozen HD catalog: `browser/gallery-hd-v1.json`; previous default: `browser/gallery-quality-v2-portable.json`. **Browser check passed15 grouped checks/all16 skeletal clips**, zero external requests, cold ready4.615s, warm selections140–395ms. Reduced motion/mobile/failure fallback passed. Two-second software-WebGL observation is not sustained rendered FPS. Evidence: `outputs/hd-gallery-browser-v1/browser-test.json`.
- Native media times/peak sampled RSS: panda967.158s/1,647,443,968B; fox800.671s/1,422,262,272B; robot579.188s/1,098,252,288B; lamp407.949s/1,143,980,032B. At most two heavy jobs/four CPU threads each,1200s/2GiB/512MiB per render; packaging180s/768MiB; isolated portrait filter90s/2GiB/64MiB. Worker cleanup is bounded7s. Direct-child memory is sampled, not an OS or descendant-memory cap.
- All owned jobs finished: tool sessions33572(panda),79384(fox),93216(robot),58161(lamp),23067(robot/lamp final reloads),29760(browser). No task-owned persistent server or native job remains. Checkpoint4 was a historical in-flight note; this handoff supersedes its live handles.
- Preserved rejected `muse-panda-hd-build-v2`: native files completed but the old outer monitor lost measurements with EPERM; cause unconfirmed. `hd-runner-incident-v1` records the gap; final builds used the repaired checkpointed runner. Earlier radial coat, temporary atlas-path proof and failed analytic comb probes remain. Isolated denoising preserves raw images but softens fibers. Persistent-render cache changed median timing only about3%; it was not promoted as an optimization.
- Remaining fidelity limit: default-size browser card fur is substantially less visible than native hair; sculpting, fur breakup and shading still differ from Muse. Explicit unsupported results: `hd-muse-fidelity-unsupported-v1`, `hd-weft-profile-unsupported-v1`. New HD models exceed Weft's frozen vertex/material profile; do not reuse earlier mappings. Native hair is not a GLB strand representation or physical hair simulation.
- Provider calls/spend **0/$0**. No deployment, upload, new account or borrowed credentials/budget. This follow-up is **local, uncommitted and unpushed**; selected models/portable media/proofs are allowlisted in `.gitignore`. Earlier migration commits remain published. Other sessions' dirty Weft work is untouched.
- Final release source/evidence bundle: `outputs/hd-release-v3`, indexed by `evidence/hd-release-v1.json`. Preparations v1/v2 incorrectly expected pack manifests for an analytic probe and result-only `unsupported3d` folders. Their partial snapshots/failure manifests remain. V3 preflights proof paths and inventories those actual result files. No finished model/media pack was modified.
- Deferred integration drafts remain separate: `avatar_assets/weft_metadata.py` and five passing analytic alignment tests; the prior Weft compound-replay fix passed seven Node tests. No real neutral-frame capture or newly calibrated mapping was produced. These are not this visual task's completion claims.

### Next safe action

View the new set; do not repeat completed render/test jobs without a change or unresolved defect. From Weft:

```sh
./tools/avatar-assets/run gallery
```

Open http://127.0.0.1:8765/#panda. Restart a previously running gallery yourself to load the new catalog. Use **Portrait** or **Rendered clip** for native groom previews; **3D model** uses lighter cards. Source/download paths and fresh-folder reproduction commands are in [the HD guide](docs/hd-quality-pass-v1.md). Further Muse matching needs another bounded sculpt/groom/shading revision, not a relabeling or pixel-count increase. Keep optional Weft integration and unrelated video/simulation work separate.

## Migration into Weft — complete

- User explicitly authorized the move, same-session reassignment, commit/push directly to main, and removal of the standalone checkout after the push verifies.
- Full copy verified: **19,778 filesystem entries / 3,146,153,025 file bytes**, including all original outputs, rejected attempts and local runtimes. Ten virtualenv launch/config files were then rebased with backups; no dependencies were reinstalled.
- Current session remains `01a18ebe-a17c-7cc0-9117-06ca476fb31c`, assigned to the existing Weft project. Future-turn cwd is `/Users/ivan/kintsugi/weft`; goal identity, status, model and session ID are unchanged. No other session or goal was edited.
- **144 tests passed in 3.161s** from the new root. All four authoring recipes reproduce exactly. **33 complete packs**, both catalogs and both legacy consumer imports/reloads passed verification.
- Headless gallery: 15 grouped checks/all 16 detailed skeletal clips passed, no external page requests; cold ready 4.664s. Largest-model two-second software-WebGL sample advanced 2.167 clip seconds; this is not a sustained FPS guarantee. Migrated Blender independently loaded/moved the panda in 5.699s, sampled peak RSS 360,939,520 bytes.
- Seven byte-preserving portable media packs total 37,414,959 bytes. Original full packs remain local; omitted frames/logs are listed in each new manifest. Both default and basic catalogs use tracked portable media. Historical `gallery-quality-v2.json` retains the full-media paths.
- Source/reference hashes and limitations remain unchanged. Read [relocation](docs/relocation.md) and [migration receipt](evidence/weft-migration-v1.json). Selected UV-repair failure/probe folders and other unlisted old outputs remain local, not remotely backed up.
- Main implementation commit `05eb8532d1d7d4ff7cf379a931b9f13326199fb5` is verified on the existing private remote. The old `/Users/ivan/kintsugi/avatar-assets` directory is deleted after its unchanged inventory and destination copy were verified. CLI startup still passes after deletion. Final publication receipt: `/Users/ivan/kintsugi/weft/.local/avatar-assets-migration-v1/publication.json`. No task-owned persistent server or native job remains running.

## Visual-quality follow-up — verified upgrade, fidelity limit explicit

User requested Muse-like quality and explicitly reinforced “make sure to have fur and all other necessary components.” The upgraded workflow and four detailed examples now work end to end. **Equivalent Muse finish is not demonstrated**: these are stylized real-time models with short geometric fur, not the original dense groom or recovered source model. Explicit result: `outputs/muse-fidelity-limit-v1/unsupported.json`.

- [x] Schema v2: shaped triangle/quad meshes, per-vertex UVs and blended weights, bevels, real skinned fur, embedded procedural BaseColor/normal PNGs, and a portable-surface validation gate. Generic subject-independent backend; v1 behavior and outputs preserved.
- [x] Final build folders: `outputs/muse-panda-quality-build-v6`, `outputs/workshop-robot-quality-build-v3`, `outputs/fox-quality-build-v4`, `outputs/lamp-quality-build-v4`. Exact recipes: panda `muse-panda-quality-v5.json`, robot `workshop-robot-quality-v2.json`, fox `fox-quality-v4.json`, lamp `lamp-quality-v3.json`. Author scripts support `--check`; all four scene descriptions reproduced exactly.
- [x] Panda has 18,120 authored fur blades; fox 12,700. Actual exported counts: panda 85,952 vertices/22 joints; robot 47,644/17; fox 68,518/20; lamp 37,762/8. Every model has four real skeletal clips, explicit normals/UVs/tangents and skin weights. Fur counts are source-reported geometry, not semantically recovered counts from a binary.
- [x] Final exported-data proof: `outputs/quality-attribute-proof-v1/manifest.json` and `SUMMARY.md`. Native reloads: `muse-panda-quality-reload-v6`, `workshop-robot-quality-reload-v3`, `fox-quality-reload-v5`, `lamp-quality-reload-v4`, all under `outputs/`. Front/back/side and state review: `outputs/quality-visual-review-v1/review.json`; sampled assistant review, not user approval.
- [x] Matching media folders: `muse-panda-quality-media-v6`, `workshop-robot-quality-media-v3`, `fox-quality-media-v4`, `lamp-quality-media-v4`, under `outputs/`. Each has 415 hash-verified artifacts, four silent 320×320/24-fps/four-second clips with 96 frames and 95–96 unique decoded thumbnails, exact first-frame static fallbacks and opaque media. Panda/fox portraits are 1600×1600; robot/lamp 640×640. Evidence: `outputs/quality-media-verification-v1`.
- [x] Browser default updated to the reviewed detailed models, with matching media. Old collection remains `browser/gallery-basic-v1.json`; new frozen configuration also saved as `browser/gallery-quality-v2.json`. High-resolution portraits load on demand; thumbnails remain small. Do not attach old Weft mappings to the new models.
- [x] Browser proof: `outputs/quality-gallery-browser-v2/browser-test.json`, 15 grouped checks/all 16 native clip selections passed, including reduced motion, failure fallback and mobile layout; zero external page requests. Cold ready 4.407s; warm selections 99–269ms. Largest-model playback remained controllable during a two-second software-WebGL sample; not a rendered-FPS or sustained performance guarantee.
- [x] Full suite: 127 tests passed in 1.991s (`outputs/quality-python-tests-v3.txt`). The later high-resolution portrait route passed all 22 affected gallery tests (`outputs/quality-gallery-tests-v4.txt`); the subsequent narrow CSP change passed the actual browser suite. Reusable skill validation passed. Detailed results/limits/commands: [docs/quality-pass-v2.md](docs/quality-pass-v2.md).
- [x] Final builds took 34–60s, max 803MB sampled native RSS; preview renders 135–191s, max 702MB. At most two native jobs/four CPU threads each, within the existing per-job memory/output limits. Provider calls/spend: 0/$0; no remote service or publication.
- [x] Immutable source/evidence release: `outputs/quality-release-v2/manifest.json`; index: [evidence/quality-release-v2.json](evidence/quality-release-v2.json). The source ZIP and both old Weft model hashes were rechecked unchanged. Separate detailed-model conversion limit: `outputs/quality-weft-profile-unsupported-v1/unsupported.json`.

### Preserved failures and fidelity limits

- Panda iterations v2/v3/v4 preserve sparse/corrugated fur, exposed shoulder caps and prop occlusion. Final authored wrists keep the orb visible. Fox iteration v3 preserves flat shoulders; v4 rounds and blends them. Exact rejected scripts/reasons live in `outputs/quality-panda-iteration-*` and `outputs/fox-quality-iteration-v3`.
- Backend analytic probe v1 had double-darkened sRGB texture bytes; corrected proof v2 is labeled analytic. Attribute checks exposed invalid tangent frames on collapsed/cancelling cap UVs and microscopic bevel remnants. Native repair probes preserved in `outputs/quality-attribute-proof-v1`; final exporter repairs only those faces and checks real MikkTSpace frames automatically.
- `quality-gallery-browser-v1` failed on blocked in-memory texture fetches. The final CSP permits local `blob:` reads in addition to same-origin requests, not remote HTTP. Browser v2 loaded textures and passed.
- One fox native reload passed but its outer runner returned an OS permission error before saving completion records. `outputs/quality-runner-incident-v1` preserves the incident; a new bounded reload passed. Cause remains unconfirmed, and no missing job record was fabricated.
- Full 720p/eight-second media has **not** been newly rendered for these detailed editions. The older basic full Muse packs remain preserved. Larger renders can hit the current 1200s/512MiB limit; estimate before expanding work. Identical loop endpoints do not certify seamless velocity/playback.
- Detailed general/PBR editions are **not Weft-compatible replacements**: they exceed the vertex/material profile. Existing matte panda/robot models and importer-accepted mappings remain unchanged. No Weft edits or new runtime-alignment claim.

### Ownership and next safe action

All task-owned native jobs, temporary browser tests and servers completed. Historical final handles: builds 31441/14334/88586/90539; reloads 40232/11978 plus completed direct calls; media 25128/87924/39587/34243; browser 92631 (rejected)/48469 (passed). PID/commands/bounds remain in each output's job/handle records. Do not resume or kill stale IDs. A user-started gallery was not managed by this task.

Start `.venv/bin/python -m avatar_assets gallery`, then visit `http://127.0.0.1:8765/`. Restart an existing server to load the new catalog. For the old collection: `gallery --catalog browser/gallery-basic-v1.json --port 8766`.

The next fidelity step is a scoped denser groom/sculpt and more organic face/dragon detail, or a separately authorized external 3D source followed by the same checks. Do not promise a generator will match Muse, borrow Weft credentials/budget, or declare a screenshot to be a model. Human visual acceptance remains for the user; faithful reconstruction/film-quality parity remains explicit unsupported scope, not a completed claim.

## Initial browser gallery — completed historical release

User asked to browse the assets without opening Blender. The local gallery now presents robot, panda, fox, lamp and the original 2D reference, with one lazy-loaded interactive model, orbit/zoom, camera presets, four-state animation selection, play/pause/seek/speed/repeat, rendered videos, portraits and GLB downloads.

- Start yourself: `.venv/bin/python -m avatar_assets gallery`, then visit `http://127.0.0.1:8765/`. Ctrl+C stops it; `--port 8766` avoids a port conflict. **No persistent server or user browser was left running.**
- Guide: [docs/browser-gallery.md](docs/browser-gallery.md). Collection: `browser/gallery.json`. Dependencies are installed, pinned in `browser/package-lock.json`, with lifecycle scripts disabled. Model-viewer4.3.1 is locally served; Playwright-core 1.64.1 is test-only. No new provider calls, subscriptions or uploads; $0 provider spend.
- Final Python suite: **103 tests passed in 1.692s**, `outputs/gallery-python-tests-v2.txt` (82 existing +21 gallery checks). Covers localhost-only routing, source-manifest matching, exact file bytes, video ranges, rejected mutation/traversal/symlinks/changed assets, vendor route and request limits.
- Final actual browser proof: `outputs/gallery-browser-v6/browser-test.json` and its desktop/mobile/four-character screenshots. Chrome156.0.8078.76 with a fresh headless profile/software WebGL; 14 grouped checks passed, all 16 skeletal clip selections worked, video playback/seek worked, 390px layout had no horizontal document overflow, reduced-motion mode assigned no model source and made no GLB request, zero external page requests.
- Tests include two explicitly injected UI races: delayed play-promise rejection and an accelerated loading timeout with a delayed real video. Both recover without stale errors. These are state-management tests, not claims of real network outages.
- Measured browser cold ready 4.069s; warm model selections 67–126ms. A one-second paused observation used 0.0256 renderer-task seconds with 8.34MB JavaScript heap. These are software-renderer snapshots, not total browser/GPU memory or sustained-use guarantees. Model cache is 1; invisible/hidden playback is paused.
- All temporary test processes finished. Historical tool sessions: 40095(v2),35799(v3),41174(v4),51745(v5),9177(v6); Python suites 92246/61545. Do not treat those stale handles as live jobs.
- Preserved failed checks: `gallery-browser-v1` could not bind under sandbox; v2 could not load the scoped `@google` vendor route; v3 used a faulty already-sideways camera test and exposed the embedded-WASM policy requirement. Fixed with safe scoped paths, an explicit Front→Side test, and only `wasm-unsafe-eval`—not arbitrary script evaluation. v4/v5 passed; v6 adds final camera reset/framing verification.
- The original asset packs and ZIP are unchanged. The prior 25 implementation/test/lockfile bytes are preserved in `outputs/core-v1-source-snapshot` and match the original release hashes, including pre-gallery CLI bytes. The v1 core release is historical; current gallery source/evidence is cataloged separately in `evidence/gallery-release-v1.json`.
- Scope remains a browser asset preview, not Weft's importer, physics renderer, replay or recording. All existing consumer alignment follow-up below remains unchanged.

## Verified core deliverables

The historical core hash-bound catalog is [evidence/release-v1.json](evidence/release-v1.json). It records ten delivery/proof bundles, 18 verification records, 25 implementation/test/lockfile hashes, actual tools/settings, media formats, fallback checks and costs.

- [x] Reference archive preserved byte-for-byte. Final SHA256 `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Inspection v1/v2 manifests matched exactly. Five media entries: opaque 1600-square WebP; four silent H.264 720-square, 192-frame, 24fps/eight-second videos. Bounded inspection: 2.336 seconds, 128,942,080 bytes sampled peak RSS.
- [x] Reusable data-only scene compiler, CLI, material reskin route and [skill](skills/avatar-asset-workflow/SKILL.md). Optional selected references, subject/style, arbitrary configured state names, camera/background and bounded output overrides are supported. Skill validation and independent forward testing passed; identified defects were corrected with regression evidence.
- [x] Two genuinely different authored examples: `outputs/fox-refined-v2` and `outputs/lamp-refined-v1`. Real meshes/materials/skeletons/weights, four skeletal actions, exact prompts and editable `.blend`. Independent native GLB reload/moving weighted vertices passed in `outputs/fox-reload-v1` and `outputs/lamp-reload-v1`. These are not analytic fixtures.
- [x] Full media: `outputs/fox-muse-v1` and `outputs/lamp-muse-v2`. Each has a native opaque 1600-square portrait/WebP, four genuine 720-square/24fps/eight-second clips with 192 distinct decoded thumbnails, static PNG fallbacks identical to first rendered frames, editable-source links and retained native frames. All artifact hashes verified; visual reviews are separate immutable records. Portraits and all state contact sheets were inspected.
- [x] Panda and humanoid robot: `outputs/muse-panda-weft-build-v1` and `outputs/humanoid-robot-weft-build-v1`. Both native reloads pass; sampled front/back/side and action views reviewed. Panda dragon is on anatomical right/viewer-left and stays attached to a torso-side shoulder helper.
- [x] Weft bundles: `outputs/weft-muse-panda-v1` and `outputs/weft-humanoid-robot-v1`. Real skin/normals/weights; explicit diffuse-only materials; ten body bindings; actual node IDs, world rest matrices, axis/scale and provenance. Both accepted by the existing frozen importer: `outputs/weft-import-proof-v1/results.json`. [Exact handoff and IDs](docs/weft-delivery-v1.md). No Weft code changed.
- [x] Resource and invalid-input checks: **82 tests passed in 1.542 seconds**, `evidence/tests-final-v2.txt`. Covers hostile/oversized archives, bounded media, static-animation rejection, corrupt rigs/weights/mappings, runtime/memory/output stop conditions, child cleanup and prompt-preserving skins. Fixtures remain labeled analytic.
- [x] Unsupported recovery recorded: `outputs/reference-3d-unsupported-v1/unsupported.json`. Original rig/topology/unseen geometry cannot be faithfully recovered from this pack. No nominal mesh is presented as recovered geometry.

## Measurements and job handles

| Job | Completed evidence | Elapsed | Sampled peak RSS |
| --- | --- | ---: | ---: |
| Fox full media, tool session 5452 | `outputs/fox-muse-v1/job.json` | 1122.540 s | 936,394,752 B |
| Lamp full media, session 72887 | `outputs/lamp-muse-v2/job.json` | 1097.050 s | 935,673,856 B |
| Panda build, session 65037 | `outputs/muse-panda-weft-build-v1/job.json` | 20.889 s | 474,103,808 B |
| Robot build, session 46932 | `outputs/humanoid-robot-weft-build-v1/job.json` | 20.645 s | See job record |
| Current robot preview, session 34795 | `outputs/robot-preview-v1/job.json` | 17.604 s | 363,446,272 B |
| Robot packaging worker | `outputs/robot-preview-v1/packaging-job.json` | 0.793 s | 97,370,112 B |
| Improved short-contact packaging proof | `outputs/robot-packaging-proof-v2/job.json` | 0.769 s | 98,123,776 B |

All listed jobs exited 0 with no bound failure. PID/command handles remain alongside logs. Full renders ran two jobs concurrently, four CPU threads each, on a ten-logical-CPU host; these timings are not service guarantees. Local retained disk snapshot before the final report: outputs approximately 910 MiB, Blender tools 1,118 MiB, virtual environment 107 MiB. Nothing was pruned.

The two full renders began before packaging moved to its own worker. A real 192-frame lamp clip passed the new bounded worker in `outputs/lamp-packaging-proof-v1`; all four short robot clips passed the current end-to-end path and its updated contact sampler. The final Muse preset exactly matches the measured eight-sample full-export settings: `evidence/default-preset-v1.json`.

Blender 4.5.13 arm64 is local at `.tools/Blender.app/Contents/MacOS/Blender`; installer checksum `663ce944257c61ff1d6aa09e15c8f57bbd8d59023adb2fa7edde33a9ed960b53`. Installer was detached. Dependencies are isolated and hash-locked. Blender requires native execution outside the sandbox for Metal initialization on this host even for CPU rendering; no viewer or development server is opened.

## Preserved failures and limits

[Rejected attempts](evidence/rejected-attempts-v1.json) records exact folders, reasons and available manifest/log hashes. Earlier output folders remain untouched.

- Initial sandbox download/mount/Metal initialization restrictions were resolved with narrow authorized local setup/execution. Initial fox output lacks the later failure-manifest format; its logs remain preserved.
- Wrong-axis/double-conversion geometry, weak fox reading poses, and native-verifier display-helper/sampling bugs were corrected in new versions. No superseded version is the accepted example.
- `lamp-muse-v1` stopped at 2,201,321,472 bytes sampled RSS during 1600px denoising. Current portraits use at least 32 samples without denoising; video uses eight denoised samples. The 2GiB limit was not raised.
- Resource limits are sampled, not an instantaneous OS allocation cap: memory every 250ms, output roughly every two seconds plus completion; termination has a five-second grace. Current worker limits: builds 300s default/2GiB; media render 1200s/2GiB; packaging 180s/768MiB; output 512MiB. Large overrides can fail honestly at the limit.
- `evidence/reproducibility-final-v1.json`: fixed triangulation preserves the tested topology, but repeated Blender exports differ in index byte order and tiny normals. **No bit-for-bit GLB rebuild guarantee.** Retain the accepted hashed originals; do not repeat unchanged experiments. Source inspection manifests were reproducible.
- Exact loop endpoints match, but velocity continuity/seam playback is not certified. All loop declarations remain not certified. Static PNGs are the reduced-motion option.
- Stylized intersecting primitive surfaces and rigid-part weights are complete authored geometry, not production fur, smooth organic deformation, cloth, photorealistic reconstruction or guaranteed manufacturing topology. New unsupported requests need a concrete unsupported result or separately authorized capability work.
- Reference prompts/model/version/license are unknown. Generated-media metadata is provenance, not independent rights verification. Visual review is sampled agent inspection, not human approval or every-frame playback.
- Immutable creation manifests may still say `needs_visual_review` or `frozen_importer_verified: false`; later hash-bound review/import evidence supplies the result. Do not rewrite old manifests.

## Consumer follow-up — separate from this migration

- [ ] Weft integration owner: verify the delivered model/mapping against actual frozen neutral and articulated simulation poses, including dragon attachment and full-body alignment. Acceptance requires target display/pose evidence; importer acceptance alone is insufficient.
- [ ] If its actual frame transforms differ, provide those transforms and recompute offsets into a new mapping version using [the exact equation](docs/weft-neutral-proposal.md). Do not alter Weft physics, actions, rewards or the 19-angle control space to fit the asset.
- [ ] Weft integration owner: verify selected-appearance replay/native export through its existing paths. This project neither implements nor claims those features.

These are explicit integration handoff items, not unfinished local asset generation. No authorization for external publication, new providers or spending is implied.

## Next safe action and exact commands

For a new character, invoke the repository skill with the requested description. Author a new scene (do not merely rename an existing subject), then use new output folders:

```sh
.venv/bin/python -m avatar_assets request --subject 'Blue workshop robot' --prompt 'A friendly blue workshop robot that handles a circuit board' --out outputs/request-next
.venv/bin/python -m avatar_assets build3d examples/scenes/humanoid-robot-weft-v1.json --out outputs/robot-next
.venv/bin/python -m avatar_assets reload3d outputs/robot-next --out outputs/robot-next-reload
.venv/bin/python -m avatar_assets export2d outputs/robot-next --preset muse --out outputs/robot-next-media
.venv/bin/python -m avatar_assets verify outputs/robot-next-media
```

The sample build command uses the already authored robot scene; `request` alone does not create a model. Review meaning/identity/attachments and record the result with `review` before claiming acceptance.

To reproduce a metadata derivation against the explicit proposal without changing geometry:

```sh
.venv/bin/python -m avatar_assets weft-metadata outputs/muse-panda-weft-build-v1 examples/weft/muse-panda-weft-v1-mapping-design.json --neutral examples/weft/neutral-proposal-v1.json --out outputs/weft-muse-panda-next
.venv/bin/python -m avatar_assets verify outputs/weft-muse-panda-next
```

That `verify` checks bundle hashes; the existing Weft importer owns consumer validation. Consumer inputs and critical node IDs are in `docs/weft-delivery-v1.md`.

Run relevant tests after changes, not repeatedly against unchanged inputs:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python /Users/ivan/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/avatar-asset-workflow
```

Git now belongs to the parent Weft repository. The user authorized committing and pushing this migration directly to main. Do not recreate a nested repository, discard failed attempts or open files automatically. See [migration record](docs/relocation.md) for tracked versus local-only artifacts.

Completion gate: reusable local workflow and two distinct real examples work end to end; 3D loading/skin/motion and explicit consumer mapping are verified at the stated scope; references, failures, costs and unsupported cases remain truthful and resumable. The recorded evidence satisfies this local gate without claiming completed Weft runtime integration.
