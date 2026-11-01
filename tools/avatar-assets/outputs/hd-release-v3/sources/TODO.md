# Avatar assets handoff

## Session and ownership

- Workspace: `/Users/ivan/kintsugi/weft/tools/avatar-assets`, migrated into Weft by explicit user request. Preserve unrelated Weft work and its active goal.
- Thread: `01a18ebe-a17c-7cc0-9117-06ca476fb31c`; title: Integrate avatar assets (core avatar goal remains completed).
- Recorded: 2026-10-30 19:37:36 PDT.
- Core workflow, gallery and detailed-model workflow upgrade are verified. Muse-quality parity remains an explicit unsupported fidelity target. Weft’s goal is untouched.
- User clarification preserved: complete controllable 3D from text is primary. The language step is agent-assisted scene authoring, not an unattended neural generator. Keep the portrait/four-state workflow too.
- User authorized defining a neutral pose. It is an authored proposal, not an observed Weft pose or permission to change its physics.
- Provider generation calls/spend: **0 / $0**; no recurring service, deployment, new remote repository, account or borrowed credentials/budget. The user subsequently authorized a push to Weft’s existing private remote. A proposed four-call raster batch was not executed.
- Resume guard: preserve original ZIP and existing output folders. Recheck job handles below before resuming or terminating anything. No persistent background service was created; gallery test servers were temporary.

## Current follow-up: HD fur and surface upgrade — verified

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
