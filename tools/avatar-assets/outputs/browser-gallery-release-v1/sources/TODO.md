# Avatar assets handoff

## Session and ownership

- Workspace: `/Users/ivan/kintsugi/avatar-assets` only. Never edit Weft or replace its goal.
- Thread: `01a18ebe-a17c-7cc0-9117-06ca476fb31c`; title: Build browser asset gallery (core avatar goal remains completed).
- Recorded: 2026-10-30 16:21:45 PDT.
- Core avatar workflow is complete. The subsequent browser-gallery feature is also implemented and verified; neither replaces Weft’s goal.
- User clarification preserved: complete controllable 3D from text is primary. The language step is agent-assisted scene authoring, not an unattended neural generator. Keep the portrait/four-state workflow too.
- User authorized defining a neutral pose. It is an authored proposal, not an observed Weft pose or permission to change its physics.
- Provider generation calls/spend: **0 / $0**; no recurring service, deployment, remote repository, account or borrowed credentials/budget. A proposed four-call raster batch was not executed.
- Resume guard: preserve original ZIP and existing output folders. Current recorded build/render/reload/packaging jobs finished; do not resume or terminate stale PIDs without rechecking ownership. No persistent background service was created; gallery test servers were temporary.

## Browser gallery follow-up — completed

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

## Consumer follow-up — outside this workspace's ownership

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

Local Git exists with no commits or remote. Sources, examples, skills, docs and evidence are untracked pending a separately requested commit; `.tools`, `.venv` and outputs are ignored. Do not push, create a remote, delete failed attempts or automatically open files.

Completion gate: reusable local workflow and two distinct real examples work end to end; 3D loading/skin/motion and explicit consumer mapping are verified at the stated scope; references, failures, costs and unsupported cases remain truthful and resumable. The recorded evidence satisfies this local gate without claiming completed Weft runtime integration.
