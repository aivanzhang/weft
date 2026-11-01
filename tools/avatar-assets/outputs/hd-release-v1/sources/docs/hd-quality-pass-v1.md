# HD fur and surface follow-up

## Requested result

The user clarified: “you're working on making these avatars better right? I want the same textures and hd quality as the muse ones for example fura and hari”. Visual fidelity is the active task. Runtime alignment work is deferred; no physics or simulator changes are part of this pass.

The existing 1600-pixel portrait has sparse, wide triangular bristles, simple eyes and an angular dragon. Higher pixel dimensions did not establish Muse parity. Preserve it as comparison evidence, along with every existing output and the unchanged source archive.

## Bounded implementation

- General scene-v2 extensions: native 1024/2048 procedural surface maps, radial iris texture, curved skinned alpha-masked fur cards, optional native editable hair and grounded studio lighting. Existing recipes retain their original algorithms/settings.
- Two distinct representations: browser GLB uses cutout ribbons with multiple fine fibers in their atlas; Blender source uses native hair, with the browser ribbons hidden for native rendering. Native hair is **not** exported into GLB. Neither representation is a recovered Muse groom or physical hair simulation.
- Texture budget: 32 Mi pixels across all basecolor/normal maps and card atlases. Maximum texture edge2048; fur-card atlas1024. General GLB remains capped at100,000 exported vertices and64 MiB; native hair at240,000 configured parent-plus-child fibers. The frozen20k-vertex diffuse-only Weft contract is unchanged.
- Local-only work: provider calls/spend0/$0, no recurring service, credentials, uploads or paid generation. Native jobs use at most two workers/four CPU threads each,2 GiB sampled RSS,512 MiB output. Look-development builds use a single portrait, up to600s; independent reloads180s; media1200s. At most four character look-development revisions before reassessing a repeated failure. Preserve rejected revisions instead of replacing files.
- Actual limits are sampled and allow the documented seven-second bounded termination cleanup; these are not instantaneous OS allocation caps. Renderer settings and actual usage live in each job manifest.

## Final model candidates

| Subject | Editable scene recipe | Model/source bundle | Exported vertices / triangles / joints |
| --- | --- | --- | --- |
| Panda + shoulder dragon | `muse-panda-hd-v4.json` | `muse-panda-hd-build-v3` |76,851 /79,180 /22 |
| Fox craftsperson | `fox-hd-v3.json` | `fox-hd-build-v2` |60,418 /67,776 /20 |
| Workshop robot | `workshop-robot-hd-v1.json` | `workshop-robot-hd-build-v2` |47,644 /38,058 /17 |
| Articulated lamp | `lamp-hd-v1.json` | `lamp-hd-build-v2` |37,762 /44,754 /8 |

Recipes are under `examples/scenes/`; bundles are under `outputs/`. Every model has four real skeletal clips. The panda has7,688 curved browser cards and238,328 configured native fibers; the fox has5,000 cards and155,000 native fibers. These are separate representations, not interchangeable hair counts. Mechanical objects have no animal fur.

The panda now has a curved muzzle/smile, larger convex radial-UV eyes, furred inner ears/crown/paws, a darker charcoal coat, and a more organic dragon with raised scales, shaped horns and wing membranes. Native hair curves preserve roots and strand arc lengths and follow the existing emitter skin. The dragon remains on the viewer-left/anatomical-right shoulder helper; no newly recovered anatomy or finger control is implied.

## Reproduce locally

From `tools/avatar-assets`, use fresh output names:

```sh
.venv/bin/python scripts/author_combed_groom.py --source examples/scenes/muse-panda-hd-v3.json --id muse-panda-hd-v4 --out examples/scenes/muse-panda-hd-v4.json --check
.venv/bin/python -m avatar_assets build3d examples/scenes/muse-panda-hd-v4.json --out outputs/panda-hd-new --resolution 384 --samples 32 --portrait-only --timeout 600
.venv/bin/python -m avatar_assets reload3d outputs/panda-hd-new --out outputs/panda-hd-new-reload
.venv/bin/python -m avatar_assets export2d outputs/panda-hd-new --out outputs/panda-hd-new-media --preset preview --size 720 --portrait 1600 --seconds 2 --samples 8 --denoise-portrait
.venv/bin/python -m avatar_assets verify outputs/panda-hd-new-media
```

Substitute any of the other validated scene recipes. `--portrait-only` skips sampled pose renders during model construction; it does not omit skeletal clips. Optional portrait denoising runs after Blender exits, preserves `portrait-raw.png` and its exact processing receipt, and can soften fine hair. The videos use the native groom; the interactive GLB uses cards. Rebuild the GLB from the scene JSON, since the editable native-render `.blend` hides its browser-card materials.

The four-second authored actions are retimed in full into short preview videos, not truncated or repeated stills. The selected final target is1600-square portraits plus four720-square/24fps/two-second previews (robot three seconds). This is not the supplied eight-second Muse timing. A measured720p native-panda frame takes about3.6s on this host; an eight-second/four-state export is not promised within the1200s job bound.

## Technical evidence

- `outputs/hd-card-probe-v1`: analytic two-part fixture, not a generated character.2000 curved ribbons, embedded MASK texture, real inherited skinning, valid normals/tangents. Native job3.088s,591,560,704B sampled RSS,4,622,914B output. At512px this fixture has sparse/subpixel hairs; it does not establish character visual quality.
- `.local/native-groom-probe-v1` (relative to the Weft root): native hair API and skin-deformation proof.48,000 configured fiber upper bound; copied weights, sampled roots and native reload verified. The initial probe's radius setting was a diameter multiplier; the helper now maps a physical radius explicitly. This analytic proof is not Muse parity.
- `tests/test_surfaces_hd.py`: native-resolution texture checks, deterministic seeds, normal/range/invalid-input tests and resource probes. Procedural maps are authored textures, not neural output or reference-extracted textures.
- Deferred integration edits remain uncommitted and separate: `avatar_assets/weft_metadata.py` and five analytic alignment tests; no real captured-frame calibration was produced. The unrelated replay compound-equipment fix remains in the Weft worktree. Neither is a completion claim for this visual task.

- `outputs/hd-final-attribute-proof-v1` and `hd-final-attribute-proof-v2`: final GLB skins, motions, normals, UVs, tangents, exact embedded texture hashes, alpha-mask cards and diffuse/PBR settings passed. New HD assets exceed the frozen Weft profile; old Weft bundles/mappings stay unchanged.
- `outputs/hd-quality-verification-v1/python-tests.txt`:209 tests passed in8.191s, including invalid inputs, size limits, denoising, native comb controls and monitor cleanup. The gallery HTTP checks require local loopback access; earlier sandbox-only permission failures are preserved, not treated as code regressions.
- Weft `.local/native-comb-probe-v1/attempt-3`: analytic native-key persistence, unchanged roots/weights, arc-length error≤2.11e-7m, skeletal bend and reload passed. Configured hairs are not simulated dynamics. First two failed probe attempts are retained.
- `outputs/hd-render-profile-v1`: eight actual720p render samples,30.125s/1,668,612,096B sampled peak RSS. Persistent cache changed median timing by only about3%; paired pixels were identical. No meaningful optimization claim or cache-setting change was promoted.
- `outputs/hd-runner-incident-v1`: preserves a completed native panda render whose old outer monitor lost its result with EPERM. Its exit/RSS cannot be reconstructed; it is not the final accepted pack. The repaired runner checkpoints measurements and records bounded cleanup failures without fabricating success.

## Completion checks

- [x] Inspect new panda native portrait against Muse and the preserved v6 portrait; inspect the independent GLB reload separately.
- [x] Verify new fox and appropriate material-only robot/lamp editions, with no fur added to mechanical objects.
- [x] Verify all real skeletal clips, native groom attachment, texture/cutout export, static fallbacks and matching source hashes.
- [x] Keep gallery descriptions honest about native-render versus interactive-GLB appearance; preserve old catalogs.
- [x] Record measured job limits, exact recipes, hashes, failures, remaining fidelity limits and runnable commands in README/TODO.

## Final verification and limits

All final jobs exited successfully; no task-owned renderer or server remains running. Human-readable times below are measured wall time, not service guarantees.

| Native media export | Seconds | Sampled peak direct-child RSS |
| --- | ---: | ---: |
| Panda |967.158 |1,647,443,968 B |
| Fox |800.671 |1,422,262,272 B |
| Robot |579.188 |1,098,252,288 B |
| Lamp |407.949 |1,143,980,032 B |

Packaging took4.177–6.451s with peaks below338MB. Every portrait is1600 square; all16 clips are silent H.264,720 square at24fps. Panda/fox/lamp have48 frames per clip; robot72. Every decoded thumbnail is distinct. Fallback PNGs are byte-identical to the first native frame. Endpoints match exactly, but seamless playback and velocity continuity remain uncertified. The source ZIP hash remains unchanged.

`outputs/hd-media-verification-v1` checks all four full packs and their portable subsets, hashes, dimensions/timing declarations, raw/filter provenance and fallback equality. It reuses the hash-bound packaging worker's full-frame decoding instead of decoding unchanged videos again. Runtime0.986s, process peak42,647,552B. All four final scene recipes reproduce exact bytes in `hd-recipe-reproducibility-v1`.

`outputs/hd-gallery-browser-v1/browser-test.json` passed15 grouped checks and all16 skeletal clip selections with zero external page requests. Chrome156.0.8078.76/software WebGL: cold ready4.615s; warm selections140–395ms. A two-second panda playback observation advanced2.1165 clip seconds, with36 animation callbacks and83.3ms maximum callback gap. Callbacks are not measured rendered frames; this is not a sustained FPS, GPU-memory or hardware-independent guarantee. Reduced motion, failed-model static fallback, mobile layout and injected stale/slow-load cases passed. The temporary browser/server were closed.

Default catalog: `browser/gallery.json`, frozen as `gallery-hd-v1.json`. Previous default: `gallery-quality-v2-portable.json`. The selected portable media preserve every retained byte, including raw portraits and processing receipts; full native frames/logs remain in their original local folders. Run `./tools/avatar-assets/run gallery` from Weft and open http://127.0.0.1:8765/#panda yourself. Restart an existing gallery to load the new catalog.

[Sampled visual review](../outputs/hd-visual-review-v1/review.json) records visible improvements and remaining limits. Native hair is denser and directional, but portrait filtering softens fibers. Browser card fur is much less prominent and can look smooth at default size. Sculpting, eye shading, fur breakup and dragon detail still differ from Muse. Explicit results: `outputs/hd-muse-fidelity-unsupported-v1` and `outputs/hd-weft-profile-unsupported-v1`.

The final [release index](../evidence/hd-release-v1.json) binds source snapshots, selected model/media/proof hashes, exact prompts, rejected revisions and costs. Provider calls/spend remain0/$0; local CPU/storage only. This follow-up is local and uncommitted/unpushed. Existing migration commits, source archive, prior packs and unrelated Weft work remain unchanged. No physical accuracy, new Weft alignment, automatic neural text-to-3D generation or Muse-equivalence claim is made.
