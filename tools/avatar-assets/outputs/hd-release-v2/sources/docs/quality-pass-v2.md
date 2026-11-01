# Quality pass v2

Started 2026-10-30 16:33 PDT. The user rejected the gap between the simple articulated examples and the polished Muse reference. This pass improves actual exported geometry, materials and deformation; it does not replace a rig with a generated picture.

## Target and constraints

- Panda: compact proportions, broad cheeks, expressive inset eyes, a shaped muzzle and smile, continuous weighted limbs, short exported fur and a more complete shoulder dragon.
- Robot: designed shell forms, beveled edges, layered joints and panels, contrasting painted/metal surfaces and detailed tools.
- Fox: continuous limbs, more expressive face, shaped ears/tail, short fur and better scarf/book details.
- Lamp: a hollow shade, machined joints, beveled metal surfaces and modeled wiring.
- Keep four real skeletal actions, editable sources, static portraits and a browser preview. Inspect full models, not just front renders.
- The reference contains no recoverable rig or unseen surfaces. New geometry remains authored interpretation. Do not claim Muse-quality parity without visual evidence or user acceptance.
- Detailed browser models use the general PBR profile. They are not automatically Weft-compatible: its diffuse-only/material/vertex limits remain unchanged. Preserve existing Weft deliveries unchanged.

## Work and spending envelope

No paid generation or remote uploads. Provider calls: 0; provider spend: $0; recurring service: none. Plugin discovery found no appropriate connected 3D-generation capability. Image generation was not used as a substitute for geometry.

Initial local work envelope: at most five build attempts (600 seconds each), four independent reload checks (180 seconds each), and four bounded preview media jobs (up to 1200 seconds each). At most two native jobs concurrently, four CPU threads and 2 GiB sampled RSS per job; existing 512 MiB output cap. Reassess failures before expanding this envelope. Local electricity/wear is not priced; these are runtime ceilings, not measured costs or performance promises.

Reassessment 2026-10-30 16:49 PDT: five builds completed within bounds. Allow two further local build attempts, with the same per-job limits, to fix observed panda celebration occlusion and fox shoulder caps. No new provider cost or recurring service. Rejected iterations remain preserved; no unchanged failed job is retried.

The wider panda arm pose cleared its face, but exposed a different cause of the hidden orb: the wrist rotated the prop behind the paw. One additional bounded build corrects wrist orientation analytically at every authored key (total envelope eight builds); it does not repeat the failed arm-position hypothesis. Provider spend remains $0.

Export validation then found invalid normal/tangent frames on a small number of collapsed or cancelling-UV caps. Three bounded no-render repair probes tested real MikkTSpace frames; the corrected full robot and lamp passed in `outputs/quality-attribute-proof-v1/uv-repair-probe-v3`. The exporter now preserves valid seams, removes microscopic zero-area remnants and reprojects only invalid UV faces. A separate bounded surface-attribute validator is an automatic v2 build/verify gate. Final rebuilds are required for all four characters; total local envelope becomes11 builds,6 native reloads and6 preview exports, with unchanged per-job bounds and no paid work. Earlier affected models/media remain preserved as rejected or superseded evidence.

## Final results

The local quality workflow upgrade is implemented and verified. It is **not a demonstrated match for Muse's cinematic finish**. The panda/fox have short geometric fur, not a dense film groom; facial and dragon sculpting remain stylized. `outputs/muse-fidelity-limit-v1/unsupported.json` records this explicit fidelity limit. No original 3D source was recovered.

| Final build folder under `outputs/` | Vertices | Triangles | Joints | Authored fur blades | Embedded PNGs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `muse-panda-quality-build-v6` | 85,952 | 68,788 | 22 | 18,120 | 12 |
| `workshop-robot-quality-build-v3` | 47,644 | 38,058 | 17 | 0 | 4 |
| `fox-quality-build-v4` | 68,518 | 60,476 | 20 | 12,700 | 8 |
| `lamp-quality-build-v4` | 37,762 | 44,754 | 8 | 0 | 2 |

All have four actual skeletal clips, real skin weights and complete authored side/back surfaces. The generic scene compiler is not hard-coded to these subjects. `sources/scene.json`, prompt, backend scripts, textures and `.blend` remain alongside each immutable GLB.

Evidence:

- Exported attributes, actual weights/motion and embedded texture hashes: `outputs/quality-attribute-proof-v1/manifest.json`. Final checks2.8624s, max122,814,464B process peakRSS. Exact fur counts come from hash-bound backend reports; independent narrow-triangle counts also include some non-fur detail and UV splits, so they are not mislabeled as exact hair counts.
- Native loading and motion: `outputs/muse-panda-quality-reload-v6`, `workshop-robot-quality-reload-v3`, `fox-quality-reload-v5`, `lamp-quality-reload-v4`. All passed. Front/back/side/contact review: `outputs/quality-visual-review-v1/review.json`; reviewer is the assistant, not user approval.
- Matching media: `outputs/quality-media-verification-v1/manifest.json`, four 415-artifact packs. All 16 clips are silent H.264,320×320,96 frames/24 fps/four seconds,95–96 distinct decoded thumbnails each. Every static fallback exactly matches its first native frame. All media are opaque. Endpoint image error is0, but seamless loops remain uncertified.
- Panda/fox portraits are1600×1600; robot/lamp 640×640. These are preview-video editions, not new720p/eight-second exports. Larger detailed-model renders remain unmeasured and may hit the existing1200s/512MiB cap; do not treat old simple-model full-render timings as guarantees.
- Final builds34.086–60.038s, max802,848,768B sampled native RSS. Final media renders134.682–190.796s, max701,284,352B; packaging2.099–3.618s. Two native jobs at most, four CPU threads each.
- 127-test full suite passed in1.991s (`outputs/quality-python-tests-v3.txt`). The later high-resolution portrait route passed all22 affected gallery tests (`outputs/quality-gallery-tests-v4.txt`). Skill schema validation passed; recipe `--check` commands reproduce all four scene descriptions exactly, not bit-identical Blender containers.
- Final real browser check: `outputs/quality-gallery-browser-v2/browser-test.json`,15 grouped checks and all16 native clip selections; Chrome156.0.8078.76/software WebGL, fresh profile, zero external page requests. Cold ready4.407s; warm selections99–269ms. Largest model's two-second playback advanced2.15clip seconds;37 browser RAF callbacks, max83.3ms gap,11.42MB JS heap. These are responsiveness samples, not measured rendered FPS, whole-browser/GPU memory or sustained-use guarantees.
- High-resolution portrait loading is lazy; thumbnails remain512px. Reduced-motion mode starts with a static portrait and no GLB request. One active viewer/cache entry, no persistent test server or user browser left running.

## Preserved failures and limits

- `quality-panda-iteration-v2`, `-v3`, `-v4`: sparse/corrugated fur and flat shoulder caps, then prop occlusion. Corrected in new versions; none overwritten.
- `fox-quality-iteration-v3`: flat shoulder roots and insufficiently smooth head/elbows; v4 preserves the prior source and fixes those forms.
- `quality-backend-probe-v1`: rejected analytic texture-color encoding. Corrected v2 proof is explicitly analytic, not a completed character.
- Attribute proof subfolders preserve invalid tangent cases and repair probes. Final gates verify real native tangent frames rather than inventing normals or dropping maps.
- `quality-gallery-browser-v1`: failed because `connect-src` blocked in-memory GLB texture fetches. Final policy adds only `blob:` to same-origin connections; remote HTTP remains disallowed. v2 passed with textures present.
- `quality-runner-incident-v1`: one fox native reload passed but its outer CLI returned an OS permission error before completion records were saved. Files remained readable; a new bounded reload passed. The transient cause is unconfirmed. No silent completion record was fabricated.
- These detailed editions exceed Weft's vertex/material profile; existing v1 matte models and mappings remain unchanged. No Weft import, alignment, renderer or physics changes are part of this pass.

Provider calls/spend remain0/$0. Native render timings exclude local electricity/wear, and memory caps are sampled, not an instantaneous OS allocation limit. Prior outputs and source archive remain unchanged.

## Commands

```sh
.venv/bin/python -m avatar_assets gallery
# In your browser: http://127.0.0.1:8765/
.venv/bin/python -m avatar_assets gallery --catalog browser/gallery-basic-v1.json --port 8766
.venv/bin/python scripts/author_quality_panda.py --check
.venv/bin/python scripts/author_quality_robot.py --check
.venv/bin/python scripts/author_quality_companions.py --only fox --check
.venv/bin/python scripts/author_quality_companions.py --only lamp --check
.venv/bin/python -m avatar_assets build3d examples/scenes/muse-panda-quality-v5.json --out outputs/panda-next --resolution 512 --samples 24 --timeout 600
.venv/bin/python -m avatar_assets reload3d outputs/panda-next --out outputs/panda-next-reload
.venv/bin/python -m avatar_assets export2d outputs/panda-next --preset preview --portrait 1600 --out outputs/panda-next-media
```

Use new output folders. Restart an existing gallery to load the changed catalog; do not start/stop unrelated servers. The next fidelity step is an explicitly scoped denser groom/sculpt and finer face/dragon work, or an authorized external 3D source with a cost estimate—not silently declaring these models equivalent to the Muse reference.
