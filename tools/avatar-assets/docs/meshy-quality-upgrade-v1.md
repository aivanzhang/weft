# Meshy quality upgrade — local refinement

Historical dragon/local-refinement record. The later authorized panda generation,
fur controls and current acceptance status are in [fur workflow](fur-workflow.md)
and the top of [TODO](../TODO.md); the older pending request below is superseded.

Started October 31, 2026, 11:13 PM PDT. Baseline source, editable models,
portable media and verification are saved in Weft commit
`050deb54933d052abcca1b786dabdd00ad5aaa53`, with the private remote verified in
`.local/avatar-assets-publication-20261031-v1/receipt.json` at the Weft root.
Old outputs and the original Muse archive remain immutable.

## Scope and cost

Improve the reusable finishing stage and produce new reviewable derivatives of
the existing Meshy panda and dragon. The panda remains the demanding face/fur
reference; the dragon checks that the changes are not panda-specific. Do not
claim Muse parity from resolution, fiber counts or encoding checks.

Initial work uses local comparisons and existing geometry: no new Meshy or Meta
calls, no new paid rendering service. The canonical Meshy ledger still records
15 calls and 167 credits; the previous 35-credit dragon grant is already spent.
New paid synthesis requires a prepared estimate within actual authorization.

At most two native jobs at once, each four CPU threads and 2 GiB sampled child
RSS. Diagnostic renders: 300 seconds and 256 MiB output per job. Finished model
builds retain the existing 600-second / 512-MiB output bounds. Start with paired
640px stills; do not buy new geometry or render full videos to diagnose shading.

## Ownership and decision sequence

- Main: finishing integration, generic AI authoring contract, new recipes,
  final artifact verification and Git checkpoint. Do not touch active camera
  reconstruction files outside `tools/avatar-assets`.
- `agent_runner`: reusable safe local look-development comparison and panda
  baseline in `outputs/meshy-lookdev-panda-v1`.
- `pipeline_contracts`: bounded source-preserving material-response profiles
  and focused tests, in three new isolated files.
- `pipeline_review`: independent read-only visual/code diagnosis.

Compare native fur, base surface alone, portable cards and base surface with
normal/specular effects disabled using the same saved camera and lighting.
Choose changes from those observations, not from the assumption that extra
render samples or more fur automatically improve the model.

## Current status

The paired diagnostics are complete. Panda cheek defects remain in the base
mesh even with uniform gray materials and no fur/normal maps. Existing raw
source images show chunky coat relief; the isolated flecks appear clearly after
the earlier color-selected smoothing. This supports a geometry/mask problem,
not a diagnosis that more hair or samples will fix it.

- `outputs/meshy-lookdev-panda-v1/attempt-2`: five matched variants, 12.669s,
  about 1.58 GiB sampled peak. First attempt rejected an empty sequence-editor
  container before rendering; failure retained. Its manifest inventories the
  initial failure files, not the later nested retry.
- `outputs/meshy-lookdev-panda-clay-v1`: one uniform-gray view, 3.644s, about
  1,061 MiB sampled peak; triangular defects remain.
- `outputs/meshy-lookdev-dragon-v1`: two matched views, 3.876s, about 981 MiB
  sampled peak; coarse geometry remains when sheen/normal maps are disabled.
  Source has 1,877 flat-shaded faces among 36,501 faces.

All source pack hashes remain unchanged. No provider call was made.
`outputs/meshy-quality-upgrade-v1/diagnosis.json` binds initial observations;
`review-v1.json` records selection and failures. The selected local derivative
is **`outputs/agent-dragon-polish-v1`**, with independent native reload passed.
Its HD media render completed under handle60224. All owned jobs and temporary
test servers have exited. Do not start a duplicate or associate old media with
the new model.

| Candidate | Decision | Native finish | Sampled peak RSS |
| --- | --- | ---: | ---: |
| Panda polish-v1 | Rejected: cheek defects persist, belly coarser | 59.327s | 1.69GB |
| Panda polish-v2 | Rejected: smoother coat but damaged companion;89 invalid tangents | 60.465s | 2.03GB |
| Dragon polish-v1 | Selected local refinement; reload passed | 18.552s | 1.53GB |
| Dragon polish-v2 | Rejected: angular glossy forehead/iris patches | 17.820s | 1.33GB |

Dragon v1/v2 clay renders are pixel-identical with identical camera/light/pose:
the v2 regression comes from region-based material assignment. The original
dragon comparison has a tiny0.08742% camera-scale difference; do not call it
pixel-matched. V1 smooths the neck/belly/tail but does not create new scale detail.

Panda v2's stronger smoothing damages the small dragon.89 export tangents fail,
including81 on the coat; none of those81 is at a material-split boundary.
A UV-matched geometry comparison finds252 triangles below1% of their source area
(41 ambiguous matches excluded). This is not a proof of self-intersection, but
it is evidence against promoting the candidate or widening tangent fallback.
Prior accepted panda remains unchanged. No Muse-quality parity is claimed.

## Reusable controls and exact commands

From `tools/avatar-assets`:

```sh
# Inspect one variable at a time using the saved camera/light/pose.
.venv/bin/python -m avatar_assets.lookdev outputs/agent-dragon-polish-v1 \
  --out outputs/my-dragon-lookdev-001 --variants baseline_native,base_clay

# New output required. Source hashes, prompts and backend files are frozen.
.venv/bin/python -m avatar_assets.finish_generated \
  examples/meshy/agent-dragon-polish-v1.json --out outputs/my-dragon-refinement-001
.venv/bin/python -m avatar_assets reload3d outputs/my-dragon-refinement-001 \
  --out outputs/my-dragon-reload-001
```

- `surface_smoothing`: Taubin by default, or explicit shrinking Laplacian;
  bounded displacement/iterations with feathered spatial exclusions. Protects
  positions, not sharp normals; global smooth normals are authored. Do not use
  this path to promise preserved mechanical creases or unseen anatomy.
  An early triangle-area regression gate rejects new degeneracy or excessive
  collapse before fur/rendering; its JSON measurements reach the repair agent.
  Source defects remain disclosed, not silently certified as repaired.
- `surface_profiles`: up to8 named material regions selected by normalized
  ellipsoids and sampled source color. Copies material response, not texture
  pixels; last matching profile wins. Face boundaries are hard, so review eye
  regions carefully. Unknown/multi-image/custom UV routes fail explicitly.
- `groom_shader`: independently controls native-hair roughness and radial
  roughness; does not add strand count or alter portable-card shading.
- `lookdev`: six bounded variants isolate native hair, cards, base surface,
  normal/specular effects and uniform clay. A diagnostic is not a new asset.

The AI authoring contract exposes these controls without allowing an agent to
override the subject, source, state list, execution limits or paid budget.
Existing completed agent jobs remain bound to their original source snapshots;
do not mutate their manifests or resume them under changed core code.

## Verification and preservation

- Final suite:368 tests passed in10.358s, `outputs/quality-upgrade-tests-v3.txt`.
  The first run had8 loopback socket permission failures; the24 gallery tests
  passed outside that sandbox restriction, and the final full run passed.
  These tests include analytic fixtures; they are not generated artwork.
- Real native regression check: `outputs/surface-integrity-native-v2`,2.084s,
  about459MB sampled peak. Dragon v1 has0 newly degenerate/near-collapsed faces;
  panda v2 is rejected before rendering for252 severely collapsed faces. The
  gate is conservative, not a self-intersection, identity or quality proof.
- `texture-source-proof.json` verifies the selected dragon's three embedded
  image payloads are byte-identical to the original generated model, and the
  original Muse ZIP still matches its supplied SHA256. No texture-resolution
  improvement is claimed.
- Full failed outputs remain local and unchanged. The tracked
  `outputs/meshy-quality-review-v1` is explicitly a review-only subset: exact
  recipes, backend snapshots, source inventories, reports/logs and portraits.
  It is not a loadable model pack. Selected model/source and diagnostic packs
  are saved separately; full video frames stay local, compact media is portable.

## Delivered dragon and browser comparison

- Model/source: `outputs/agent-dragon-polish-v1`; 36,322 exported vertices,
  36,501 triangles,9 joints. Mesh node9, armature10, root bone8; complete IDs,
  bind/rest matrices and skin order are in `exported-nodes.json` and the rig
  contract. Meters, +Y up/+Z forward,1.6m authored rest height. No Weft mapping
  is supplied: the mesh/PBR materials exceed that consumer's frozen profile.
- New full media: `outputs/agent-dragon-polish-media-v1`. Render522.509s,
  1,176,043,520B sampled peak RSS,217,199,174B output at native exit.
  1600×1600 portrait, four720×720 silent H.264 clips,24fps/4s/96 distinct
  decoded thumbnails each. Static PNG fallbacks match source frame0. Endpoints
  match, but seamless loops remain uncertified; all outputs are opaque.
- Portable subset: `outputs/agent-dragon-polish-media-v1-portable-v1`:
  40 artifacts /10,087,576B, byte-preserving. Full frames remain local.
  Portrait and all four contact sheets were visually reviewed: gentle idle,
  head turns, small wing motion and bow/wing/tail celebration. This is not
  flight, grasping, finger control or locomotion.
- Browser: `outputs/browser-polish-v2` passed13 grouped checks, all8 before/after
  skeletal clips, video playback/seek, mobile, static/reduced-motion and loading
  failure handling. Initial ready4.994s; warm selection61–384ms; no external
  requests/unexpected console errors. Two-second software-rendered playback
  remained controllable; this is not a sustained FPS guarantee.
- First browser run exercised playback but the full test required a2D-only
  reference entry. Its report and original catalog are retained; the corrected
  before/after catalog includes the original Muse2D pack and passed. No gallery
  application code changed. All temporary browser/server processes exited.

Exact media and preview commands from `tools/avatar-assets` (new output names
required when repeating):

```sh
.venv/bin/python -m avatar_assets export2d outputs/agent-dragon-polish-v1 \
  --out outputs/agent-dragon-polish-media-v1 --preset preview \
  --size 720 --portrait 1600 --seconds 4 --samples 8 --denoise-portrait
.venv/bin/python -m scripts.package_portable_media \
  outputs/agent-dragon-polish-media-v1 --out outputs/agent-dragon-polish-media-v1-portable-v1
.venv/bin/python -m avatar_assets gallery --catalog browser/gallery-polish-v1.json
```

The final hash index is `evidence/meshy-quality-upgrade-v2.json`. Model/reload/
portable inventories independently rechecked114 artifacts. Original reference
ZIP, source models and previous packs are unchanged. Git checkpoint/remote
verification is recorded outside the commit in the Weft-root
`.local/avatar-quality-publication-v1/receipt.json`; do not infer a push from
the presence of local files. No active job, recurring service or deployment.

## Pending paid branch — not authorized or executed

The account ledger remains15 calls /167 credits, with8 credits available under
the saved allowance. A question for35 additional credits is pending: one new
panda base, no paid retry or credit purchase. No grant is recorded.

The prepared `outputs/muse-panda-smooth-base-v1` is **do not submit**. Review
found that image-mode `texture_prompt` changes texture guidance, not geometry;
the same furry image cannot reliably request a clean undercoat through text.
The next paid branch requires a separately authored/reviewed smooth-surface
reference guide and a new frozen plan. Original reference bytes stay intact.
Meshy7.1 plus4K geometry/4K textures is estimated at35 credits; the account's
dollar-per-credit cost is unknown. Do not spend without the pending approval.
[Official image API](https://docs.meshy.ai/en/api/image-to-3d),
[official pricing](https://docs.meshy.ai/en/api/pricing).

Final code review caught float32 storage rounding after a double-precision check.
The gate now checks the exact stored coordinates and uses them for normals. A
regression fixture proves a face that collapses only after rounding is rejected
before mutation. Native v2 recheck passes dragon/ rejects panda as expected;
previous proof and367-test run remain unchanged. See delivery-v2.json for the
final verification records; delivery-v1.json is the earlier checkpoint.
