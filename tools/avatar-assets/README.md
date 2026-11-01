# Avatar assets

Local workflow for editable, rigged 3D avatars and their 2D portrait/animation exports, now maintained in Weft under `tools/avatar-assets/`. Authoring stays separate from Weft physics, actions, rewards and provider budget. See [relocation and preservation](docs/relocation.md).

## Prompt-driven AI pipeline

`generate` connects an AI planner, rig/motion author and independent visual reviewer to the existing generation and local build tools. It accepts one text prompt, optional reference/style/states, and an output preset. The coordinator enforces budgets, validates the generated configuration, preserves failed attempts, and exports a real skinned GLB, editable Blender source, portrait, videos, static fallbacks and browser catalog only after review passes. Unsupported requests return an explicit result.

```sh
# From Weft: prepare the exact request and cost estimate without calling a model/provider.
./tools/avatar-assets/run generate \
  --prompt 'A friendly emerald dragon with amber eyes, small wings and a long tail' \
  --states idle,look_around,wing_flap,celebrate \
  --preset preview --max-credits 35 --prepare-only \
  --out outputs/my-dragon-001

# Executes within both this cap and the existing authorized provider ledger.
./tools/avatar-assets/run generate --resume outputs/my-dragon-001 --execute
```

Fresh Meshy geometry/textures are estimated at35 credits; AI-agent account charges are separate and currently not exposed by the authenticated CLI. The previous dragon-specific allowance is spent: setting `--max-credits` does not create a new account allowance. `--reuse-model PATH --max-credits 0` tests new automatic rigging/motions on existing generated geometry with no new Meshy calls. Agent calls are still real model usage.

**Verified with real agents on two different generated sources:** the dragon completed after two automatic repairs; the robot completed after one. Both exported real skins, four motions, portrait/videos/static fallbacks and passed actual browser playback checks. The robot test inferred style and state IDs from the text alone. These runs reused preserved Meshy geometry; fresh geometry generation was not purchased again. Full results, rejected attempts and measured limits are in [the verification report](docs/prompt-pipeline-results-v1.md).

View both completed examples with `./tools/avatar-assets/run gallery --catalog browser/gallery-agent-v1.json`, then open `http://127.0.0.1:8765/` yourself. Dragon media is1600px/720px/4s; the bounded robot test uses512px/256px/2s. Both retain their original4K texture maps.

See [commands, roles, outputs and limits](docs/prompt-pipeline.md) and the [current handoff](TODO.md). Agents cannot execute tools or spend directly; local scripts own execution. This is not a guarantee of Muse-equivalent quality, arbitrary interactions, hair physics or Weft compatibility. This release checkpoint includes source, selected editable models, compact media and verification evidence; full frame sequences and rejected binary intermediates remain local.

## Current scope

**Delivery target remains open:** a Muse-quality, controllable 3D avatar with convincing fur, matching proportions and expression, the shoulder dragon, meaningful actions and verified browser/Weft variants. The current models are prototypes below that visual target. A working compiler or a provider recommendation is not completion; a generation service would supply inputs to the workflow, not replace the remaining authoring and verification.

There are now **two implemented generation routes**:

- **Meshy7.1 text/reference-to-3D:** real paid generation using the user-designated project key, an independent non-resetting credit ledger, resumable tasks, original GLB/FBX downloads and local validation. [Repeatable commands, costs and limits](docs/meshy-workflow.md).
- **Local scene authoring:** agent-written JSON creates geometry/materials/skins/actions without paid generation. This remains useful for articulated objects, precise controls and repairs.

The new Meshy panda is closer in shape/identity than the procedural prototype, but **Muse-quality parity is not established**. Its auto-weights incorrectly assigned belly geometry to a hand; generated motions visibly distorted it and are retained as rejected candidates. The finishing path supports explicit spatial weight corrections, accessory anchors, local replacement actions, native hair and separate browser fur cards. These are authored repairs, not secretly relabeled provider animation or recovered anatomy. Text-to-3D is automatic; final visual acceptance is not.

The independently text-generated blue/copper robot exercises the same finishing workflow. Earlier fox/lamp/HD models and all failed attempts remain available. Native hair and portable cutout cards have different appearance. No cloth/hair physics, recovered fingers or unchanged-fidelity Weft HD conversion is claimed. The original reference and hidden surfaces cannot establish the original 3D mesh or license.

**Standalone dragon:** Meshy generated the emerald mesh and scale textures; the local workflow authored its19-bone creature skeleton, skin weights and four motions: idle, look around, wing flap and celebration. This adds a reusable non-humanoid finishing route, not a universal automatic rig detector. [Dragon files, exact commands and limitations](docs/dragon-delivery-v1.md).

### What our code contributes

| Stage | Responsibility |
| --- | --- |
| Base shape and texture synthesis | Meshy, for the neural-generation route |
| Prompts, selected references, budgets and resumable jobs | Local workflow |
| Creature bones/weights, repairs, accessories, fur and explicit motions | AI author proposes subject-specific configuration; local tools validate/build |
| Structural checks, editable sources, renders and browser playback | Local workflow |
| Sampled visual review and bounded repair decisions | Separate AI reviewer; not human approval |

The procedural scene route creates geometry without Meshy, but has not matched the Muse target. Neural base generation currently uses Meshy only; another provider adapter is not implemented. Reuse of the downstream workflow is supported, while hands-off production quality for arbitrary prompts is not established.

The earlier [platform research](docs/generation-platform-research-2026-10-31.md) predates this authorized Meshy trial; it is not a report of tested alternatives. Current generation/results and costs supersede its previous “no backend connected” status.

## Migration verification

The relocated workflow passes 144 tests, both gallery catalogs, all four detailed browser models and the two legacy Weft importer checks. The complete local archive remains under this tool; selected portable assets are versioned. [Migration receipt](evidence/weft-migration-v1.json).

## Local setup

Python 3.14 on Apple silicon was used for this workspace. Dependencies are isolated in `.venv`. The lock contains exact hashes for the downloaded macOS wheels; other platforms require their own reviewed wheel hashes.

```sh
# From the Weft repository root:
cd tools/avatar-assets
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Blender 4.5.13 is installed only under `.tools/Blender.app`; use an explicit Blender executable on another machine. Official download SHA256: `663ce944257c61ff1d6aa09e15c8f57bbd8d59023adb2fa7edde33a9ed960b53`. No global app install or subscription. Local runtimes and intermediate outputs are ignored by Git. Selected complete model packs, portable media, source/configuration, tests and evidence records are tracked; see `.gitignore` for the explicit release allowlist.

The default, agent, HD, Meshy v1/v2/v3 and portable-quality catalogs include their required assets in Git. Historical `gallery-fur-comparison-v1/v2/v3.json`, `gallery-meshy-candidates-v1.json` and `gallery-quality-v2.json` depend on local-only experimental or full-frame packs; they are not fresh-checkout demos. Fur-comparison v2 also fails the current rig-facts validator. Use the current catalogs or `gallery-quality-v2-portable.json`; these historical records are retained without relabeling them as valid releases.

## View everything in the browser

**Dragon plus the previous generated candidates:** from Weft, use `./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v3.json`. The dragon is first; the panda, robot and original reference remain available. Open `http://127.0.0.1:8765/` yourself. The gallery makes no paid calls and does not drive physics.

**New Meshy candidates:** real generated panda/robot meshes, four locally authored clips each, and the original reference for comparison. These are review candidates, **not accepted Muse-equivalent assets**. From Weft:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v2.json
```

Use `http://127.0.0.1:8765/`. No Blender is needed. Current local bundles: [panda](outputs/muse-panda-meshy-refined-v6/manifest.json), [robot](outputs/workshop-robot-meshy-refined-v2/manifest.json), [workflow](docs/meshy-workflow.md). Each has a matching 1600px portrait and four 720px/24fps/four-second clips, with static fallbacks. Crafting/working remain weak pantomimes, not accepted task interactions. Generation used **132 Meshy credits**; local finishing used no further provider calls. Dollar purchase cost is unknown. Viewing the gallery makes no paid call.

The local gallery defaults to the HD panda, robot, fox, lamp and original 2D reference. From the Weft root, use `./tools/avatar-assets/run gallery`. No Blender or rendering job is needed to view them. Selected editable HD bundles and compact media are included in this release checkpoint alongside the earlier detailed release.

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -m avatar_assets gallery
```

Open **http://127.0.0.1:8765/** yourself. Choose a model, orbit/zoom, switch animations, play/pause, scrub the timeline, or view existing videos and portraits. Playback starts paused; reduced-motion settings default to a portrait. Ctrl+C stops the server. Nothing is uploaded by the gallery; it does not drive Weft physics.

Dependencies are already installed here. See [browser setup, controls and adding assets](docs/browser-gallery.md) for a fresh checkout or custom collection. The gallery is a read-only preview, not Weft physics or alignment verification.

If the gallery was already running, stop it with Ctrl+C and restart it to load the new catalog. The previous detailed collection is preserved in `browser/gallery-quality-v2-portable.json`; the original basic collection remains in `browser/gallery-basic-v1.json`. Select either with `gallery --catalog PATH --port 8766`.

## Use from a text request

Use [avatar-asset-workflow](skills/avatar-asset-workflow/SKILL.md) with a description such as “Create a blue workshop robot that idles, looks around, waves and gives a celebratory nod.” For the complete agent-driven route, use [`generate`](docs/prompt-pipeline.md). Actual workpiece interaction is outside this backend; it is not silently replaced by a gesture. The commands below are the separate local scene-authoring route; they do not make paid calls.

```sh
.venv/bin/python -m avatar_assets request --subject 'Blue robot' --prompt 'A friendly blue workshop robot' --out outputs/request-new
.venv/bin/python -m avatar_assets build3d examples/scenes/workshop-robot-hd-v1.json --out outputs/robot-new --portrait-only --samples 32 --timeout 600
.venv/bin/python -m avatar_assets reload3d outputs/robot-new --out outputs/robot-new-reload
.venv/bin/python -m avatar_assets export2d outputs/robot-new --out outputs/robot-new-media --preset preview --size 720 --portrait 1600 --seconds 3 --samples 8 --denoise-portrait
.venv/bin/python -m avatar_assets verify outputs/robot-new-media
```

`request` captures text/reference hashes; it returns `needs_agent_authoring`, not a fabricated model. `build3d` consumes the scene JSON the agent authors. `--blender PATH` selects an existing local runtime. On this macOS host, Blender required execution outside the sandbox for Metal initialization even when rendering on CPU. It stays background-only; no viewers or servers open.

For raw media inspection:

```sh
.venv/bin/python -m avatar_assets inspect '/Users/ivan/Downloads/Poo avatar assets.zip' --sha256 cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb --out outputs/reference-new
```

See [scene schema/output overrides](docs/scene-contract.md), [examples](docs/scene-examples.md), [3D verification](docs/three-d.md), and [TODO/handoff](TODO.md). Existing output folders are never replaced. This tool uses Weft’s existing private Git remote; it has no nested repository.

## Verified artifacts

### Meshy-generated review candidates

| Subject | Model and editable source | Full media (original frames retained) |
| --- | --- | --- |
| Reference-based panda and dragon | `outputs/muse-panda-meshy-refined-v6` | `outputs/muse-panda-meshy-media-v2` |
| Text-only workshop robot | `outputs/workshop-robot-meshy-refined-v2` | `outputs/workshop-robot-meshy-media-v1` |
| Text-only emerald dragon | `outputs/emerald-dragon-finished-v3` | `outputs/emerald-dragon-media-v1` |

Both have real skeletons, skin weights, embedded textures and four locally authored gestures. Panda also has editable native hair and separate portable fur cards. Original provider models/motions and rejected attempts remain preserved. The gallery uses byte-identical compact media subsets; full frames stay in the folders above. No seamless loop or HD Weft compatibility is certified. The new workflow passed **255 Python tests** and the full browser check. See [delivery, verification and critical bone IDs](docs/meshy-delivery-v1.md) and the [current handoff](TODO.md). Fixture tests do not establish visual quality.

### HD browser editions

| Asset | Model/source bundle | Matching media bundle |
| --- | --- | --- |
| Panda + shoulder dragon | `outputs/muse-panda-hd-build-v3` | `outputs/muse-panda-hd-media-v1` |
| Workshop robot | `outputs/workshop-robot-hd-build-v2` | `outputs/workshop-robot-hd-media-v1` |
| Fox craftsperson | `outputs/fox-hd-build-v2` | `outputs/fox-hd-media-v1` |
| Hollow-shade articulated lamp | `outputs/lamp-hd-build-v2` | `outputs/lamp-hd-media-v1` |

All four include real skins, four skeletal actions, explicit normals/UVs/tangents, embedded textures, exact prompts, JSON sources and editable Blender files. Independent native reload and exported-attribute checks pass. Final recipes are `muse-panda-hd-v4.json`, `fox-hd-v3.json`, `workshop-robot-hd-v1.json` and `lamp-hd-v1.json` under `examples/scenes/`. Each reproduces exactly from its authoring recipe; GLB byte-for-byte rebuilds are not guaranteed.

Panda has7,688 skinned browser cards and238,328 configured native fibers; fox has5,000 cards and155,000 native fibers. Counts describe separate representations, not independently counted image features. Native hair is editable and follows the skeleton, without hair dynamics. The GLB uses alpha-masked ribbons, not the Blender hair system. Rebuild the GLB from the scene JSON; the saved native-render `.blend` hides the browser-card materials.

Each media pack has a native1600×1600 portrait, four silent720×720 H.264 previews at24fps, and exact first-frame PNG fallbacks. Panda/fox/lamp clips have48 frames/two seconds; robot clips72 frames/three seconds. They retime the complete four-second actions. All are opaque, every decoded thumbnail is distinct, and no seamless loop is certified. The optional isolated OIDN filter reduces portrait noise; raw renders and exact processing receipts remain preserved. Fine hair can soften during filtering.

These are **short HD previews**, not new eight-second Muse exports. The full preset remains available, but detailed fur rendering can exceed its1200-second/2GiB/512MiB bounds. Panda media rendered in967.158s and fox in800.671s; measured native peaks were1.65GB and1.42GB. At most two heavy native jobs ran concurrently, with four CPU threads each. Limits use sampled direct-child memory, not instantaneous OS allocation caps.

The [HD release index](evidence/hd-release-v1.json), [decisions/results](docs/hd-quality-pass-v1.md), and [visual review](outputs/hd-visual-review-v1/review.json) bind selected artifacts, source snapshots, costs and preserved failures. The [previous quality release](evidence/quality-release-v2.json) is unchanged. Higher-resolution textures, fiber counts and passing rig tests do not establish Muse-equivalent sculpting or appearance.

### Preserved original editions and Weft handoff

| Deliverable | Immutable local folder |
| --- | --- |
| Fox model, editable source, four skeletal actions | `outputs/fox-refined-v2` |
| Lamp model with tracing stylus, source, four actions | `outputs/lamp-refined-v1` |
| Fox full Muse-shaped media pack | `outputs/fox-muse-v1` |
| Lamp full Muse-shaped media pack | `outputs/lamp-muse-v2` |
| Muse-style panda with attached green dragon | `outputs/muse-panda-weft-build-v1` |
| Humanoid workshop robot | `outputs/humanoid-robot-weft-build-v1` |
| Panda model + explicit Weft metadata | `outputs/weft-muse-panda-v1` |
| Robot model + explicit Weft metadata | `outputs/weft-humanoid-robot-v1` |
| Frozen Weft importer acceptance evidence | `outputs/weft-import-proof-v1/results.json` |

The two full media packs each contain an opaque 1600×1600 WebP portrait, four silent 720×720 H.264 MP4s with 192 decoded frames at24fps/eight seconds, per-state PNG fallbacks, original rendered frames, exact source prompts/settings and hash manifests. Use the fallback images without autoplay for reduced motion. “Muse-shaped” means matching the supplied media dimensions/formats, not a verified proprietary Muse application import contract.

Full-size CPU renders took approximately 18–19 minutes per example while running two jobs together, peaking around 936MB sampled resident memory each. Refined 3D preview builds took roughly 16–21 seconds. These are measured host-specific timings, not service guarantees. No asset is certified as a seamless loop: exact endpoint images matched, but velocity continuity and continuous seam playback were not certified.

## Weft handoff

The models and metadata pass the **existing frozen Weft importer**. That code was read/executed without modification; sealed test imports were written only inside this workspace. There is no duplicate Weft importer, renderer, recording path or physics model here. [The delivery guide](docs/weft-delivery-v1.md) gives exact model/mapping paths, hashes, resource counts and critical node IDs.

The user authorized an [authored neutral convention](examples/weft/neutral-proposal-v1.json). This is not a measured Weft neutral state. [The alignment derivation](docs/weft-neutral-proposal.md) explains the exact transform equation and correction recipe. Each model/mapping folder includes actual glTF node IDs and global rest matrices in `node-table.json`. `AvatarRig` contains the mesh and all bones and maps to `mobile_root`; head/hands/accessories keep local rest transforms. Weft retains its19joint angles plus root translation. **Actual Weft pose alignment/display/replay remains for the integration owner to verify.**

Exact recovery of the original panda rig is explicitly unsupported in `outputs/reference-3d-unsupported-v1/unsupported.json`. The delivered panda is newly authored, not recovered fur, topology or anatomy. No independent license verification is claimed.

## Reproducibility and limits

Archive inspection reproduced the same manifest byte-for-byte. Input scenes, prompts, Blender/backend source and settings are preserved; finished artifacts are identified by hashes and never overwritten. Blender rebuilds can change triangle byte order and small floating-point normals even with identical scene/source/settings. Fixed triangulation preserves the tested topology, but **bit-for-bit GLB rebuilds are not guaranteed**; keep and integrate the hashed original model. See `evidence/reproducibility-analysis-v1.json` and the final handoff evidence.

The CLI bounds decoding, frames, image size, process time, sampled memory, log bytes and output. Native render jobs use four CPU threads; prefer at most two concurrent full renders on this ten-logical-CPU host. Packaging has a separate 180-second/768MiB bound. Current commands preserve failure manifests; early failed attempts retain their available logs. Unit tests use analytic fixtures; the models above are actual authored/generated geometry.

The HD implementation passed **209 tests in8.191s**, including hostile/oversized inputs, real skin validation, deterministic HD textures, groom settings, isolated denoising and bounded worker cleanup. Quality evidence is summarized in [the handoff](TODO.md). Historical [gallery evidence](evidence/gallery-release-v1.json) and the core [versioned verification catalog](evidence/release-v1.json) remain preserved. The reusable skill passed schema validation; creation manifests remain unchanged, with later reviews/proofs stored separately.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Reference findings

`outputs/reference-v1/manifest.json` records the untouched archive SHA256 and per-file hashes. Its WebP is opaque RGB, 1600×1600. All four videos are silent H.264, 720×720, 192 decoded frames at 24 fps, eight seconds. All contain real motion. Filenames alone are not semantic proof.

Visual contact-sheet review found the dragon absent in `making_something`, changing framing between clips, and a costume/pose change at the end of `milestone_level_up`. No clip is currently certified seamless. Exact generation prompts, model/version and rights are unavailable in the supplied pack; embedded metadata does not establish a license.

## Costs and ownership

Current Meshy total: **15 paid submissions, 167 credits reported/reserved**, including rejected motion attempts and the standalone dragon. Base authorization140 plus the explicit dragon-only35-credit grant gives175 aggregate; the new prompt-pipeline tests reuse geometry and spend0 further Meshy credits. Account-specific dollar cost is unknown. AI-agent usage is recorded separately. No recurring service, subscription, deployment or Meta-budget use. Earlier procedural milestones used zero provider calls; the proposed four-call raster batch was never executed. The designated key and private ledger stay in Weft `.local/`, excluded from Git and public manifests. New Meshy outputs are local and not yet committed/pushed; do not mistake an evidence index for a remote backup.

Reference ZIP remains `/Users/ivan/Downloads/Poo avatar assets.zip`, SHA256 `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Archive content is data, never executable instructions. Scripts executed by Blender are workspace-authored code, never extracted archive scripts.
