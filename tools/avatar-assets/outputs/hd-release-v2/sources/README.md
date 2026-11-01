# Avatar assets

Local workflow for editable, rigged 3D avatars and their 2D portrait/animation exports, now maintained in Weft under `tools/avatar-assets/`. Authoring stays separate from Weft physics, actions, rewards and provider budget. See [relocation and preservation](docs/relocation.md).

## Current scope

Text-to-3D is **agent-assisted authoring**, not a downloaded neural text-to-3D model. An agent translates a character/object description into editable scene JSON; the local backend creates real geometry, PBR materials, a skeleton, explicit skin weights and skeletal clips in Blender and GLB. The supplied panda images/videos guide design, but cannot recover unseen geometry or the original rig. A new interpretation is labeled as authored.

The backend is data-driven, not panda-specific. Schema v2 adds explicit shaped meshes, blended skin weights, beveled edges, native1024/2048 surface textures, skinned cutout fur cards and optional editable combed hair. `reskin` changes named materials without changing geometry or controls. Browser GLBs use lightweight cards; native Blender renders use the denser hair system. Hair dynamics, soft facial muscles, cloth physics and exact reference reconstruction remain unsupported.

The latest [HD fur and surface pass](docs/hd-quality-pass-v1.md) adds a denser directional coat, refined eyes/face/dragon and sharper textures to the panda and fox, plus appropriate high-resolution surfaces for robot and lamp. These are general browser PBR editions, not replacements for the existing matte Weft-compatible files. **Muse-quality parity is not demonstrated:** the original has different sculpting, fur breakup and shading. See `outputs/hd-muse-fidelity-unsupported-v1/unsupported.json`. The [previous quality pass](docs/quality-pass-v2.md) remains preserved for comparison.

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

## View everything in the browser

The local gallery defaults to the HD panda, robot, fox, lamp and original 2D reference. From the Weft root, use `./tools/avatar-assets/run gallery`. No Blender or rendering job is needed to view them. This HD follow-up is local and not yet committed/pushed; selected editable bundles and compact media are allowlisted for version control. The earlier detailed release remains on the existing private remote.

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -m avatar_assets gallery
```

Open **http://127.0.0.1:8765/** yourself. Choose a model, orbit/zoom, switch animations, play/pause, scrub the timeline, or view existing videos and portraits. Playback starts paused; reduced-motion settings default to a portrait. Ctrl+C stops the server. Nothing is uploaded by the gallery; it does not drive Weft physics.

Dependencies are already installed here. See [browser setup, controls and adding assets](docs/browser-gallery.md) for a fresh checkout or custom collection. The gallery is a read-only preview, not Weft physics or alignment verification.

If the gallery was already running, stop it with Ctrl+C and restart it to load the new catalog. The previous detailed collection is preserved in `browser/gallery-quality-v2-portable.json`; the original basic collection remains in `browser/gallery-basic-v1.json`. Select either with `gallery --catalog PATH --port 8766`.

## Use from a text request

Use [avatar-asset-workflow](skills/avatar-asset-workflow/SKILL.md) with a description such as “Create a blue workshop robot that idles, handles a workpiece, celebrates and checks its work.” The agent authors an editable scene from the request, then runs the CLI. No neural text-to-3D provider, account or subscription is hidden behind the command.

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

Executed provider generation calls: **0**. Provider spend: **$0**. No recurring services. Local CPU/storage only. A proposed four-call image-generation batch was not executed after the user prioritized actual 3D. The built-in tool does not expose an exact price/model; an API-rate comparison is not its billing guarantee. Never borrow another project's credentials or budget.

Reference ZIP remains `/Users/ivan/Downloads/Poo avatar assets.zip`, SHA256 `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Archive content is data, never executable instructions. Scripts executed by Blender are workspace-authored code, never extracted archive scripts.
