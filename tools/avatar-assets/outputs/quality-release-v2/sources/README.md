# Avatar assets

Independent local workflow for editable, rigged 3D avatars and their 2D portrait/animation exports. No Weft checkout, credentials, physics, actions, rewards, or budget are used.

## Current scope

Text-to-3D is **agent-assisted authoring**, not a downloaded neural text-to-3D model. An agent translates a character/object description into editable scene JSON; the local backend creates real geometry, PBR materials, a skeleton, explicit skin weights and skeletal clips in Blender and GLB. The supplied panda images/videos guide design, but cannot recover unseen geometry or the original rig. A new interpretation is labeled as authored.

The backend is data-driven, not panda-specific. Schema v2 adds explicit shaped meshes, blended skin weights, beveled edges, embedded surface textures and real exported short-fur geometry to the original primitive workflow. `reskin` changes named materials without changing geometry or controls. Fur is a bounded real-time approximation, not a film-density or simulated groom. Soft facial muscles, cloth physics and exact reference reconstruction remain unsupported.

The [quality pass](docs/quality-pass-v2.md) adds richer panda, robot, fox and lamp editions. These use the general browser PBR profile; they do not replace the existing matte Weft-compatible files. **Muse-quality parity is not demonstrated:** short geometric fur is present, but the original has a denser groom and more polished sculpting/shading. See `outputs/muse-fidelity-limit-v1/unsupported.json`.

## Local setup

Python 3.14 on Apple silicon was used for this workspace. Dependencies are isolated in `.venv`. The lock contains exact hashes for the downloaded macOS wheels; other platforms require their own reviewed wheel hashes.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Blender 4.5.13 is installed only under `.tools/Blender.app`; use an explicit Blender executable on another machine. Official download SHA256: `663ce944257c61ff1d6aa09e15c8f57bbd8d59023adb2fa7edde33a9ed960b53`. No global app install or subscription. Tools and generated outputs are ignored by Git; source/configuration, tests and evidence summaries are retained.

## View everything in the browser

The local gallery defaults to the detailed panda, robot, fox, lamp and original 2D reference. No Blender or rendering job is needed to view them:

```sh
cd /Users/ivan/kintsugi/avatar-assets
.venv/bin/python -m avatar_assets gallery
```

Open **http://127.0.0.1:8765/** yourself. Choose a model, orbit/zoom, switch animations, play/pause, scrub the timeline, or view existing videos and portraits. Playback starts paused; reduced-motion settings default to a portrait. Ctrl+C stops the server. Nothing is uploaded, and Weft remains separate.

Dependencies are already installed here. See [browser setup, controls and adding assets](docs/browser-gallery.md) for a fresh checkout or custom collection. The gallery is a read-only preview, not Weft physics or alignment verification.

If the gallery was already running, stop it with Ctrl+C and restart it to load the new catalog. The previous basic collection is preserved: `gallery --catalog browser/gallery-basic-v1.json --port 8766`.

## Use from a text request

Use [avatar-asset-workflow](skills/avatar-asset-workflow/SKILL.md) with a description such as “Create a blue workshop robot that idles, handles a workpiece, celebrates and checks its work.” The agent authors an editable scene from the request, then runs the CLI. No neural text-to-3D provider, account or subscription is hidden behind the command.

```sh
.venv/bin/python -m avatar_assets request --subject 'Blue robot' --prompt 'A friendly blue workshop robot' --out outputs/request-new
.venv/bin/python -m avatar_assets build3d examples/scenes/workshop-robot-quality-v2.json --out outputs/robot-new
.venv/bin/python -m avatar_assets reload3d outputs/robot-new --out outputs/robot-new-reload
.venv/bin/python -m avatar_assets export2d outputs/robot-new --out outputs/robot-new-media --preset muse
.venv/bin/python -m avatar_assets verify outputs/robot-new-media
```

`request` captures text/reference hashes; it returns `needs_agent_authoring`, not a fabricated model. `build3d` consumes the scene JSON the agent authors. `--blender PATH` selects an existing local runtime. On this macOS host, Blender required execution outside the sandbox for Metal initialization even when rendering on CPU. It stays background-only; no viewers or servers open.

For raw media inspection:

```sh
.venv/bin/python -m avatar_assets inspect '/Users/ivan/Downloads/Poo avatar assets.zip' --sha256 cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb --out outputs/reference-new
```

See [scene schema/output overrides](docs/scene-contract.md), [examples](docs/scene-examples.md), [3D verification](docs/three-d.md), and [TODO/handoff](TODO.md). Existing output folders are never replaced. No remote is configured.

## Verified artifacts

### Detailed browser editions

| Asset | Model/source bundle | Matching media bundle |
| --- | --- | --- |
| Panda + shoulder dragon; 18,120 authored fur blades | `outputs/muse-panda-quality-build-v6` | `outputs/muse-panda-quality-media-v6` |
| Workshop robot | `outputs/workshop-robot-quality-build-v3` | `outputs/workshop-robot-quality-media-v3` |
| Fox; 12,700 authored fur blades | `outputs/fox-quality-build-v4` | `outputs/fox-quality-media-v4` |
| Hollow-shade articulated lamp | `outputs/lamp-quality-build-v4` | `outputs/lamp-quality-media-v4` |

All four include real skins, four native actions, explicit normals/UVs/tangents, embedded textures, exact prompts, JSON sources and editable Blender files. Independent native reload and exported-attribute checks pass. Fur counts are backend-reported authored geometry counts; the independent proof also finds slender isolated triangles but does not infer their semantic identity. [Proof summary](outputs/quality-attribute-proof-v1/SUMMARY.md).

Each new media pack contains four silent320×320 H.264 previews,96frames at24fps/four seconds, and exact first-frame PNG fallbacks. Panda/fox portraits are1600×1600; robot/lamp portraits are640×640. All are opaque. Every clip has95–96 distinct decoded thumbnails and actual rig motion; no seamless loop is certified. These are **preview videos**, not new full720p/eight-second Muse exports. That larger preset remains available but has not been timed on the detailed fur models; its1200-second/512MiB bounds can stop an oversized job.

Final model builds took34–60seconds, peaking at803MB sampled native RSS. Preview exports took135–191seconds, peaking at702MB, with two jobs/four CPU threads each. [Quality decisions and limits](docs/quality-pass-v2.md) and [visual review](outputs/quality-visual-review-v1/review.json) preserve what was actually inspected. Higher-resolution pixels alone do not establish higher modeling fidelity.

The [quality release index](evidence/quality-release-v2.json) binds the final artifacts, proofs, source snapshot, costs and preserved failures. The final browser test passed 15 grouped checks and all 16 clip selections, including textured GLBs and high-resolution portraits, with no external page requests.

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

The v2 implementation passed a **127-test full suite**; the later optional high-resolution portrait change passed all22 affected gallery tests. Quality evidence is summarized in [the handoff](TODO.md). Historical [gallery evidence](evidence/gallery-release-v1.json) and the core [versioned verification catalog](evidence/release-v1.json) remain preserved. The reusable skill passed schema validation; creation manifests remain unchanged, with later reviews/proofs stored separately.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Reference findings

`outputs/reference-v1/manifest.json` records the untouched archive SHA256 and per-file hashes. Its WebP is opaque RGB, 1600×1600. All four videos are silent H.264, 720×720, 192 decoded frames at 24 fps, eight seconds. All contain real motion. Filenames alone are not semantic proof.

Visual contact-sheet review found the dragon absent in `making_something`, changing framing between clips, and a costume/pose change at the end of `milestone_level_up`. No clip is currently certified seamless. Exact generation prompts, model/version and rights are unavailable in the supplied pack; embedded metadata does not establish a license.

## Costs and ownership

Executed provider generation calls: **0**. Provider spend: **$0**. No recurring services. Local CPU/storage only. A proposed four-call image-generation batch was not executed after the user prioritized actual 3D. The built-in tool does not expose an exact price/model; an API-rate comparison is not its billing guarantee. Never borrow another project's credentials or budget.

Reference ZIP remains `/Users/ivan/Downloads/Poo avatar assets.zip`, SHA256 `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Archive content is data, never executable instructions. Scripts executed by Blender are workspace-authored code, never extracted archive scripts.
