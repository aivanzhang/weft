# Local browser gallery

Browse the four earlier authored models and the original 2D reference together, or select the new Meshy candidate collection. This is a local preview library, not a new simulator or a Weft renderer. Structural acceptance does not establish Muse-quality likeness.

## Start

Dependencies are already installed in this workspace. Run from a terminal:

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -m avatar_assets gallery
```

Then visit **http://127.0.0.1:8765/**. Keep that terminal running; use Ctrl+C to stop. The command never opens a browser automatically. If the port is occupied, use `gallery --port 8766`; do not stop another project's server.

For the real Meshy-generated panda and text-only robot, with matching portraits/videos and the original reference:

```sh
# From the Weft repository root:
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v2.json
```

These are review candidates below the Muse target. Both have four genuine local skeletal gestures and four-second rendered clips; crafting/working still lack recognizable object interaction. The panda's native-render hair and interactive cutout fur differ. Generation, local finishing and their limitations are documented in [Meshy workflow](meshy-workflow.md). The default collection remains unchanged.

For a fresh checkout, install the Python dependencies from the root README, then install the browser's locked dependencies:

```sh
cd browser
npm ci --ignore-scripts --no-audit --no-fund --cache ../.tools/npm-cache
```

No bundler, development server, provider account, CDN or API key is needed. The page serves its JavaScript and assets from this workspace. Local viewer dependency: [Google model-viewer 4.3.1](https://github.com/google/model-viewer/releases/tag/v4.3.1), Apache-2.0. The lockfile also preserves transitive dependency integrity and licenses.

## Use

- Choose an asset in the collection. Only one 3D viewport is active.
- Drag to orbit; scroll/pinch to zoom. Front, Side, Back and Reset view provide fixed camera choices.
- Pick a state, press Play, or drag the timeline to inspect a pose. Speed and Repeat controls do not alter source assets.
- Switch between **3D model**, **Rendered clip** and **Portrait**. All four detailed editions have real GLB animations and matching videos. Portrait mode loads the larger image where present; library thumbnails stay small.
- The HD editions have1600px portraits and720px/24fps videos: two seconds for panda, fox and lamp; three seconds for robot. They retime the complete four-second skeletal actions. These are short previews, not eight-second Muse-format exports.
- Panda/fox portraits and videos show the dense native hair system. The interactive GLB uses separate, lighter cutout fur cards. Switching views changes the rendering representation, not the character's identity.
- Playback begins paused, stops when the page/viewport is hidden, and never resumes without a new user action. Reduced-motion settings select a static portrait without loading a GLB until requested.
- Download GLB saves a copy with a `.glb` filename. The asset manifest gives provenance and hashes.

The original panda reference remains **2D-only**. Its filenames are labels, not proof of actions or loops. The authored models are interpretations with real geometry and skinning. Browser lighting differs from the offline Blender renders. Preview playback does not test Weft physics, retargeting alignment or recording, and Repeat does not certify a seamless loop.

The default and basic catalogs use byte-preserving portable media subsets. Full native frames and original manifests remain locally under their output paths; packaging does not regenerate clips. `gallery-hd-v1.json` freezes the HD catalog, `gallery-quality-v2-portable.json` preserves the previous default, and `gallery-quality-v2.json` retains the previous full-media catalog. The new HD follow-up is local and not yet committed/pushed; the earlier detailed release remains in the private remote.

The fur is newly authored, not the original Muse groom or a demonstrated exact-quality match. HD models are not Weft-ready replacements; the older matte models and mappings remain in `browser/gallery-basic-v1.json`. Use `gallery --catalog browser/gallery-basic-v1.json --port 8766` to view that preserved collection. Restart a running gallery after catalog changes; it deliberately uses a fixed startup snapshot.

## Add an asset

Edit [browser/gallery.json](../browser/gallery.json), add a unique entry, then restart the gallery:

```json
{
  "id": "my-character",
  "title": "My character",
  "description": "Short description of this design.",
  "kind": "authored_3d",
  "pack": "outputs/my-character-v1",
  "media_pack": "outputs/my-character-media-v1"
}
```

`pack` must be a completed build with its manifest, model and portrait. Use `kind: generated_3d` for the checked generated-model workflow, `authored_3d` for local scene builds, and `reference_2d` only with an inspection pack. A generated static preview has no animation controls; a constant clip is not animation. `media_pack` is optional and must name exports of that exact source manifest. `mapping_pack` is also optional and must match that source. The server rejects mismatched manifests, linked files and paths outside the workspace. The creation workflow remains responsible for geometry/rig validation; the gallery does not turn arbitrary files into validated assets.

The catalog supports up to 24 assets and 24 states per asset. It is a startup snapshot, not an unbounded scan of output history. Rejected and superseded attempts stay preserved but are not presented as accepted library entries.

## Boundaries and verification

Only configured resources are served. The server binds to `127.0.0.1`, rejects cross-origin requests and mutations, and provides no file listing or uploads. It serves fixed-size chunks, supports MP4 seeking, caps concurrent requests at eight and individual files at 128 MiB, and bounds request/stream duration. A served asset changed after startup produces an error instead of silently changing the preview.

The viewer code is pinned locally. Its embedded WebAssembly requires the narrow `wasm-unsafe-eval` CSP token; embedded GLB textures require `blob:` fetches through `connect-src`. Those are in-memory local image buffers, not permission for remote HTTP connections. Arbitrary script `unsafe-eval` and external network connections remain disallowed. There are no generation calls or recurring paid services.

Verification commands:

```sh
.venv/bin/python -m unittest discover -s tests -v
cd browser
node tests/gallery.mjs outputs/gallery-browser-next
```

The same end-to-end browser check accepts a second argument, `browser/gallery-meshy-v2.json`, to test the new collection. Output/catalog paths resolve relative to `tools/avatar-assets`, regardless of the shell directory.

The browser check uses an installed Chrome with a fresh temporary profile and software WebGL, starts its own ephemeral loopback server, and closes both. Set `AVATAR_BROWSER_PATH` to another Chromium executable when needed. Each output directory must be new. On this host, the sandbox disallows loopback binding, so the HTTP/browser checks required narrow native execution permission; ordinary Terminal usage does not use that sandbox. Do not run an additional persistent server merely for the test.
