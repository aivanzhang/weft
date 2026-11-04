# Weft website

The product landing page pairs source video with a reconstructed rolling-motion
example. It runs independently of the [interactive sandbox](../sandbox/README.md).

## Quick start

Requires **Node.js 22 or later**. From the repository root:

```sh
npm run dev
```

Open `http://127.0.0.1:3000/`. No dependency installation or build step is needed;
refresh the page after editing. To choose another port, use
`npm run dev -- --port 3001`.

`/demo` and `/demo/runs` redirect to the local sandbox on port 8765. Use
`npm run demo` to start both servers, or `npm run sandbox` separately. Open the
sandbox's walkthrough and unlock it with your demo code; no token URL is needed.
The session remains local to that browser tab. Set `WEFT_SANDBOX_PORT` when
the sandbox uses another port.
The combined demo launcher sets that routing automatically for its selected ports.

You can also open [index.html](index.html) directly for the offline gallery.
Local landing-page uploads are disabled; use the sandbox to process a video.
The server exposes only files listed in [public-assets.mjs](public-assets.mjs),
not private projects or the simulation API.

## Source layout

| Files | Purpose |
| --- | --- |
| `index.html`, `styles.css`, `site.js` | Page content, layout and interactions |
| `environments.*`, `environments/` | Source/simulation gallery and exported example |
| `footer.*` | Animated mascot story |
| `fonts/` | Bundled Inter font and license |

Use the shared [simulation engine](../src/weft/assets/engine.js) and
[brand assets](../assets/brand/README.md). Keep animation finite and respect
reduced-motion preferences. Font attribution is in [fonts/README.md](fonts/README.md).

## Update and verify the gallery

Export a verified, supported source run from the repository root:

```sh
PYTHONPATH=src python scripts/export_website_environment.py <verified-run> --out website/environments
npm run test:local
```

The gallery contains one CC0 scene with two model stages. Push/brake controls
are authored additions; research graphics are conceptual. Private upload results
are never added to the public catalog automatically.

The local checks cover routes, launchers and video seeking. Browser animation,
accessibility and device behavior require separate review before release.
