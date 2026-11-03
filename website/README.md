# Weft website

## Run locally

From the repository root, run `npm run dev` and open `http://127.0.0.1:3000/`.
Requires Node.js 22+; no install, Python or build step. Refresh after edits.
Use `npm run dev -- --port 3001` if needed. The user manages the server.

The landing server exposes only files listed in `public-assets.mjs`, supports video seeking and
disables uploads. It never starts Python, proxies the sandbox or exposes private
projects. Run the [sandbox](../sandbox/README.md) separately on port 8765.
The [hosted service](../hosting/README.md) remains offline; local commands do not deploy it.

[index.html](index.html) also opens directly without a server or network access;
its gallery works offline, but hosted visitor uploads do not.

## Maintain

- Keep this as the single website source. [AGENTS.md](AGENTS.md) defines the approved layout, finite/reduced-motion animation and footer behavior.
- `site.js` owns hero/research interactions; `environments.*` owns the gallery; `footer.*` owns the hidden story.
- Reuse [the simulation engine](../src/weft/assets/engine.js) and canonical [brand assets](../assets/brand/README.md). Do not duplicate them for styling.
- Keep bundled [Inter and its license](fonts/README.md); no font CDN or analytics.
- Generate the public catalog only from a verified, approved source run:

```sh
PYTHONPATH=src python scripts/export_website_environment.py <verified-run> --out website/environments
```

The gallery represents one CC0 rolling-ball scene with two model stages, not
multiple observed worlds. Push/brake controls are authored; research graphics
are conceptual. Private visitor results never enter the public catalog automatically.

`npm run test:local` checks launchers, public routes and video ranges using a
short-lived loopback listener, without AI calls. These checks do not qualify
browser animation, real-device behavior or sustained performance.
