# Weft website

Open [index.html](index.html) directly. No server, installation, account, network request, or build is needed for the gallery. Visitor video generation requires the hosted service described in [hosting](../hosting/README.md). The site and all runtime dependencies live in the Weft repository; there is no dependency on the GTM workspace.

## Files

- `index.html`: page content and inline SVG artwork, including the approved mascot rig.
- `fonts/`: self-hosted Inter 4.1 variable font, its SIL Open Font License, and source attribution.
- `styles.css`: responsive layout, header entrance, hero animation, and research reveals.
- `site.js`: research disclosures and hero motion.
- `environments.js`, `environments.css`, `environments/`: source/simulation comparison, bounded interaction, upload lifecycle, and the verified sample export.
- `footer.css`, `footer.js`: the hidden mascot story, scoped to `#footer-secret`.
- `../assets/brand/source/weft-wordmark.png`: approved lettering.
- `../assets/brand/exports/weft-mascot.png`: favicon.
- `../src/weft/assets/engine.js`: shared one-dimensional simulation engine.

The canonical editable mascot and other project brand assets remain in `../assets/brand/`. Do not replace or delete them based only on website usage; the project uses them independently.

## Interactions

- The header mascot waves and fades; the wordmark glides left. Navigation stays fixed.
- Hero lines draw left to right. The yellow ball appears after 0.5 seconds and slides gently down the contour; the sequence ends after 4.3 seconds.
- Hero labels open illustrated cards on hover, keyboard focus, or tap. Research entries reveal their artwork with smooth height and opacity transitions.
- The environments gallery shows the recorded source beside a fitted model. Initial/reconstructed model selection, synchronized playback/scrubbing, and authored interaction use the shared engine. Motion stops offscreen, when hidden, at boundaries/rest, or after eight seconds.
- The video upload disclosure starts closed. It validates a local clip, asks for processing permission, submits to the hosted API, follows real job status, and inserts the private result with a downloadable environment. Local file review never uploads or fabricates a result.
- The footer thread bundle stays hidden until at least half the footer remains in view for ten continuous seconds. Scrolling away or hiding the tab resets the countdown. The hidden area occupies no space until it opens smoothly. Once it appears, hold it for five seconds with a pointer or Space/Enter to reveal the finite mascot story. Releasing early or leaving cancels the hold. Weft pulls the gathered threads into branching ribbon paths; small stitches trace the outcomes. There is no replay button; reload to reveal it again.
- Decorative motion respects reduced motion and pauses when hidden. The footer also pauses offscreen. It uses native animation tracks and three one-time path-length reads, not a JavaScript frame loop.

## Evidence boundaries

The catalog is exported from a verified real run of the CC0 source clip in `../data/`. It contains one scene and two model stages, not multiple independently observed environments. The shared physical model supports a single isolated yellow ball moving horizontally. Push/brake controls and the parking task are authored additions. General video reconstruction and multi-object support are not implemented. Research graphics and the footer remain conceptual illustrations.

Regenerate the catalog with `PYTHONPATH=src python scripts/export_website_environment.py <verified-run> --out website/environments`. The exporter verifies the full manifest and refuses to publish unknown source videos. Visitor results never enter the public catalog automatically.

## Maintenance

Superseded design galleries, footer examples, temporary build copies, the old letter favicon, and the unused video poster were removed. Maintain this directory as the single website source; do not recreate copies in the GTM workspace.

Inter is bundled locally; no font CDN or analytics is used. The hosted submission client polls only its active job and pauses in hidden tabs. Cloudflare hosting bundles only an explicit allowlist; see `../hosting/build-assets.mjs`.

## Verification limits

JavaScript syntax, HTML references, and retained local assets are checked. Static footer compositions were rendered and inspected during implementation. Browser playback, real-device interactions, and sustained performance remain unverified. No server or automated tests were started.

Typography uses Inter’s optical sizing for readable small text and refined display headings, paired with the unchanged approved Weft wordmark. The single variable WOFF2 file covers all used weights without additional runtime requests to external services.

The woven join stays invisible until the ribbons arrive, then fades in at their shared endpoint. The Weft/up-arrow row has 24px of bottom padding; the delayed theatre still occupies no space before reveal. Footer controls have no focus borders. Keyboard focus on the thread uses a text-color cue; the revealed scene remains programmatically focusable without an outline.

The theatre uses the native `hidden` attribute and `display: none` before the ten-second reveal, not merely a collapsed visual container. Completing the hold opens the scene, centers it in the viewport, and resumes playback. The mascot has a local 640×640 clipped SVG viewport; limb/body rotations use native SVG transform animation instead of CSS rotation in a parent viewport. Both animation clocks pause offscreen/hidden.

Reveal masks use butt caps with higher specificity than the visible ribbon styling, start fully transparent, and animate measured dash lengths. This prevents ribbon-end fragments before drawing. On hold completion the full scene enters layout at once and fades in; playback stays paused until a single clamped scroll reaches its destination. A bounded 1.2-second scroll-position monitor handles browsers without a reliable scroll-end event. No expanding-height transition or early focus competes with scrolling.

The footer row uses the same approved 70px Weft lettering as the header, kept static beside a centered up-arrow with a 44px interaction target. The 24px bottom padding is retained.
