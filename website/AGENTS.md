# Weft website

- This is Weft’s local, buildless product website, not a second GTM dashboard.
- Design: the approved narrow sans-serif editorial layout, evidence illustration, and three research graphics. Hover, keyboard focus, and tap reveal graphics with finite micro-animations.
- Preserve warm cream surfaces, deep teal type, clay links, generous whitespace, and simple navigation. The header places the layered mascot to the left of the approved source wordmark. The mascot waves once, then fades away; the lettering glides left afterward to close the gap without moving navigation. Reduced motion shows only the left-aligned wordmark. Use the approved mascot for the favicon; do not modify canonical brand artwork.
- This directory is the single website source. Superseded concepts and GTM workspace copies were removed at the user’s request; do not recreate them. Keep all website work and dependencies in the Weft repository.
- All SVG graphics are editable site-native artwork, not extracted screenshots. They are conceptual illustrations, not measured trajectories.
- Reuse `../src/weft/assets/engine.js` and `../data/rolling-ball.mp4`; do not duplicate or change the shared simulation engine for website styling.
- The demo uses explicitly illustrative parameters. It is not a fitted replay, model training, video upload, multi-object reconstruction, or physical validation.
- The site must work directly from `index.html` without a server, externally hosted fonts, dependencies, API calls, telemetry, or credentials.
- Keep animation finite, respect reduced motion, and stop simulator frames offscreen or when hidden. Keep the source video in the repository, not an embedded panel. Only user actions start simulation.
- No public deployment, GitHub publication, or submission is authorized by local preview work.

- The experiment is collapsed by default, with icon-only end controls when expanded. Keep preview labels and work-in-progress status off the page; preserve factual capability limits without presenting the site as a mockup.

- The main footer contains the approved hidden mascot story. Keep its thread bundle hidden until the footer has stayed at least half visible for ten continuous seconds, resetting on scroll-away or hidden tabs. Then hold its gathered threads for five seconds to reveal; preserve pointer/keyboard cancellation, one-time playback without a replay button, reduced motion, and visibility pausing. Keep the bundle gathered away from the mascot's face and scope its styles/behavior to `#footer-secret`. No review-only copy belongs in the main footer.

- Use locally bundled Inter with optical sizing for page typography; preserve the approved Weft wordmark. Retain the font license and attribution.

- Keep the footer theatre collapsed with no reserved height before its delayed reveal. Keep the woven endpoint hidden until the ribbons arrive, allow 24px below the footer row, and omit focus borders on the footer trigger/scene; retain keyboard operation and its subtle text-color focus cue.

- Use a true hidden theatre before the delay, auto-scroll the expanded story into view after the hold, and keep mascot rotations in its own clipped SVG coordinate space.

- Keep reveal-mask caps flat and hidden before drawing. Reserve the complete scene height before auto-scrolling; resume playback only after scroll settlement. Do not restore nested height transitions or focus-driven scrolling during this handoff.
