# Real non-ball object source

Acquired one complete, openly licensed **real toy-van handling clip**. It meets the size/duration/frame bounds and contains at least 20 consecutive readable object views with physical hand/crank interaction. It does **not** establish a clean fixed-camera or uninterrupted rigid-body dynamics example: the original contains crossfades, camera changes, hand occlusion and 3D object rotation.

## Source and rights

[**Corgi Ice Cream Van with chime (Music Box Toy Car)**](https://commons.wikimedia.org/wiki/File:Corgi_Ice_Cream_Van_with_chime_(Music_Box_Toy_Car).webm), by **OrangeLark64**, is the uploader’s own work, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The complete original Commons upload—not a local crop or transcode—is saved as `data/general/corgi-hand-crank-van.webm`. Credit the author and source, retain the license, and identify derivatives when sharing the review images.

- 6,036,776 bytes; 1920×1080; 59 fps; 6 seconds.
- All 354 frames decoded; this matches the reported frame count.
- SHA-256: `caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8`.
- SHA-1 matches the official source-page checksum: `e6e6ce4b46ba15672884cba4f445fe5f8098a423`.

The original page HTML, exact media/revision URLs, author, license, source hashes and source-only images are retained in `artifacts/general-object-source-01`. Frame crops are review aids, never replacement pipeline input.

## Frozen source-only initialization and observations

The full source interval **0–353 inclusive** is retained. Initialization is frame 0, with the toy body’s manually estimated source-pixel box **`[970,538,108,68]`**. This box is an image region, not reconstructed physical geometry. The selected object is the main van body; hands, crank mechanism and the person are not modeled as rigid-body facts.

Six dispersed observation slots were fixed before any tracker output:

| Original frame | Approximate body center, pixels | Visibility / difficulty |
| --- | --- | --- |
| 0 | (1024,572) | Readable; hand-supported, underside partly hidden |
| 70 | (1104,538) | Readable during cranking; hand overlap |
| 140 | **null** | Source crossfade blends two temporal views; no unique physical center |
| 210 | (760,550) | Rotated toward camera, held in both hands |
| 280 | (1029,479) | Changed viewpoint; hand support and crank operation |
| 353 | (999,875) | Final lowering; top view, larger projected size, hand overlap |

Centers are assistant visual estimates with **±16-pixel radial uncertainty**, not expert labels or physical centers of mass. They describe the visible main-body envelope, excluding hands/crank where practical. The crossfade slot stays ambiguous/null; it was not replaced with an easier frame. Source frames **60–79** were individually inspected and preserved as a 20-frame review sheet: the main body is readable throughout, but hands partially occlude/support it. “Readable” does not mean entirely unoccluded.

The `.manual.json` and `.initialization.json` sidecars pin these source-only observations. A `.scene.json` exposes the same full interval and initial region for later integration. Its zero gravity, damping, restitution and target are explicitly authored simulation placeholders, not inferred forces or calibrated dynamics. No tracker, model, scorer or simulator was run. Current planar reconstruction should treat the camera changes, crossfades and human handling as unsupported conditions rather than report a validated physical reconstruction.

## Rejected candidate and bounded acquisition

The first download, [**Friction example**](https://commons.wikimedia.org/wiki/File:Friction_example.webm) by **Amethyst80**, is also CC BY-SA 4.0, but its source pixels show a **rendered block animation**. It is retained as `data/general/friction-example.webm` with a source sidecar and review sheet, explicitly rejected as real-world evidence. It must not be counted as a physical trial.

The two downloads total **10,931,485 bytes**, within the two-download / 30 MB bound. Search and source review ran October 28, 2026, approximately **3:08–3:13 p.m. PDT**. No accounts, browser interaction, paid services, external messages or new dependencies were used. The source-only archive preserves the rejection rather than presenting the synthetic clip as a fallback success.

The later [Physics 101 ramp study](phys101-ramps.md) adds complete real stationary and sliding/contact block clips without source edits, with source-reported angle, mass and volume. Camera calibration and executable ramp/contact validation remain open. The toy-van acquisition above supplies appearance/handling coverage, not that stronger dynamics evidence.
