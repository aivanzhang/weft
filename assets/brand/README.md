# Weft brand assets

The layered vector is the single mascot master. All PNG mascots derive from it. The logo retains the approved lettering and layout; earlier design alternatives, historical prompts, and unused export tools have been removed.

## Layout

- **[source/](source/):** editable mascot SVG, rigging notes, original wordmark, palette, and sprite timing data. Keep these to rebuild the exports.
- **[exports/](exports/):** ready-to-use logo, mascot, sprite sheet, forward/reverse animations, and generated CSS/metadata.
- **[preview.html](preview.html):** local animation review; no server required.
- **[../../scripts/brand/](../../scripts/brand/):** rebuild scripts. Run commands from the repository root.

Packaged copies in `src/weft/assets/` are required by installed/offline evidence viewers, not additional design variants. Existing experiment folders remain unchanged.

## Approved palette

| Role | Color |
| --- | --- |
| Primary actions and links | Deep clay `#965644` |
| Primary hover | `#7F4738` |
| Canvas and primary-button text | Warm cream `#F5EADC` |
| Small accents | Soft terracotta `#C7795D` |
| Text and outlines | Deep teal `#214F50` |
| Secondary surfaces | `#FBF7F1` |
| Secondary text | `#596562` |
| Borders | `#D8C9BA` |

Cream on deep clay has a calculated contrast ratio of 4.77:1. Terracotta is decorative, not small text on cream. Keep real video, scene colors, and success/failure indicators independent of branding; the fixture ball stays yellow.

## Deliverables

| File | Use |
| --- | --- |
| [Logo](exports/weft-logo.png) | 1254×1254 transparent logo |
| [Mascot](exports/weft-mascot.png) | 1280×1280 transparent neutral character |
| [Sprite sheet](exports/weft-mascot-sprites.png) | Idle, blink, wave A, wave B; four 640×640 cells in reading order |
| [Sprite metadata](exports/weft-mascot-sprites.json) | Rectangles, shared pivot `[320, 580]`, and one-shot timings |
| [Layered vector](source/weft-mascot-rig.svg) | Master artwork with separately named body, limbs, weave, stitch, eyes, cheeks, and mouth |
| [Rigging notes](source/weft-mascot-rig.json) | Suggested pivots and movement constraints |
| [Formation](exports/weft-mascot-forming.svg) | Self-contained 2.15-second on-load entrance |
| [Unweave](exports/weft-mascot-unweaving.svg) | Exact reverse timeline, ending empty |

Formation is decorative, not a loading gate or evidence of processing progress. It plays once. Reduced motion applies the selected endpoint immediately. The preview cancels earlier playback, stops when hidden, and uses one shared timer. Unweave becomes available after formation finishes; Replay weave rebuilds it.

The SVG animations require no JavaScript or external assets:

```html
<img src="weft-mascot-forming.svg" width="160" height="160" alt="Weft">
```

`weft-mascot-empty.svg` is the reverse animation's static endpoint. Generated JavaScript metadata supports the local preview; it is not required to embed the animated SVG alone. Evidence-report headers use the static logo.

For sprite playback, apply the frame pivot to a fixed destination anchor:

```js
const { rect, pivot } = atlas.frames[name];
context.drawImage(sheet, ...rect,
  footX - pivot[0] * scale, footY - pivot[1] * scale,
  rect[2] * scale, rect[3] * scale);
```

The layered SVG is source for Rive or a vector-to-Lottie workflow, not a finished `.riv` or Lottie animation. Target-editor import, rigging, and export remain unverified. Preserve the limb overlaps behind the body and keep the weave attached to it.

## Rebuild

Native PNG export requires macOS and Swift/AppKit. The other scripts use Python's standard library. Do not hand-edit generated exports.

```sh
python3 scripts/brand/build_mascot_poses.py .local/brand-export
swift -module-cache-path .local/swift-cache scripts/brand/export_brand.swift assets/brand .local/brand-export
python3 scripts/brand/build_mascot_intro.py
python3 scripts/brand/sync_brand.py
```

Check generated animation and packaged resources without rewriting:

```sh
python3 scripts/brand/build_mascot_intro.py --check
python3 scripts/brand/sync_brand.py --check
```

The original visual direction was image-generated. The current mascot is a locally authored vector; its PNGs are direct native exports. The wordmark is retained as a raster source. No hosted resources or recurring services are required.
