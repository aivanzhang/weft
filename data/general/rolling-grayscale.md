# Grayscale rolling-ball derivative

`rolling-ball-grayscale.avi` is a lossless grayscale **derivative of the existing real recording**, not a new independent physical source. Attribution: Ecole Science (provider; individual author not listed), [Rollender Ball](https://hubbs.schule/mediathek/rollender-ball), [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).

All 157 source frames are retained at 640×360. OpenCV decoded the original, converted each frame with `COLOR_BGR2GRAY`, and wrote lossless FFV1 video. Verification checked that every decoded grayscale byte matched the original conversion. AVI rounds the frame rate from 23.976023976 to 23.976; the final-frame timestamp differs by less than 7 microseconds. Audio is not copied. Parent/derivative hashes and exact transformation details are pinned in `rolling-ball-grayscale.source.json`.

`annotations/rolling-grayscale.scene.json` configures the appearance profile. The selected source interval, frames **39–60 inclusive**, was authored from original-frame inspection **before tracking or fitting**: the first complete visible pass with the ball inside the authored boundary geometry. The clip starts with empty frames, shows two separate passes, and ends with a cut/black tail. This run does not reconstruct the second pass; all exclusions are explicit in the selection sidecar and run evidence. The derivative itself is not trimmed.

Direct inspection plus the unchanged planar fitting routine produced:

- 157/157 source frames decoded; 135 outside the authored tracking interval.
- Frame 39: authored initial rectangle; excluded from scored measurements.
- Frames 40–60: 21/21 accepted tracking observations.
- 12 generation observations, followed by 9 development observations.
- Development reconstruction RMSE: **3.1205 pixels**, or 0.004876 image widths; the existing per-object replay gate passed.
- No large background motion detected by the samples; this does not prove a fixed camera.
- No paid calls; no manual scoring labels or benchmark truth supplied to tracking/fitting.

Artifacts, including source/config/code snapshots and SHA inventory: `artifacts/appearance-rolling-grayscale-01/`. The reconstruction contact sheet was visually inspected. This direct study did not run the full lab worker or policy training. Root integration can use the exact saved source/config without changing tracker thresholds.

The circle size, unit mass, planar world, target, acceleration/braking controls, and world properties remain authored/inferred hypotheses. The image-space fit does not recover physical friction, depth, forces, or establish real intervention accuracy.
