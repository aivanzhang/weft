# Source-derived ramp hypotheses

The ramp experiment now uses the video's opening motion and visible supports to construct **nine playable 2D hypotheses**. It does not establish an accurate physical reconstruction: settings that explain the opening slide disagree after the block reaches the table and hits the green box.

Open the [offline comparison](../artifacts/source-ramp-reconstruction-01/index.html) to see the complete source, six aligned inspection panels, every hypothesis's errors and recorded controls. The existing exercise sandboxes are unchanged.

The later [3D rigid-object experiment](spatial-rigid.md) adds depth/rotation and a second source-camera inspection, with complete interactive recordings. It preserves this 2D study and still fails to match the real collision; additional dimensions alone do not resolve the unknown physical parameters.

## Interact and record

From the repository root, using a new recording directory:

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/source-ramp-reconstruction-01/friction-025-mass-1 \
  --record artifacts/my-source-ramp-session
```

Type `right 6`, `coast 6`, `brake 6`, `reset` or `quit`. Each command advances only the requested simulation steps. The recording directory's `current-frame.png` shows the current state; quitting exports the video and step-by-step replay. The parking target and controls are authored, not recorded physical interventions. No policy is trained in this study.

The middle friction/mass setting is the predetermined inspection example, **not a selected best result**. Saved examples: [coast replay](../artifacts/source-ramp-reconstruction-01/representative-coast/replay.html) and [controls/reset replay](../artifacts/source-ramp-reconstruction-01/terminal-session/replay.html).

## Construction and comparison

The original 44-frame [Physics101 clip](phys101-ramps.md) is unchanged. Existing Meta tracking and decoded masks are reused; there are no new paid calls or model inferences.

1. Select approximate ramp/table top lines in source frame 8. The ramp endpoints come from an image line proposal; table endpoints combine two visible top-edge segments. These are source-specific choices, not a general support detector.
2. Assume both supports and tracked centers share one camera plane. A rotation and vertical scale map the table to horizontal and the ramp to its provider-declared 20° angle. This is an affine camera hypothesis, not metric calibration.
3. Fit position, velocity and acceleration to the eleven saved wooden-block centers at frames 8–18. Keep the tangent component of initial velocity; record the discarded normal component. The green box starts at rest.
4. Use initial mask widths and effective heights that place both box centers above their supports. That imposed contact is an assumption, not recovered 3D shape. Retain the visible gap between ramp and table; support thickness is authored.
5. Freeze all nine scenes before comparison: pair friction .20/.25/.30 crossed with green-to-wood relative mass .25/1/4. Table friction is assumed equal to ramp friction; restitution is zero. Each scene can be loaded by the existing terminal sandbox.
6. Simulate every later frame 19–43, including ramp exit, table travel and collision. Retain all 88 object/frame slots per hypothesis: 22 construction, 50 later comparisons and 16 pre-initialization slots without a simulation or returned tracking.

The complete clip and sparse labels were exposed during earlier development. These later frames are excluded from this fit but are **not an untouched final holdout**. Saved model identity becomes unresolved after the objects approach; comparisons retain the original IDs rather than rematching by error.

## Why several settings explain the opening slide

For an ideal sliding box, downhill acceleration is `gravity × (sin(angle) − friction × cos(angle))`. With unknown real-world scale, the observed motion does not uniquely identify friction and normalized gravity.

The fitted downhill acceleration is 1.00637 scene widths/s². Three explicit assumptions explain it:

| Assumed pair friction | Inferred gravity, scene widths/s² | Conditional meters per scene width if gravity is 9.81 m/s² |
| --- | ---: | ---: |
| .20 | 6.5314 | 1.5020 |
| .25 | 9.3968 | 1.0440 |
| .30 | 16.7415 | .5860 |

These scales are conditional calculations, not measurements. The simulator's two shape coefficients multiply to the pair coefficient; each shape uses its square root, following [Pymunk's contact rule](https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.friction). The 10° clip is not reconstructed or rescored in this study.

## Results

**October 29, 2026, 8:08 AM PDT.** All nine scenes execute through the complete comparison interval without numerical caps. Six record the block/target collision; the three highest-friction scenes stop before it. Lower center error means closer agreement with saved model tracks, not verified physical behavior.

| Assumptions | Later wood center RMSE, pixels | Later green center RMSE, pixels | Simulated collision |
| --- | ---: | ---: | --- |
| Friction .20, target mass .25 | 27.73 | 7.55 | Yes |
| Friction .20, target mass 1 | 15.64 | 48.97 | Yes |
| Friction .25, target mass 1 — fixed inspection example | 43.42 | 75.39 | Yes |
| Friction .30, target mass 1 | 108.00 | 78.19 | No |

The fixed inspection example is close to the unchanged approximate source labels at frame 27: wood 10.70 px and green .19 px. At frame 43 its errors grow to 61.75 px and 115.96 px, outside the labels' 12-pixel uncertainty. The [frame-27 panel](../artifacts/source-ramp-reconstruction-01/frame-0027.png) and [frame-43 panel](../artifacts/source-ramp-reconstruction-01/frame-0043.png) show the difference. Across the family, only 3–5 of the six sparse label slots pass; two slots are inside the construction interval. No candidate is promoted.

## Verification and evidence

- Eleven analytic checks cover camera round trips, support angles, known quadratic motion, friction/scale ambiguity and invalid inputs: `.local/ramp-inference-tests-01.xml`.
- Nine 70-action simulations plus 18 manual-control actions retain 648 transitions. The fixed coast recording has 71 states/video frames; controls plus reset have 20 states. All 91 recorded states replay exactly.
- Two bounded local workers complete fitting, simulation, exports, terminal checks and panels in 1.468 s, excluding initial source verification/snapshot preparation. Per-case step/snapshot p95 is .336–.455 ms. These are local measurements, not browser/HTTP responsiveness.
- Final verification retains all 792 object/frame slots, including all 450 later slots; checks exact source IDs, timestamps and original model identity; decodes all 44 original frames; checks six panel source bands, 28 local links and the unchanged 109-file runtime snapshot. No successful fitting, simulation, recording or test is repeated.
- The [170-file study](../artifacts/source-ramp-reconstruction-01/report.json) seal is `efa7cce62eb6127c1cccee743454c78eea6f0eeaaa807c0372d0a7eca74162e4`. Final audit: `.local/source-ramp-final-verification-01.json`.

`ramp_inference.py` remains an explicit experimental helper, not an ordinary lab option or replacement for existing sandboxes. Before broader integration, constrain camera/scale, actual collision shapes, table friction and impact response with additional evidence. Preserve missing intervals and unresolved identities, and independently test changed physical actions. The new aligned comparison does not complete general-video reconstruction or exercise anatomy/contact accuracy.

To reproduce into a new directory without API calls:

```sh
.venv-lab/bin/python scripts/studies/check_ramp_reconstruction.py \
  --out artifacts/source-ramp-reconstruction-new
```
