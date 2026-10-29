# Tilted supports and a recordable ramp sandbox

Weft can now load tilted rigid boxes, set friction per surface, retain angular motion and record the resulting contacts. **The two new ramp sandboxes are authored, source-linked hypotheses—not accurate video reconstructions.** The exercise sandbox remains separate.

The [source-derived follow-up](source-ramp-reconstruction.md) adds nine playable hypotheses using opening tracked motion and approximate camera-aligned supports. Its complete later-frame comparison exposes collision errors; it does not replace or validate these authored examples.

## Try it

Open the [offline ramp overview](../artifacts/oriented-ramp-sandbox-01/index.html). It links the complete original videos, simulation replays, changed-action examples and assumptions. The views are not camera-aligned or time-synchronized.

For actual interaction, run this from the repository root with a new recording directory:

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/oriented-ramp-sandbox-01/ramp-20 \
  --record artifacts/my-ramp-session
```

Enter `coast 6`, `right 6`, `left 6`, `up 6`, `down 6` or `brake 6`. Each command advances six simulation steps; nothing moves while you type. `status` shows the state, `reset` restarts, and `quit` saves the video and replay. Open the recording folder's `current-frame.png` for the current picture. Arrows apply virtual forces, not a learned policy or measured physical intervention.

Saved examples:

- [10° stationary replay](../artifacts/oriented-ramp-sandbox-01/ramp-10-coast/replay.html)
- [20° sliding and collision replay](../artifacts/oriented-ramp-sandbox-01/ramp-20-coast/replay.html)
- [20° right-push intervention](../artifacts/oriented-ramp-sandbox-01/ramp-20-push/replay.html)
- [Verified terminal session](../artifacts/oriented-ramp-sandbox-01/terminal-session/replay.html)

## What changed

`planar-rigid-v2` adds required units, initial angle/angular velocity, per-object friction and boundary friction. Length remains normalized to scene width; mass is relative. It does not silently reinterpret either as meters or kilograms. Coordinates remain right/down, angles are radians, and time is seconds.

Rotated collision bounds and separating-axis checks reject initial intersections and corners outside the world. Static boxes can span longer supports. Integration uses local thickness and accounts for rotation; snapshots retain angular velocity, substep count and any speed/integration limits. Recordings reject missing rotation or diagnostics. The old `planar-rigid-v1` inputs retain their behavior.

Contact friction is the product of the two shape coefficients: 0.5 × 0.5 gives 0.25. This follows [Pymunk's contact rule](https://www.pymunk.org/en/latest/pymunk.html#pymunk.Arbiter.friction). One coefficient covers sticking and sliding; separate static/kinetic material properties, deformation and recovered contact forces are not implemented. The existing four-value RL observation is still target offset and controlled-body velocity; full geometry and contacts are available in snapshots. No policy was trained here.

## Verification

**October 29, 2026, 2:47 AM PDT.** There are 151 distinct passing checks across `.local/oriented-scene-tests-01.xml`, `oriented-integration-tests-01.xml` and `oriented-recording-tests-01.xml`. These cover analytic stationary/sliding cases, equal-mass rotated collision, thin-support impact, schema rejection, recording integrity and existing planar integrations. Three pre-change v1 coast/control/jitter traces retain exactly the same hashes.

The [saved study](../artifacts/oriented-ramp-sandbox-01/report.json) freezes both scenarios before running two bounded local workers. Four 120-action runs plus an 18-action/reset terminal session preserve **504 states**, all replayed exactly and fully decoded from their exported videos. All declared qualitative development checks pass:

| Authored condition | Coast block travel | Target-box travel | Right-push difference |
| --- | ---: | ---: | ---: |
| 10° | 0.00017 scene widths | 0.00017 | 0.21585 |
| 20° | 0.50709 | 0.03532 | 0.02892 |

The 20° run records block/target contact; the 10° run remains still within solver tolerance. None of these four runs reaches a speed or integration cap. These are simulation results, not measured source-video errors.

Four-object step/snapshot p95 latency is 0.068–0.091 ms. A separate 12-body rotating/contact load check takes p95 0.247–0.251 ms including JSON serialization across two repeated 600-step runs. Seven steps per load run reach an explicit envelope cap; none escapes. That stress case measures runtime, not unconstrained physical fidelity. Evidence: `.local/oriented-load-check-01.json`. These measurements exclude browser/HTTP responsiveness; no browser interaction or server was started.

The 163-file study seal is `d4c0bc70d468507a9dec7389c5111aa05950f8b04a76fcc0b06101ed0c49fc06`. Original sources, frozen scenes and the implementation snapshot remain unchanged. No new API request was made; aggregate Meta usage remains **$0.176152 of $5**.

## What the video does not establish

The [Physics 101 source](phys101-ramps.md) supplies ramp angles, object identity and reported block mass/volume. Camera calibration, physical dimensions, friction and green-box mass remain unknown. The sandbox's geometry, mass ratio, scale and initial rest state are authored. Saved tracking labels and masks were not used to fit it, and hand release is not simulated. Both complete original clips remain available.

Next work must constrain those unknowns with construction evidence, retain all source intervals and evaluate trajectory/contact predictions separately. Changed virtual controls demonstrate an interactive environment; they do not validate predictions under changed physical actions. Broader video coverage and exercise limb identity, foot contact and muscle/anatomy accuracy remain unfinished.

To reproduce into a new directory, without API calls:

```sh
.venv-lab/bin/python scripts/studies/build_ramp_sandbox.py \
  --out artifacts/oriented-ramp-sandbox-new
```
