# Initial-speed repair for longer clips

The new construction-only velocity proposal reduces long-clip drift without moving the starting body or shortening evaluation. It is an additional candidate, not a replacement for the original model. Later-frame scoring rejects it when it worsens a result.

## Try the improved environment

```sh
.venv-lab/bin/python -m weft lab --resume \
  artifacts/initial-motion-01/labs/appearance-constant
```

Use the evidence panel to inspect full frame coverage, the repair panel to compare proposals, and the sandbox to act and record. This example is a deterministic 20-second synthetic video, not a physical recording. The [saved interaction replay](../artifacts/initial-motion-01/labs/appearance-constant/sessions/session-f0b6bc24d6d3/replay.html) opens without a server.

## Method and limits

The existing initializer keeps its original first-five-frame estimate. After the bounded damping search, `initial_motion.propose_initial_motion` receives **construction observations only**, never later development positions or scores. It fits at most the first second under the selected gravity/damping hypothesis.

- Require 8–241 samples over at least 0.25 seconds, with no gap longer than 0.2 seconds. Do not interpolate or discard outliers to improve the fit.
- Screen for possible wall/object contact using the authored 2D geometry. Missing neighboring-object evidence makes this screen unavailable.
- Fit initial velocity with a continuous free-flight equation. Pymunk uses discrete integration, so this is a proposal rather than an exact inverse of the engine.
- Keep the actual initial position fixed. A nuisance intercept estimates velocity but cannot shift the body. Construction residual limits are 1 px RMS / 2 px maximum; speed is bounded to three image widths per second.
- Preserve gravity, geometry, mass, task, source frames and scoring thresholds. Static objects remain unchanged.
- Apply the existing later-frame rule: at least 2% error improvement with no object-level regression under the unchanged allowance. A better aggregate result cannot hide a worse object.

The audit records the contract, construction IDs, source frames, rejected assumptions, proposed speeds and final decision in `initial-motion-proposal.json`. At most one velocity proposal follows the existing three damping proposals. No force, contact, metric scale or real-world accuracy is inferred from a passing image fit. Small hidden forces and between-frame contacts may remain unobservable.

## Frozen development comparison

The same saved evidence, complete intervals, initial geometry and later-frame scores were reused from the [full-interval study](full-interval-reconstruction.md). No labels, tolerances or source clips changed.

| Source | Previous error | Selected error | Decision |
|---|---:|---:|---|
| 20-second appearance control | 15.72 px | **3.22 px** | Velocity proposal accepted; now passes the 10 px study target |
| 20-second colored control | 14.68 px | **1.64 px** | Velocity proposal accepted; now passes the 10 px study target |
| Real rolling ball, declared pass | 3.12 px | **3.12 px** | Candidate worsens error to 4.32 px and is rejected |
| Appearance clip with later direction change | 39.84 px | **36.74 px** | Partial improvement accepted, but reconstruction still fails |
| Colored clip with later direction change | 35.91 px | **35.91 px** | Improvement too small; reconstruction still fails |

All 200 frame slots remain in each long synthetic comparison. An accepted parameter change does not override a failed reconstruction gate. [Improved constant motion](../artifacts/initial-motion-01/fits/appearance-constant/reconstruction-contact-sheet.png) and [still-failing later motion](../artifacts/initial-motion-01/fits/appearance-late-motion/reconstruction-contact-sheet.png) were inspected locally.

The four-case full pipeline completes in 4.31 seconds with two local workers: three declared gate passes and the expected extended-real-clip coverage rejection. Each runnable case trains for 200 episodes, evaluates ten trials per policy, verifies changed-action effects and records **12 transitions / 13 exported frames**. Run manifests remain unchanged by recording. These are short implementation checks, not robust control or physical-transfer validation.

## Verification and reproducibility

**77 distinct Python checks pass:** 18 new analytic checks plus 59 integration checks. They include quantization, damping/gravity, impulses, acceleration, outliers, gaps, contacts, fixed geometry, evidence isolation and rejection of object-level regressions. Eight analytic ball-video controls retain six reconstruction passes and two expected ambiguity/camera rejections. Their already-passing execution was copied with matching implementation hashes instead of rerun.

The [598-file study](../artifacts/initial-motion-01/report.json) has seal `b94840119082f20cd3d7d649026f59cba3b0715f93d23a3f967cfdcabbe7cd23`. The original failures and implementation snapshots remain intact. Repeat only into a new directory:

```sh
.venv-lab/bin/python scripts/studies/check_initial_motion.py \
  --out artifacts/initial-motion-repeat \
  --analytic artifacts/initial-motion-01/analytic-controls
```

The runner refuses to reuse analytic results if their implementation no longer matches. The saved original runner snapshot records its original temporary test-source path; the current runner accepts the durable copied bundle above.

Three local timing samples per prepared workload give a 0.096 ms median for one object and an **11.56 ms median / 11.64 ms maximum** for 8,640 construction rows, eight objects and a 240 fps source. The latter fits 241 samples per object from the first second. This measures the proposer only, not replay or sustained UI/CPU/memory behavior. [Timing evidence](../artifacts/initial-motion-performance-01/report.json): five-file seal `6618d0f4665d3defc92bce115c9c65d333bd3473e9efca505447af109f09d623`.

No API calls or recurring services were added. Exercise floor/contact, articulated feet, camera/projection and broader object/force recovery remain separate unresolved work.
