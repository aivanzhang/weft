# Ball and rigid-scene development probes

## What changed

The planar tracker now removes **configured** gravity's quadratic displacement before estimating initial velocity. Previously the first-five-frame straight-line fit estimated velocity near the interval midpoint, then the simulator applied gravity again from time zero. This fix does **not** infer gravity from video.

A bounded background-motion screen samples at approximately 2 Hz, fits a robust 2D transform to at most 120 low-saturation background corners, and requires at least 12 matching points spread across 20% of the image. Large translation, rotation or scale change rejects the fixed-camera profile with a saved `camera-motion.json`. Insufficient features produce **unknown**, not “camera verified.” Widespread moving background can also trigger rejection; this is conservative abstention, not camera recovery or compensation.

## Independent synthetic videos

`weft.ball_benchmark.create_analytic_probe(name, out)` creates a fresh `video.mp4`, explicit `config.json`, `truth.json`, and `provenance.json`. The trajectories use closed-form equations, **not Pymunk/SceneEnv output**. The tracker receives video/config only; analytic coordinates are used afterward for scoring. Every video fully decodes after encoding. The exercise/body pipeline is not exercised by these probes.

```python
from weft.ball_benchmark import run_analytic_probes
run_analytic_probes("artifacts/ball-diversity-new")  # directory must not exist
```

Each suite preserves source and tracker implementation copies, source hashes, rejected cases, complete measured tracks, fitting ledgers, previews, numerical results and a verifiable file manifest. Existing directories are never overwritten.

### Retained verification

`artifacts/ball-diversity-verified-01/results.json` records eight probes, completed in **5.14 seconds** on this machine. `weft verify artifacts/ball-diversity-verified-01` verified **108 files**; manifest SHA-256 `820d00c2bc1f67cd17d6decccfd25a39244cd785334d998b4f12b9514cd991fe`.

All coordinates below are pixels in **480 × 288** videos. “Analytic replay” compares simulated positions against independent authored equations over all visible scored frames. “Development replay” compares against the tracker's later same-video observations. Neither establishes real physical accuracy.

| Case | Detections | Tracking RMSE | Analytic replay RMSE | Development replay RMSE | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Constant motion | 45 | 0.33 | 0.57 | 0.94 | Reconstructed |
| Fixed camera with textured background | 45 | 0.31 | 0.33 | 0.54 | Reconstructed; no large background motion detected |
| Gravity flight | 45 | 0.41 | 2.69 | 3.71 | Reconstructed; gravity supplied |
| Gravity bounce | 66 | 0.41 | 5.78 | 8.58 | Reconstructed; gravity/restitution supplied |
| Equal-mass elastic collision | 108 | 0.36 | 2.74 / 5.80 per ball | 6.55 | Reconstructed; restitution supplied |
| Brief four-frame occlusion | 41 | 0.32 | 0.59 | 0.96 | Reconstructed; missing frames retained/excluded |
| Two same-color shapes | — | — | — | — | Explicit ambiguous-identity rejection |
| Translated camera/background | — | — | — | — | Explicit moving-background rejection at frame 15 |

The existing per-object replay gate remains 0.08 image widths; it was **not loosened**. The tests also check subpixel tracking and per-object analytic replay below 10 pixels for these small probes. These are development regression thresholds, not preregistered physical acceptance limits.

### Before and after

Unchanged source hashes and comparison outputs remain in `artifacts/ball-diversity-baseline-01` and `artifacts/ball-diversity-improved-01`:

- Gravity-flight development error: **16.13 → 3.71 px**.
- Gravity-bounce development error: **16.39 → 8.58 px**.
- Camera translation previously produced an apparently passing 1.62 px same-video replay while confusing camera motion with object motion. It now rejects the unsupported input.
- Constant-motion, elastic-contact and brief-occlusion results remained unchanged.

The first two exploratory suites predate suite-level implementation-copy/manifest preservation. Their artifacts are retained as development observations, not fully restorable implementation snapshots. The final verified suite includes source copies.

## Additional real source

[ScienceCOLA's Bouncing Ball](https://commons.wikimedia.org/wiki/File:Bouncing_Ball.webm), licensed CC BY 3.0, is preserved unchanged under [data/balls](../data/balls/README.md). Inspection shows a **suspended electrostatic ball**, not an ordinary gravity bounce. It expands the unsupported-scene inventory but does not demonstrate successful reconstruction.

The original exploratory run failed the then-current strict decode gate: metadata reports 661 frames while OpenCV decodes 656. The preserved failure is `artifacts/ball-real-electrostatic-01/failure.json`. That historical rejection is a decode/count limitation, not evidence of semantic understanding or correct electrostatic inference. The later [full suite](diversity-verification.md) independently verifies encoded frame counts and reaches the actual unsupported low-saturation/color-tracker error; it still does not produce an environment. The original [real rolling-ball input](../data/README.md) remains separate.

## Verification and remaining work

- `pytest tests/test_scene_vision.py tests/test_demo.py -q`: **18 passed** after the tracker fixes.
- `pytest tests/test_ball_benchmark.py -q`: **2 passed**, covering eight analytic cases plus provenance/no-overwrite behavior.
- Final saved suite rerun after implementation-copy support: all eight expectations met; every supported case has tracking RMSE < 1 px and every object's analytic replay RMSE < 10 px; 108-file manifest verified.
- No paid calls or cloud resources; local CPU only. No browser interaction.

Still missing: calibrated real bounce/collision trials; same-color multi-object identity tracking; robust long occlusion; automatic gravity/friction/restitution estimation; camera compensation; nonplanar objects; external intervention validation; diverse real video success beyond the original rolling fixture. Bounce/contact discrepancies remain measurable and should guide subsequent physics and geometry work. Passing these synthetic probes does not complete generalized-video reconstruction.
