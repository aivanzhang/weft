# Full-interval reconstruction checks

New appearance and colored-object runs no longer silently select the first convenient episode or shorten replay to six/ten seconds. They retain every frame in the declared interval and run one continuous simulation from the first common object measurement through the last frame. Missing observations stay missing; the simulation is not reset at a gap.

This improves coverage accounting, not general physical accuracy. A requested source section is still a section, not the whole original video. Old saved runs retain their original evidence and scoring semantics.

## Try the verified real-video run

Start the local lab yourself:

```sh
.venv-lab/bin/python -m weft lab --resume \
  artifacts/full-interval-reconstruction-01/labs/real-rolling-declared-pass
```

The evidence panel now shows the requested frame range, planned simulation range, frames before model initialization, and measured/missing object-position counts. **Inspect every frame** opens the detailed coverage artifact. The summary loads once, without fetching the full trajectory on every status update. Old runs without this report are explicitly labeled as lacking coverage information.

The real example uses the existing CC0 grayscale rolling-ball derivative, frames 39–60. Frame 39 is authored initialization, not an independent measured position; simulation covers 40–60. The later pass is not included or claimed reconstructed.

## What changed

- `reconstruction-coverage.json` records all requested frames, missing objects, chronological partitions and separate tracked episodes. `coverage-summary.json` supplies the small UI view.
- The first 60% of the modeled source timeline supplies construction evidence; later observations supply development checks. Gaps cannot move late observations into construction or disappear from replay.
- All available per-object observations in the modeled interval are used, including frames where another object is missing. Each object needs construction and development evidence.
- At least ten jointly tracked frames must support the initial continuous section. A later, easier section cannot silently replace the beginning. The appearance path's 80% coverage requirement remains unchanged.
- Replay retains missing-frame predictions as **unscored**. Before common initialization, no simulated state is invented. Shorter preview captions distinguish source, simulation and measurement availability.
- Limits remain 1,800 source frames, 60 seconds, and 1,800 physics steps. No tolerance, identity or camera gate was relaxed.

## End-to-end results

The [saved study](../artifacts/full-interval-reconstruction-01/report.json) uses two bounded local workers, 200 training episodes and ten evaluation episodes per policy on each runnable case. These are short pipeline checks, not evidence of robust learned control or physical transfer.

| Case | Interval and observations | Result |
|---|---|---|
| 20-second appearance control | All 200 frames retained; 195 measured positions | Runnable and recorded; 15.72 px error fails the stricter 10 px study gate |
| 20-second colored control | All 200 frames retained; 196 measured positions | Runnable and recorded; 14.68 px error fails the stricter 10 px study gate |
| Real rolling ball, declared pass | 22 frames retained; 21 measured positions | 3.12 px error; declared replay/action/recording gates pass |
| Real rolling ball, extended through frame 156 | 118 frames retained; 21 measured positions | Rejected for insufficient coverage and missing later evidence |

All three runnable cases verify changed-action effects and **12 recorded transitions / 13 exported frames**, with original run manifests unchanged. Their engine replay gate remains 0.08 image widths per object; at 240-pixel width this is 19.2 px. Passing that gate does not mean passing the study's separately declared 10 px target. Both failures remain failures.

Two additional synthetic clips change motion after the opening section. Full replay now rejects both: appearance **39.84 px**, color **35.91 px**, with all 200 frame slots preserved. A compressed colored frame also loses detection and remains unavailable. [Inspect the late-motion comparison](../artifacts/full-interval-preview-01/late-motion/reconstruction-contact-sheet.png).

The long controls expose a specific next issue: the existing first-five-frame initial velocity estimate is **4.00 px/s** for appearance and **(3.88, 0.27) px/s** for color, versus the generator's **(3.00, 0.00) px/s**. Quantized/localized centers can therefore create accumulating drift. That diagnostic did not apply a repair. The subsequent [bounded initial-speed candidate](initial-motion.md) now improves both long controls while preserving all later-frame evidence, gates and this original failed study. The real rolling candidate is rejected for regression.

## Verification and resource evidence

126 distinct Python checks pass after updating the old test that treated occluded frames as excluded from simulation. The initial assertion failure remains in `.local/full-interval-integration-tests.xml`; the corrected ball checks are in `.local/full-interval-ball-fixed-tests.xml`. Twenty-six deterministic UI checks pass. Actual browser/device interaction was not performed.

A maximum-frame/object stationary workload verifies all **14,400 positions** across 1,800 frames and eight objects: 1,799 physics steps, 0.0063 seconds planning and 0.211 seconds replay. Its serialized trajectory is 4,365,525 bytes. This is one local offline observation, not a sustained CPU/memory or browser-responsiveness guarantee; large JSON files remain downloadable rather than automatically rendered.

The 462-file study seal is `6bf290bfae6e1fb600abb02cd4d163f8fdccbc9ab969a935316d046904d530d0`. The presentation-only rerender uses saved states, with no new tracking, fitting or physics: eight-file seal `7f88e71711552a29d5854f37786973a3bbc13687813f5ae3b7ae282a9672f7b9`. Source and simulation comparison sheets were inspected locally. No API calls, deployment or recurring service was created.
