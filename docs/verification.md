# Iteration setup verification

Observed **October 27, 2026, 12:29 a.m. PDT** in `/Users/ivan/kintsugi/weft-v1` on `feat/weft-v1`, base revision `a35070c25ec3e00a063e711a0958497d14c6676f`. Verification preceded the first implementation commit; Git history records subsequent commits and remote tracking. This verifies the iteration foundation and restricted fixture, not the full video-to-world target.

## Checks

- `.venv/bin/python -m pytest --junitxml=.local/iteration-setup-tests.xml`: **46 passed**, exit 0. Coverage includes bounded proposal rejection, analytic motion, Gymnasium reset/step semantics, Python/JavaScript physics parity, real-video provenance/partitions, artifact tampering, config limits, source pinning, and incompatible comparison guards.
- `.venv/bin/python -m pip check`: no broken requirements, exit 0.
- Both checked-in experiment presets completed, with **15,000 training episodes** and **300 evaluation episodes per policy**, using distinct output directories.
- Both runs passed **8 restricted-engine analytic checks** and their development replay gate.
- Each run verified **45 artifact hashes**. All **21 copied implementation/build files** matched `implementation.json`.
- Generated viewer: all **9 local asset/link references** resolved, embedded JSON parsed, executable JavaScript passed Node syntax checks, and the terminal interface completed **3 action steps** and closed normally. The contact-sheet rendering was visually inspected. Browser interaction/visual QA was not performed; no development server was started.

## Measured results

| Metric | No repair | Repaired | Interpretation |
| --- | --- | --- | --- |
| Development replay RMSE | 11.4991 px | 4.2795 px | Matched evidence/scorer; development fit improved |
| Second-pass audit RMSE | 11.5164 px | 11.1592 px | Smaller improvement; same video, development-inspected tracker |
| Accepted repairs | 0 | 2 | Only velocity/drag were eligible |
| Learned task success | 300/300 | 300/300 | Simulated parking only; different fitted worlds |
| Random task success | 25/300 | 26/300 | Same seed list, but different world dynamics |
| Pipeline time reported | 3.3594 s | 3.4402 s | Single concurrent runs, not a speed benchmark; excludes final source copying/hashing |
| Artifact bytes | 2079849 | 2101784 | Actual logical file sizes, including source snapshots |
| External inference/cloud spend | $0 | $0 | Local CPU-only; no Meta request |

The comparator marks reconstruction comparable and RL **not comparable across these runs**. Neither the 100% success rates nor the replay improvement establishes real-world control or improved policy transfer.

## Evidence identities

Source SHA-256: `75621f8624ce677336f9bbd6c110b638063d0f75e1678460d85602026e4d3c86`.

- Baseline: `artifacts/iteration-baseline-02`; manifest `a1f1238dd0447cbe761b9945ce8b5686906b955e6cf7297cd8cd567fbb3984aa`.
- Candidate: `artifacts/iteration-repaired-02`; manifest `0b21d7582de846a13cd6f00d093b649d073ac404f70e1ad0df575ad6a93a9300`.
- Comparison: `artifacts/comparison-02.json`.
- Tests: `.local/iteration-setup-tests.xml`.

Run artifacts and test reports are local ignored evidence, not published assets. Preserve them; use new directory/file names for subsequent experiments. The [presets](../experiments/rolling-ball.json), pinned source, dependency lock, and source snapshots support reproduction.

## Unverified or intentionally absent

Multi-object/3D reconstruction, general video understanding, true hidden-parameter recovery, new physical intervention evidence, real-world policy transfer, live Meta access, broader input/platform performance, browser UI QA, and public/deployed functionality remain unverified. The [TODO](../TODO.md) retains the corresponding acceptance gates.
