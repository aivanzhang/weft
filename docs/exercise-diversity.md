# Exercise diversity study

Three additional real videos now expose both executable cases and important reconstruction failures. They do **not** establish accurate full-body physics. All are development inputs inspected during implementation; none is a final holdout or a measured intervention trial.

## Inputs and results

Original bytes and attribution are retained in [data/exercise](../data/exercise/README.md). These three clips are Taco Fleur / Cavemantraining works under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The set varies movement and setting but does not provide demographic diversity.

| Video | Pose result | Environment result / remaining mismatch |
| --- | --- | --- |
| [Kettlebell crush-grip push-up](https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm), 8.6 seconds | 127/129 usable samples; 125 partially tracked; far arm mostly uncertain | Body executes all 48 joint/action checks. The upright anchored torso does **not** reconstruct the horizontal, supported body or kettlebell load. |
| [Kettlebell lunge and twist](https://commons.wikimedia.org/wiki/File:Kettlebell_Lunge_and_Twist.webm), 20.6 seconds | 293/308 usable samples; 13 invalid geometries; 2 multiple-pose estimates | Explicit abstention, not a successful reconstruction. The two estimates overlap near the clip’s end; both remain preserved rather than silently selecting one. |
| [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm), 45.3 seconds | 572/678 usable samples; 40 multiple-pose estimates, 56 low-confidence, 6 invalid geometry, 4 no-pose samples | Explicit abstention. Lateral motion, floor transitions, jumps and torso movement exceed the current body’s mechanics even where tracking succeeds. |

“Usable” means a documented visibility/geometry rule accepted the model estimate, **not** that an independent annotation verified it. Inspect the source overlays and all per-joint confidence values.

## Problems found and corrected

1. **False incomplete-decode errors.** OpenCV metadata reported 259/618 frames for push-up/lunge, but each original contains exactly 257/616 unlaced encoded video blocks, all of which decoded. The fix counts bounded Matroska video blocks independently when metadata differs by more than one frame. It rejects truncated elements, ambiguous video tracks and unsupported laced video rather than simply relaxing a frame-count threshold. Audio blocks do not count as video frames. The parser follows the [Matroska element specification](https://www.matroska.org/technical/elements.html) and [block-layout notes](https://www.matroska.org/technical/notes.html). This is a bounded frame-count audit, not a complete container validator.
2. **Cropped joints could appear reliable.** A landmark outside the supported image bounds now contributes zero joint confidence even when the model gives it high visibility. The original inferred landmark and confidence are still retained.
3. **Unsupported movement lacked measured diagnostics.** `pose-quality.json` / `evidence.json` now report the fraction of usable poses with lateral limb components above 0.35, plus camera-relative torso-tilt estimates. The fitted body preserves these diagnostics and joint-limit clamping coverage. These diagnose discarded pose information; they are not physical-accuracy scores. Lateral fractions were 16.5% push-up, 93.5% lunge, and 40.9% jumping-jacks/burpees.
4. **Reference testing silently stopped after 15 seconds.** Servo evaluation now covers the full reference within the existing 60-second / 1,800-step bounds and reports evaluated duration, coverage and per-joint errors. Its comparison remains against inferred poses, not independent motion capture.

## Preserved evidence

- `artifacts/exercise-diversity-kettlebell-pushup-01/` and `artifacts/exercise-diversity-lunge-twist-01/`: original decode failures, complete sampled landmarks and source previews.
- Corresponding `-02/` directories: post-fix decode evidence and quality results. Push-up also includes all 48 trials (eight joints × six actions × 30 physics steps) and the reference servo trace.
- `artifacts/exercise-diversity-jumping-jacks-burpees-02/`: complete source/landmarks, explicit quality failure and model-coverage diagnostics.
- `artifacts/exercise-diversity-diagnostics-03/`: final diagnostics recomputed from hash-pinned saved landmark evidence without repeated inference; current implementation copies and a local integrity manifest. Push-up servo RMSE is 0.16754 radians across eight joints, with full 8.47-second usable-reference coverage; left/right elbow errors are 0.25689/0.27651 radians. These values measure programmed controller tracking, not recovered muscle control.

The direct studies intentionally did not rerun training or claim full lab completion. The benchmark runner performs separate fresh end-to-end runs. No old run or source was overwritten.

## Verification and limits

Twenty-four focused tests passed across `test_exercise_vision.py` (15), `test_human.py` (7), and `test_exercise_source.py` (2). New tests cover encoded video/audio counting, truncation/lacing rejection, out-of-frame confidence, horizontal-torso disclosure, reference evaluation beyond 15 seconds, source pins and preservation of conflicting files. Pose contact sheets were visually inspected; no browser interaction was used.

All work used local CPU and free public downloads, approximately 10.8 MB for the three originals; external inference/cloud cost was $0. The saved failures remain useful next-step evidence: identity association, more complete joints, root motion, contacts/load coupling, whole-exercise task learning, independently annotated pose tests, and measured physical validation are still needed. An executable anchored model is not completion of the exercise or arbitrary-video goal.

The subsequent [full suite](diversity-verification.md) verifies elbow-task selection, learning, interaction and recording on the push-up source. This fixes the hardcoded-knee task, not the anchored-body fidelity limitation.
