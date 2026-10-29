# Fresh full-body squat: baseline ready, reference-floor check failed

The unchanged baseline built an interactive mobile body from the complete new squat clip in **20.87 seconds**. The selected single-joint learning task passed, but the full inferred pose trajectory still conflicts with the authored floor: **13/34 source poses** exceed the unchanged 0.025-model-unit limit, with maximum penetration **0.073842**. The predeclared gate therefore prevented additional whole-body training and manual recording.

Evidence: [study report](../artifacts/fullbody-exercise-01/report.json), [source comparisons](../artifacts/fullbody-exercise-01/source-grading.json), [full-reference floor audit](../artifacts/fullbody-exercise-01/baseline-feasibility.json), [frozen command/settings](../artifacts/fullbody-exercise-01/pre-run-freeze.json), and [study script](../scripts/studies/check_fullbody_exercise.py). The study verified **190 files**; its baseline run separately verified **108 files** and remained unchanged. Study manifest SHA-256: `c91a60d15d78d66a77a46bf3de982ce7aa66298b7fb2451bc00c348806bfb1fd`.

## Source and fixed run

The source is Danielflefil's *Bodyweight Squats*, a complete official 500×500 GIF cycle converted to a lossless RGB MP4. The retained [source package](../artifacts/exercise-fullbody-source-01/report.json) records attribution, CC BY-SA 4.0 provenance, and frame-by-frame pixel/timestamp identity. This is an edited animation derivative—not raw-camera timing or calibrated physical observations.

- All **68 frames** were decoded: no trimming, resizing, replacement, or successful-interval selection.
- The unchanged inference stride produced **34 pose samples**, at frames 0, 2, …, 66.
- Settings: MediaPipe Lite, `video` mode, `temporal` identity handling, `mobile` body, 100 training episodes, 100 evaluation episodes, seed 1729, no server, 240-second study limit.
- No fitting candidate, root lifting, floor adjustment, parameter tuning, API calls, or paid inference was used.

The original source labels retain their **GIF hash**, `14f1febdd9827951c0099ae2df6288fa5afb7b83c1fb99882f7505486982af85`. The verified conversion bridge links them to MP4 hash `ec636e28b4711887ecd9ff7014b64801d16055faf2f2b0bf6ed9221ea8280118`; the label identity was not overwritten.

## Source-only image comparisons

All five preselected labels remain in the result. Raw pose samples are matched within one source frame, with earlier-frame tie breaking. Labels are approximate assistant visual estimates, including a subjective 20-pixel pelvis allowance—not measured anatomy or expert truth.

| Label frame | Compared raw pose | Time offset | Pelvis error | Body-reference projection |
|---|---:|---:|---:|---|
| 0 | 0 | 0 ms | 6.61 px | Available |
| 13 | 12 | −40 ms | 13.82 px | Available |
| 33 | 32 | −40 ms | 1.70 px | Available |
| 53 | 52 | −40 ms | 4.31 px | Available |
| 67 | 66 | −40 ms | 6.45 px | **Unavailable: outside reference time coverage** |

Raw pelvis estimates are within the subjective allowance at **5/5** slots; RMSE is **7.72 pixels**. This supports approximate image-location agreement only. Frame 67 is at 2.68 seconds, beyond the final body reference at 2.64 seconds: its body projection remains missing rather than clamping the last pose to manufacture coverage.

### Feet and floor are not interchangeable labels

The manual labels identify a near/larger and far/smaller visible sole-boundary proxy, with **no anatomical left/right assignment**. The pose model provides heel/toe landmarks; their midpoints are different point definitions. The diagnostic retains both possible anonymous pairings and the full distance matrix. Minimum-pair mean discrepancies span **14.23–21.25 pixels**, but are not scored as anatomical-foot or contact successes. At the deep-squat comparison, one proxy has confidence **0.406**, which remains visible in the audit.

The body model has ankle endpoints and lower-leg capsules, not anatomical feet or sole geometry. Its x/−z screen projection uses an inferred pelvis origin and uncalibrated scale. Under that authored projection, the horizontal simulated floor would appear at approximately **y=452.17 px**, while the manual near/far sole proxies are at y=484/450 px. Perspective, depth, ground-plane orientation, and model base height are unresolved; this difference is not a measurement of actual floor height, contact, or force. Model ankles were not relabeled as soles.

## Separate simulator outcomes

| Check | Result | What it establishes |
|---|---|---|
| Selected `right_hip` target task | Learned 100/100; random 0/100; coast 0/100 | A single authored angle-target task, not whole-exercise control |
| Authored reference servo | Angular RMSE 0.06998 rad; root RMSE 0.02270 model units | Tracking under virtual support and contact dynamics |
| Servo's dynamic maximum penetration | 0.001731 model units | Contact-constrained rollout, not feasibility of every target pose |
| Static source-reference floor audit | **13/34 failed; maximum 0.073842** | Inferred target geometry conflicts with the authored floor |
| Static 30 Hz interpolation audit | **30/80 failed** | The conflict also exists between sampled references |

The affected geometry is `left_knee_geom`, which names the **shin capsule extending from knee to ankle**, not an isolated knee sphere. Dynamic contact can keep a servo-controlled body above the floor while it deviates from an infeasible target; the small dynamic penetration does not override the static reference failure.

## Decision and inspection

Keep this baseline and all failures. Full-body visibility improved source coverage, but did not by itself establish correct depth, shape, root scale, floor alignment, or support. The optional 20-step, eight-generation/16-candidate whole-body search and 12-step manual recording were **not run**, because the predeclared full-reference feasibility condition failed. No alternative settings or trimmed source were substituted.

The saved baseline can still be inspected interactively:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/fullbody-exercise-01/baseline
```

Its inferred poses, illustrative motor/muscle displays, virtual support, and recordings must remain distinct from physiological measurements or independently validated balance. This source is a development case selected before model inference, not a final holdout.
