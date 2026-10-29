# Image-direction refinement: visual gain, unresolved floor regressions

The optional image-direction fit improved the frozen independent visual check from **18/28 to 24/28 passing segments**, but it is **not promoted**. Single-leg squat floor failures increased, push-up peak penetration worsened, and all three complete reference trajectories remain infeasible. No dynamic rollout was started.

Evidence: [sealed study](../artifacts/body-image-refinement-01/report.json), [independent limb grading](../artifacts/body-image-refinement-01/limb-grading.json), [frozen settings](../artifacts/body-image-refinement-01/pre-fit-freeze.json), and [study script](../scripts/studies/check_body_image_refinement.py). The manifest verified **47 files**, including exact original models/pose evidence, candidate references, full audits, annotations, implementation helpers, and tests. Manifest SHA-256: `d4383c5be24b90b24106838c585d5d6881d4fd03701666289915bfd42a18e4c1`.

## Frozen method and provenance

Twenty-one analytic tests passed before real fitting. The fitter compared eight projected model-bone unit directions with directions from retained image-landmark estimates. Normalized image coordinates were converted to pixels using source width and height, preventing aspect-ratio distortion. Each source-video checksum and the first decoded frame's actual dimensions were verified against the model and OpenCV metadata:

- Single-leg squat: 480×640.
- Squat: 1280×720.
- Push-up: 640×480.

The model projection stayed fixed at simulation x/−z; no camera alignment was fitted. Weights used the minimum endpoint visibility/presence, with missing, cropped, zero-confidence, or degenerate directions explicitly excluded from fitting and retained in the audit. Source-world points were preserved but **not used as targets**.

The frozen objective used a unit-direction Huber transition of 0.1, original-angle regularization 0.001, and temporal regularization 0.0005 on corrections. Bounds were ±0.15 radians per axis, original joint limits, twelve iterations and four backtracks per iteration, 900 frames, and 60 seconds per fit. Root references, floor, body lengths, orientation anchor, motor limits, and timestamps remained exactly unchanged. Floor distances and manual annotations were not optimization inputs.

The original-angle prior selects a nearby 3D solution; it **does not measure depth**. Multiple 3D poses can have the same image directions, and better projection agreement does not increase the original pose model's confidence or validate contact.

## Construction-fit results

These errors compare the candidate with the **model's own estimated image landmarks**, not independent labels.

| Clip | Mean direction error | Confidence-weighted angular RMSE | Summed weighted robust loss | Fit runtime |
|---|---:|---:|---:|---:|
| Single-leg squat | 3.878° → 0.326° | 5.077° → 1.374° | 2.2504 → 0.1685 | 0.192 s |
| Squat | 8.365° → 1.665° | 13.320° → 4.105° | 6.2267 → 1.0714 | 0.143 s |
| Push-up | 3.742° → 0.390° | 4.632° → 1.116° | 2.0420 → 0.1254 | 0.183 s |

All 345 existing reference poses received bounded fits, with no deadline failures. Each clip reached the 0.15-radian change bound somewhere. The audit preserves 944, 800, and 1,016 segment samples respectively, plus raw evidence frames unused by the original references. Runtime is a local measurement for these inputs, not a general guarantee.

## Independent source-view check

Only after all candidates were fitted did the study run the unchanged grader against the existing four annotated source frames. Its projection, time matching, angular tolerances, and unavailable slots stayed unchanged.

| Clip | Segments within tolerance | Mean angular error |
|---|---:|---:|
| Squat | 11/16 → **16/16** | 11.808° → 6.675° |
| Push-up | 7/12 → **8/12** | 12.410° → 10.196° |
| Combined | 18/28 → **24/28** | 12.066° → 8.184° |

Four additional source-occluded slots remain unavailable. Single-leg squat has no matching existing annotations, so no independent direction improvement is claimed for it. These are previously exposed assistant visual development estimates—not expert ground truth, new held-out data, or proof of whole-video physical accuracy.

## Unchanged floor checks

The full-reference preflight checked every original reference pose and the 30 Hz interpolation grid without physics integration. Penetration values are model units; the unchanged maximum limit is **0.025**.

| Clip | Failed source poses | Failed grid poses | Maximum penetration |
|---|---:|---:|---:|
| Single-leg squat | **8 → 14 / 118** | **16 → 27 / 235** | 0.08313 → 0.06762 |
| Squat | 40 → 35 / 100 | 77 → 72 / 213 | 0.25390 → 0.23442 |
| Push-up | 25 → 23 / 127 | 46 → 43 / 255 | **0.06648 → 0.07425** |

A lower maximum does not imply fewer failures, and better image alignment does not imply better contact geometry. The single-leg candidate intersects the floor at more poses; the push-up has a deeper worst intersection. All three remain infeasible under the existing contract.

## Decision

Keep this fitter optional and its outputs separate. The study retains `accepted:false` and `promoted:false`; no original model, default behavior, failed trial, or success threshold changed. No API calls, new pose inference, policy training, dynamics, root lifting, or floor refitting occurred.

The independent visual gain supports further investigation of image-based constraints, not automatic adoption. Remaining depth, body-shape, projection, and support ambiguities require a separately declared experiment with independent evidence. Do not tune this candidate against the exposed labels or hide its floor regressions.
