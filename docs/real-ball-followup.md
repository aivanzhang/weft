# Independent real collision follow-up

Two original public clips were acquired and inspected; neither qualifies as a reconstructed interactive environment under the current checks. They add real collision evidence beyond the existing rolling-ball source and synthetic fixtures. No checks were loosened and no favorable tracking-only section was presented as physical success.

**Later follow-up:** [color compatibility removes the pool false-visible prediction](appearance-guards.md), retaining all seven visible labeled positions. Both reconstruction failures below remain; the pool camera disturbance is still unknown to the screen. This document preserves the original baseline.

## Sources and rights

- **Overhead air-table collision:** MikeRun's [original source page](https://commons.wikimedia.org/w/index.php?title=File:2d-collision-air-table.webm&oldid=1291911231), **CC BY-SA 4.0**, downloaded unchanged as `data/balls/air-table-collision.webm` (70,794 bytes). This is a filmed collision between two disks, not rolling spheres. The caption supplies nominal masses, but Weft did not measure or calibrate them.
- **Pool break:** MarkBuckawicki's [original source page](https://commons.wikimedia.org/w/index.php?title=File:Quick_Pool_Break.webm&oldid=462819491), **CC0-1.0**, downloaded unchanged as `data/balls/quick-pool-break.webm` (approximately 257 KB). The low side view has overlap, strong perspective, and a later camera disturbance.

Source sidecars pin attribution, license URLs, media URLs, retrieval timestamps, file sizes, and SHA-256. Air-table previews and annotations are adaptations under CC BY-SA 4.0. Pool-break adaptations remain CC0. This acquisition used two bounded downloads, less than 0.4 MB combined, no account and no paid API.

## Label independence and decode audit

Original source frames were inspected before invoking either tracker or scorer. Small center/visibility labels, coarse contact-event intervals, explicit initialization regions, and assumptions were frozen in `artifacts/real-ball-followup-01/pretracking-freeze.json`. No predicted coordinates entered those labels. These are assistant visual development annotations, not independent expert measurements or a final holdout. Labels are never inputs to scene construction or fitting.

The air-table source has 13 decoded frames at 25 fps. Its page reports display width 768; OpenCV decodes a 720×576 pixel raster. Labels use those decoded pixels without perspective or metric calibration. Frames 0–4 contain hand-held/partly obscured disks; the authored interval is 5–12, leaving only eight unobscured frames.

Pool metadata advertises 141 frames, but the independent WebM block counter and decoder both find **118 video frames**. The full original file and all encoded video frames are preserved; no trimming repairs this discrepancy. The authored interval covers the whole available video.

## Results

| Check | Air-table disks | Front yellow pool ball |
| --- | --- | --- |
| Visible source labels accepted | 10/10 | 7/7 |
| Accepted-center RMSE | 2.02 px | 3.02 px |
| Maximum accepted-center error | 3.81 px | 5.10 px |
| Visual annotation uncertainty | ±4 px | ±6 px |
| Explicitly unlocatable labels accepted | No absent labels in this set | 1/3 |
| All tracked observations, excluding authored initialization | 14 (seven frames × two disks) | 70 |
| Current environment result | Rejected: fewer than ten jointly tracked frames | Rejected: under 80% whole-interval coverage |

The pool tracker publishes a position at frame 64 where source inspection could not confidently locate the partly occluded yellow object's center. This is an abstention/visibility failure against the frozen annotation, not proof of a measured identity swap. It must not be hidden by reporting only errors on accepted visible points.

The air-table source appears steady, but its short selected interval yields only the initialization camera sample; that does **not** verify a fixed camera. The pool background stays stable through frame 72, then the screen reports an inconsistent background at frame 84 and insufficient tracked features later. The source visibly shows a camera disturbance, but the current screen does not classify it as `moving_background`. It remains `not_verified_fixed`; the tracking-coverage guard independently blocks scene construction.

## Contact evidence and limits

The air-table labels place the projected collision candidate between frames 6 and 7 (0.24–0.28 s), based on closest approach and subsequent direction changes. Actual contact is not resolved in a sampled frame. Pool labels place the rack-break onset between frames 60 and 61; individual pairwise contacts are unresolved. Neither interval measures impulse, restitution, friction, depth, or metric force.

No scene passed the prerequisites for the unchanged fitter, so no fitted physics, policy-training success, or interactive reconstruction is claimed. The air-table sample is useful new tracking/contact-timing evidence; a longer unobscured overhead recording is needed for the existing temporal split. The pool source is a useful occlusion/camera-change negative case. Adding duplicated frames, including hand-driven setup as passive motion, or removing difficult frames would not supply the missing evidence.

Saved originals, frozen labels/configuration, tracking overlays, decode audits, evaluations, explicit failure records, and checksum inventories are under `artifacts/real-ball-followup-01/`. The air-table overlays were visually checked. No full lab worker ran, and no shared source modules were edited.
