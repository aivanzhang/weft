# Cartesian body refinement: rejected for promotion

The bounded joint-angle fit improved agreement with its construction landmarks but **failed independent checks**. It is rejected for promotion: source-view limb-direction passes fell from **18/28 to 15/28**, and single-leg squat floor violations increased from **8 to 17 source poses**. All three candidate trajectories remain infeasible under the unchanged floor limit.

Evidence: [sealed study](../artifacts/body-refinement-01/report.json), [limb grading](../artifacts/body-refinement-01/limb-grading.json), [frozen settings](../artifacts/body-refinement-01/pre-fit-freeze.json), and [study script](../scripts/studies/check_body_refinement.py). The manifest verified **44 files**, including original models/pose evidence, candidate models, full audits, source annotations, and implementation/test snapshots. Manifest SHA-256: `41a6f82285da32043e01be604bab383a376b781d04300fecf611ffab6f8ac25f`.

## Frozen method

The optional fitter adjusted 19 angles against twelve hip-centered model-world landmark positions using actual MuJoCo forward geometry and analytic position derivatives. It used confidence-weighted Huber loss (0.05 model-unit transition), original-angle regularization 0.001, and temporal regularization 0.0005 on **corrections**, not original movement. Changes were bounded to ±0.15 radians per axis, existing joint limits, twelve iterations and four backtracks per iteration, 900 frames, and 60 seconds per fit.

Sixteen analytic tests passed before real fitting. Root references, floor placement, body lengths, body orientation anchor, motor limits, and timestamps remained unchanged. The objective used no floor distances, evaluator labels, or policy rewards. Original estimates and confidence, including low-confidence and unused source frames, remain retained. These are construction estimates—not measured anatomy or physical truth.

## Results

Position errors are twelve-joint RMSE in model units. Floor failures count original reference poses; interpolation checks are retained separately in the study.

| Clip | Unweighted RMSE, original → candidate | Confidence-weighted RMSE | Failed floor poses | Maximum penetration | Fit runtime |
|---|---:|---:|---:|---:|---:|
| Single-leg squat | 0.06238 → 0.04627 | 0.06065 → 0.04442 | **8 → 17 / 118** | 0.08313 → 0.07746 | 0.122 s |
| Squat | 0.05033 → 0.04148 | 0.04861 → 0.04044 | 40 → 32 / 100 | 0.25390 → 0.22559 | 0.114 s |
| Push-up | 0.04809 → 0.03305 | 0.04651 → 0.03315 | 25 → 2 / 127 | 0.06648 → 0.02983 | 0.136 s |

The unchanged maximum-penetration limit is **0.025**. Smaller peak overlap does not imply fewer failing poses, as the single-leg result demonstrates. At the 30 Hz interpolation checks, failure counts changed 16→31, 77→65, and 46→2 respectively.

All 345 reference poses received bounded fits; each clip reached the 0.15-radian change bound somewhere. No compute deadline was reached. Reported runtime is measured locally for these inputs, not a general performance guarantee. No dynamics, training, API calls, source re-extraction, or parameter tuning occurred.

## Independent direction check

After fitting, the unchanged source-view grader compared four previously annotated source frames: 28 scorable limb segments and four unavailable slots. Its projection remained simulation x/−z, with no camera alignment or label fitting.

- Squat: **11/16 → 10/16** within the frozen angular tolerances.
- Push-up: **7/12 → 5/12**.
- Combined mean error: **12.066° → 13.417°**.
- Single-leg squat had no matching existing annotations; no independent direction result is claimed for it.

These annotations are frozen assistant visual development estimates, not expert ground truth or a final holdout. They nevertheless expose a regression that the optimized 3D landmark loss does not reveal. Per-segment direction tradeoffs are retained; agreement with the fitted targets is not sufficient evidence of a better reconstruction.

## Decision and next hypothesis

Keep the original models and defaults. Preserve these candidates as a rejected experiment; do not tune their settings against the exposed labels or relabel their failures. The study records `accepted:false` and `promoted:false`.

**Plan only:** test whether an image-space objective combined with explicit segment-length uncertainty avoids fitting overconfident, distorted 3D targets. It would retain uncertain/cropped observations, distinguish camera projection assumptions from measured coordinates, and use bounded fixed-shape constraints rather than per-pose limb resizing or root lifting. Any new objective must be frozen before evaluation and checked against both image-direction fidelity and unchanged floor feasibility. No such method has been implemented or tested here.
