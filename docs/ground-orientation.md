# Ground orientation: implemented, not recovered reliably

The mobile body can now use an explicit inclined floor, with gravity pointing along its negative normal. The simulator, live canvas, recorded video and offline replay use the same plane. Existing models keep their horizontal floor unless the optional hypothesis is supplied.

**None of the four tested exercise clips qualified for an inferred floor orientation.** No real body model was changed, no floor-error limit was relaxed, and no extra training or paid request ran.

## What changed

`mobility.ground_plane` records a unit normal, source hash, construction evidence IDs and explicit provenance. It requires `gravity_assumption: authored_normal_alignment_not_measured`. Coordinates remain camera-relative; this does not recover camera calibration or reference depth.

MuJoCo uses the inclined plane for actual collision and gravity calculations. The first body pose sets only the plane/body clearance offset; later poses cannot raise the body or move the floor. The saved fit explicitly distinguishes offset calibration from the separately supplied orientation. Joint angles, lengths, root trajectories and independent scoring limits are unchanged.

Recording snapshots preserve the plane normal and its assumption label. Old recordings need neither new field. The visual floor grid follows the saved normal rather than drawing a horizontal floor over inclined physics.

## Source-only orientation test

The estimator uses the first 60% of ordered pose observations. It centers each four-point heel/toe set separately before averaging normalized covariance matrices, so pelvis motion cannot masquerade as ground spread. Minimum endpoint confidence is 0.5; cropped, missing, nonfinite and degenerate inputs remain rejected. Later observations do not affect fitting.

Authored screening limits require at least three usable frames, sufficient two-dimensional spread, limited plane thickness, positive camera-up normal, median normal dispersion at most 15°, and 90th-percentile dispersion at most 30°. These are engineering checks, not measured physical tolerances. A plane through inferred heels/toes is **not necessarily the ground**: a lifted foot or raised heel violates that hypothesis even if pose estimation is accurate.

| Source | Usable construction poses | Result |
|---|---:|---|
| Single-leg squat | 70/70 | Excessive thickness and dispersion; normal nearly perpendicular to camera-up |
| Squat | 0/64 | No usable four-foot-landmark set |
| Kettlebell push-up | 73/77 | Normal nearly perpendicular to camera-up; unstable orientation |
| Fresh full-body squat | 12/20 | Median dispersion **15.070°**, just beyond the unchanged **15°** screen |

The fresh clip's diagnostic orientation is 42.44° from camera-up. It is retained as a rejected estimate, not applied or called a calibrated floor. The small margin beyond the screening limit is visible; it does not prove that a true floor orientation is impossible to infer with better evidence.

Chronology matters: an earlier unnormalized diagnostic on the fresh clip had already been computed when the stricter equal-frame contract was specified. The tests were frozen before this four-source run, but this was **not** an unseen-source threshold design or final holdout. No setting was retuned after its result.

Evidence: [four-source report](../artifacts/ground-hypotheses-01/report.json), [study script](../scripts/studies/check_ground_hypotheses.py), [earlier diagnostic](../artifacts/fullbody-foot-plane-diagnostic-01/report.json). The four-source study sealed 27 files, manifest `2d1ae6b7f6930e24c6d6f9a0a689007622499bdb8fd198bc8d4eb76408ba0810`, with zero physics steps and zero API spend. Candidates were unavailable, so downstream floor comparisons were not run.

## Interaction and recording verification

A separate **authored analytic fixture**, not a recovered video scene, exercised 30 actions and exported 31 decoded video frames. Replaying every action reproduced all states, rewards and termination flags exactly. Actual plane-contact distances matched the analytic geometry within `8.33e-17` model units. Live and offline JavaScript projection checks and the image renderer agreed on the inclined floor.

Maximum dynamic penetration was **0.0264754**, above the existing 0.025 whole-body success limit. This verifies recording and geometry integrity, **not** a passing exercise trajectory. A first harness attempt failed on an incorrect snapshot-field expectation and remains preserved; no simulator settings were tuned.

[Interaction evidence](../artifacts/ground-plane-interaction-01/report.json): 45 files, manifest `d968e6af081d08672345621100cc798dc5c6881629c205ff818ecc762571e4b3`. Analytical verification: 43 estimator tests, 30 new plane-physics tests within 72 focused model checks, 25 deterministic UI checks, and scoped recording/whole-body regression checks. No browser interaction was performed.

## Next evidence needed

Use scene geometry or calibration to constrain the ground separately from body pose. Infer which feet are supporting the body before treating their landmarks as ground samples; preserve raised heels and lifted feet. Actual ankle/foot/sole geometry, camera perspective and root depth remain unresolved. A Meta explanation or a lower simulator error alone cannot establish these quantities.

To reproduce the bounded study in a new directory:

```sh
.venv-lab/bin/python scripts/studies/check_ground_hypotheses.py --out artifacts/ground-hypotheses-new
```
