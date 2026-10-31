# Ground orientation: implemented, not recovered reliably

The mobile body can now use an explicit inclined floor, with gravity pointing along its negative normal. The simulator, live canvas, recorded video and offline replay use the same plane. Existing models keep their horizontal floor unless the optional hypothesis is supplied.

**The four original exercise clips did not qualify for an inferred floor orientation.** A later unfamiliar clip passes the conditional support-selected screen described below, but physical ground is still unverified. No real body model was changed or floor-error limit relaxed.

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

### Opt-in temporal support candidates

The existing estimator now accepts an explicit `support_prior` to screen heel/toe
points for image stationarity before fitting its orientation. Its default path,
first-60% construction split and all original plane checks remain unchanged.

```python
from weft.ground_orientation import estimate_ground_orientation

result = estimate_ground_orientation(evidence, support_prior={
    "mode": "fixed_source_camera",
    "status": "authored_unverified",
    "source_sha256": evidence["source_sha256"],
    "rationale": "My explicit fixed-camera assumption; not measured calibration",
})
```

Use the saved complete `evidence.json`, not the smaller landmark-only export:
dimensions, both clocks and decoded/container counts are required. A centered
0.4-second construction-only window needs at least three observations spanning
0.2 seconds, with no gap above 0.12 seconds or missing/low-confidence endpoint.
The image-coordinate bounding range must be at most 4% of the construction
median torso span. These fixed engineering limits were set before this comparison;
they are not calibrated measurement uncertainty or physical tolerances.

At least three stationary candidate heel/toe points must support a non-degenerate
per-frame plane. One stationary foot alone has only two points and cannot identify
ground orientation. No ankle point is treated as a ground sample. Missing,
moving and later-excluded slots remain explicit; no interpolation or fallback
normal is created. Native timestamp gaps remain separate from the nominal pose
clock used for these windows.

The reusable `exercise_support_evidence.estimate_exercise_support` function also
accepts exact canonical scene-observation **bytes** and selected equipment IDs.
It retains source/document/observation identities and reports nearby hand/box
co-motion candidates. Occluded, missing, withheld or clock-incompatible object
slots remain unavailable. An inferred image box is not a physical surface;
co-motion is not grasp/contact/load, and a handheld prop is never relabeled floor.

**Four-source result: no reconstruction gain.** The new selector leaves all four
known sources unavailable. It does not change any body, floor, task or default.
Unchanged baseline orientations were reused after verifying identical complete
pose rows and settings; no earlier fit or simulator audit was repeated.

| Source | Construction poses | Stationary heel/toe candidates | Moving | Unavailable | Plane-usable poses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-leg squat |70|1 /280|279|0|0|
| Squat |64|0 /256|0|256|0|
| Kettlebell push-up |77|0 /308|280|28|0|
| Full-body squat |20|1 /80|61|18|0|

This does not prove the feet physically moved: model jitter and small image
foot/torso spans can fail the screen. Stationary elevated feet can also pass a
conditional plane screen without establishing a floor height; height and physical
contact remain unknown. Do not loosen the limits or tune these exposed clips into
claimed final validation.

Evidence: `artifacts/exercise-support-selection-01`, 26 files / 7,939,783 bytes,
seal `89543a20ee9eff86fc9c46174d268ad49716da44ea8ca2b3ec732f089ee4bb8c`.
One observed run took 0.195 seconds with 71,401,472 bytes peak process RSS; this is not
browser performance or a hard memory guarantee. There was no physics, inference,
video decoding, network or paid call. The first run's source-index shadowing bug
and corrected focused receipt remain preserved. [Current workstream handoff](handoffs/exercise-reconstruction.md).

### Unfamiliar Interval Push-ups: conditional candidate, not recovered ground

The already acquired complete clip produced 772 Full CPU IMAGE/raw poses across
1,543 decoded frames. The container's 1,544-frame metadata is retained separately.
There are 24 fully tracked and 748 partially tracked poses; passing the existing
one-complete-arm/leg visibility rule does not make the occluded side reliable.
The two handheld props still have no source-bound equipment tracks or grasp proof.

Both estimators used the unchanged first 463 poses. The remaining 309 poses and
771 unsampled decoded frames remain explicit. No thresholds were tuned.

| Estimate | Usable / construction poses | Thickness ratio | Median / p90 normal dispersion | Screen |
| --- | ---: | ---: | ---: | --- |
| Original pooled heel/toe plane | 433 / 463 | 0.45754 | 20.612° / 119.405° | Rejected |
| Stationary-selected heel/toe plane | 126 / 463 | 0.01619 | 3.369° / 8.862° | Conditional candidate |

The selector retains 640 stationary, 1,085 moving and 127 unavailable construction
foot-point slots, and rejects 337 construction poses. The candidate is **73.570°
from camera-up**. Stationary heel/toe points can describe a tilted foot surface,
not the floor. Ground height, gravity, contact, load and grasp remain unknown;
the candidate has not been installed or passed independent floor/task checks.

Evidence: `artifacts/interval-pushups-full-pose-03`, seal
`bec762dd6747c01f540f617fc3d76b5f8b4a90cd506debc5e2f04954db3d7614`,
39 files / 41,372,515 bytes; 11.965 seconds wall, 13.415 CPU seconds and
400,293,888-byte observed child peak RSS. Exactly one decode and 772 model calls;
no downloads, network, paid calls, body construction or physics. Earlier runs
`interval-pushups-full-pose-01` and `-02` preserve optional-import startup failures
with zero decoded frames or inference calls; they are not failed pose results.

### Input-envelope check

The selector rejects excessive pose counts before serialization. It checks JSON
depth, node count and the exact canonical byte budget using bounded string chunks
and iterator stacks before allocating the full canonical representation. Accepted
source hashes remain unchanged. This is input validation, not a new pose method.

All 29 affected support checks pass, including ten new pre-serialization rejection
and hash/deadline checks; the 43 unchanged ground checks retain their earlier passing
receipts. A single saved-only 772-pose comparison measured 0.104 seconds before and
0.404 seconds after validation, with the same canonical hash and 102,072,320-byte
whole-process peak RSS. This adds bounded validation work; it is not a speedup or
hard memory guarantee. No source inference or ground fit was repeated.

To reproduce the original four-source orientation study (not the new temporal selector) in a new directory:

```sh
.venv-lab/bin/python scripts/studies/check_ground_hypotheses.py --out artifacts/ground-hypotheses-new
```
