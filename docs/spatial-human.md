# Spatial exercise body: 19 controllable rotation axes

Verified **October 28, 2026, 1:29 PM PDT**. Experimental local simulation, not validated biomechanics.

**Later follow-up:** [explicit temporal identity correction](temporal-identity.md) repairs the two diagnosed jumping-jack swaps and provides a newer ready-to-open environment. Results below preserve the original raw-identity study, not the later candidate.

**Mobile option:** [moving-pelvis model and assumed floor](mobile-human.md) add release/push/support controls and contact recording. Three real clips pass; missing root evidence still rejects jumping jacks. This does not turn the retained spatial model into a balanced or anatomically validated human.

## Try it

```sh
# Completed jumping-jack environment: inspect, interact and record without rebuilding.
.venv-lab/bin/python -m weft lab --resume artifacts/spatial-human-e2e-04/jumping-jacks-burpees

# Build from a new exercise video.
.venv-lab/bin/python -m weft lab --video /path/to/exercise.mp4 \
  --profile exercise --body-model spatial --out artifacts/my-spatial-exercise
```

The completed jumping-jack example explicitly used `--pose-mode image`. The default remains `video`; neither mode verifies anatomical identity or person count. Existing eight-axis models remain readable and reproducible with `--body-model hinge` (the default). No silent model or inference fallback occurs.

Choose an axis, tap **Rotate + / Rotate −**, or start and hold a control. **Follow video** is an authored motor controller; **Learned** is a policy trained on one selected target axis. Recordings preserve all state and action transitions. Source, assumptions, frame confidences and alternate model versions remain inspectable artifacts.

## What changed

`exercise-v2` has three torso axes, three axes at each shoulder and hip, and one flexion axis at each elbow and knee: **19 scalar degrees of freedom**. Twelve connected display segments remain; 38 colored bands illustrate antagonist motor commands. These are not anatomical muscles or measured activation.

`spatial_pose.py` builds torso-relative and limb-relative rotation frames from retained model landmarks. The simulation applies Z/Y/X hinges in order, matching [MuJoCo's joint transformation convention](https://mujoco.readthedocs.io/en/3.3.3/XMLreference.html). Analytic tests reproduce rotated/lateral bent-limb endpoints through actual MuJoCo geometry. Near-straight limbs have unidentified twist; assumed twist, Euler singularities, chart wraps and clamped flexion remain explicit. Cropped/missing image endpoints have zero confidence. Rotation-chart bounds are not anatomical range-of-motion limits.

Torso rotation is now dynamic, but **pelvis position remains anchored**. No root translation, ground contact, balance, foot/wrist mechanics or validated anatomy is claimed. Depth and scale remain monocular estimates; camera up is assumed to align with gravity.

The authored controller adds target-velocity feed-forward and [simulated gravity/Coriolis compensation](https://mujoco.readthedocs.io/en/stable/computation.html). Initial 30-Nm torso motors could not hold the authored model: static model loads reached 42–76 Nm across four clips. Virtual torso support motors now allow 120 Nm per axis; limb limits remain 30 Nm. These are **simulation design choices, not inferred human strengths or contact forces**. Manual release still removes the selected motor command.

## End-to-end evidence

`artifacts/spatial-human-e2e-04/`, [pinned input](../benchmarks/spatial-human-v4.json): four source videos, 47.073 seconds, at most two workers, all four declared gates passed. Each run trained 1,000 episodes and evaluated 50 matched episodes per policy. Every run recorded 12 transitions and fully decoded 13 video frames; run manifests were unchanged.

| Clip | 19-axis servo RMSE | Learned / random / coast success |
| --- | ---: | --- |
| Single-leg squat | 0.120 rad | 100% / 90% / 14% |
| Squat | 0.081 rad | 100% / 24% / 0% |
| Kettlebell pushup | 0.150 rad | 100% / 92% / 0% |
| Jumping jacks/burpees | 0.231 rad | 100% / 4% / 0% |

The high random baseline on two clips demonstrates an easy authored task, not meaningful exercise mastery. One training seed is not a stable cross-seed result. The metric is named `spatial_reference_rmse_rad`; **it cannot be compared directly with the old eight-axis coordinate error**. Both retain a declared 0.35-rad threshold under their explicit contracts.

Intermediate same-19-axis studies are preserved: plain PD (`spatial-human-direct-01`), compensation with insufficient torso torque (`spatial-human-compensated-02`), and bounded virtual support (`spatial-human-supported-03`). Jumping-jack servo error improved 0.484 → 0.368 → 0.231 radians on the same inferred reference, not on physical force measurements.

Final full-study seal: **409 files**, SHA-256 `a8f06d80b69f45371a29f8c70d43290e88e8e5672dae744e3b3f3a8c25b24c95`.

## Independent source-view check and remaining failure

[Source-only limb labels](limb-fidelity.md) were frozen before candidate inspection: 12 frames across jumping jacks, squat and pushup; 84 visible segment slots, 12 source-occluded slots. Four visible slots have no usable nearby pose and remain failures. These are assistant visual development estimates, **not expert ground truth or a final holdout**.

- Mean direction error on the 80 scored segments: **40.69° → 14.09°**.
- Within preset tolerances: **26/84 → 60/84** visible segments.
- Worst error remains **174.77°**. A mean alone would hide this failure.

The final outputs reproduce the improvement under a fixed original-camera projection, without fitting a camera or swapping labels to improve scores. Evidence: `artifacts/limb-spatial-evaluation-final-02/`, 14-file seal `72e5ec7172807389655ec58bd0dc4b0e14ff06a79e26fe13d44b291139ccb3da`. Reproduction script: `scripts/studies/grade_spatial_limbs.py`.

`artifacts/pose-frame330-diagnosis-01/` identifies the worst case: the pose detector flips left/right anatomical identities at jumping-jack frames 330–332, restoring them at 334. Both raw image and world landmarks are affected; the source still faces forward. Simulator directions match the retained world estimates within 0.000002°. Legacy VIDEO inference has the same frame-330 error. This needs temporal identity handling, not a model-coordinate or label rewrite. Lunge ambiguity and missed second people also remain unresolved.

## Display, recording and resource checks

Spatial snapshots now use fixed conservative reach bounds, centered at the anchored pelvis. Raised hands stay inside the view as the camera rotates. Sandbox, video and offline replay use the same bounds; legacy snapshots keep their old framing. The final contact sheet was inspected locally. `artifacts/spatial-view-export-01/` preserves a new **120-step / 121-frame** recording, full decode verification and parse-checked offline replay; 13-file study seal `0847102fbb8bd2c09f4c40b2c1e32126658c930e0dea4e9d1d65bdba5d7acd34`.

The browser-facing scene now contains a compact model summary; full evidence remains in `human.json`. On the jumping-jack model, minified scene size fell from **1,935,491 to 3,996 bytes**. A 300-step local control/JSON measurement had **0.727 ms p95**, with a 145-ms cold model/manifest load. This measures direct Python calls on one machine, **not network/browser latency or sustained rendering**. Evidence: `artifacts/spatial-interaction-check-01/`; measured before the small view-bound fields were added.

**325 distinct Python tests verified**: 320-test full run, then affected advisor/view/recording checks adding five tests. Fifteen deterministic UI checks pass. Saved XML: `.local/spatial-full-tests.xml`, `spatial-advisor-tests.xml`, `spatial-view-tests.xml`, `spatial-view-validation-tests.xml`. Actual browser/device interaction is still unperformed.

Meta receives a whitelisted authored body-model summary, not evaluation or reference-audit results. `artifacts/meta-spatial-body-dryrun-03/` prepares a bounded request but sends none: the approved $5 total ledger has $0 committed and no project key. General-object work also preserved a failed rotation/flow pendulum prototype rather than promoting unmeasured early tracking as accuracy; see [tracking follow-up](general-tracking.md).

Next gates: resolve temporal anatomical identity without hiding ambiguities; validate fresh poses and dynamic/root/contact behavior independently; establish useful tasks across training seeds. Passing this scoped development suite does not complete the broad physical video-to-RL goal.
