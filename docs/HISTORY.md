# Documentation history

Current setup, architecture, status and remaining work live in [PROJECT.md](../PROJECT.md).
This index replaces superseded design notes, study narratives and handoffs. Original text
is preserved in Git, not copied into another multi-megabyte Markdown file. Historical
claims, pending tasks, budgets and ownership are dated records, not current instructions.

All links below are pinned to pre-cleanup commit `d0338eb926ef97ba05d59418ade028461ccbc9ae` in the private repository.
They preserve the original files and heading anchors. GitHub access requires repository access.
For offline reading from a full clone, use:

```sh
git show d0338eb926ef97ba05d59418ade028461ccbc9ae:docs/portal.md
```

Retired from the working tree: **350 Markdown files**, 4,828,923 bytes.
No history was rewritten. Source records, licenses, scoped agent instructions, immutable
evidence bundles, avatar operational README/TODO and six source-linked reference pages remain.
Existing README edits from concurrent product work were preserved; only reference links
and the current-guide pointer were changed by this cleanup.

## Root plans and prior handoffs

| Original file | Historical title |
| --- | --- |
| [AGY_GOAL_PROMPT.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/AGY_GOAL_PROMPT.md) | AGY_GOAL_PROMPT |
| [AGY_HANDOFF.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/AGY_HANDOFF.md) | Weft: teammate handoff for agy |
| [TODO.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/TODO.md) | Weft iteration backlog |
| [goal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/goal.md) | Weft — from observation to an executable world |
| [v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/v1.md) | Weft v1 — the first trainable world |

## Simulation and pipeline references

| Original file | Historical title |
| --- | --- |
| [docs/appearance-guards.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/appearance-guards.md) | Appearance evidence and camera checks |
| [docs/architecture.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/architecture.md) | Weft: one video to an interactive RL environment |
| [docs/avatar-authoring.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/avatar-authoring.md) | Avatar authoring and browser library |
| [docs/avatars.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/avatars.md) | Local avatars |
| [docs/ball-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/ball-diversity.md) | Ball and rigid-scene development probes |
| [docs/body-camera-root-refinement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-camera-root-refinement.md) | Camera/root proposal boundary |
| [docs/body-fidelity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-fidelity.md) | Body-orientation fidelity check |
| [docs/body-geometry-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-geometry-diagnosis.md) | Body geometry diagnosis |
| [docs/body-image-refinement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-image-refinement.md) | Image-direction refinement: visual gain, unresolved floor regressions |
| [docs/body-model-options.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-model-options.md) | Body-model options and access |
| [docs/body-pixel-refinement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-pixel-refinement.md) | Image-position fitting: inspect the rejected candidates |
| [docs/body-refinement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-refinement.md) | Cartesian body refinement: rejected for promotion |
| [docs/calibrated-human-sample.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/calibrated-human-sample.md) | Three-camera human-motion sample: calibration still unqualified |
| [docs/camera-aware-body.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-aware-body.md) | Camera-aware exercise sandbox |
| [docs/camera-ground.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-ground.md) | Supplied ground: correct coordinates, unresolved body height |
| [docs/camera-stabilization.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-stabilization.md) | Bounded camera registration |
| [docs/conditional-rigid-camera.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/conditional-rigid-camera.md) | Second-view camera and box-depth checks |
| [docs/cylinder-source.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/cylinder-source.md) | Real cylinder: depth-motion tracking test |
| [docs/decoded-masks.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/decoded-masks.md) | Inspect actual object silhouettes |
| [docs/dense-exercise-poses.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/dense-exercise-poses.md) | Every-frame exercise evidence |
| [docs/diversity-verification.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/diversity-verification.md) | Diverse-video end-to-end verification |
| [docs/equipment-proposals.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/equipment-proposals.md) | Build and inspect an exercise scene |
| [docs/exercise-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/exercise-diversity.md) | Exercise diversity study |
| [docs/exercise-fullbody-source.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/exercise-fullbody-source.md) | Fresh full-body squat source |
| [docs/feature-track-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/feature-track-diversity.md) | Appearance histories across different videos |
| [docs/floor-calibration.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/floor-calibration.md) | Camera and floor calibration |
| [docs/foot-image-refinement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-image-refinement.md) | Image-directed foot candidate |
| [docs/foot-reference.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-reference.md) | Inspectable foot references |
| [docs/foot-sandbox.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-sandbox.md) | Ankle and foot sandbox |
| [docs/full-interval-reconstruction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/full-interval-reconstruction.md) | Full-interval reconstruction checks |
| [docs/full-pose-model.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/full-pose-model.md) | Optional Full pose model and a new jogging sandbox |
| [docs/fullbody-exercise-results.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/fullbody-exercise-results.md) | Fresh full-body squat: baseline ready, reference-floor check failed |
| [docs/fullbody-geometry-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/fullbody-geometry-diagnosis.md) | Fresh full-body geometry diagnosis |
| [docs/general-object-evaluation.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/general-object-evaluation.md) | Toy-van full-interval development check |
| [docs/general-object-source.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/general-object-source.md) | Real non-ball object source |
| [docs/general-tracking.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/general-tracking.md) | Appearance tracking beyond colored masks |
| [docs/ground-orientation.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/ground-orientation.md) | Ground orientation: implemented, not recovered reliably |
| [docs/hackathon-demo.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/hackathon-demo.md) | Run the local Weft demo |
| [docs/heavy-pose-model.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/heavy-pose-model.md) | Heavy pose model: playable squat comparison |
| [docs/initial-motion.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/initial-motion.md) | Initial-speed repair for longer clips |
| [docs/iteration-followup.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/iteration-followup.md) | Orientation and appearance follow-up |
| [docs/iteration.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/iteration.md) | Iterating on Weft |
| [docs/joint-loss-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/joint-loss-diagnosis.md) | Joint tracker: remaining loss diagnosis |
| [docs/joint-tracking.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/joint-tracking.md) | Experimental joint object tracking |
| [docs/lab-verification.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/lab-verification.md) | Local lab verification |
| [docs/limb-fidelity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/limb-fidelity.md) | Limb-direction development check |
| [docs/local-lab.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md) | Weft local lab |
| [docs/material-transfer.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/material-transfer.md) | Contact materials and a reserved repeat |
| [docs/meta-advisor.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-advisor.md) | Meta scene advisor |
| [docs/meta-ball-plan.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-ball-plan.md) | Meta three-ball comparison input |
| [docs/meta-ball-results.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-ball-results.md) | Three-ball Meta tracking: locations retained, continuity IDs unresolved |
| [docs/meta-body-hypotheses.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-body-hypotheses.md) | Independent checks of Meta's body/contact hypotheses |
| [docs/meta-segmentation-plan.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-segmentation-plan.md) | Bounded Meta SAM toy-van comparison |
| [docs/meta-tracking-cli.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-tracking-cli.md) | Inspect video tracking locally |
| [docs/meta-tracking-results.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-tracking-results.md) | Meta tracking: first full-clip result |
| [docs/mobile-human.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/mobile-human.md) | Mobile exercise body with an assumed floor |
| [docs/moving-equipment-plan.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/moving-equipment-plan.md) | Moving and handheld equipment in the existing coupled world |
| [docs/moving-equipment.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/moving-equipment.md) | Moving-equipment exploration |
| [docs/moving-placement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/moving-placement.md) | Authored moving-object tasks |
| [docs/multi-view-body.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/multi-view-body.md) | Three- and four-camera body candidates |
| [docs/oriented-supports.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/oriented-supports.md) | Tilted supports and a recordable ramp sandbox |
| [docs/parallel-workstreams.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/parallel-workstreams.md) | Weft workstreams and integration |
| [docs/phys101-ramps.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/phys101-ramps.md) | Real ramp videos and full-clip tracking |
| [docs/pixel-body-shape.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/pixel-body-shape.md) | Static body-length experiment |
| [docs/point-flow-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/point-flow-diversity.md) | Original-point flow: accurate control, worse real coverage |
| [docs/portal-body-detail.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/portal-body-detail.md) | More detailed body extraction in the portal |
| [docs/portal-tasks.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/portal-tasks.md) | Create and train a task in the portal |
| [docs/portal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/portal.md) | Weft web portal |
| [docs/pose-identity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/pose-identity.md) | Exercise pose identity investigation |
| [docs/real-ball-followup.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/real-ball-followup.md) | Independent real collision follow-up |
| [docs/real-collision-long.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/real-collision-long.md) | Longer real collision acquisition — bounded result |
| [docs/real-collision-transcodes.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/real-collision-transcodes.md) | Real collision video: bounded official transcode follow-up |
| [docs/root-scale-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/root-scale-diagnosis.md) | Root-scale diagnosis: foreshortening matters for one clip, not a universal repair |
| [docs/root-source-labels.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/root-source-labels.md) | Source-only exercise root labels |
| [docs/runtime-qualification.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/runtime-qualification.md) | Check an executable environment |
| [docs/scale-tracking.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/scale-tracking.md) | Opt-in scale-aware appearance tracking |
| [docs/scene-observations.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/scene-observations.md) | Shared scene-observations contract |
| [docs/segmentation-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/segmentation-evidence.md) | Saved tracking as uncertain scene evidence |
| [docs/segmentation-planar-candidates.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/segmentation-planar-candidates.md) | Explicit planar candidates from segmentation evidence |
| [docs/selective-crop-exercises.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/selective-crop-exercises.md) | Complete-video exercise crop results |
| [docs/shared-exercise-lab-plan.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/shared-exercise-lab-plan.md) | Human-only exercise task in the local lab |
| [docs/shared-exercise.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/shared-exercise.md) | Human-only exercise tasks |
| [docs/shared-world.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/shared-world.md) | Coupled human and object sandbox |
| [docs/source-intervals.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/source-intervals.md) | Selected sections of longer videos |
| [docs/source-ramp-reconstruction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/source-ramp-reconstruction.md) | Source-derived ramp hypotheses |
| [docs/spatial-human.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-human.md) | Spatial exercise body: 19 controllable rotation axes |
| [docs/spatial-rigid.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-rigid.md) | Inspectable 3D rigid-object sandbox |
| [docs/spatial-scene-construction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-scene-construction.md) | Turn a saved object scene into a 3D lab |
| [docs/static-support.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/static-support.md) | Authored static supports for exercise bodies |
| [docs/stereo-ground.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/stereo-ground.md) | Ground from two calibrated images |
| [docs/surface-feature-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/surface-feature-diversity.md) | Automatic image features across saved videos |
| [docs/temporal-crop-recovery.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-crop-recovery.md) | Recovering failed poses from original video pixels |
| [docs/temporal-identity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-identity.md) | Temporal identity development evidence |
| [docs/three-ball-tracking.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/three-ball-tracking.md) | Three-ball tracking diagnosis |
| [docs/track-shape-diversity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/track-shape-diversity.md) | Can the tracked points determine a 3D shape? |
| [docs/triple-collision-input.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/triple-collision-input.md) | Triple-collision lab input mapping |
| [docs/two-view-body.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-body.md) | Partial two-view body geometry |
| [docs/two-view-correspondence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-correspondence.md) | Two-view limb correspondence diagnosis |
| [docs/two-view-root.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-root.md) | Two-view body position, unchanged monocular shape |
| [docs/two-view-segments.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-segments.md) | Partial limbs with explicit uncertainty |
| [docs/v1-cost-and-scope.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/v1-cost-and-scope.md) | V1 execution envelope |
| [docs/verification.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/verification.md) | Iteration setup verification |
| [docs/whole-body-floor.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/whole-body-floor.md) | Squat floor failures: reset-only diagnosis |
| [docs/whole-body-learning.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/whole-body-learning.md) | Whole-body sandbox and recording |
| [docs/world-construction-boundary.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/world-construction-boundary.md) | Shared-world construction: evidence and coordinate boundary |

## Session checkpoints

| Original file | Historical title |
| --- | --- |
| [docs/checkpoints/2026-10-29-collaboration.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/checkpoints/2026-10-29-collaboration.md) | Collaboration checkpoint — October 29, 2026 |
| [docs/handoffs/agy.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/agy.md) | agy — Track B handoff |
| [docs/handoffs/aspset-distal-pixels.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-distal-pixels.md) | Distal image constraints — verified optional candidate |
| [docs/handoffs/aspset-ground-construction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-ground-construction.md) | ASPset source-qualified ground construction |
| [docs/handoffs/aspset-limb-fit.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-limb-fit.md) | Fixed-length source body fit |
| [docs/handoffs/aspset-next-clip.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-next-clip.md) | ASPset different-motion reconstruction |
| [docs/handoffs/aspset-partial-limb.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-partial-limb.md) | Partial multi-view limbs — verified checkpoint |
| [docs/handoffs/aspset-reference-assessment.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-reference-assessment.md) | Frozen body versus published joints |
| [docs/handoffs/aspset-reference-dynamics.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-reference-dynamics.md) | Source-only motion-force diagnostic |
| [docs/handoffs/aspset-source-readiness.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-source-readiness.md) | Calibrated sports-video source readiness |
| [docs/handoffs/aspset-third-view.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/aspset-third-view.md) | ASPset third-view reconstruction |
| [docs/handoffs/automatic-equipment-console.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/automatic-equipment-console.md) | Automatic-equipment terminal acceptance |
| [docs/handoffs/automatic-equipment-contact-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/automatic-equipment-contact-diagnosis.md) | Automatic-equipment contact diagnosis |
| [docs/handoffs/automatic-equipment-launcher.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/automatic-equipment-launcher.md) | Launch the automatic-equipment example |
| [docs/handoffs/automatic-silhouette-screen.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/automatic-silhouette-screen.md) | Automatic-frame silhouette screen |
| [docs/handoffs/avatar-motion-quality.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/avatar-motion-quality.md) | Avatar preview quality correction |
| [docs/handoffs/body-worker-denials.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/body-worker-denials.md) | Whole-body worker blocked-call diagnosis |
| [docs/handoffs/calibrated-foot-consumer-jog.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/calibrated-foot-consumer-jog.md) | Saved jogging calibrated-foot acceptance |
| [docs/handoffs/calibrated-foot-consumer.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/calibrated-foot-consumer.md) | Calibrated foot consumer |
| [docs/handoffs/camera-body-fit.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-body-fit.md) | Fixed-camera body angle proposals |
| [docs/handoffs/camera-body-reference-inputs.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-body-reference-inputs.md) | Camera-body reference input custody |
| [docs/handoffs/camera-lab-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-lab-evidence.md) | Restore camera-lab evidence inspection |
| [docs/handoffs/camera-multiview-fit.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-multiview-fit.md) | Optional calibrated two-view body fitting |
| [docs/handoffs/camera-multiview-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-multiview-report.md) | Two-camera angle-proposal inspection |
| [docs/handoffs/camera-multiview-review.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-multiview-review.md) | Camera multiview consumer review |
| [docs/handoffs/camera-multiview-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-multiview-workflow.md) | Optional second-view workflow integration |
| [docs/handoffs/camera-policy-runtime-resume.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-policy-runtime-resume.md) | Resume the original camera policy lab |
| [docs/handoffs/camera-reference-controller.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-controller.md) | Camera reference-controller adapter |
| [docs/handoffs/camera-reference-lab.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-lab.md) | Camera fitted-reference lab |
| [docs/handoffs/camera-reference-real-acceptance.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-real-acceptance.md) | Real camera-reference acceptance |
| [docs/handoffs/camera-reference-recording.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-recording.md) | Camera fitted-reference recording |
| [docs/handoffs/camera-reference-replay-ui.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-replay-ui.md) | Camera fitted-reference replay UI |
| [docs/handoffs/camera-reference-report-storage.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-report-storage.md) | Recover the completed camera fit report |
| [docs/handoffs/camera-reference-review.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-review.md) | Camera reference trial integration review |
| [docs/handoffs/camera-reference-trial-result.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-trial-result.md) | Camera fitted-reference trial result |
| [docs/handoffs/camera-reference-trial.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-reference-trial.md) | Camera angle-overlay controller trials |
| [docs/handoffs/camera-shared-candidate.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-candidate.md) | Camera-body → shared-world candidate |
| [docs/handoffs/camera-shared-contract.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-contract.md) | Camera shared-world admission contract |
| [docs/handoffs/camera-shared-lab.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-lab.md) | Camera-based shared-world lab integration |
| [docs/handoffs/camera-shared-learning.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-learning.md) | Camera shared learning |
| [docs/handoffs/camera-shared-recording.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-recording.md) | Camera shared recording |
| [docs/handoffs/camera-shared-runtime.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-runtime.md) | Camera-body shared runtime |
| [docs/handoffs/camera-shared-task.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-task.md) | Camera shared task adapter |
| [docs/handoffs/camera-shared-ui.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-shared-ui.md) | Camera task controls and saved replay |
| [docs/handoffs/camera-surface-binding.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-surface-binding.md) | Camera-body surface binding |
| [docs/handoffs/camera-surface-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/camera-surface-workflow.md) | Camera-aware surface fitting |
| [docs/handoffs/combined.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/combined.md) | Combined Weft implementation handoff |
| [docs/handoffs/compact-body-portal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/compact-body-portal.md) | Dense body through the portal and real perception APIs |
| [docs/handoffs/complete-goal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/complete-goal.md) | Complete Weft goal — video to an interactive, trainable world |
| [docs/handoffs/conditional-equipment-and-constraints.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/conditional-equipment-and-constraints.md) | Equipment candidates, portal selection and constrained body fitting |
| [docs/handoffs/contact-reconstruction-resume.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/contact-reconstruction-resume.md) | Conditional finite-support reconstruction |
| [docs/handoffs/continuous-reference-screen.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/continuous-reference-screen.md) | Continuous source-reference screen |
| [docs/handoffs/control-contact-parallel.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/control-contact-parallel.md) | Parallel controller and contact work |
| [docs/handoffs/correlated-surface-region-proposal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/correlated-surface-region-proposal.md) | Preserve correlated support geometry |
| [docs/handoffs/current-coupled-learning.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/current-coupled-learning.md) | Current coupled learning — no feasible policy |
| [docs/handoffs/dense-pose-and-contact-targets.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/dense-pose-and-contact-targets.md) | Dense source poses and constrained contact targets |
| [docs/handoffs/distal-depth-branches.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/distal-depth-branches.md) | Temporally supported depth branches — verified checkpoint |
| [docs/handoffs/distal-pixel-uncertainty.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/distal-pixel-uncertainty.md) | Explicit distal-image uncertainty — verified optional candidate |
| [docs/handoffs/equipment-action-controller-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-controller-native.md) | Single-decision equipment projection pilot |
| [docs/handoffs/equipment-action-controller.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-controller.md) | Bounded joint-action projection |
| [docs/handoffs/equipment-action-local-scale-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-local-scale-native.md) | Fixed-center derivative-scale diagnostic |
| [docs/handoffs/equipment-action-multiscale-proposal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-multiscale-proposal.md) | Saved multiscale action proposal |
| [docs/handoffs/equipment-action-projection.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-projection.md) | Detached action prediction for equipment control |
| [docs/handoffs/equipment-action-relinearization-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-relinearization-diagnosis.md) | Saved two-round response diagnosis |
| [docs/handoffs/equipment-action-relinearization-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-relinearization-native.md) | Two-round action decision at saved control 50 |
| [docs/handoffs/equipment-action-relinearization.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-relinearization.md) | Bounded two-round action refinement |
| [docs/handoffs/equipment-action-rollout-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-rollout-native.md) | Full nominal equipment projection rollout |
| [docs/handoffs/equipment-action-small-scale-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-small-scale-native.md) | Small-scale two-round controller decision |
| [docs/handoffs/equipment-action-velocity-probe-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-action-velocity-probe-native.md) | Equipment signed-velocity diagnostic |
| [docs/handoffs/equipment-anticipatory-control-proposal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-anticipatory-control-proposal.md) | Two-control anticipatory correction |
| [docs/handoffs/equipment-box-consensus.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-box-consensus.md) | Construction-only equipment frame selection |
| [docs/handoffs/equipment-clearance-controller.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-clearance-controller.md) | Experimental equipment-clearance controller |
| [docs/handoffs/equipment-clearance-results.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-clearance-results.md) | Joint-only clearance comparison |
| [docs/handoffs/equipment-depth-prior-core.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-depth-prior-core.md) | Explicit equipment depth prior |
| [docs/handoffs/equipment-depth-prior-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-depth-prior-native.md) | Explicit depth-prior ratio-one acceptance |
| [docs/handoffs/equipment-depth-prior-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-depth-prior-workflow.md) | Guided equipment depth priors |
| [docs/handoffs/equipment-feedback-feature-timing.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-feedback-feature-timing.md) | Equipment-feedback feature timing |
| [docs/handoffs/equipment-feedback-learning.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-feedback-learning.md) | Versioned equipment-feedback training |
| [docs/handoffs/equipment-feedback-root-pair-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-feedback-root-pair-native.md) | Fixed root-only equipment intervention |
| [docs/handoffs/equipment-feedback-training-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-feedback-training-native.md) | Equipment-feedback training pilot |
| [docs/handoffs/equipment-feedback.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-feedback.md) | Equipment-conditioned feedback actor |
| [docs/handoffs/equipment-frame-config.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-frame-config.md) | Guided equipment frame selection |
| [docs/handoffs/equipment-hand-constraint.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-hand-constraint.md) | Restrict co-motion with named hand points |
| [docs/handoffs/equipment-hand-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-hand-evidence.md) | Named hand-point equipment evidence |
| [docs/handoffs/equipment-hand-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-hand-workflow.md) | Optional named-hand equipment evidence |
| [docs/handoffs/equipment-lab-launcher.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-lab-launcher.md) | Named experimental equipment sandbox |
| [docs/handoffs/equipment-proposal-config.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-proposal-config.md) | Equipment proposal configuration extraction |
| [docs/handoffs/equipment-root-search-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-root-search-native.md) | One nonlinear root-authority screen |
| [docs/handoffs/equipment-root-search.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-root-search.md) | Nonlinear root-assistance search |
| [docs/handoffs/equipment-sequence-control-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-sequence-control-native.md) | One pre-49 two-action search |
| [docs/handoffs/equipment-sequence-prediction-native.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-sequence-prediction-native.md) | Two-action native prediction qualification |
| [docs/handoffs/equipment-sequence-prediction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-sequence-prediction.md) | Detached fixed-action sequence prediction |
| [docs/handoffs/equipment-silhouette-constraint-review.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-silhouette-constraint-review.md) | Equipment silhouette constraint review |
| [docs/handoffs/equipment-silhouette-constraint.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-silhouette-constraint.md) | Silhouette constraint for equipment co-motion |
| [docs/handoffs/equipment-silhouette-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-silhouette-evidence.md) | Equipment silhouette evidence |
| [docs/handoffs/equipment-silhouette-inspection.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-silhouette-inspection.md) | Saved equipment-evidence inspection |
| [docs/handoffs/equipment-silhouette-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-silhouette-report.md) | Equipment silhouette inspector |
| [docs/handoffs/equipment-support-hand-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-support-hand-report.md) | Equipment hand-point inspection |
| [docs/handoffs/equipment-support-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-support-report.md) | Equipment support comparison report |
| [docs/handoffs/equipment-support-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-support-workflow.md) | Saved equipment-support comparison |
| [docs/handoffs/equipment-temporal-silhouette.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/equipment-temporal-silhouette.md) | Conditional temporal equipment geometry |
| [docs/handoffs/exercise-contact-source-readiness-03.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-contact-source-readiness-03.md) | Additional contact-data screen |
| [docs/handoffs/exercise-equipment-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-equipment-evidence.md) | Public exercise/equipment evidence composition |
| [docs/handoffs/exercise-equipment-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-equipment-workflow.md) | Public exercise/equipment composition |
| [docs/handoffs/exercise-floor-calibration-gap.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-floor-calibration-gap.md) | Exercise floor calibration gap |
| [docs/handoffs/exercise-geometry-resume.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-geometry-resume.md) | Reuse saved conditional equipment geometry |
| [docs/handoffs/exercise-projection.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-projection.md) | Exercise projection consistency |
| [docs/handoffs/exercise-reconstruction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-reconstruction.md) | Exercise reconstruction workstream |
| [docs/handoffs/exercise-source-readiness-02.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-source-readiness-02.md) | Exercise source readiness 02 |
| [docs/handoffs/exercise-source-readiness-04.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-source-readiness-04.md) | New body–ground source screen |
| [docs/handoffs/exercise-surface-inputs.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-surface-inputs.md) | Exercise surface input collection |
| [docs/handoffs/exercise-surface-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-surface-report.md) | Exercise surface collection report |
| [docs/handoffs/exercise-surface-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/exercise-surface-workflow.md) | Exercise surface collection workflow |
| [docs/handoffs/fitted-control-lab-ui.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/fitted-control-lab-ui.md) | Fitted controls in the graphical lab |
| [docs/handoffs/floor-target-calibration-contract.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/floor-target-calibration-contract.md) | Floor-target calibration contract |
| [docs/handoffs/floor-target-calibration-core.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/floor-target-calibration-core.md) | Floor-target calibration core |
| [docs/handoffs/floor-target-calibration-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/floor-target-calibration-report.md) | Floor-target calibration inspector |
| [docs/handoffs/floor-target-calibration-worker.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/floor-target-calibration-worker.md) | Floor-target calibration worker |
| [docs/handoffs/floor-target-calibration-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/floor-target-calibration-workflow.md) | Source-bound floor calibration workflow |
| [docs/handoffs/foot-admission-gap.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/foot-admission-gap.md) | Foot admission gap |
| [docs/handoffs/hackathon-demo-goal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/hackathon-demo-goal.md) | Active goal: deliver the Weft hackathon demo |
| [docs/handoffs/hackathon-demo.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/hackathon-demo.md) | Hackathon demo integration |
| [docs/handoffs/heavy-video-kettlebell.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/heavy-video-kettlebell.md) | Explicit single-person pose tracking |
| [docs/handoffs/joint-motion-prior.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/joint-motion-prior.md) | Source-image motion refinement |
| [docs/handoffs/joint-pixel-trajectory.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/joint-pixel-trajectory.md) | Joint image trajectory — verified checkpoint |
| [docs/handoffs/lab-loader-retention.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/lab-loader-retention.md) | Shared exercise loader buffer lifetime |
| [docs/handoffs/limb-twist-gauge.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/limb-twist-gauge.md) | Source-bounded limb orientation |
| [docs/handoffs/local-environments.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/local-environments.md) | Separate local landing page and sandbox |
| [docs/handoffs/mode-recovered-equipment.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/mode-recovered-equipment.md) | Reconstruct equipment around the mode-recovered body |
| [docs/handoffs/motion-forces-cli.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/motion-forces-cli.md) | Motion-force check in the normal CLI |
| [docs/handoffs/moving-equipment-foundation.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/moving-equipment-foundation.md) | Moving-equipment foundation and active parallel work |
| [docs/handoffs/moving-placement.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/moving-placement.md) | Authored placement task integration |
| [docs/handoffs/multiview-foot-reference.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/multiview-foot-reference.md) | Calibrated two-view foot references |
| [docs/handoffs/multiview-landmark-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/multiview-landmark-evidence.md) | Additional-view landmark evidence |
| [docs/handoffs/native-clock-integration.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/native-clock-integration.md) | Native-clock reconstruction and preserved portal runtimes |
| [docs/handoffs/native-equipment-root-transfer.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/native-equipment-root-transfer.md) | Native body roots through equipment construction |
| [docs/handoffs/native-interval-and-association.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/native-interval-and-association.md) | Native endpoints and conditional equipment reconstruction |
| [docs/handoffs/parallel-reconstruction-and-muse.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/parallel-reconstruction-and-muse.md) | Parallel reconstruction and imported-world Muse integration |
| [docs/handoffs/partial-foot-evidence-jog.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/partial-foot-evidence-jog.md) | Saved jogging partial-foot inspection |
| [docs/handoffs/partial-foot-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/partial-foot-evidence.md) | Partial calibrated foot evidence |
| [docs/handoffs/partitioned-exercise-observations.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/partitioned-exercise-observations.md) | Partition-safe exercise observations |
| [docs/handoffs/portal-api.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-api.md) | API-backed product portal |
| [docs/handoffs/portal-auto-body.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-auto-body.md) | Restore body sandbox actions for AI-routed exercise uploads |
| [docs/handoffs/portal-body-detail.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-body-detail.md) | Detailed body attempts and collision-first contact search |
| [docs/handoffs/portal-checks.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-checks.md) | Local experiments beside Muse review |
| [docs/handoffs/portal-inventory-and-import.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-inventory-and-import.md) | Explicit scene scope and playable equipment imports |
| [docs/handoffs/portal-muse.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-muse.md) | Muse review in the product portal |
| [docs/handoffs/portal-native-and-repair.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-native-and-repair.md) | Portal timing, candidate testing and reconstruction diagnostics |
| [docs/handoffs/portal-perception.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-perception.md) | Unfamiliar-video perception in the local portal |
| [docs/handoffs/portal-support-experiments.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-support-experiments.md) | Portal support experiments |
| [docs/handoffs/portal-tasks.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-tasks.md) | Portal task creation and learning |
| [docs/handoffs/portal-world-review.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-world-review.md) | Muse review of the coupled world |
| [docs/handoffs/pose-identity-windows.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/pose-identity-windows.md) | Preserve competing-pose identity breaks |
| [docs/handoffs/pose-mode-recovery.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/pose-mode-recovery.md) | Source-only temporal pose recovery |
| [docs/handoffs/pose-precursor-custody.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/pose-precursor-custody.md) | Saved pose precursor custody |
| [docs/handoffs/pose-precursor-qualification.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/pose-precursor-qualification.md) | Explicit pose precursor qualification |
| [docs/handoffs/pose-surface-annotations.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/pose-surface-annotations.md) | Source-pose surface annotations |
| [docs/handoffs/radial-surface-envelopes.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/radial-surface-envelopes.md) | Certified positive-radial surface envelopes |
| [docs/handoffs/reconstructed-camera-coupling.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/reconstructed-camera-coupling.md) | Reconstructed bodies in shared physics |
| [docs/handoffs/reconstructed-camera-learning.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/reconstructed-camera-learning.md) | Reconstructed camera-body learning |
| [docs/handoffs/reference-equipment-overlap.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/reference-equipment-overlap.md) | Saved reference–equipment overlap |
| [docs/handoffs/sampled-equipment-initialization.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/sampled-equipment-initialization.md) | Clearer-frame equipment initialization |
| [docs/handoffs/saved-perception-recovery.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/saved-perception-recovery.md) | Saved perception recovery and equipment initialization |
| [docs/handoffs/selected-world-evidence.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/selected-world-evidence.md) | Explicit construction-frame admission |
| [docs/handoffs/shared-exercise-config-report.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/shared-exercise-config-report.md) | Guided shared-exercise configuration report |
| [docs/handoffs/shared-exercise-config-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/shared-exercise-config-workflow.md) | Guided shared-exercise configuration |
| [docs/handoffs/shared-exercise-config.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/shared-exercise-config.md) | Guided shared-exercise configuration |
| [docs/handoffs/shared-exercise-lab-capacity.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/shared-exercise-lab-capacity.md) | Shared exercise lab package capacity |
| [docs/handoffs/shared-world-feet.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/shared-world-feet.md) | Shared-world feet integration |
| [docs/handoffs/source-direction-diagnostic.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/source-direction-diagnostic.md) | Source-only simultaneous motion directions |
| [docs/handoffs/source-ground-contact.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/source-ground-contact.md) | Source-ground and working-equipment reconstruction |
| [docs/handoffs/source-surface-anchors.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/source-surface-anchors.md) | Source-qualified surface anchors |
| [docs/handoffs/source-surface-interval-construction.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/source-surface-interval-construction.md) | Source-surface interval construction |
| [docs/handoffs/split-limb-contact-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-contact-diagnosis.md) | Saved equipment failure and next control experiment |
| [docs/handoffs/split-limb-feedback-actor.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-actor.md) | Split-limb feedback actor |
| [docs/handoffs/split-limb-feedback-lab.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-lab.md) | Split arm/leg feedback: saved policy and lab |
| [docs/handoffs/split-limb-feedback-play.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-play.md) | Split arm/leg feedback: bounded lab acceptance |
| [docs/handoffs/split-limb-feedback-results.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-results.md) | Split-limb feedback — improved tradeoffs, no task success |
| [docs/handoffs/split-limb-feedback-review.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-review.md) | Split-limb feedback integration review |
| [docs/handoffs/split-limb-feedback-ui.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-ui.md) | Split arm/leg feedback: saved interface verification |
| [docs/handoffs/split-limb-feedback-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/split-limb-feedback-workflow.md) | Separate arm/leg feedback — integration in progress |
| [docs/handoffs/surface-interval-input-readiness.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/surface-interval-input-readiness.md) | Surface-interval fit input readiness |
| [docs/handoffs/surface-interval-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/surface-interval-workflow.md) | Surface-interval body fitting workflow |
| [docs/handoffs/surface-only-fitting-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/surface-only-fitting-workflow.md) | Missing-image fitting and usable fitted controls |
| [docs/handoffs/surface-only-reference-fitting.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/surface-only-reference-fitting.md) | Surface-only reference fitting |
| [docs/handoffs/temporal-arm-control.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/temporal-arm-control.md) | Temporal arm control |
| [docs/handoffs/track-a-goal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/track-a-goal.md) | Track A goal — video understanding and scene reconstruction |
| [docs/handoffs/track-a.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/track-a.md) | Codex — Track A handoff |
| [docs/handoffs/unfamiliar-exercise-input-20261102-01.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/unfamiliar-exercise-input-20261102-01.md) | New portrait exercise input |
| [docs/handoffs/visible-person-and-selected-muse.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/visible-person-and-selected-muse.md) | Visible-person tracking and selected-world Muse |

## Avatar authoring references

| Original file | Historical title |
| --- | --- |
| [tools/avatar-assets/docs/adaptive-undercoat.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/adaptive-undercoat.md) | Adaptive short-pile surface detail |
| [tools/avatar-assets/docs/api-model-profiles.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/api-model-profiles.md) | Explicit API model profiles |
| [tools/avatar-assets/docs/api-streaming.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/api-streaming.md) | Bounded API streaming and stopped-job status |
| [tools/avatar-assets/docs/api-website-backend.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/api-website-backend.md) | Website and API backend |
| [tools/avatar-assets/docs/browser-gallery.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/browser-gallery.md) | Local browser gallery |
| [tools/avatar-assets/docs/compound-motion.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/compound-motion.md) | Compound avatar motion |
| [tools/avatar-assets/docs/contour-contact-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/contour-contact-diagnosis.md) | Contour and contact diagnosis |
| [tools/avatar-assets/docs/dragon-delivery-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/dragon-delivery-v1.md) | Emerald dragon delivery |
| [tools/avatar-assets/docs/edge-fringe-experiment.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/edge-fringe-experiment.md) | Portable edge-fringe experiment |
| [tools/avatar-assets/docs/export-fidelity-and-staging.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/export-fidelity-and-staging.md) | Export fidelity and staged native rendering |
| [tools/avatar-assets/docs/exported-preview-renderer.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/exported-preview-renderer.md) | Exported-model preview renderer |
| [tools/avatar-assets/docs/fresh-avatar-acceptance-plan.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/fresh-avatar-acceptance-plan.md) | Fresh text-to-avatar acceptance plan |
| [tools/avatar-assets/docs/fur-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/fur-workflow.md) | Furry avatar workflow |
| [tools/avatar-assets/docs/generation-platform-research-2026-10-31.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/generation-platform-research-2026-10-31.md) | Choosing a 3D generation engine |
| [tools/avatar-assets/docs/hands-off-studio.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/hands-off-studio.md) | Hands-off avatar quality and creation studio |
| [tools/avatar-assets/docs/hd-quality-pass-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/hd-quality-pass-v1.md) | HD fur and surface follow-up |
| [tools/avatar-assets/docs/material-owned-fur.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/material-owned-fur.md) | Material-owned fur |
| [tools/avatar-assets/docs/meshy-delivery-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/meshy-delivery-v1.md) | Generated-avatar trial: deliverables and limits |
| [tools/avatar-assets/docs/meshy-quality-upgrade-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/meshy-quality-upgrade-v1.md) | Meshy quality upgrade — local refinement |
| [tools/avatar-assets/docs/meshy-workflow.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/meshy-workflow.md) | Repeatable generated-avatar workflow |
| [tools/avatar-assets/docs/motion-coat-admission.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/motion-coat-admission.md) | Motion evidence and coat admission |
| [tools/avatar-assets/docs/multipart-sources.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/multipart-sources.md) | Separate source parts for controllable avatars |
| [tools/avatar-assets/docs/normal-transfer-guard.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/normal-transfer-guard.md) | Reject damaging surface-normal transfers |
| [tools/avatar-assets/docs/portable-fringe.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/portable-fringe.md) | Reusable portable edge fringe |
| [tools/avatar-assets/docs/portable-fur-attachment.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/portable-fur-attachment.md) | Automatic portable-fur attachment audit |
| [tools/avatar-assets/docs/portable-fur-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/portable-fur-diagnosis.md) | Portable fur: measured diagnosis |
| [tools/avatar-assets/docs/preserve-current-appearance.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/preserve-current-appearance.md) | Keep the current look; finish motion |
| [tools/avatar-assets/docs/prompt-pipeline-results-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/prompt-pipeline-results-v1.md) | Prompt pipeline: actual results v1 |
| [tools/avatar-assets/docs/prompt-pipeline.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/prompt-pipeline.md) | Prompt-driven avatar pipeline |
| [tools/avatar-assets/docs/protected-skin-repairs.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/protected-skin-repairs.md) | Protected-skin repairs |
| [tools/avatar-assets/docs/quality-next-step.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/quality-next-step.md) | Next quality work: clean shapes, portable coat, heterogeneous materials |
| [tools/avatar-assets/docs/quality-pass-v2.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/quality-pass-v2.md) | Quality pass v2 |
| [tools/avatar-assets/docs/relocation.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/relocation.md) | Relocation into Weft |
| [tools/avatar-assets/docs/rig-quality-diagnosis.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/rig-quality-diagnosis.md) | Automatic rig spike diagnosis |
| [tools/avatar-assets/docs/scene-contract.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/scene-contract.md) | Scene authoring contract v1/v2 |
| [tools/avatar-assets/docs/scene-examples.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/scene-examples.md) | Authored scene examples |
| [tools/avatar-assets/docs/separate-assets.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/separate-assets.md) | Separate characters and new-animal checks |
| [tools/avatar-assets/docs/skin-consistent-fur.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/skin-consistent-fur.md) | Skin-consistent portable fur |
| [tools/avatar-assets/docs/source-uv-control.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/source-uv-control.md) | Actual-source UV2 control |
| [tools/avatar-assets/docs/standing-reference-retries.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/standing-reference-retries.md) | Reference-bound standing retries |
| [tools/avatar-assets/docs/studio-blocked-audit.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/studio-blocked-audit.md) | Studio delivery blocker audit |
| [tools/avatar-assets/docs/studio-jobs.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/studio-jobs.md) | Durable studio jobs |
| [tools/avatar-assets/docs/three-d.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/three-d.md) | Controllable 3D assets and verification |
| [tools/avatar-assets/docs/tuft-profile.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/tuft-profile.md) | Portable tuft experiment |
| [tools/avatar-assets/docs/typed-review-recovery.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/typed-review-recovery.md) | Typed review and rejected-review recovery |
| [tools/avatar-assets/docs/undercoat-uv2-preservation.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/undercoat-uv2-preservation.md) | Preserve source textures while adding undercoat |
| [tools/avatar-assets/docs/weft-authored-designs.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/weft-authored-designs.md) | Weft-profile authored designs |
| [tools/avatar-assets/docs/weft-delivery-v1.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/weft-delivery-v1.md) | Weft asset delivery v1 |
| [tools/avatar-assets/docs/weft-delivery-v2.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/weft-delivery-v2.md) | Studio avatar → Weft ZIP |
| [tools/avatar-assets/docs/weft-neutral-proposal.md](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/tools/avatar-assets/docs/weft-neutral-proposal.md) | Authored visual-neutral proposal v1 |
