# Mobile exercise body with an assumed floor

Verified **October 28, 2026, 2:58 PM PDT**. Three real exercise clips produce interactive mobile environments; two desired clips still fail. This is an experimental simulated body with virtual assistance, not recovered balance or validated biomechanics.

## Open and try it

```sh
# Open the completed squat environment without rebuilding.
.venv-lab/bin/python -m weft lab --resume artifacts/mobile-exercise-e2e-10/squat

# Build another short exercise video with explicit mobile-body assumptions.
.venv-lab/bin/python -m weft lab --video /path/to/exercise.mp4 \
  --profile exercise --body-model mobile --pose-identity temporal \
  --out artifacts/my-mobile-exercise
```

In **Move it yourself**, choose a rotation axis or use the new body controls:

- **Release body support:** disables all virtual joint/root motors. Gravity and simulated floor collisions remain active.
- **Push body left/right:** applies a bounded horizontal force. It does not turn support back on.
- **Restore support:** stabilizes the current pose rather than teleporting to the initial pose.
- **Follow video movement:** restores assistance and follows the current reference time. This is a programmed controller, not learned balance.

Use **Record & start** before interacting, then **Stop & save**. The recording includes positions, joint rotations, support state, applied commands and simulated contacts. Muscle bands remain illustrative rotation-motor proxies; root support forces are not painted as muscles.

A completed release/push/restore/follow trial is saved in `artifacts/mobile-interaction-02/lab/sessions/session-cebfc2a5df9f/replay.html`, with `video.mp4` beside it. It contains 120 actions and 121 frames. [Contact sheet](../artifacts/mobile-interaction-02/interaction-contact-sheet.png).

## What the video supports—and what it does not

MediaPipe world landmarks are hip-centered, so their hip midpoint is not a room-position trajectory. The [official output definition](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python#handle_and_display_results) distinguishes normalized image coordinates from hip-origin world coordinates.

`root_motion.py` uses confident, uncropped image hips and shoulders. The hip midpoint supplies horizontal/vertical image displacement. One fixed scale is computed from model torso length divided by the median projected torso length. This is an **uncalibrated fixed-camera/weak-perspective assumption**, not recovered metric scale. Camera motion, foreshortening and perspective can invalidate it. No per-frame scale fitting or image-depth estimate is used. Reference depth displacement stays zero; simulated depth motion is free to change under forces.

Every source frame ID, time, image estimate, confidence, scale sample and rejected root estimate is retained in `root-motion.json`. The origin must be the first reference, at least 80% of aligned references must be directly estimated, and at least three scale samples are required. At most two interior missing samples may be interpolated across a bracket no longer than 0.25 seconds, explicitly marked derived. Longer gaps—including omitted pose samples between otherwise valid endpoints—reject the mobile model. There is no silent fallback to an anchored body.

The first-pose floor is placed below the lowest authored capsule/sphere surface, with 0.005 model-unit clearance. It is **not a detected floor or verified foot/hand contact**. Later reference poses do not move the floor to hide inconsistencies. Feet, wrists, kettlebells and bars are not reconstructed articulated/contact objects; body capsules can contact the floor, but body self-collision remains disabled.

The [four-source foot audit](../artifacts/exercise-foot-evidence-01/interpretation.md) confirms that the current compiled bodies have ankle endpoint sites but **no ankle joints, heel/toe sites or foot/sole collision shapes**. The shin capsule ends are not reconstructed feet. Across 388 retained pose samples, cropped feet can still receive high model confidence; crop and visibility must remain separate. An [opt-in kinematic foot-reference artifact](foot-reference.md) is now implemented and tested across four sources. It is not attached to the simulator: all missing feet stay unavailable, and visible projection errors prevent promotion to foot collisions or controls.

## Simulation and controls

`exercise-v3` has 19 rotational axes plus three genuine translational slide joints: 22 scalar MuJoCo degrees of freedom. The world-level root frame keeps translation axes separate from torso rotation. [MuJoCo joints and collision settings](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) implement dynamics; there is no per-step root-position teleport.

An explicitly authored 0.1 mass-unit root wrapper has inertia. Root assistance uses mass-scaled proportional/damping gains of 60/12, simulated gravity compensation, and a per-axis force cap of three times model weight or 2,000 N, whichever is smaller. Pushes request mass × 6 N. These are virtual controls, not estimated human strengths. Root translation is bounded to ±5 model units; the 19 rotation axes keep their existing motor/angle limits. Root positions and forces stay separate from angle-error metrics and the 38 illustrative muscle bands.

Snapshots report at most 32 contact details, total contact count, aggregate simulated normal force and maximum penetration. Reference snapshots expose geometric overlap but zero dynamic-force claims. MuJoCo uses [soft contact constraints](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model), so small penetration is measured rather than concealed. Forces are simulation outputs under assumed geometry/mass/friction, not measured ground reactions.

The RL task remains one selected rotation axis reaching an authored target. Root assistance is not learned, and high task success is not whole-exercise imitation or balance control.

## Frozen real-video results

`benchmarks/mobile-exercise-v9.json` declared five desired cases before processing. The unchanged angular gate is 0.35 radians; additional model-check gates are root-reference RMSE ≤0.15 model units and floor penetration ≤0.025 model units. Those tolerances are engineering checks, not medical or physical-accuracy criteria.

| Source | Rotation servo RMSE | Root servo RMSE | Maximum floor penetration | Learned / random / coast |
| --- | ---: | ---: | ---: | ---: |
| Single-leg squat | 0.117 rad | 0.0353 units | 0.01265 units | 86% / 80% / 40% |
| Squat | 0.100 rad | 0.0355 units | 0.00485 units | 100% / 28% / 0% |
| Kettlebell pushup | 0.146 rad | 0.0386 units | 0.01086 units | 100% / 92% / 0% |
| Jumping jacks/burpees | Failed before model creation: unfilled root sample and reference gaps, including 1.40 seconds | — | — | — |
| Lunge/twist | Failed perception: multiple people or competing pose hypotheses | — | — | — |

The three successful cases pass changed-action checks and 12-action/13-frame recording verification. Released-body trials also show nonzero displacement, actual simulated floor contacts and bounded forces. Sparse visible-pose success does not remove occluded limbs or competing identities. Existing anchored models are retained, not silently upgraded.

The complete suite took **56.522 seconds**, with two local workers and no paid calls. Each successful task trained 1,000 episodes and evaluated 50 matched episodes per policy. This remains one training seed and an easy single-axis task; two random baselines are high. Suite: `artifacts/mobile-exercise-e2e-10`, 375-file seal `a8743fd16da21a07a32507140cbf59ce0a28945b6b37f96c3815d62c23ea599c`.

## Independent image-position check

[Twelve source-only slots](root-source-labels.md) were selected and annotated before mobile candidates were inspected. Eleven have rough pelvis/torso estimates; one occluded slot remains null. These are low-confidence assistant visual labels with explicit pixel allowances, not expert anatomy or final holdouts. All original label images were subsequently verified against sequential decoding at their absolute source indices.

All **11 estimable pelvis slots** have estimates within their subjective 12–30 pixel allowances. Per-clip pelvis RMSE: single-leg 5.65 px, squat 7.96 px, pushup 11.59 px, jumping jacks 4.67 px. The null jumping-jack slot remains without an estimate. Its other two good sampled positions do not override the full mobile-reference failure.

Relative movement is scored from each clip's first estimable labeled frame, without fitting camera, rotation or scale to labels. For the pushup, a fixed-pelvis baseline misses later displacements by 43.28/67.74 px; the image-motion proxy errors are 6.20/10.02 px, within the 32-pixel combined endpoint allowance. Other sparse frame pairs often fall within annotation uncertainty even for an anchored baseline; they do not independently prove improved motion fidelity. No 3D or physical-translation claim follows from this 2D check.

Grade: `artifacts/root-motion-source-grades-01`, six-file seal `f000c3d15f74c7466b68390ea896737841b176e78ca2e16a08bc68f122e82b5d`. Code/label freeze: `artifacts/mobile-design-freeze-01`, 60-file seal `198704fc1ba8b3ead862f1893d50d27d1e29359bc0ef5d9116e906d81c1fa191`.

## Interaction, recording and remaining work

All **577 Python tests** pass, with no failures/errors/skips, in 16.313 seconds (`.local/mobile-body-tests.xml`); **17 deterministic UI checks** pass (`.local/mobile-ui-tests.log`). Actual browser/device interaction remains unperformed.

`artifacts/mobile-interaction-02` verifies authenticated loopback HTTP for release, left/right push, restoration and reference following. All 120 returned transitions exactly match the saved journal; all 121 exported video frames decode; original and copied 96-file run manifests remain unchanged. Twenty six-step HTTP batches measured median 7.10 ms and p95 7.71 ms. This is one short local sequence, not sustained browser responsiveness. The contact sheet was inspected locally. Interaction seal: 110 files, `e7d5eeeebb3b0db5e37d1a7e547f9f8a10667c292600aafc31b8090b42b4899c`.

The first interaction attempt is preserved in `mobile-interaction-01`: physics and the 121-frame export completed, but the verification script compared dictionaries with mismatched keys. The corrected script includes the step event type on both sides and saves HTTP transitions before comparison; no simulator behavior or threshold was changed.

Meta construction advice is prepared in `artifacts/meta-mobile-body-dryrun-04` about gravity/support alternatives and falsifiable contact tests. Only whitelisted authored support settings and construction evidence enter the prompt; no scoring labels/evaluation are exposed. No request was sent from that historical preparation. The key is now configured; subsequent live reasoning and tracking evidence is recorded in the [project overview](../README.md). Five settled calls commit $0.152552 of the shared $5 budget, with $4.847448 remaining; no model advice automatically changes this body.

Still required: reliable longer root trajectories, camera/metric calibration, supported contact geometry, explicit external equipment, anatomical/force validation, meaningful whole-body tasks and multi-seed evaluation. Virtual support must not be described as recovered balance. These scoped passes do not complete arbitrary-video reconstruction.
