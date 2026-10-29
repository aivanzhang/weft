# Recovering failed poses from original video pixels

**The denser jogging sandbox now builds and records correctly: 224 body poses across all 240 source frames, with 16 missing poses retained explicitly.** Four inconsistent poses were re-estimated from image crops. This improves coverage, not established anatomical accuracy or foot contact.

This experiment requires **two supplied calibrated camera views with assumed aligned clocks**. It does not recover camera calibration or verified 3D anatomy from an arbitrary single exercise video. The earlier [failed every-frame build](dense-exercise-poses.md) and all original sandboxes remain saved.

The subsequent [complete five-exercise study](selective-crop-exercises.md) produces separate playable single-camera squat/push-up candidates, preserves three rejected clips, and checks controls, learning and recording. It does not resolve anatomical or physical accuracy.

## Try the new sandbox

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/temporal-crop-jog-body-lab-01
```

Run that command yourself, then open the printed local URL. Use the joint, push, release and support controls. **Record & start** / **Stop & save** preserve your actions and the resulting states. No learned exercise policy is available; the displayed muscle bands are illustrative, not measured activation or recovered muscle forces.

Inspect without starting a server:

- [Source/body panels](../artifacts/temporal-crop-jog-interaction-01/index.html), [recorded action replay](../artifacts/temporal-crop-jog-interaction-01/session/replay.html) and [recorded video](../artifacts/temporal-crop-jog-interaction-01/session/video.mp4).
- Frame 191: [original bad image labels](../artifacts/dense-jog-rejection-01/view-1-frame-0191.png), [new image estimate](../artifacts/temporal-crop-jog-01/inference/view-1/pose-0191.png), [new source/body panel](../artifacts/temporal-crop-jog-interaction-01/reference-0191.png).
- [All crop decisions](../artifacts/temporal-crop-jog-01/decisions.json), [fixed-slot comparison](../artifacts/temporal-crop-jog-comparison-01/report.json) and [remaining frame-148 body error](../artifacts/temporal-crop-jog-interaction-01/reference-0148.png).

## What happens step by step

1. Keep the original every-frame observations and triangulated joint slots unchanged. Select every frame where the original two-camera hip rays disagree: **139, 171, 187 and 191**. Missing/competing-person frames are not silently selected as one person.
2. Find an original usable pose on each side within 0.2 seconds. Their visible image bounds define one crop, expanded by 25%. Reject missing context, unstable boxes, crops smaller than 64 pixels per side or larger than 80% of the frame. No corrected observation supplies another crop's anchors.
3. Run the same pinned Full IMAGE-mode model on the **actual target-frame crop**, once per view. Map its image coordinates back to the full image; keep its newly inferred hip-relative world coordinates. No joint interpolation or side permutation is performed. Coordinate conventions follow the [official Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
4. Retest both new observations using the unchanged calibrated-ray checks. Require usable single-pose hypotheses, all four torso/hip points, at least eight of twelve major joints, no loss of previously supported major joints, and a root within the declared original-neighbor motion bound. This last check is a heuristic consistency condition, not proof of physical motion.
5. Preserve all proposals and decisions. Accepted rows retain their complete original observation, crop/anchor identities, proposal hash and decision hash. Unselected rows remain exactly unchanged. The body reference carries the refinement metadata into the sandbox; original and current tracking previews remain separate.
6. Build the existing camera-ground/partial-limb body, then independently compare it and exercise its controls and recording. Ground, controller, motor limits and support settings are unchanged. Newly sampled evidence changes fitted body dimensions and its initial coordinate frame; this is not a crop-only body ablation.

`src/weft/pose_crop.py` implements planning, coordinate mapping, acceptance and exact evidence preservation. It allows at most 32 target frames, with one proposal per view and no retry/search loop. The source-specific study uses two local workers with 95-second process limits. Original inputs are checked before and after construction; provider body joints and third-camera evidence are excluded from proposal generation and selection.

The initial known-frame probe recovered frame 191 with 2.01–3.88 px hip reprojection errors, versus 25.33–37.56 px originally. The complete study reused those two exact crop inferences and made **six new local calls**, taking **5.646 seconds** including candidate packaging. All four proposals pass the fixed source-only checks. Each view preserves its other **236 observations exactly**. The second view still has only 150/240 usable whole-body poses and fails the unchanged 65% visibility check; per-joint two-view construction does not erase that failure.

## Coverage and accuracy are separate

All comparisons retain **2,880 source/joint slots**. These are development comparisons against third-party reconstructed body/camera references, not measured anatomy or a fresh holdout. The crop diagnosis has already been inspected.

| Image-based two-view geometry | Before | After |
| --- | ---: | ---: |
| Available major-joint slots | 1,964 | 1,997 |
| Lost previously available slots | — | 0 |
| RMSE on the same 1,964 available slots | 0.077853 m | 0.077859 m |
| Third-camera RMSE on the same 935 slots | 23.9419 px | 23.9419 px |

The 33 newly recovered slots have reconstructed-reference RMSE **0.05990 m**, maximum **0.13919 m**. Four old estimates change: one improves and three worsen; the largest regression is **0.01672 m** at frame 139's left wrist. All other 1,960 previously available major-joint estimates are unchanged. The recovered frames are odd-numbered; the sparse third-camera observation set has **no observations at these frames**, so it cannot independently corroborate them.

| Native body comparison | Previous 112-pose sandbox | New 224-pose sandbox |
| --- | ---: | ---: |
| Body-joint RMSE on the same 1,344 joints | 0.128058 m | 0.127972 m |
| Third-camera RMSE on the same 1,117 slots | 24.4385 px | 24.4224 px |
| Maximum joint error on the paired slots | 0.60533 m | 0.60683 m |

**Accuracy on the old poses is essentially unchanged.** On those paired slots, 778 errors improve and 566 worsen; the largest individual increase is 0.00215 m. The new poses add 1,344 body-joint estimates, with separate RMSE 0.12169 m. Do not mix their different slot set into a claimed paired improvement.

All **224 bodies still float** above the supplied ground: minimum 0.01607 m, median 0.12969 m, maximum 0.25277 m. The 224 source-pose and 240 control-grid penetration checks pass, but these checks do not establish correct contact. Existing left/right ambiguity, frame-104/frame-148 limb failures, hidden-arm priors, uncertain anatomy and authored forces remain unresolved.

## Recording, performance and evidence

The new sandbox's **39 actions, one reset and 41 states/video frames replay exactly**, including rewards and completion flags. Recording session `session-10c56add0e52` has seal `4cc720c43405811887735c33b9e058e296fd9f35271371f2c0eef0c8b9b1630d`. Source/body panels for frames 191 and 148 were visually inspected without browser interaction.

Across 120 in-process action/state/JSON calls, median latency is 0.800 ms, p95 0.830 ms, maximum 0.961 ms and largest response 35,126 bytes. This excludes HTTP, browser painting and sustained idle behavior. A separate 900-pose/view, 32-target analytic workload takes 0.248 s for planning, assessment and evidence assembly, after 4.416 s for the two baseline/proposal triangulations. No model inference is included in that maximum-count check.

**90 distinct software checks pass**: 26 crop/geometry/provenance/abstention checks and 64 affected vision, reference, human-model and cache checks. Records: `.local/pose-crop-tests-01.xml`, `.local/pose-crop-integration-tests-01.xml`; load evidence: `.local/pose-crop-load-01.json`. Missing/competing hypotheses, genuine consistent motion, changed inputs, repeated refinement, lost joint support, implausible root jumps, exact coordinate transforms and unchanged original rows are covered.

| Bundle under `artifacts/` | Files | Manifest SHA-256 |
| --- | ---: | --- |
| `temporal-crop-probe-01` | 117 | `abfd556c759e67be498f9e59a9cecfe7a6106a7b8dbfff9cb5cf71d5504fe747` |
| `temporal-crop-jog-01` | 173 | `f4b501180c9f8add4cceac5ef43c36db432bf282155a87f8f9a24875e0015fa1` |
| `temporal-crop-jog-body-build-01` | 134 | `e643fc30dac98938fa3c0e540484640a6849c25dabf0196fcf94a1b8e05be011` |
| `temporal-crop-jog-body-lab-01/run` | 258 | `f859713850325c812ebab1645073ccd4ecd799a081f32aedd912b91fc724432c` |
| `temporal-crop-jog-comparison-01` | 8 | `3bf04d1419118aa41a77304ab9eac63a4d90f993b6513964ce60064200b08759` |
| `temporal-crop-jog-interaction-01` | 24 | `62a8fa4e137cb4af9a303a249736454c07fc4d309d5130a0360bdd68110c22bd` |

No paid call, model download, policy training, server startup, deployment or default workflow change occurred. At **October 29, 2026, 4:16 AM PDT**, the existing application-side Meta ledger remains **$0.176152 of $5**, with **$4.823848 remaining** and no unsettled reservation. The full balls/exercise/general-video goal remains open; this run adds no new source-video diversity or independent changed-physical-action evidence.

## Follow-up: five single-view exercises

At **October 29, 2026, 4:50 AM PDT**, a separate development screen compared whole-frame and cropped inference on single-leg squat, squat, kettlebell push-up, jumping-jacks/burpees and lunge/twist. All 2,677 source frames were decoded, but only **28 selected samples** were inferred. The same pinned Full IMAGE-mode model made 53 local calls: 28 whole-frame and 25 crop calls; three crops lacked usable context. Two bounded workers completed in 3.023 seconds.

Unconditionally using crops improves some torso-side estimates but loses limb coverage and worsens some push-up estimates. A new experimental selector therefore keeps an ordinary usable pose, considers a crop only for an unusable pose or a clear short torso reversal, and rejects proposals that lose previously visible joints or disagree with reliable original neighbors. Competing hypotheses, long gaps and unclear context remain unchanged. No label coordinates enter selection.

| Fixed development comparison | Whole frame | All crops | Selective crops |
| --- | ---: | ---: | ---: |
| Torso-side checks passing / 39 observable | 29 | 36 | 33 |
| Limb checks passing / 84 observable | 55 | 54 | 63 |
| Missing observable limb estimates | 18 | 26 | 10 |
| Torso-side / limb regressions from whole frame | — | 3 / 17 | 0 / 0 |

The selector replaces **only jumping-jacks frame 40**, recovering eight limb checks and four torso-side checks; the latter use two adjacent labels matched to the same sampled estimate, not four independent poses. The other 27 samples remain exactly unchanged. Previously scored errors do not improve, and confident wrong estimates remain. The selector was designed after inspecting the crop screen, so this is **development evidence, not independent validation**. Labels are approximate assistant-made image annotations, not measured anatomy.

This screen combines historical raw neighboring poses with newly inferred Full-model targets. Complete-video inference would change those neighbors and crop plans; it must be qualified separately. `pose_crop_selection.py` remains outside the default pipeline. It preserves source identity, clocks, original observations and decision hashes, with a 64-target limit. Its 21 analytic checks cover abstention, genuine motion, missing context, competing people, lost joints, changed inputs and repeat refinement; they do not establish physical accuracy.

- [Matched crop report](../artifacts/crop-diversity-screen-01/report.json): 256 files, seal `17f8cccb1107bbdca64eba5b7125f62b90254d4c30616742371c9fe7b0d15ce4`.
- [Selective report](../artifacts/selective-crop-screen-01/report.json) and [all decisions](../artifacts/selective-crop-screen-01/decisions.json): 120 files, seal `c89caa8c47d3c10d239db889e7a00675c54e13a97ff0d1ce4372198e1930ed30`.
- `.local/selective-crop-final-verification-01.json` verifies both bundles, current frozen runtime, exact unselected rows/clocks and the 21 saved passing checks.

Selection reused saved proposals without another inference call. This follow-up made no paid request, built no new body and changed no defaults or existing sandbox. Whole-video qualification, remaining joint errors, missing observations and physical contact validation remain open.
