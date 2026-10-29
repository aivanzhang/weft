# Fixed limb lengths and the third-camera check

**The fixed-length correction is rejected.** It loses supported joints and worsens agreement on the same third-camera observations. The [playable squat](heavy-pose-model.md) is unchanged.

Inspect the frozen comparison at [standing frame 550](../artifacts/freemocap-third-comparison-01/third-view-comparison-0550.png) and [blurred frame 966](../artifacts/freemocap-third-comparison-01/third-view-comparison-0966.png). Cyan circles are the third camera's original image-model observations. Orange crosses are predictions made before that camera was opened. Neither is measured anatomy.

## What was tested

`constrained_stereo.py` fits up to twelve joints to the original Cam1/2 image observations while holding inferred segment lengths fixed. Lengths come only from supported original estimates in frames 400–620, inclusive. Cameras, clocks, side labels, source coverage and image limits remain unchanged. No missing joint is filled.

The first solver often reached its iteration cap. A bounded curvature update improves numerical convergence from **248 to 512 of 522** fits without changing observations, lengths or the 40-iteration limit. That numerical improvement does not establish better human geometry.

| Complete-source result | Original two-view estimates | Converged fixed-length candidate |
| --- | ---: | ---: |
| Supported major-joint slots | 5,069 | 3,966 |
| Samples with all twelve major joints | 143 | 57 |
| New / lost supported major joints | — | 15 / 1,118 |

All 1,108 source frames, 554 analyzed moments and 36,564 joint slots remain. Inactive, absent-person, unsampled and rejected observations are not removed to create an easier exercise episode.

## Known geometry exposes incorrect length priors

The 72-case controlled comparison uses identical inputs for both solver versions. It is a software experiment, not a physical capture.

| Control, 24 cases each | Original 3D RMS error | Constrained 3D RMS error |
| --- | ---: | ---: |
| Independent two-pixel observation noise | 67.39 mm | 45.51 mm |
| One hip observation biased by fifteen pixels | 126.72 mm | 70.67 mm |
| Every length prior five percent too long | 67.39 mm | **83.91 mm** |

Only three biased-hip fits satisfy the fixed-length/convergence requirements. All 24 wrong-length cases converge and pass their image limits, despite worse true 3D error. Image agreement plus stable bone lengths is not enough to establish anatomy.

## Predictions were frozen before Cam3 was opened

The [human prediction protocol](../artifacts/freemocap-third-predictions-01/protocol.json) freezes all 13,296 major-joint slots per method, code hashes, source identity, model, clocks, scoring rules and limits. The separate [board protocol](../artifacts/freemocap-third-board-predictions-01/protocol.json) freezes 234 projected corners across eighteen poses. No camera, timing, length, label or score is fitted after opening Cam3.

The comparison finishes on **October 29, 2026, at 2:25 PM PDT**: 1,108 pose-frame decodes, 554 local model calls, eighteen board-frame reads and four post-grading inspection reads, in 19.23 seconds. No paid request is made.

- **Board:** all 234 predictions are compared; p95 error **1.57 px**, maximum **2.41 px**, passing the frozen 3/10-pixel limits. All 500 prefix slots, including 482 without supported board placements, remain. This checks stereo board placement in another view; it does not waive the original failed monocular placement check.
- **Human, same 3,918 comparable slots:** original RMS **16.74 px**, fixed-length RMS **17.02 px**; 1,800 improve and 2,118 worsen. The candidate also loses 1,092 observable predictions and adds fifteen. Comparing each method only on its own available subset would hide these losses.
- **Active interval, same 2,791 slots:** **10.77 → 11.10 px** RMS, with 690 observable predictions lost and ten added.
- The secondary diagnostic includes converged proposals that failed the construction-image limits. Its shared-source error also worsens; it never overrides admission rules.

**Cam3 is now exposed development evidence.** It is not an untouched final test for subsequent repairs. The 6-fps derivative of the same recording is not a fresh holdout either. The third view uses the same pose-model family, so its estimates are not independent anatomical measurements.

## Verification and retained evidence

[Independent verification](../.local/freemocap-length-comparison-verification-01.json) checks seven manifests / **92 files**, exact matched inputs between solver versions, all source slots, saved control errors, third-view projections and shared-cohort scores. OpenCV projection accounting differs by at most 3.17e−13 pixels. All 120 runtime files match the frozen comparison; fifteen solver and ten scorer tests passed and were reused unchanged.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| First / final source fits | 17 / 17 | `4a3abd279716f69c9abd69f8ff431b0ebc6ae29d5ed2eb180915e0571d572c30` / `72990428ebb125150ee46bc9985fc6ee713710af7bd9283438a06ea2c791527b` |
| First / final known-geometry controls | 6 / 6 | `b672d5c53fc993c9167a499df8736af68f921f5c60e0a6fd8b0fa6d0c7c59cfe` / `40525cb24b7277ea06afc85d0bce0f1010c08cf8e45c339ea53c370999cbce3f` |
| Frozen human / board predictions | 7 / 4 | `72706982464be5b677c33452c6b52c85e9cc91de20a82004810b01dce8bd6790` / `cb5c2f064dc47e77ca471f3685f7a3f306975a607444751f9b21bf4c57eb8de2` |
| Third-camera comparison | 35 | `9a7b2a0aa0afa4f6e8dea6e1cb7f0b2743a6d5db17cb11cd6a5e115a91693d4d` |

Earlier numerical failures remain saved, including the near-feasible length-tolerance failure. Verification corrected two report-reader assumptions (`truth`, and cohort/index pairing); no source, fit or score changed. No new body, floor, policy, deployment or default is produced.

The next [all-observation triangulation study](freemocap-three-view.md) drops the fixed-length prior and explicitly uses the already exposed third camera as construction evidence. Source landmarks, floor/contact evidence and independent changed-action validation remain unresolved; balls and general-video reconstruction remain part of the active goal.
