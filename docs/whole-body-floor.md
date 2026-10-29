# Squat floor failures: reset-only diagnosis

The six failed learned-policy squat trials already violate the floor limit **before any action**. The immediate cause is an inconsistent reconstructed reference pose and authored floor, not a movement produced by the learner. This does not establish which upstream estimate is physically wrong.

Evidence: [diagnostic report](../artifacts/whole-body-floor-diagnosis-01/report.json), [reproduction script](../artifacts/whole-body-floor-diagnosis-01/diagnose.py), and the unchanged [learning study](../artifacts/whole-body-learning-02/report.json). The diagnostic recreated nine seeded evaluation resets and five unjittered comparison resets. `mj_step` was replaced with a function that raises; no dynamics, training, policy evaluation, model edits, or threshold changes occurred. Module hashes matched the frozen study. The report records source-video, model, settings, evaluation, and implementation hashes.

## Measured simulator overlap

All distances below are **model units**, not measured metres. The unchanged failure limit is 0.025.

| Study seed | Start 4.333 s | Start 4.700 s | Start 5.067 s |
|---|---:|---:|---:|
| 11 | 0 | 0.108819 | 0.215640 |
| 29 | 0 | 0.108365 | 0.213571 |
| 47 | 0 | 0.108903 | 0.211933 |

These are reset-time maximum penetrations. In all six failing windows, the initial penetration equals the saved full-episode peak for **each** of the learned, authored-feedback, random, and zero policies. Because the metric retains the initial maximum, no later action can make these episodes pass the existing floor gate. The first window starts clear; its later outcomes still depend on the controller. This finding does not imply all controllers perform equally.

The deepest intersecting shape is `left_knee_geom` on `left_knee_body`: despite its name, this is the **lower-leg/shin capsule extending from knee to ankle**, not a sphere at the knee. The right lower-leg capsule also intersects. Exact contact distances, positions, capsule sizes, and minimum geometry heights are retained in the report; reset contact forces are not interpreted as physical evidence.

## Why the reference can start underground

The model's horizontal floor is present and collision detection is active. Its height is authored as zero. Only the first reference pose sets pelvis base height, placing its lowest capsule surface 0.005 above that floor. The recorded base height is 0.785895; no later pose was used to refit it.

Root translation then follows image pelvis displacement at a fixed, uncalibrated scale of 0.00211738 model units per pixel. Joint rotations come from separate model-world landmark geometry; limb lengths are fixed. These estimates are not jointly constrained by observed foot contacts or a calibrated floor.

Without reset jitter, the root z displacement is +0.055344 at 4.333 s, −0.148167 at 4.700 s, and −0.343050 at 5.067 s. The corresponding maximum overlaps are 0, 0.107748, and 0.214285. The first source pose and the near-last comparison at 7.033 s are clear. Therefore small seeded reset noise is not the cause: the nominal reconstructed trajectory itself conflicts with the static floor during the squat.

This is **not missing floor geometry**. It is missing independently grounded floor/contact consistency across the inferred root motion, rotation, scale, and fixed limb geometry. This diagnostic cannot isolate camera calibration, landmark inference, segment lengths, or their combination as the physical cause.

## Source limitations

Visual inspection of retained [frame 0](../artifacts/whole-body-floor-diagnosis-01/source-frame-0.png) and [frame 152](../artifacts/whole-body-floor-diagnosis-01/source-frame-152.png) shows one person carrying a barbell, with the feet substantially cut off at the lower image boundary. The floor is visible elsewhere, but this is not a calibrated observation of the person's sole contacts. The simulated body omits the barbell and rack interactions.

The retained ankle image estimates lie near the lower boundary (normalized y approximately 0.97–1.01 in the inspected samples). High model confidence does not turn cropped or partly hidden feet into measured contact locations. Source frames, landmark subsets, source IDs, and image/world estimates remain in the diagnostic artifact. No claim is made that the real person penetrates the real floor, that the inferred depth is observed, or that simulated contact forces match human forces.

## Interpretation and next boundary

Keep all six trials failed and preserve the existing gate. The learned policy's tracking performance is a separate result from whether its reconstructed reference is initially feasible. Any future change needs a separately declared reconstruction/reset contract and development study, with reference feasibility reported before training. Silently lifting every reset, refitting the floor to evaluation poses, discarding these windows, or weakening the floor limit would hide this failure rather than resolve its evidence gap.
