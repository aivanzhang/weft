# Diverse-video end-to-end verification

Observed **October 28, 2026, 12:42 PM PDT**. Local development evidence; not a final holdout.

This is the preserved v1 result. Later orientation, explicit inference-mode and
grayscale-tracking work is documented in [follow-up verification](iteration-followup.md);
it does not rewrite these historical outcomes.

## Result

The 15-case suite completed in **72.176 seconds**, using at most two local pipeline workers. Ten environments passed their declared gates and all ten produced verified interactive recordings. Two synthetic negative cases were correctly rejected; three desired cases remain failed coverage.

| Case | Outcome | Learning success | Random success | Source-fit / servo error |
| --- | --- | ---: | ---: | --- |
| constant-motion | gates passed | 100% | 14% | 0.94 px |
| fixed-camera-texture | gates passed | 100% | 8% | 0.54 px |
| gravity-flight | gates passed | 100% | 0% | 3.71 px |
| gravity-bounce | gates passed | 100% | 0% | 8.58 px |
| elastic-contact | gates passed | 100% | 10% | 6.55 px |
| brief-occlusion | gates passed | 100% | 14% | 0.96 px |
| same-color-ambiguity | expected rejection | — | — | — |
| camera-translation | expected rejection | — | — | — |
| single-leg-squat | gates passed | 100% | 0% | 0.218 rad (servo) |
| squat | gates passed | 100% | 18% | 0.241 rad (servo) |
| kettlebell-pushup | gates passed | 100% | 82% | 0.168 rad (servo) |
| lunge-twist | execution failed | — | — | — |
| jumping-jacks-burpees | execution failed | — | — | — |
| real-rolling | gates passed | 94% | 10% | 4.28 px |
| electrostatic-ball | execution failed | — | — | — |

These rates are 50 matched episodes per policy, one training seed per case, on each generated world. They do not prove cross-world transfer or a statistically stable improvement across training seeds. Several baselines already succeed frequently. Pose coverage and servo tracking do not establish accurate anatomy, balance, contacts, or muscle forces.

## Evidence

- Suite input: [video-diversity-v1.json](../benchmarks/video-diversity-v1.json); [usage and bounds](../benchmarks/README.md).
- Full results: `artifacts/video-diversity-e2e-01/report.json`; readable summary: `report.md`.
- The completed study seal verified **949 files**, manifest SHA-256 `c658bd03e0fb607a33c1254ef203d907645ce9131962337e9a3f67eb79b86d2b`. Sealing was added after execution; nested implementation snapshots preserve the exact executed source, while the independently tested finalizer hashes the complete study without changing nested runs.
- Each successful case recorded 12 action transitions and 13 video frames. Recorded actions/state match the returned simulation transitions exactly; all before/after run-manifest hashes match. Export used a killable 90-second child process and full video decode.
- Full regression run: **243 passed**, zero errors/failures/skips, `.local/video-diversity-tests.xml`, 11.761 seconds. A subsequent seven-check benchmark run verified the new study sealing path, bringing distinct verified Python tests to **244**. UI checks: **14 passed**.
- Source pins, failed cases, attribution, repair ledgers, model/task settings, saved implementation copies, logs and exported recordings are retained. No live Meta request, cloud deployment or submission occurred in this development pass.

## Fixes verified in this pass

- Configured gravity no longer biases estimated initial velocity; fixed-camera assumptions have an explicit background-motion screen. [Independent ball checks](ball-diversity.md).
- Matroska frame counts are independently checked when duration metadata disagrees; cropped joints lose reliable-confidence status, while original estimates remain preserved. The electrostatic clip now reaches the actual color-tracker limitation instead of failing on its container metadata.
- Reference servo testing covers the full bounded input duration. Lateral motion, torso tilt and joint clamping are disclosed. [Exercise study](exercise-diversity.md).
- Exercise task choice follows reliable robust joint motion rather than always selecting the left knee. These runs chose right hip for single-leg squat, right knee for squat, and left elbow for push-up. This is an authored target-control heuristic, not recovered intent or whole-exercise learning.
- A failed planar per-object reconstruction gate now prevents the lab from labeling the generated world ready.
- [Shared observation contracts](scene-observations.md) preserve identities, units, time, visibility, evidence partitions and unknowns across all current profiles.

## Still failing

- **Lunge/twist:** ambiguous extra pose estimates cause rejection. The source is not established to contain multiple people merely because the detector says so.
- **Jumping jacks/burpees:** ambiguous detections plus out-of-plane/root/ground movement beyond the current body model.
- **Electrostatic ball:** low-saturation objects and unmodeled electrostatic support/forces exceed the colored planar tracker.
- **Push-up fidelity:** an executable elbow controller still uses an upright anchored body; the horizontal torso, floor/kettlebell contacts and load are not reconstructed. Passing the declared execution/servo gates is not passing a full-body fidelity gate.

## Next evidence gates

Independently label the ambiguous pose frames before changing identity logic; add general appearance-based masks/tracks and calibrated camera handling; extend body root/orientation/contact mechanics; test on fresh real motion/contact annotations and multiple training seeds. Browser/device rendering remains unverified.

## Meta access

User approved a **$5 total** development budget. It is configured in the private project ledger, preserving reservations and known/unknown charges across both Meta adapters. At this observation: $0 committed, no reservations, no key configured and no live calls. [Setup and cost limits](meta-advisor.md).
