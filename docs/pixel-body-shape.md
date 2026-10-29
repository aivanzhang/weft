# Static body-length experiment

**No candidate is promoted.** Fitting body lengths to image measurements improves three clips' later-pose image errors, but all four candidates still intersect the floor beyond the existing limit. Two candidates make floor collisions worse.

Open the [offline comparison](../artifacts/pixel-body-shape-view-01/index.html) to inspect source images beside original and candidate skeletons. No server or API call is needed. The view aligns both skeletons to the observed hip center, isolating shape; it deliberately does **not** test root-position accuracy. It shows four evenly spaced retained poses per clip, not selected best frames. All 379 poses are included in the numerical study.

## Frozen method

The experiment follows the rejected [world-landmark shape fits](fullbody-geometry-diagnosis.md). It estimates the four existing shared left/right length parameters—upper arm, forearm, thigh and shin—from the first 60% of source timestamps. Later poses cannot affect the estimates. The original full-clip reconstruction and scale are inherited, so this is development evidence, not a final holdout.

For each segment, divide its source-image length by its original model's projected length. Weight that ratio by the lower endpoint confidence times the squared visible projection fraction. Exclude confidence below 0.5, projection fractions below 0.2 and image segments shorter than two pixels. Require three usable segment observations per shared parameter. Add a prior toward the original length with 10% of the usable weight, then cap the change at ±15%. Unidentifiable lengths retain their originals.

These rules were frozen and covered by 16 analytic tests before any of the four real fits. The fit never reads later-pose residuals, manual labels, floor contacts or dynamic results. It does not estimate new angles, attachments, per-frame bone sizes, camera scale, root motion or floor height.

Each candidate is an actual MuJoCo model with changed static lengths. Its forward kinematics are evaluated at every original pose. Floor checks use the same complete source and 30-Hz interpolation schedules and the unchanged 0.025-model-unit limit. The original floor-fit claim is removed from the candidate because it describes old geometry; its numeric base height is preserved, not recalibrated. The old claim remains in the diagnostic evidence.

## Four-source results

Image RMS below measures hip-centered model joints against pose-estimator image coordinates, not independent anatomical labels. Floor counts include every retained source pose; interpolation checks are also saved.

| Source | Poses | Later image RMS, pixels: original → candidate | Floor-failing source poses: original → candidate | Peak overlap: original → candidate |
|---|---:|---:|---:|---:|
| Fresh full-body squat | 34 | 10.637 → 11.445 | 13 → 3 | 0.07384 → 0.04432 |
| Single-leg squat | 118 | 7.427 → 7.393 | 8 → 12 | 0.08313 → 0.09686 |
| Cropped squat | 100 | 37.034 → 31.983 | 40 → 23 | 0.25390 → 0.19012 |
| Kettlebell push-up | 127 | 19.167 → 13.155 | 25 → 51 | 0.06648 → 0.12386 |

The push-up fit reaches the +15% cap for all four lengths. Its inferred-world endpoint error rises from 0.04800 to 0.07861 model units despite better image agreement. This conflict is consistent with projection/pose uncertainty; it does not establish that the person has longer bones. The fresh squat has fewer floor failures but worse later-pose image agreement. Neither improvement licenses changing the acceptance definition.

No independent anatomical endpoint labels were added or consulted, no dynamics or training ran, and no existing model changed. All four sources are known development clips; they are not four new downloads or unseen tests.

## Evidence and next action

- Study: `artifacts/pixel-body-shape-01`, 52 files, seal `ed48a2751d3634a0de8cd348f04d3dd89f6bb71493307398048ab008e9b1dda6`.
- Inspection: `artifacts/pixel-body-shape-view-01`, 16 three-column images, attribution records and offline HTML; 23-file seal `1e9708b6e35ca2dc77728227311294e87c99eb1a05f6446bfa66b204639fa61c`.
- The four-source study took 1.199 seconds locally; no paid calls. This is one observed runtime, not a performance guarantee.
- Sixteen analytic tests cover known length recovery with prior shrinkage, unavailable/foreshortened segments, split isolation, caps, stale floor claims, source identity/timing, finite inputs, immutability and prohibition of floor fitting or physics steps.

Next, separate camera projection from inferred body geometry using independently supported camera orientation and visible anatomical endpoints. Any further fit must preserve these rejected candidates and unchanged later-pose/floor checks. Do not repeatedly resize bones to compensate for floor or camera errors. Illustrative muscles remain separate from measured anatomy or activation.
