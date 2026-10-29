# Three-ball Meta tracking: locations retained, continuity IDs unresolved

The bounded SAM 3.1 call returned three separate box regions throughout the post-initialization interval. Against the frozen identity labels, however, IDs 2 and 3 follow opposite post-contact branches. This is not a validated recovery of physical identity or dynamics. Independent tracking remains the default.

## Frozen comparison

The [sealed run](../artifacts/meta-three-ball-01/report.json) used the original 426×240 interval, source frames 2580–2760, as an explicitly lossy CRF18 MP4 with all 181 frame mappings and original timestamp gaps verified. All three boxes were initialized at source frame 2620 / request frame 40: ID 1 upper-passive, ID 2 lower-passive, ID 3 incoming-active. One forward-only request processed 141 frames, cost approximately $0.0282, and completed in 13.69 seconds. No retry was made.

Accuracy excludes the pre-initialization and initialization slots. All 24 manual slots remain in the report; the common accuracy denominator is **17 visible post-initialization center labels**. Labels have ±5 px tolerance, and post-contact physical identity is inferred rather than independently known.

| Frozen metric | SAM 3.1 | Saved joint tracker v2 |
|---|---:|---:|
| Returned eligible center slots | 17/17 | 9/17 |
| Within 5 px under fixed IDs | 8/17 | 9/17 |
| Conditional center RMSE | 42.283 px | 1.054 px |
| Maximum accepted center error | 98.653 px | 1.414 px |
| All-three returned post-init frames | 140/140 | 47/140 |
| Returned post-init object slots | 420/420 | 264/420 |
| Predictions conflicting with labeled absence | 1 | 0 |

Missing observations do not receive zero error. The v2 figures were recomputed from its saved trace on the same eligible labels; its older 9/22 denominator included five visible pre-initialization/initialization labels and is not directly comparable. Boxes use the same half-open geometric-midpoint convention. Source hashes, original grades, trace hashes and exact frame mapping are preserved in the [comparison](../artifacts/meta-three-ball-01/comparison.json).

## Post-hoc location diagnostic — not a replacement grade

For each labeled post-initialization frame, a separate diagnostic assigns returned box centers to visible label centers by minimum total squared distance, without using identity continuity. Each prediction can be assigned at most once; unmatched predictions remain explicit. This diagnostic uses the labels after the call and **does not relabel the accepted result or turn the frozen evaluation into a pass**.

- At frame 2640, the closest assignment agrees with IDs 1, 2 and 3.
- At labeled frames 2650, 2670, 2700 and 2720, the closest assignment is prediction 1 → upper label, prediction 3 → lower label, prediction 2 → incoming label.
- At frame 2760, prediction 1 matches the upper label and prediction 3 matches the lower label; prediction 2 remains unmatched because the frozen incoming label has no center.
- All 17 visible label locations are within 5 px under this anonymous assignment: RMSE **0.831 px**, maximum **1.803 px**. This establishes compatibility with the sparse locations, not correct physical identities, accurate masks, or accuracy on unlabelled frames.

The divergence is present by the first post-contact labeled frame, 2650; these sparse labels do not establish an exact identity-switch frame. Source inspection shows separate ball-like regions around the event. Since the balls look alike, the source cannot uniquely determine physical continuity through contact. The result disagrees with the **declared continuity labels**, not independently established object identity. See the [per-frame assignments](../artifacts/meta-three-ball-diagnosis-01/anonymous-location-diagnostic.json) and [source contact sheet](../artifacts/meta-three-ball-diagnosis-01/source-contact-sheet.png).

## No duplicated box regions found

Across all 140 post-initialization frames and 420 object pairs:

- No identical boxes or centers; no pair of centers within 2 px.
- Minimum center separation: **7.566 px**.
- Only one pair has positive box overlap, at frame 2644; its intersection-over-union is **0.0241**. No pair reaches 0.5.

Intersection-over-union measures shared box area divided by their combined area. These post-hoc descriptive checks show distinct returned regions, not duplicated assignments to the same location. They do not prove that every region is a physical ball or that identities are correct. Full pair measurements are retained in [pairwise-boxes.json](../artifacts/meta-three-ball-diagnosis-01/pairwise-boxes.json).

Confidence does not resolve this identity conflict. The nine fixed-ID center errors above 5 px have box confidences **0.793–0.847**, mean **0.815**; the eight within tolerance have mean confidence **0.801**. These are model output confidences, not calibrated probabilities of physical identity correctness.

## Frame 2760: frozen absence label conflicts with the source

| Returned ID | Box center, source px | Box confidence | Location-only observation |
|---|---|---:|---|
| 1 | (178.5, 11.5) | 0.763 | Separate bright ball-like region above the paper |
| 2 | (36.5, 151.5) | 0.801 | Separate bright ball-like region on the left wooden strip |
| 3 | (99.0, 226.5) | 0.745 | Separate bright ball-like region over the lower yellow label |

The frozen annotation says incoming-active is `out_of_frame`. Post-hoc inspection of the [native source frame](../artifacts/meta-three-ball-diagnosis-01/source-2760.png) and [unannotated enlarged regions](../artifacts/meta-three-ball-diagnosis-01/final-frame-regions.png) shows three visually distinct ball-like highlights inside the frame. Two agents independently reviewed these source crops. The image supports a location-level annotation concern; it does not establish which identical physical sphere occupies each region.

The original annotation and grade remain unchanged, including the one labeled-absence conflict. **Do not describe it as a verified false positive or known physical absence.** This review is post-hoc development evidence, not a revised independent holdout. The diagnostic leaves prediction 2 unmatched at this frame instead of inventing a new scored center.

## Provenance and limits

The [diagnosis artifact](../artifacts/meta-three-ball-diagnosis-01/report.json) verifies the 34-file live run and 105-file input bundle. Ten selected native YUV frames were individually checked against the saved original pixel hashes before RGB conversion and lossless PNG export. Unannotated crops are separate from rendered model overlays. No masks were decoded, no tracker was rerun, and this diagnosis made no API calls or additional paid requests.

The sealed live source copies remain authoritative for this call. A later transport cost-arithmetic correction affects future float serialization only; it is not evidence of source drift during the call. The historical ledger remains $0.138511 committed / $4.861489 remaining, including its existing conservative microdollar rounding.

The useful result is improved observed-location coverage with unresolved continuity IDs. It does not solve calibrated 3D reconstruction, camera motion, contacts, forces, restitution, simulator validity, or generalized-video accuracy. No default change, repair promotion, retrospective score replacement, or further API request follows from this study.
