# Joint tracker: remaining loss diagnosis

## V2 follow-up: search exclusion addressed, appearance losses remain

The retained v2 traces in `artifacts/joint-appearance-02` were compared against v1 without rerunning tracking or changing labels. `artifacts/joint-v2-comparison-01` verifies both 48-file input seals and preserves all 26 changed status/position rows, focused source-frame records and recomputed grades. Its four-file seal is `d5d45c04c193dac2553e42b693660edd41a1e98201050d4db837662395655af6`.

At incoming frame **2646**, v2 accepts box **[180,114,8,8]**, center **(184,118)**, on the branch identified in the earlier development inspection. Its explicit search rectangle is **[158,86,62,62]** with radius **27**, compared with v1's radius 12. The best visual score is **0.849930**; the old box **[192,120,8,8]** remains an explicitly rejected association hypothesis at **0.755360**. Two viable candidates produce an assignment margin **0.123196**, including a bounded temporal bonus **0.063441** on the selected candidate. The earlier accepted history also differs, so the change cannot be attributed solely to one wider window. This addresses the retained search-exclusion example, not uniquely verified physical identity. Frame 2646 still has no frozen numerical truth label.

The overall result is mixed, not a coverage win: accepted visible labels remain **9/22**, accepted-center RMSE **1.05 px**, maximum **1.41 px**, and all-three coverage **47/180 (26.1%)**, below the unchanged 80% gate. Incoming at 2650 becomes accepted with a 1.41-pixel labeled error; incoming at 2670 becomes ambiguous, losing an equally accurate labeled acceptance. Accepted rows decrease **267→264**. Eight rows gain acceptance, seven without frozen labels; eleven lose acceptance; two unlabelled accepted positions move. Do not describe the additional rows as independently verified improvements. All five non-three-ball clips retain identical statuses and accepted positions.

V2 now records below-threshold maxima as **hypotheses only**, which distinguishes the searched-region maximum from the score at a labeled center:

| Lower-ball frame | Best searched hypothesis box | Visual score | Accepted position |
| --- | --- | ---: | --- |
| 2650 | [177,113,8,8] | 0.687261 | None |
| 2670 | [144,119,8,8] | 0.688230 | None |
| 2700 | [143,168,8,8] | 0.672276 | None |

The first two maxima are not centered on the frozen lower-ball label. Their higher searched-region scores do not demonstrate better localization of that ball. All remain below 0.70.

Permanent-loss timing is unchanged:

| Object | Last accepted | V2 rejected frames | V2 rejection evidence | First permanent loss |
| --- | ---: | --- | --- | ---: |
| Lower | 2698 | 2699–2707 | Nine visual misses, best scores 0.6285–0.6956 | 2708 |
| Incoming | 2729 | 2730–2738 | Six visual misses and three color rejections; searched maxima 0.6481–0.7125 | 2739 |
| Upper | 2744 | 2745–2753 | Nine visual misses, best scores 0.5571–0.5840 | 2754 |

The wider incoming search now sometimes finds an above-threshold but color-incompatible hypothesis; this is not a recovered ball or evidence that the boundary problem is solved. Once lost, the exact search rectangle and best hypothesis are null and no search is performed. The earlier source inspection remains the evidence for visible balls crossing the dark border; this comparison does not decode fresh images or isolate its cause. Incoming's later out-of-frame label at 2760 still cannot validate its earlier loss as a correct exit detection.

Current audit fields distinguish `proposal_status` from `final_status`; legacy v1's nested `status` remains an intermediate label. Preserve these version differences when reading old artifacts. Further appearance/reacquisition work needs separate analytic controls and unchanged-source evaluation; neither lower thresholds nor motion-only positions are justified here. Independent tracking remains the default.

## Historical v1 diagnosis

The direct probe and end-to-end run have **identical saved observations**. Their remaining failures include low visual similarity, a narrow-search contact error, and permanent loss after repeated misses. A physical ball being visible is distinct from an accepted tracker measurement.

Sources: `artifacts/joint-appearance-01/triple/appearance-tracks.json` and `artifacts/joint-association-e2e-09/triple-collision-interval/run/appearance/appearance-tracks.json`. Source video SHA-256: `f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887`. Original manual labels remain unchanged, including all 24 visibility slots and the ±5-pixel center tolerance.

## Lower-ball misses: low similarity, not excluded search

At the three frozen visible centers below, the manual center is inside the candidate search bounds reconstructed from the retained radius and last accepted box. There are no above-threshold candidate hypotheses. These are not assignment ambiguity or color rejection.

| Frame | Manual center | Candidate-center search bounds, x / y | Best searched-region visual score |
| --- | --- | --- | ---: |
| 2650 | (195,125) | 188–212 / 108–132 | 0.661464 |
| 2670 | (173,145) | 169–193 / 126–150 | 0.605713 |
| 2700 | (146,173) | 137–161 / 157–181 | 0.672277 |

The configured visual threshold is 0.70. Intermittent later recoveries occur, but nine consecutive `not_detected` frames **2699–2707** lead to permanent `lost` at **2708**. The original pixels show the ball still visible at 2707 and 2720. Appearance variation in a tiny reflective object and its 8×8 fixed reference is a plausible limitation; this read-only inspection does not isolate glare, blur, background texture or center placement as the sole cause.

The retained best score covers the whole searched region. Its below-threshold coordinate is **not** recorded, so it must not be described as the exact score at the manual ball center. Empty candidate lists mean nothing exceeded the threshold, not that no physical object existed.

## Late losses: appearance changes at the surface boundary

| Object | Last accepted | Rejected sequence | Permanent loss | Original-source inspection |
| --- | ---: | --- | ---: | --- |
| Incoming | 2729 | 2730–2738, scores 0.5534–0.5788 | 2739 | Visible sphere crosses the light grid’s left edge onto a dark border; still visible at 2740. |
| Upper | 2744 | 2745–2753, scores 0.5314–0.5840 | 2754 | Visible sphere crosses the light grid’s top edge onto a dark border; still visible at 2760. |

At onset, the visible balls lie within the reconstructed search area. No above-threshold proposal or association ambiguity is reported. The border/background change is directly visible and consistent with the score drop; it is not a separately controlled causal experiment. The search subsequently remains anchored to the last accepted location. A ball approaching or leaving that stale window can compound failure, but does not explain the initial drop. Once `lost`, `search_pixels` is zero: the tracker stops looking and cannot demonstrate disappearance or recovery.

Frozen labels mark the incoming ball out of frame at 2760. Its earlier tracker loss must not be counted as an accurate exit detection. The upper and lower remain manually visible at 2760 despite both being `lost`.

## Additional contact caveat: suspected incoming wrong branch at 2646

The sparse frozen checks do not cover every accepted frame. Source frame **2646** shows an important limitation:

- Incoming’s last accepted box is `[199,116,8,8]`; its search radius is 12 pixels. Candidate centers are restricted to **x=191–215, y=108–132**.
- The left-moving branch visible in the source, consistent with the frozen incoming identity convention, lies outside that center range. No candidate centered on that branch can be considered in this frame.
- Incoming is instead accepted at box **`[192,120,8,8]`**, center **(196,124)**, on the lower/right visible branch. The source crop, search rectangle and accepted box are preserved separately.
- The accepted visual score is **0.755361**; temporal bonus is **0**. The recorded association margin is also **0.755361**, with only **one** viable candidate. That apparently strong margin compares the available assignments, not the excluded correct-branch hypothesis. It cannot establish identity.
- At frame 2647, a broader window includes two candidate branches and the association becomes ambiguous. At 2651, the incoming track is accepted on the left-moving branch again.

This supports a **suspected transient wrong-branch acceptance**, with a concrete search-exclusion mechanism, not a uniquely known swap of physical spheres. The steel balls have no distinctive markers and post-contact manual identities are inferred continuity. No new truth label or numerical error was assigned to frame 2646. Improved error on sparse labeled frames does **not** justify a general no-identity-switch claim.

## Historical next action (implemented analytically in v2 above)

Before changing thresholds or template behavior, preserve the below-threshold best location and exact search rectangle in future bounded candidate audits. Independently test repeated/near-identical frames followed by motion and light-to-dark boundary crossings: the current local window can shrink around a stale accepted box, exclude an identity alternative, then make its surviving assignment look decisive. Such probes should separate search coverage, immutable-reference appearance failure, association ambiguity and bounded re-acquisition. Keep ambiguous identities explicit rather than treating motion predictions or rejected candidates as observations. This diagnosis makes no tracker changes and does not propose accepting lower scores on this video.

## Artifacts and limits

`artifacts/joint-loss-diagnosis-01` pins both trace hashes, the video/manual hashes, and the inspected search-rule hash. It preserves all 24 manual slots, accepted measurements separately from candidate hypotheses, reconstructed search bounds, 25 original frames with decoded RGB hashes, unchanged-pixel crops, and clearly marked viewing enlargements/overlays. Search reconstruction uses retained geometry only, not a new visual match. Some nested proposal-audit `status` values remain `not_detected` even when the top-level observation is `tracked`; interpretation uses the top-level result plus candidate disposition, not that nested intermediate label.

Only source frames 2620–2760 were decoded for inspection. No tracker rerun, label change, threshold tuning, package/test edit or saved-run modification occurred. This remains development diagnosis, not an independent physical-identity or simulation-accuracy validation.
