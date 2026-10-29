# Three-ball tracking diagnosis

**The retained run temporarily assigns the upper ball’s identity to the lower visible ball. The incoming and lower tracks stop after repeated rejected matches, despite visible balls remaining.** This is a diagnosis of the saved run, not a new tracking experiment or revised ground truth.

Input trace: `artifacts/source-interval-final-08/triple-collision-interval/run/appearance/appearance-tracks.json`. Frozen labels: `data/balls/triple-collision-240p.manual.json`. Source SHA-256: `f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887`.

## Accepted wrong-ball location

At source frame **2650**, the accepted `upper-passive` box is `[191,121,8,8]`, centered at **(195,125)**. The frozen upper center is **(190,102)**: **23.537 pixels** away, exceeding the manual ±5-pixel tolerance. The accepted center exactly equals the frozen **lower** center. Unmodified source pixels show three separated balls, not an occluded upper ball. The wrong accepted box occurs at frames **2648–2650**; by frame 2656 the accepted upper track is back on the upper branch, and at the next frozen label, frame 2670, its error is 1 pixel.

The erroneous acceptance passes the recorded settings:

| Measurement at frame 2650 | Value | Required |
| --- | ---: | ---: |
| Template similarity | 0.824068 | ≥0.70 |
| Best-versus-alternative peak margin | 0.066679 | ≥0.06 |
| Regional color histogram intersection | 0.990069 | ≥0.50 |

These scores measure local appearance, not identity probability. Similar small steel balls share texture and color; passing those checks does not establish which ball was found. The 8×8 reference patches are preserved for inspection. Their limited detail and the collision explain a plausible identity-association weakness, not a measured causal isolation of every failure.

The source has no unique visible markers. Post-contact identities in the frozen labels are explicitly inferred continuity. Thus this is a visually supported wrong-branch assignment **relative to the frozen identity convention**, not independent proof of the physical identity of indistinguishable spheres. The 23.54-pixel discrepancy cannot be explained by the stated center tolerance alone. Labels were not changed to make the run pass.

## Why the other tracks stop

| Object | Retained sequence | Meaning |
| --- | --- | --- |
| Incoming | Last accepted frame 2635; ambiguous matches 2636–2644; `lost` from 2645 | Nine consecutive rejected matches exceed the configured eight-frame gap. Rejections start before the balls visibly separate. |
| Lower | Last accepted frame 2643; `ambiguous_overlap` 2644–2647; `not_detected` 2648–2650; `ambiguous` 2651–2652; `lost` from 2653 | Another nine-frame rejection sequence. At 2650 its similarity is 0.687259, below 0.70; at 2651–2652 its peak margin is below 0.06. |

All three balls remain visually present in the inspected collision-region frames through 2675. The hand is outside these separated ball locations. Here, `lost` is a tracker state, not an observed disappearance or occlusion. No identity switch can be assigned to a rejected track without an accepted location.

The saved trace includes scores and statuses but **discards rejected candidate boxes and alternative peak coordinates**. It therefore cannot establish exactly which competitor caused each ambiguity or overlap. No tracker rerun, threshold tuning or inferred replacement boxes were used to fill that gap.

The saved camera check reports `not_verified_fixed`, but its samples detect no large motion: the frame-2650 estimated center shift is about **0.13 pixels**. That does not explain the 23.54-pixel wrong-ball location. Earlier acquisition cautions about camera drift remain unchanged; this narrower diagnosis does not treat them as measured cause of the collision failure or claim calibrated camera stability.

## Preserved evidence

`artifacts/three-ball-tracking-diagnosis-01` contains:

- All **24 frozen visibility slots**, including the initial occluded incoming ball and final out-of-frame incoming ball. There are 22 manual centers; none were dropped. Nine slots have accepted centers, including three authored initialization centers and the one wrong-ball match.
- Original decoded PNGs and RGB pixel hashes for frame 2620 and every frame 2640–2675; unchanged-pixel collision crops; original 8×8 reference patches.
- Clearly labeled nearest-neighbor enlargements for viewing only, plus a separate annotated frame-2650 comparison. These are not enhanced input or new evidence of detail.
- Retained trace excerpt, original settings, status transitions and per-slot errors; hashes pin the complete original trace, source and frozen manual file.

Only 56 local source frames were decoded for inspection. No package/test files, thresholds, labels or saved run artifacts were changed. No tracker, simulator or policy was run. The diagnostic manifest seals the new evidence separately.
