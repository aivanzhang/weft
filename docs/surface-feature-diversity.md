# Automatic image features across saved videos

**Verified October 29, 2026, 3:58 PM PDT.** One fixed method processed 684 frames from five views of four scene recordings. It finds repeatable image texture in some clips but does not establish general physical correspondence or reconstruct a new environment. Tiny balls produce no features, and important rigid-object motion loses matches. Existing exercise and object sandboxes are unchanged.

[Open all five inspection videos](../artifacts/surface-feature-diversity-01/index.html). Each panel compares the declared initialization with the current source frame. Lines show at most the first 24 raw reference IDs per object; the complete journal retains every point, missing match and ambiguity. The preview uses a fixed crop and source-derived average playback rate; exact source timestamps remain in the journal.

## Method and bounds

`surface_features.py` detects local image-texture descriptors inside supplied masks, then matches each frame directly to the unchanged initialization. It does not hand-label feature points, chain transformations, refresh the reference, fit camera motion, or discard points using a geometric fit. Object selection, initialization and masks remain supplied inputs—not automatic discovery.

- Reuse unchanged saved Meta masks for the front ramp, van, balls and exercise; the side ramp uses its existing color-component masks. Original identity uncertainty remains attached to every frame/object slot.
- Inset masks by three pixels without filling holes. Detector support can still include adjacent texture outside the mask. A disappearing small mask means no supported features, not no object.
- Keep all raw detector rows, capped at 128, and mark duplicate orientations within one pixel. Require two-way nearest matches, a nearest/second-nearest distance ratio below 0.7 in both directions, and descriptor distance at most 220. These are appearance checks, not proof of physical identity.
- Bound detection to 262,144 source-crop pixels without resizing the source. Two workers, 60 seconds and 96 MiB per case; at most 500 frames per case. Archive raw points, descriptors and duplicate identities in 32-frame batches.
- No new model inference, API calls, simulation, training, default changes or model promotion.

## Complete-source results

The initialization itself is excluded from the following counts. Four pairs is a descriptive count, not a camera/pose acceptance gate: points may be wrongly associated, clustered or collinear.

| Source / object | Full source frames | Later frames | Later frames with any pair | With at least four pairs |
| --- | ---: | ---: | ---: | ---: |
| Ramp side / green box | 37 | 36 | 27 | 25 |
| Ramp front / wood block | 44 | 35 | 9 | 3 |
| Ramp front / green box | 44 | 35 | 35 | 21 |
| Toy van | 354 | 353 | 202 | 134 |
| Three balls / each object | 181 | 140 | 0 | 0 |
| Squat / person | 68 | 67 | 67 | 60 |

All 1,090 object/frame slots remain, including frames before initialization. All 32,726 raw reference-point slots remain after initialization, including duplicate, missing and rejected matches. There are 4,344 later appearance pairs, plus 221 initialization self-pairs. The two ramp views depict the same event; they are not independent scene diversity. All recordings were already exposed development evidence, not unseen-video tests.

Important counterexamples remain visible:

- The side-view box has no pairs at frame 29 while it rotates after contact. The moving wood block mostly loses its original texture. These matches alone cannot support complete motion reconstruction.
- The tiny ball masks vanish under the fixed inset and supply no usable initialization features. Changing thresholds per clip was not used to manufacture coverage.
- The edited van source contains a crossfade. Appearance matches across that edit do not establish continuous physical motion.
- Squat matches can follow clothing or texture. They do not identify joints, muscle activity, rigid anatomy or camera calibration.

The study completed in **8.378 seconds** with two local workers. Median / p95 feature-step times were 9.37 / 15.77 ms for the side ramp, 2.90 / 7.83 ms for the front ramp, 6.34 / 15.34 ms for the van, 0.23 / 0.90 ms for the balls and 21.84 / 26.84 ms for the squat. Each step includes feature preparation and matching, not video decoding/export or UI latency. Empty ball regions are cheap failures, not a throughput success.

## Verification and reproducibility

[Saved report](../artifacts/surface-feature-diversity-01/report.json), [frozen settings and inputs](../artifacts/surface-feature-diversity-01/pre-run.json), and [independent verification](../.local/surface-feature-diversity-verification-01.json).

- Fifteen feature tests pass: known translation/perspective, partial occlusion, missing-frame recovery, duplicate texture, one-to-one matching, empty regions and input/resource bounds. Verification reuses the unchanged passing result rather than rerunning extraction or tests.
- All 91 sealed files and 124 runtime-file hashes match. Seal: `a0833ca84645cccde2e62da61b74edd7bd37f1ed7aab06a20aa47ff3ae978a11`.
- Independent double-precision descriptor distances reproduce every saved match/rejection; maximum distance difference from OpenCV is 0.00003047. Every original source hash, object/frame slot, raw reference slot and per-object summary agrees.
- All five inspection videos fully decode: 684 frames. Four saved preview images were visually inspected, including missing matches and the van crossfade.
- The project ledger remains eight settled calls, **$0.186152 / $5**, with no unsettled reservation. This work added no paid request.

To reproduce, choose new output names; do not overwrite the saved evidence:

```sh
.venv-lab/bin/python scripts/studies/check_surface_feature_diversity.py --out artifacts/my-surface-feature-study
.venv-lab/bin/python scripts/studies/verify_surface_feature_diversity.py --out .local/my-surface-feature-verification.json
```

The verifier audits the fixed `surface-feature-diversity-01` baseline. It is not a generic grader for newly named studies.

## What remains

These are reusable image observations, not a generalized video-to-simulation system. Resolve missing and ambiguous correspondence and combine separately fixed camera/shape/pose evidence across time before attributing residual errors solely to physics. Textureless small objects need other observations, while articulated humans need joint/identity/contact evidence rather than rigid-point assumptions. The [two-view geometry regression](conditional-rigid-camera.md) and [human multi-view conflicts](freemocap-three-view.md) remain unresolved. Qualify complete reconstructions against fresh videos and independent changed-action outcomes before claiming physical generalization.
