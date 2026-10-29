# Appearance histories across different videos

**Verified October 29, 2026, 4:14 PM PDT.** The same bounded method links features between neighboring frames while checking the unchanged opening reference. It improves local image continuity, but does not establish continuous physical identity, recover a new 3D world or qualify real-video accuracy. Existing labs and defaults are unchanged.

[Inspect all five videos](../artifacts/feature-track-diversity-01/index.html). The left panel is the preceding frame; cyan lines show adjacent appearance pairs and magenta marks show conflicts. Only the first 24 raw current IDs are drawn; every point, rejected match, birth, break and conflict remains in the journal. Exact source clocks are retained separately from average-rate inspection playback.

## Method

`feature_tracks.py` consumes existing `surface_features.py` outputs. Adjacent links use the same two-way descriptor checks as the [fixed-reference baseline](surface-feature-diversity.md). Each linked feature keeps a local history ID and age. New points start new histories; missing observations, skipped frames or source-clock gaps over 0.2 seconds break continuity. Lost IDs are never silently reused or merged.

The opening reference never changes. A direct opening match and a propagated anchor may disagree; competing claims to one anchor also count as conflicts. All affected histories split and lose their anchor rather than silently choosing an identity. Source object/mask uncertainty remains attached to every observation.

State is bounded to original/previous feature arrays and at most 128 active histories; callers journal the output. Limits are 500 frames and 64,000 allocated history IDs per instance. The study uses two workers, 60 seconds and 96 MiB per real case. It reuses all saved descriptors—no new source-model inference, API call, physics rollout or training.

## Real development results

Counts exclude initialization. “Adjacent” and “opening” compare different reference frames; more adjacent matches is not a measured accuracy improvement or one continuous opening-to-end track.

| Source/object | Later frames | Opening matches on frames | Adjacent links on frames | Any opening anchor retained on frames |
| --- | ---: | ---: | ---: | ---: |
| Ramp side / green box | 36 | 27 | 36 | 31 |
| Ramp front / wood | 35 | 9 | 25 | 11 |
| Ramp front / green | 35 | 35 | 35 | 35 |
| Toy van | 353 | 202 | 352 | 221 |
| Three balls / each | 140 | 0 | 0 | 0 |
| Squat / person | 67 | 67 | 67 | 67 |

All **684 source frames / 1,090 object-frame slots** remain. The new journal retains **35,596 raw current-point slots**, including 5,940 duplicate orientations, 17,283 adjacent links, 12,138 new appearances, 221 initialization points and **14 conflicts** (four side-box, seven van, three squat). Current-point counts are not the baseline's 32,726 fixed-reference slots; both journals remain intact.

Important limits:

- Side-box frames 28–32 still have no opening anchor. The count of later frames with at least four anchors remains 25/36; its longest uninterrupted history is 27 frames. Complete 3D pose is not recovered by filling a table with local links.
- The tiny balls have no adjacent links. One ball produces eleven isolated feature births, not persistent tracking or resolved identity.
- The van's crossfade still has appearance pairs. Matching texture does not detect every source edit or establish physical continuity across it.
- Squat texture may follow clothing or skin, not joint centers. No new anatomy, floor contact or muscle evidence is inferred.
- These are five views of four already exposed scenes; the ramp views share an event. No fresh holdout was consumed or generalization established.

Processing took **6.691 seconds** after input verification. Median/p95 history-update time: side ramp 0.587/4.247 ms; front ramp 0.212/0.467 ms; van 0.632/0.913 ms; balls 0.041/0.055 ms; squat 1.478/6.128 ms. These timings exclude source decoding, feature extraction and export and are not UI latency measurements.

## Known-motion control and independent verification

A frozen 75-frame synthetic texture rotates through 185 degrees. Masks are deliberately unavailable on frames 31–32; histories end and restart without bridging the gap. Known transforms grade every eligible point:

| Check | Points | p95 error | Maximum error |
| --- | ---: | ---: | ---: |
| Adjacent point motion | 4,807 | 0.147 px | 0.506 px |
| Opening-anchor location | 4,366 | 0.743 px | 1.338 px |

Both pass the frozen 1-pixel p95 / 3-pixel maximum checks. This is a planar software control, not real physical accuracy. An explicit identical-looking-replacement test retains the limitation that appearance alone cannot verify identity.

Twelve tests pass in `.local/feature-track-tests-01.xml`. [Independent verification](../.local/feature-track-diversity-verification-01.json) checks all link/birth/break/conflict decisions without rerunning the tracker, independently recomputes descriptor distances and all 9,173 known-motion errors, and decodes every one of the 684 exported frames. Maximum independent warp-error difference is 1.59e−13 px. All 71 new sealed files, 91 baseline files and 125 runtime hashes match at verification. Three preview images were visually inspected, including the crossfade and newly linked side-box frame.

Study seal: `6408e3b7c87aa1fed8424be1a9e2d193b1e80371d2058190d14f28aad37489e6`. [Frozen inputs/settings](../artifacts/feature-track-diversity-01/pre-run.json), [full report](../artifacts/feature-track-diversity-01/report.json). The shared ledger remains eight settled calls / **$0.186152 of $5**, no unsettled reservation.

Reproduce only when inputs or the hypothesis change, using new output paths:

```sh
.venv-lab/bin/python scripts/studies/check_feature_track_diversity.py --out artifacts/my-feature-histories
.venv-lab/bin/python scripts/studies/verify_feature_track_diversity.py --out .local/my-feature-history-verification.json
```

The verifier audits the fixed `feature-track-diversity-01` study, not a newly named run. Source evidence and study exports are local-only unless explicitly transferred. Next reconstruction work must connect trustworthy correspondences to camera/shape/pose evidence while preserving missing anchors and the [two-view geometry regression](conditional-rigid-camera.md); it must not promote link counts into recovered physics.
