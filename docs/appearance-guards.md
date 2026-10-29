# Appearance evidence and camera checks

The pool-break source exposed a false-visible prediction: grayscale matching accepted a similar ball-shaped region while the labeled yellow object's center was not locatable. This is a development failure, not proof of a specific identity switch. The source labels remain unchanged.

## Color compatibility

New appearance tracking stores an immutable hue distribution from each source rectangle. If enough source pixels carry color, a candidate must retain at least half that color support and half the smoothed distribution overlap. Thresholds and 13 analytic checks were fixed before scoring real candidates; a separate integration test checks that rejection cannot publish a position or update the immutable identity template.

The distribution uses 36 circular hue bins weighted by saturation. Source regions with less than 15% sufficiently saturated/bright pixels have no usable color evidence. Nearby hue bins overlap to tolerate small codec/lighting changes. The full contract, source signatures and per-candidate decisions are saved in `appearance-tracks.json`. Rejected matches have status `appearance_changed` and no accepted box; they may recover during the existing bounded short-gap window.

Color compatibility is **not** object segmentation, calibrated confidence, verified visibility or semantic identification. Partial occlusion can still preserve compatible color; same-color objects cannot be distinguished by this check. Illumination/material changes can reject a true object. It never manufactures a location after a failed grayscale match.

## Retained real-source comparison

`artifacts/color-appearance-01/` preserves the pre-evaluation implementation/threshold freeze, source/input hashes, original annotations, complete tracks and separate grades. No annotations enter construction. All source-visible predictions below retain the same center errors; missing positions remain missing.

| Source | Accepted visible labels, before → after | False-visible labels, before → after | Center RMSE after |
| --- | --- | --- | --- |
| Pool break | 7/7 → 7/7 | 1/3 → 0/3 absent labels | 3.02 px |
| Air-table disks | 10/10 → 10/10 | No absent labels available | 2.02 px |
| Electrostatic scene | 8/11 → 8/11 | No false-visible labels in this small sample | 2.18 px |
| Metallic pendulum | 0/11 → 0/11 | No absent labels available | Undefined: no accepted labels |

All 21 accepted grayscale rolling positions are exactly unchanged. The grayscale clip is a transformed existing source, not an independent physical trial. Pool frame 64 is now rejected (hue overlap 0.287 < 0.5); frame 68 recovers and remains accurate against its frozen source label. Neither the pool nor short disk source satisfies current physical reconstruction prerequisites. Pendulum tracking remains unsuccessful.

Study seal: 38 files, `0f7f4d5c395ebf051335d707193977ed09d83d9fd0f1951216e5b99122866ac4`. Reproduce in a new directory with `scripts/studies/check_color_appearance.py --out artifacts/color-appearance-new`. These are exposed development clips and assistant estimates, not expert truth or a final holdout.

## CPU overhead

`artifacts/color-appearance-performance-01/` compares the saved earlier tracker with the candidate on 300 analytic 960×540 frames and eight simultaneous textured targets. Both accepted all 2,400 positions. Per-frame tracking p95 changed **2.207 → 2.412 ms**; summed tracking time **0.625 → 0.682 seconds**. This is a measured modest CPU increase, not a speedup.

The one-process sequential measurement excludes video decoding, fixture generation, preview writing, network and browser work. It does not establish sustained browser responsiveness or every supported ROI size. Study seal: five files, `0f412fff1be707bb335b18bfa1e667e790a179aa93f1ecd04f1c520252f8968a`; script `scripts/studies/profile_appearance.py`. No paid/API calls or recurring resources were used.

## Camera evidence

The shared camera screen now verifies forward/backward feature tracking and spatial spread. When that fails, bounded [SIFT feature matching](https://docs.opencv.org/4.13.0/da/df5/tutorial_py_sift_intro.html) and [robust projective alignment](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html) compare directly with the original reference. At least 12 consistent matches, broad image coverage and low reprojection error are required. The existing movement thresholds are unchanged. Missing correspondences remain unknown, never proof of movement or a fixed camera.

`artifacts/camera-screen-followup-01/` freezes nine analytic tests before real scoring and retains 72 paired samples across five real clips. Nine previously unknown samples gain positive movement evidence. Three older movement results become conservatively unknown; other samples still reject their intervals. No old movement becomes a no-motion result, and no unknown becomes a no-motion acceptance. Overall interval decisions are unchanged.

The **pool disturbance remains unresolved**: original-reference correspondences disappear during the impact/blur. Even extra per-frame-triggered samples could not recover positive motion. The experimental trigger helper is not connected to profile loops; the existing two-samples-per-second cadence remains. Pool tracking coverage independently prevents reconstruction.

Summed camera alignment time for the 72 paired samples increased from 73.8 to 499.3 ms; the longest new call was 31.7 ms. These are local alignment-only measurements, not full pipeline or UI latency. The fallback is bounded to 640-pixel analysis images, 600 features, 120 flow points and 500 robust-fit iterations. It runs in the pipeline worker, not the UI.

## Final end-to-end verification

`artifacts/appearance-guards-e2e-06/`, [pinned manifest](../benchmarks/appearance-guards-v6.json): seven cases, **10.856 seconds**, at most two workers. Three environments pass unchanged gates: fixed textured-camera analytic scene, analytic elastic contact and real-source grayscale rolling. All three have 12 recorded transitions and 13 fully decoded video frames, with unchanged run manifests. The analytic moving-camera negative is rejected with its required perception error.

Three desired real cases remain failures: air-table disks have fewer than ten jointly tracked frames; pool tracking covers less than 80% of the interval; the newly acquired rail experiment has detected camera movement. They are not relabeled successful rejections. The rail video's frozen labels yield 8/19 accepted visible centers, 5.70-px RMSE on those eight and no acceptance on its three absent labels. It is not a working reconstructed environment.

The full suite seal covers 351 files: `12178b073f72543a39a596c0d708e6ac60d4cf1bff0f4925ec2300af698d8046`. The final source-only grades reproduce the pool/disk results in `artifacts/appearance-guards-source-grades-01/`; five-file seal `9eff12ecb3c1371bfc27c83719b1f91820341bd41fd796ac896dbf89b4f13dca`.

**370 Python tests passed** with no failures/errors/skips (`.local/appearance-guards-full-tests.xml`, 16.419 seconds); all 15 deterministic UI checks passed (`.local/appearance-guards-ui.log`). Browser/device QA remains unperformed. No paid/model requests or deployments occurred; the $5 Meta ledger remains configured with $0 committed and no project key at this observation.

```sh
# Open a verified appearance-based simulation; interact and record without retraining.
.venv-lab/bin/python -m weft lab --resume artifacts/appearance-guards-e2e-06/rolling-grayscale
```

Two additional [complete official collision-video transcodes](real-collision-transcodes.md) are retained with source-only review. Neither establishes clean fixed-camera footage; both exceed the current whole-video ingestion bounds. A source-pinned bounded interval decoder is needed before testing the independently selected real-time event. The repeated slow-motion replay is not independent physical timing evidence.
