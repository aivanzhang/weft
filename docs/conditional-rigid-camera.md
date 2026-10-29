# Second-view camera and box-depth checks

**The new depth candidate is playable, but it does not qualify as an accuracy improvement.** It slightly improves front-view center error while worsening both side-view alternatives. Existing sandboxes and the failed prospective repeat stay unchanged.

[Open the side-by-side comparison](../artifacts/ramp-conditional-depth-01/index.html), [rotate its 3D replay](../artifacts/ramp-conditional-depth-01/candidate-coast/replay.html), or [inspect the recorded controls/reset](../artifacts/ramp-conditional-depth-01/candidate-controls/replay.html).

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/ramp-conditional-depth-01/candidate \
  --record artifacts/my-conditional-depth-session
```

Use a new recording directory. Try `right 4`, `forward 4`, `brake 4`, `reset`, then `quit`. These are virtual forces, not independently validated physical interventions. No policy is trained in this comparison.

## Reusable implementation, explicit source assumptions

`affine_camera.py` fits a linear 3D-to-image projection from **6–1,024 construction correspondences**. Uniform normalization improves numerical conditioning without hiding thin or coplanar geometry. Every input point and residual remains; there is no outlier deletion, view selection or evaluation-data fitting. Planar, ill-conditioned and collapsed-image inputs are rejected.

The real-source input is [six approximate opening corners](../data/general/phys101-side-corners.json) of the green box, with an authored ±6-pixel-per-axis annotation envelope. Corner selection is manual, not automatic detection. The broad face is assumed to be the cuboid top, and both possible depth-axis directions remain. The box dimensions and initial pose come from the existing conditional scene. These are not independently measured 3D correspondences.

[Inspect the opening labels and original-depth projections](../artifacts/ramp-side-camera-01/index.html). Both cameras fit the six corners at **1.93 px RMS**, but their image-axis scale ratio is **1.516**. Equal-scale, orthogonal rows would be required under the additional scaled-orthographic, square-pixel camera assumption. That assumption is not established for the source.

The ambiguity controls change assumed depth to **0.5×, 1× and 2×**. Compensating camera changes preserve the same fitted image points. A small fitted residual cannot identify the real depth.

`orthographic_shape.py` derives one bounded dimension factor under that explicit camera assumption. It minimizes the unequal-scale/nonorthogonal part of the image projection while changing only one supplied shape axis. It rejects unobservable axes, nonpositive scales and changes outside 0.25×–4×; it does not clip them into acceptance. Other dimensions and the object center remain fixed.

For this source, both depth directions yield the same factor, **0.65937**: green depth changes **0.08000 → 0.05275 scene widths**. The camera-axis ratio becomes **1.0118** and its row cosine is −0.01173. Opening pixel predictions remain identical. This is a conditional shape proposal, **not metric calibration or a measured dimension**.

## Complete-motion comparison

The candidate is frozen before simulation/comparison. Mass, friction, gravity, contact solver, initial state, other dimensions, front camera, controls and task remain fixed. Changing depth at fixed mass changes cuboid inertia. No material search or repeat-trial fitting occurs.

| Same-source check | Original depth | Conditional depth |
| --- | ---: | ---: |
| Front center RMS, 72 comparisons | 6.006 px | 5.968 px |
| Side centroid RMS, positive depth direction | 12.473 px | 14.334 px |
| Side centroid RMS, negative depth direction | 11.751 px | 13.377 px |
| Front wood silhouette overlap | .7810 | .7716 |
| Front green silhouette overlap | .8778 | .8934 |

All **88 front object/frame slots** remain, including sixteen pre-initialization slots. Of 72 paired centers, 34 improve, thirteen worsen and 25 are unchanged. Every front mask slot is retained separately from center error.

Each side alternative retains **37 source frames: thirty compared and seven before simulation initialization**. The 25-fps side source is paired to the nearest saved state, with no interpolation or fitted clock shift; maximum mismatch is 8.033 ms. Provider synchronization remains assumed. Side shape and centroid checks both compare to saved color masks, not measured center of mass or 3D rotation.

Neither original nor conditional side alternative passes the frozen nominal image check: p95 ≤12 px, maximum ≤24 px and minimum mask overlap ≥.65. Corner-annotation propagation alone reaches **23.03 px per axis** on original projected vertices and **21.24 px** on the candidate. These bounds exclude geometry, clock, correspondence and physics uncertainty. The mismatch therefore does not identify a unique physical cause.

This first trial is exposed development data, not a fresh generalization test. The separately failed [prospective repeat](material-transfer.md) is not read or retuned here. Both depth directions and all regressions remain; neither is selected by its later score.

## Verification and cost

Verified **October 29, 2026, 3:28 PM PDT** in [.local/ramp-side-geometry-verification-01.json](../.local/ramp-side-geometry-verification-01.json):

- **33 new passing tests:** nineteen camera tests and fourteen conditional-shape tests, including known geometry, coordinate changes, nonidentifiable depth and invalid-input rejection. No unchanged passing suite is repeated.
- **85 exact replay states:** 71 coast states and fourteen manual-control/reset states. These establish recording consistency, not physical accuracy.
- **159 decoded export frames** across both comparisons and both candidate recordings.
- Independent OpenCV projection of all 120 compared side states; maximum numerical discrepancy **3.41e−13 px**. Independent source-mask centroids, all 176 front center slots, four final mask overlaps, unchanged scene settings and opening-pixel equivalence are checked.
- All **123 runtime files** match their applicable study snapshots. The only runtime additions are the two new helpers; no existing simulator, scorer or exercise behavior changes.
- Forty maximum-count numerical workloads, each with 1,024 correspondences plus the shape calculation, have median **.218 ms**, p95 **.256 ms**. This is function latency, not browser or end-to-end performance.

| Saved study | Files | Manifest SHA-256 |
| --- | ---: | --- |
| `ramp-side-camera-01` | 62 | `5b7fa4c9ab81e6f309f40e02344b03c428ed237bd805390a1443c09b2b87eb27` |
| `ramp-conditional-depth-01` | 44 | `653d129366c6d85499e8cb76356a4d270e425652188f5c19e0d63fe3fda23ee5` |

The studies take .463 s and 1.644 s internally, excluding later verification. No new model inference, API request, download, deployment or server is involved. Meta remains **eight settled calls, $0.186152 of $5**, with no unsettled reservation.

## What must improve next

Constrain camera/shape/pose across time with separately fixed source evidence before another material fit. Preserve corner-identity and projection ambiguity; do not choose hidden geometry because it looks good from one view. A reliable automatic pipeline must obtain those inputs without this study's manual annotations and pass independent unseen-video and changed-action checks. Exercise anatomy, ground contact, whole-exercise control and broader ball/general-video accuracy remain separate unfinished requirements.
