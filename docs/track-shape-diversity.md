# Can the tracked points determine a 3D shape?

**The estimator works on known 3D controls, but none of the 140 tested real-video windows supports a shape proposal.** This is a reconstruction limit, not a new qualified world. Existing scenes, materials, bodies, lab defaults and independent scorers remain unchanged.

[Inspect every window](../artifacts/track-shape-diversity-01/index.html). The report links the original tracked videos and retains all source/object slots, partial histories and failed fits. The cyan/orange diagrams compare observations with fitted projections; they are not independent accuracy measurements.

## Implemented reconstruction step

`track_shape.py` attempts a relative visible-point shape from persistent image tracks. It uses low-rank motion/shape factorization and a scaled-orthographic camera constraint, following [Tomasi and Kanade](https://www.ri.cmu.edu/pub_files/pub2/tomasi_c_1991_3/tomasi_c_1991_3.pdf) and [Poelman and Kanade, Appendix I](https://www.ri.cmu.edu/pub_files/pub1/poelman_conrad_1993_1/poelman_conrad_1993_1.pdf). The camera model assumes parallel projection within an object, with a shared image scale for each frame. It does not model general perspective or lens distortion.

The implementation requires 8–32 consecutive frames and 6–128 tracks observed in every supplied frame. Incomplete tracks retain their original points and missing-slot counts; no point is interpolated or rejected because its residual is inconvenient. Source frame/time gaps, inadequate depth motion, ambiguous camera constraints and nonpositive shape metrics prevent a proposal. Accepted camera rows must have equal scale and perpendicular axes, and every point must pass the fixed 2-pixel p95 / 6-pixel maximum fit limits.

A proposal contains relative 3D points, per-frame camera rows, feature centroids, all projections and residuals. Its radius is normalized to one. **Metric scale, depth reflection, hidden surfaces, collision bounds, physical identity and camera-versus-object motion remain unknown.** A mirrored shape/camera pair has identical images. Passing this model does not verify rigidity or anatomy.

The method and limits were frozen before this study's fits. Eight-frame non-overlapping windows start at original frame zero; pre-initialization frames and short suffixes remain. A preliminary coverage-only screen informed this window size. These are exposed development sources, not a fresh holdout.

## Complete real-source results

All **684 frames / 1,090 object-frame slots** are accounted for. Input journals retain **35,596 raw feature slots**, including 5,940 duplicate orientations. The new windows preserve all **29,656 distinct observed point slots**. Only **3,472 point/frame observations in 37 windows** meet the complete-track prerequisites for fitting.

| Result | Windows | Meaning |
| --- | ---: | --- |
| Too few complete tracks | 95 | Insufficient persistent observations; includes the tiny-ball failure |
| Insufficient depth motion | 21 | A small image fit cannot identify a third shape dimension |
| Inconsistent camera constraints | 15 | The observations do not support the tested scaled-orthographic model |
| Inconsistent affine tracks | 1 | Even the less restrictive low-rank image model fails |
| Short source suffix | 8 | Fewer than eight frames; retained, not silently dropped |
| Conditional shape | **0** | No real geometry exported or promoted |

Side-box windows have inadequate depth motion or track coverage. The front wood/green boxes also lack coverage or depth motion. Van and exercise windows additionally conflict with the camera constraints. These results do not uniquely attribute failure to tracking, perspective, deformation or actual motion; those causes remain distinct hypotheses.

## Known-geometry controls

The separate 56-case software study uses eight fixed seeds:

- All 24 rigid controls reconstruct a relative shape, with 0, 0.1 or 0.25-pixel observation noise. The worst normalized pairwise-distance RMS error is **0.002111**; the unchanged limit is 0.03. This checks relative shape, not metric scale.
- All eight no-depth-motion cases and eight insufficient-view-direction cases reject geometry.
- All eight wrong-correspondence cases reject rather than deleting the bad point.
- All eight coherently expanding objects still pass the camera model: image scaling is indistinguishable from camera-distance scaling here. Their **physical rigidity is false by construction**, despite a good fit.

These controls are analytic data, not physical trials or evidence of exercise anatomy recovery.

## Verification, performance and reproduction

Verified **October 29, 2026, 5:28 PM PDT**:

- **25 focused tests pass**, including missing points, untouched inputs, source gaps, bounds, reflected geometry and ambiguous depth. Evidence: `.local/track-shape-tests-01.xml`.
- [Independent verification](../.local/track-shape-diversity-verification-01.json) checks 40 study files, 71 input files, five used source snapshots, every source slot/window and all 56 controls. OpenCV independently reconstructs the low-rank fits, camera-constraint spectra and proposed projections; maximum projection-accounting difference is **5.68e−13 px**. The estimator and passing tests are not rerun for this audit.
- The complete study takes **0.440 s** after input verification and writes **7,886,190 bytes** before the final report/index/manifest. Two bounded workers share prepared evidence; no video decoding, model inference, simulation, training or paid request occurs.
- Twenty maximum-size numerical calls (32 frames × 128 tracks) have median **2.459 ms**, p95 **12.111 ms**, maximum **13.184 ms**. These are function measurements during a local study, not UI latency or a responsiveness guarantee.

Manifest SHA-256: `f95561b9eb4ba78abe96685a7045aab164a0c31243b29cd5e72691324aaed659`.

```sh
.venv-lab/bin/python scripts/studies/check_track_shape_diversity.py --out artifacts/my-track-shape-study
.venv-lab/bin/python scripts/studies/verify_track_shape_diversity.py \
  --study artifacts/my-track-shape-study --out .local/my-track-shape-verification.json
```

Use new paths and rerun only after changing the hypothesis or inputs. Local artifacts are not included in Git.

## Next acceptance evidence

Longer, source-supported or cross-view correspondences and observable out-of-plane motion are needed before another shape/material proposal. Fitting the same short windows again cannot supply missing information. Preserve the [two-view depth regression](conditional-rigid-camera.md), source-edit ambiguity and textureless-ball failures. Exercise landmark and floor/contact qualification remain open. New physical observations and changed-action outcomes are still required before generalization or physical-transfer claims; no synthetic check completes that goal.
