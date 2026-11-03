# Two-view human-motion evidence

**The complete motion is processed, but it is not a qualified exercise body.** The two-view board diagnostic passes, while full-body joint coverage and floor evidence remain incomplete. Existing interactive exercise labs are unchanged.

Inspect the [standing source/joint comparison](../artifacts/freemocap-camera-space-inspection-02/source-and-joints-0550.png) and [knee-lift comparison](../artifacts/freemocap-camera-space-inspection-02/source-and-joints-0824.png). These show original camera images beside estimated 3D joints, not muscles, collision shapes or a learned controller.

The newer [joint and ground diagnosis](freemocap-joint-ground-diagnosis.md) tests bilateral associations, timing, room edges, carpet features and six crop proposals. None qualifies a body or floor repair; source-error panels and mixed crop results remain inspectable.

## 1. Separate camera consistency from single-view pose sensitivity

The [original calibration checks](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/calibrated-human-sample.md) remain failed. On 18 supported board poses, a board placed using Cam1 alone produces a 35.46-pixel cross-camera p95 error. Replacing the camera file or using only shared corners does not repair that result.

The new [rigid-geometry diagnostic](../artifacts/freemocap-full-geometry-01/report.json) places the known-size board onto the saved two-view points, without scaling it or changing either camera. Its 3D residual is 1.93 mm RMS; reprojected corners have p95 errors of 1.45/1.56 pixels. The separately reported, **unapplied** scale hypotheses range .99025–1.00927. The original monocular placement differs from stereo by 30.9 mm RMS.

The single-view pose Jacobian has a local cross-camera error amplification of 26–57× in combined pixel-error norm. This supports pose sensitivity as an explanation; it does not uniquely identify camera, board or corner errors.

A subsequent [omitted-corner test](../artifacts/freemocap-stereo-prediction-01/report.json) places the board from all other shared corners, then predicts the excluded corner. All 234 predictions are written before their pixels are scored. Every target is absent from its own placement fit.

| Development prediction check | Cam1 | Cam2 |
| --- | ---: | ---: |
| Predicted corners | 234 | 234 |
| Error, p95 | 1.64 px | 1.82 px |
| Maximum error | 2.11 px | 2.89 px |
| Existing numeric p95 limit | 2 px | 3 px |

All 18 supported poses and 482 unavailable prefix slots remain. No threshold, calibration, clock or scale changes. This **different stereo diagnostic passes**; it does not replace the failed single-camera test or establish independent physical calibration. The recording was already exposed construction evidence, not a fresh holdout.

`rigid_correspondence.py` supplies the bounded, scale-preserving fit. Thirteen analytic tests cover known placements, planar geometry, scale disagreement, inverse transforms, reflections and invalid/degenerate inputs. Evidence: `.local/rigid-correspondence-tests-01.xml`.

## 2. Reject unsupported floor placement

The visible red cover on the carpet at frame 550 provides a separate scene-only hypothesis. Fixed color regions produce four corners per camera; all eight cyclic/reversed correspondences are retained. No human joints are used.

The [best correspondence](../artifacts/freemocap-floor-proxy-02/report.json) still fails the preregistered checks:

- Maximum reprojection error: **4.41 px**, limit 3 px.
- Departure from a plane: **7.75 mm RMS**, limit 5 mm.
- Opposite-edge length difference: **9.92%**, limit 5%.

The cover does not set the floor. Its thickness, true flatness, corner errors and camera uncertainty remain unresolved. No floor is shifted to match inferred feet. The first export failed on a NumPy boolean; all completed contours/geometry and the exception are preserved. Recovery only summarizes saved results using native JSON values; it repeats no detection or fitting.

## 3. Process both whole videos once

[Pose extraction](../artifacts/freemocap-full-pose-01/report.json) uses the pinned local MediaPipe Full model in independent IMAGE mode, with two bounded CPU workers. Each camera retains all 1,108 decoded frames, 554 sampled frames at 15 fps and original 30 fps timestamps. The pass takes 20.477 s, including preview export, with no API request.

| Sample status | Cam1 | Cam2 |
| --- | ---: | ---: |
| Tracked | 419 | 371 |
| Partially tracked | 93 | 123 |
| Low confidence | 35 | 31 |
| Multiple/competing hypotheses | 1 | 1 |
| No pose | 6 | 28 |
| Total sampled | 554 | 554 |

The ambiguous frames are 176 and 162 respectively, during the board sequence. Cam2 also misses five crouching/board frames; the remaining no-pose frames occur near the end. Both ordinary monocular admission checks reject the complete source because of competing hypotheses. These failures remain failures; the study does not quietly crop them away or build a continuous body through absent-person frames.

## 4. Preserve complete 3D availability

[Two-view triangulation](../artifacts/freemocap-camera-space-joints-01/report.json) reuses those image observations and the existing confidence, ray-angle and reprojection rules. It retains **36,564 joint slots**: 33 joints × 1,108 original frames. There are 14,341 estimated joints, 20,649 unavailable image-evidence slots, 1,560 inconsistent-ray slots and 14 unavailable distortion inversions.

Among 554 sampled moments:

| Body support | Moments |
| --- | ---: |
| All 12 major body joints | 143 |
| Torso present, incomplete limbs | 171 |
| Both hips, incomplete torso | 17 |
| One or both hips unavailable | 223 |

Another 554 original frames are explicitly unsampled, not interpolated. The longest complete-body sample span is frames 448–538: 46 samples over three seconds. This is a coverage finding, not an automatically selected exercise episode or proof of accuracy.

All segment-length slots are saved. The inferred shin lengths vary substantially: p95 deviations from their respective medians are 73/64 mm. A geometrically plausible frame does not establish stable anatomy through the whole sequence. At the knee-lift panel, one elbow fails cross-view consistency and three hand landmarks are outside one image; the missing connections stay visible.

Coordinates remain **camera 0: x right, y down, z forward**. They are not labeled gravity-aligned or passed to a body builder that requires z-up ground coordinates. No floor, body, policy, mass, contact force or muscle activation is generated here. Triangulation and inspection export take 1.825 s; only two new source frames are decoded for the knee-lift panels. The final panel correction reuses all geometry and source PNGs with shared plot bounds, preserving the earlier clipped export.

## Evidence and next work

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Rigid geometry | 5 | `50076fd027c26291c3a59596bbb8c50f284ce4001de2645e6b90dad4fb20bf81` |
| Omitted-corner prediction | 6 | `cfdf6db78e4bf19918f5919787c580f025461f27d857f04456f72f853cc8e319` |
| Full pose extraction | 23 | `cac551185bb1a2986c57d6d925f67a113d0cb4c830fb646cfa1eb9b83dc1e2e9` |
| Original floor export failure | 13 | `70158d10e529ad03cbe9886e17b068b3ddf8f43b7eec7b3444508a802a6ba90a` |
| Recovered floor rejection | 14 | `b9c5ea4ae2b54f04fdda982c5ebc3f37eadf4805079a3eb43a2344bc2c94a654` |
| Complete camera-space joints | 14 | `08d1801f6a957d0c0ae83873b43507d0f9301f4e89876dd50af332a840b234f1` |
| Unclipped comparison panels | 6 | `3f6a3367983149203339080a0ed02133970eb0ba37ee75690a2b544f7241e2e8` |

The [follow-up diagnosis](freemocap-joint-ground-diagnosis.md#next-work) rejects simple global swaps, nonzero clock offsets and the tested room/carpet floor hypotheses. The subsequent [fixed-length comparison](freemocap-fixed-lengths.md) freezes predictions before opening Cam3, then rejects the correction. Cam3 is now exposed development evidence; the [all-view follow-up](freemocap-three-view.md) uses it explicitly for construction without claiming independent validation. Stronger source-landmark and ground evidence remain necessary. Preserve the entire video, missing joints and explicit episode boundaries.

This follows the earlier [Meta camera/ground/body hypotheses](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-body-hypotheses.md), without paying for repeated advice. No new Meta calls or uploads occurred; the ledger remains eight settled calls, $0.186152 of the $5 budget. These results do not complete the broader ball, general-video, anatomical or physical changed-action requirements. Use the existing [Heavy squat sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/heavy-pose-model.md) for interactive control and recording while those checks continue.
