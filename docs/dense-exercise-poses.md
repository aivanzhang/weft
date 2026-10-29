# Every-frame exercise evidence

**Both jogging cameras now have an estimate or explicit failure for all 240 frames. This did not fix limb identity, and the denser 3D sandbox was rejected.** The existing [interactive body sandbox and recording workflow](two-view-segments.md) remain unchanged.

Follow-up: [bounded crop re-inference](temporal-crop-recovery.md) recovers four inconsistent poses and builds a separate 224-pose sandbox with exact recording. This failed run remains unchanged. The new body still floats and retains substantial joint errors.

## What was added

`weft.pose_densify` processes only previously unsampled frames with the same pinned MediaPipe Full model, runtime and IMAGE mode. Each original observation, including its confidence, status and timing, is retained exactly. No interpolated pose, repaired label, provider body reference or third camera enters inference.

Two bounded local workers added **240 observations and reused 240**, taking **15.155 seconds** total with no paid requests. Each view now has 240 sampled frames:

| View | New inferences | Unchanged cached observations | Usable pose references | Existing whole-body quality check |
| --- | ---: | ---: | ---: | --- |
| Drone1 | 120 | 120 | 221 | Passes |
| Drone2 | 120 | 120 | 150 | Fails: 62.5%, below 65% minimum |

Completion means every requested frame was processed, not that every pose is correct. Raw per-joint evidence remains available even when a view fails the whole-body check. Source clocks, model identity, failed estimates and the complete source interval are preserved. New observations are journaled as they finish; failed runs are retained.

The command accepts only sealed raw IMAGE-mode observations from the same model/runtime and exact video. It preserves the union of cached frames and the requested sampling grid, with at most 900 samples, 1,800 decoded frames, 60 seconds of source video and a 90-second local loop limit. No-work requests fail before inference. The two-camera study adds a 95-second process timeout per worker.

## Why the correspondence repair abstained

The new `limb_correspondence.py` is an opt-in experiment, not part of body generation. It compares four left/right assignments using both image rays and child-to-parent motion. A short one-to-three-sample label island requires two original supporting poses on each side, confidence at least 0.8, distinct alternatives and unchanged ray-validity checks. It never uses reconstructed comparison joints, third-camera observations, floor fitting or single-camera depth. Parent-side identity remains an assumption.

Analytic cases pass, including continuous leg crossing, genuine short swaps, missing evidence, conflicting hypotheses and unchanged global side ambiguity. On this real clip, **both sparse and dense studies make zero corrections**. Every candidate window lacks sufficient context:

| Dense evidence | Eligible frames | Longest uninterrupted eligible run |
| --- | ---: | ---: |
| Elbow pairs | 0 | 0 |
| Wrist pairs | 0 | 0 |
| Knee pairs | 45 | 4 |
| Ankle pairs | 31 | 3 |

The minimum window needs five samples. Adding frames alone therefore cannot make this method apply. Thresholds were not relaxed after inspecting the results. The earlier [left/right correspondence regressions](two-view-correspondence.md) remain unresolved.

The denser raw two-view reconstruction adds 964 of the twelve compared joint slots, increasing available slots from 1,000 to 1,964. **All 120 original sampled-frame 3D arrays are unchanged.** The dense conditional reference RMSE is 0.07785 m versus 0.08324 m on the earlier, smaller slot set; this is not a paired accuracy improvement. Third-camera agreement on the same 935 slots is unchanged at 23.9419 px. These references are reconstructed development evidence, not measured anatomy or a new holdout.

## Why the denser sandbox was rejected

The existing body builder retained 221 primary-view poses. Both hips can be triangulated at 220 of them, but **frame 191, at 6.373 seconds, fails**: projecting its inferred hips back into the cameras misses the image observations by 25.33–37.56 pixels, above the unchanged 19.2-pixel limit.

Inspect the [first-camera frame](../artifacts/dense-jog-rejection-01/view-1-frame-0191.png) and [second-camera frame](../artifacts/dense-jog-rejection-01/view-2-frame-0191.png). In the first image, knee and ankle labels lie on the upper body; both hip confidences round to 1.00. High model confidence does not establish correct anatomy. The failure is broader than just swapping left and right.

The builder preserved its no-fallback/no-dropped-pose rule. It saved the failed construction and intermediate input, but created no new interactive lab, recording or policy. No body, floor, camera, reference or acceptance threshold was changed. [All 240 root slots](../artifacts/dense-jog-rejection-01/root-coverage.json) and [the rejection report](../artifacts/dense-jog-rejection-01/report.json) remain inspectable, including neighboring frames 190 and 194.

The next method needs to handle incomplete or incorrect pose hypotheses explicitly. It must retain missing evidence, distinguish any motion prior from image measurements, and compare individual regressions on a declared development split before using new evaluation evidence. More samples or passing software tests are not sufficient to qualify a physical body.

## Run or inspect locally

Existing results need no new inference:

- [First-camera estimates](../artifacts/dense-jog-poses-01/drone1/tracks-contact-sheet.png) and [second-camera estimates](../artifacts/dense-jog-poses-01/drone2/tracks-contact-sheet.png).
- [Inference report](../artifacts/dense-jog-poses-01/report.json), [dense correspondence results](../artifacts/limb-correspondence-dense-jog-01/report.json) and [failed body build](../artifacts/dense-jog-body-build-01/report.json).
- [Existing playable sandbox instructions](two-view-segments.md#try-it-locally), with its previously verified recording.

For a different sealed compatible pose run, use a new output directory:

```sh
.venv-lab/bin/python -m weft.pose_densify \
  --run artifacts/your-cached-pose-run \
  --video data/your-exercise.mp4 \
  --model .local/models/pose_landmarker_full.task \
  --out artifacts/your-dense-pose-run --stride 1
```

To reproduce the source-only failure diagnosis without repeating inference:

```sh
.venv-lab/bin/python scripts/studies/diagnose_dense_jog.py \
  --out artifacts/dense-jog-rejection-new
```

These commands do not start a server, browser, paid request or trainer. `scripts/studies/densify_jog_poses.py`, `check_limb_correspondence.py` and `build_full_pose_body.py` preserve the exact study inputs and construction settings; do not rerun the known failed build unchanged.

## Verification and evidence

**99 distinct checks pass**: 38 densification/correspondence checks in 1.995 s, 60 existing vision/two-view checks in 0.610 s and one raw-baseline-export check in 0.110 s. Evidence: `.local/dense-and-correspondence-tests-01.xml`, `.local/dense-existing-integration-tests-01.xml`, `.local/correspondence-baseline-export-test.xml`. The initial correspondence failures and their corrected rerun are retained separately.

A 900-pose-per-view analytic workload takes 3.622 s wall/CPU, producing 9,338,249 bytes of predictions and 131,196 bytes of audit, with no corrections. This is an offline maximum-sample-count check, not browser responsiveness, physical accuracy or additional video diversity. Evidence: `.local/correspondence-load-01.json`.

| Bundle under `artifacts/` | Files | Manifest SHA-256 |
| --- | ---: | --- |
| `dense-jog-poses-01` | 132 | `f04d2b9c1e49bb70951bb8121655656bab9367abe93ff3f06fdf584c14ff04f6` |
| `limb-correspondence-jog-01` | 118 | `34cb11c63a66ce46f5914880015163722497d9ebd88d3427aa2e1f479f09a397` |
| `limb-correspondence-dense-jog-01` | 120 | `2266ec8df728933e52d873cf287c874a708f4285253f10c5a66a493012e02776` |
| `dense-jog-body-build-01` (rejected) | 131 | `28eaf1ec64ca03bf729be9d4df5e6fdf12ccd7fe9b8b231b007515e97e41590e` |
| `dense-jog-rejection-01` | 11 | `de9b5391c22e1f1b9c5e50eb6d3ddd652918c80544ba10dda93e21fd2c124f50` |

No paid call was made for this work. As of **October 29, 2026, 3:46 AM PDT**, seven settled Meta calls commit **$0.176152 of $5**, leaving **$4.823848**, with no unsettled reservation. This is the existing application-side ledger, not a provider-enforced account cap.
