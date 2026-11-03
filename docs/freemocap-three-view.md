# Three-view joint estimates: coverage is not accuracy

**Neither new method replaces the playable body.** Direct pixel fitting adds supported two-camera joints, but slightly worsens the third-view development comparison. Using all three cameras exposes additional conflicts.

Inspect [standing frame 550](../artifacts/freemocap-three-view-02/three-view-joints-0550.png) and [knee-lift frame 824](../artifacts/freemocap-three-view-02/three-view-joints-0824.png). Cyan circles show original observations; orange crosses show accepted projections. The camera-space skeleton deliberately leaves missing connections open. It is not a physics body, floor or muscle reconstruction.

## Method and complete-source results

`multiray_pose.py` supports two to four calibrated views. It minimizes squared pixel error with bounded iterations and motion, using every supplied observation. The existing confidence, two-degree ray-angle, forward-depth and one-percent image-error limits remain. No length prior, view selection, label swap, temporal smoothing or missing-joint interpolation is applied.

The source comparison reuses all three saved Full-model streams at their original clocks. There is no new video decoding or model inference. Cam3 was opened in the prior [frozen comparison](freemocap-fixed-lengths.md); **it is construction/development evidence now, not independent validation**.

| Method | Supported major-joint slots | All twelve major joints present | New / lost major-joint support |
| --- | ---: | ---: | ---: |
| Original two-ray estimate | 5,069 | 143 samples | — |
| Pixel fit, original two cameras only | 5,157 | 155 samples | 88 / 0 |
| Pixel fit, all available cameras | 4,952 | 81 samples | 473 / 590 |

Every method retains all **1,108 source frames and 36,564 joint slots**. Unsampled frames remain empty. All-view estimates distinguish exactly which two or three views supported them; a missing third observation is not a three-view success. A present conflicting view cannot be dropped to manufacture acceptance.

Two bounded workers finish **33,042 source fits** in 6.33/6.32 seconds. Per-fit median is .364 ms and p95 .488/.487 ms. These are local numerical-function/worker measurements, not browser latency or physical accuracy.

## Independent controls and their limits

The saved 288-case known-geometry control has three groups of 96 points:

- Independent .75-pixel noise: true 3D RMS improves **11.70 → 11.14 mm** with three views; all 96 estimates are available.
- A conflicting third-camera observation: all 96 candidates are rejected rather than silently ignoring the camera. Seventy-one fail image consistency; 25 do not converge within the fixed envelope.
- A coherent wrong landmark in every camera: all 96 estimates pass geometry but are still **40 mm wrong**. Cross-camera agreement cannot detect this shared error.

These are analytic software controls, not measured physical trials or proof of anatomical accuracy.

## Two-camera gains do not pass the comparison-view check

Only the condition fitted to Cam1/2 is projected into Cam3 for the [development comparison](../artifacts/freemocap-multiray-comparison-02/report.json). The all-three-view condition is not graded against one of its own fitting views as independent evidence. The same scorer, observations, clocks and 12.8-pixel limit are retained; the already exposed camera is not a new holdout.

| Same-slot comparison | Original RMS | Two-view pixel-fit RMS |
| --- | ---: | ---: |
| Complete source, 5,010 shared observed slots | 17.765 px | **17.866 px** |
| Active frames 400–1059, 3,481 shared slots | 11.933 px | **12.093 px** |

Across the whole source, 2,441 estimates improve and 2,569 worsen. There are 87 additional observable predictions and no lost observable predictions. Only **44 of those 87** new predictions meet the comparison limit; existing shared predictions lose a net 42 passes. The overall count within the limit therefore increases only **3,624 → 3,626**. In the active interval it decreases **2,611 → 2,608**, despite 27 passing new predictions. More available estimates do not establish better anatomy, and no accuracy/default promotion follows.

## Recovery, verification and cost

The first export exceeded its frozen 96 MB output envelope: **119,171,580 bytes** before its failure report. Both fitting workers had already finished. The [failed export](../artifacts/freemocap-three-view-01/report.json) remains sealed and unchanged.

Recovery verifies exact inputs and runtime, copies both completed journals, and writes logically identical compact JSON in a new directory. The [recovered study](../artifacts/freemocap-three-view-02/report.json) uses **80,471,132 bytes** before its report/manifest and completes packaging in .666 seconds. No fit, inference, source decode or control is repeated. These byte counts measure export formatting, not reduced data coverage or storage cleanup; both copies remain on disk.

Verification checks the two manifests, six exact JSON-object equivalences, byte-identical worker journals, all original baseline rows and all view-availability decisions. Every retained proposal is independently projected with OpenCV; accounting differences are below **1e−8 px**. All 288 control errors are recomputed against their known coordinates. The original 120 runtime files are unchanged, with one new module added.

A later report export used the wrong scorer-result key after those checks completed. Its [failed verification bundle](../artifacts/freemocap-multiray-verification-01/report.json) preserves the assertions and predictions. The corrected exporter completes only the unsaved grading; no geometry or passing test is repeated. The exact maximum accounting-difference scalar was not saved before that failure; only its verified bound is reported.

Sixteen software tests pass in `.local/multiray-pose-tests-02.xml`. The first run retains one analytic-fixture failure: cameras pointed outward, placing a generated observation outside the image. Correcting the synthetic camera directions leaves production limits unchanged. No whole-project regression rerun is claimed.

[Final verification](../.local/freemocap-multiray-final-verification-01.json) independently recomputes shared-score and new-prediction accounting, checks the six-file comparison manifest, confirms all 121 frozen runtime files, parses six affected Python files and checks 236 local documentation links. It reuses completed geometric checks and passing tests rather than rerunning them.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Original export-size failure | 21 | `689b39ae317fa34cb18bbf45ce9834d6108c25d93a0bbd023e5e517752a81a72` |
| Compact recovered study | 17 | `28e7547c053abd3ef8c0c0f2db8abe87aba42731e2c0012485b574c4d5db6136` |
| Verification/export failure | 4 | `ecc89eb206c1fc6b4d08548bd6c864d8c2d7fde5d79a15b301e7a287daedfa6e` |
| Completed development comparison | 6 | `418627db8e9b09da265e7fe16be539736b66e109f712ddbc898fd66047d48e73` |

No paid calls, downloads, uploads, servers or deployments occur in this study. Meta remains eight settled calls, **$0.186152 of $5**, with no unsettled reservation. Existing exercise/ball/object sandboxes and their recordings are unchanged.

## Next acceptance evidence

The next improvement needs stronger source-landmark or scene evidence, not another claim based only on smaller fitted image residuals. Preserve all missing and conflicting joints, separate inferred anatomy from observations, and establish supported floor orientation/height before body/contact qualification. Accurate changed-action behavior and broader ball/general-video coverage remain required. Use the [existing squat walkthrough](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/heavy-pose-model.md) for interaction and recording meanwhile.
