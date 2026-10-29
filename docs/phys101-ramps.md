# Real ramp videos and full-clip tracking

Two new real-object clips extend development beyond the earlier toy-van and ball sources. They show the same wooden block on provider-declared 10° and 20° ramps: it remains still in the first clip and slides into a green box in the second. **Tracking is now inspectable; neither clip has a validated simulated environment yet.**

## Inspect the results

- [Every frame and decoded silhouette](../artifacts/meta-phys101-ramp-masks-01/index.html): use the frame slider and layer toggles. Opens locally without a server or another API call.
- [Source/model comparison panels](../artifacts/phys101-tracking-comparison-01/index.html), [complete comparison report](../artifacts/phys101-tracking-comparison-01/report.json) and [every frozen comparison label](../artifacts/phys101-tracking-comparison-01/manual-comparison.json).
- Original source sheets: [10° stationary case](../artifacts/phys101-ramp-inspection-01/ramp-10-sheet.png) and [20° sliding/contact case](../artifacts/phys101-ramp-inspection-01/ramp-20-sheet.png).

The silhouette viewer shows model output, not simulation controls. The existing [exercise sandbox](two-view-segments.md) remains separately playable and recordable.

The new [tilted-support sandbox](oriented-supports.md) adds executable 10°/20° hypotheses with terminal controls and exact recordings. Its geometry, friction, scale and mass ratio are authored; it does not promote the tracking results to a validated reconstruction.

## Source and acquisition

Source: [Physics 101, MIT CSAIL](https://phys101.csail.mit.edu/), release v1.0. Attribution: Jiajun Wu, Joseph J. Lim, Hongyi Zhang, Joshua B. Tenenbaum and William T. Freeman, *Physics 101: Learning Physical Object Properties from Unlabeled Videos*, BMVC 2016.

Before viewing the clips, the plan selected `w_block_01`, first trial, front camera, settings `10_01` and `20_01`. The provider's README identifies the settings as ramp angles and a green cardboard target. Its tables report block mass **21.2 g** and volume **33.0 mL**. These are source-reported properties, not measurements independently verified here; they do not provide camera calibration, block dimensions, friction or target-box mass.

| Complete original clip | Frames | Last frame time | Bytes | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 10° | 27 | 0.867533 s | 210,721 | `0d45b96b59bc990f7c4c3b7150beeb6f74230089145f4bc42ed07047f4ff6282` |
| 20° | 44 | 1.434767 s | 326,575 | `a93fa02bf9c53c019ca8870a9532234cce6196c62efb5210a4105afffa44d710` |

Both are 1920×1080 at 30000/1001 fps. Every frame decodes with strictly increasing timestamps, matching nominal frame time within 2.23e-16 s. Files were not trimmed, resized, retimed or transcoded. Initial hand release remains in the sources; tracking starts at frame 8 with the earlier frames explicitly unobserved.

The archive is 4,808,836,283 bytes. A bounded HTTP range reader retrieved its index/README in **2,441,471 bytes and seven range requests**, then reused that index to retrieve both clips and mass/volume tables in **529,924 bytes and 12 requests**. No full archive was downloaded. Exact ranges, ETag, ZIP CRCs, content hashes and request bounds are saved. Thirteen reader checks pass, including rejection before reading an ignored full-file response, changed ETags, malformed ranges, truncated data, transfer limits and cached-index reuse. Metadata and sample passes each enforce an 8-MiB transfer cap, 20 requests, no retries and 15-second per-request timeout.

The provider publicly offers the research dataset and requests citation. No explicit redistribution license was found in the saved page or README. Keep source files local/private; do not publish them or describe them as openly licensed. The authorized Meta request used private API processing, not public publication. No account, agreement or wallet was created. [Machine-readable provenance](../data/general/phys101-ramps.source.json).

## Unchanged local tracker versus Meta

Before tracking, source-only frame-8 boxes and six approximate comparison centers per clip were frozen. Labels have **±12 source-pixel uncertainty** and are assistant visual development annotations, not laboratory truth. No later label, old prediction or repair prompt enters the Meta request. IDs 1/2 map to wooden block/green box by initialization, not by whichever returned ID minimizes error.

The unchanged local tracker processes both full clips with two workers in **1.064 s** total:

| Case | Wooden block after initialization | Green box after initialization | Frozen labels tracked |
| --- | ---: | ---: | ---: |
| Local, 10° | 18/18 | 18/18 | 6/6 |
| Local, 20° | 0/35 | 21/35 | 2/6 |
| Meta SAM 3.1, 20° | 35/35 | 35/35 | 6/6 |

The local moving-block trace retains nine ambiguous and 26 lost slots; the box has nine not-detected and five lost slots. No threshold or starting region was changed to obtain a pass. The camera screen reports `not_verified_fixed`; sparse small-motion samples are not calibration.

One Meta request completed in **5.044 s**, returned 50 billed frames and cost **$0.01**. The prepared and executed request hashes match. It uses the existing Standard-mode adapter, inline video, `store:false`, a 120-second network deadline, no persistent Files upload and no retry. The conservative estimate was $0.02 with a $0.15 temporary reservation, reconciled from returned usage. [Current SAM pricing](https://dev.meta.ai/docs/pricing-rate-limits).

SAM's six available label errors have RMSE **6.10 px**, maximum **10.61 px**, all within the declared 12-pixel uncertainty. Local RMSE **1.00 px** covers only two easy stationary-box labels, so the unmatched aggregate errors must not be compared as equally complete estimates. On those two common slots, local/SAM RMSE is 1.00/0.87 px; both are far below annotation precision. This supports better coverage on this clip, not subpixel accuracy or universal superiority.

All **88 object/frame slots** remain represented: 16 pre-initialization, two initialization and 70 later model observations. The offline decoder exports 72 masks, including initialization, in **0.659 s**, with 2,634,951 image bytes. Mask overlap was not independently annotated or graded. A proximity warning starts at frame 28 and remains unresolved after separation; model IDs are not proof of physical identity through contact. The final comparison panel was visually inspected. Browser/device interaction was not performed.

A final pixel check verifies all 368,454 decoded mask pixels against the 72 saved binary PNGs, including dimensions, hashes and foreground counts. Viewer alpha is intentionally translucent; display opacity is not part of mask geometry. Evidence: `.local/ramp-and-correspondence-final-verification.json`.

## Evidence and cost

| Bundle | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Archive metadata | 12 | `823ae6a6453f5627498872b96abc525905fd12ddb9fce89001591897d8ba1668` |
| Two clips and properties | 20 | `85c55fcfc8cd01d0765d909b265f1a6f64215332347f2f9e2a128d070fbf8dda` |
| Complete decode/inspection | 9 | `1e4e115bc42333310e40074429b17f4435511580252c61804cd39a8819a58e13` |
| Local tracking and frozen labels | 134 | `8e70c2a001451ea95a113d95adef7b74babaaa45898b120cede7efbdb123678b` |
| One paid result | 9 | `2dbf65d8e70a29717315832b73810de46f30e5f1cd64d79fc6b7bfb2d61806eb` |
| Shared evidence | 3 | `24df2833e9813ff9a4c9d17afe567855e65f7f2f6c4f8958adffb20100babae3` |
| Decoded-mask viewer | 234 | `361eb6a2e0288ec9f48f798ce57673da607395f8a074d879825c5b60b0c78d4d` |
| Paired tracking comparison | 8 | `ae4382adea4aa306e9da4bd80111afdc4bdb3fc2e8b2fc0573cbf24fdd1c1699` |

At **October 29, 2026, 2:11 AM PDT**, seven settled Meta calls commit **$0.176152 of $5**, leaving **$4.823848**, with no unsettled reservation. No recurring work, server, training or deployment was started. Source retrieval has no API fee. The initial sandbox DNS failure is retained in `phys101-source-access-01`; metadata access succeeded in `...-02`. The acquisition plan JSON preceded the download; its first seal lacked a required report, so the manifest was added after starting acquisition without changing the plan.

## Next environment gate

An explicitly versioned oriented-support/contact representation now passes analytic static, sliding and collision checks; see [the implementation and authored sandbox](oriented-supports.md). This closes the representation gap, not physical reconstruction. Camera/scale, dimensions, friction and mass-ratio uncertainty still need source-grounded constraints before a meaningful trajectory comparison. Keep observed versus assumed properties and generation/evaluation separate; preserve complete intervals, hand release, contact and failures. Independent changed-action validation, broader videos and exercise contact/anatomy remain open.

Reuse the saved Meta result and masks; another paid tracking request is not needed for this study.
