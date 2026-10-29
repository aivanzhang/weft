# Bounded camera registration

`tracking-stabilize` tests whether background features can map every initialized frame into one reference image. It runs locally, makes no API calls, and preserves failed frames instead of interpolating them.

```sh
.venv-lab/bin/python -m weft tracking-stabilize \
  --run artifacts/segmentation-evidence-01/three-ball \
  --video artifacts/meta-three-ball-input-01/attempt-05/three-ball-crf18.mp4 \
  --reference-frame 40 \
  --out artifacts/my-new-registration
```

Use a new output directory. This saved real source currently fails qualification; rerunning unchanged inputs is unnecessary. Inspect the existing [latest report](../artifacts/camera-stabilization-study-02/report.json) instead.

## What it checks

The exact hash-pinned video is decoded sequentially. ORB feature matching is attempted first, followed by forward/backward optical flow when necessary. Both compare directly with the fixed reference, never with accumulated transforms. Authored initial regions and model object boxes are excluded from background matching.

Up to 120 spatially distributed matches per method are retained. Each reference-image grid cell contributes deterministic fitting and withheld matches where possible. A similarity transform—translation, rotation and uniform scale—is fitted only to the fitting subset. Withheld errors and spatial support must pass unchanged limits, including 1.5-pixel RMS, eight grid cells and four quadrants. Single-point cells cannot supply both subsets.

Execution uses one spawned worker, a 30-second deadline plus bounded cleanup, at most 500 frames, 37 MB of video, 960-pixel analysis dimensions, bounded feature counts and bounded output. Source hashes, decoder timestamps, frame coverage, settings, correspondences and rejected attempts are retained. A successful process exit means evidence was saved, not that registration qualified.

Every frame from the latest object initialization through the end must qualify. Accepted transforms are required for every transformed observation; unavailable geometry stays unavailable. No clipping or frame removal is allowed. Original model boxes, IDs, masks and identity ambiguities remain intact in the public wrapper.

If complete registration qualifies, `stabilized-evidence.json` can enter `tracking-scene` with the explicit `similarity_stabilized_orthographic_assumed` camera hypothesis. Keep the source's declared camera motion; do not relabel it fixed. Registration does not establish common depth, perspective calibration, metric scale, physical identities, forces or material properties. `physics_ready` remains false.

## Real three-ball results

All three attempts decode all 181 frames and retain the same initialized suffix, frames 40–180. None creates a stabilized scene.

| Attempt | Qualified suffix frames | Corrected manual-label slots | Result |
|---|---:|---:|---|
| V1 global split | 25/141 | 0/20 | Unavailable |
| V2 per-cell split | 23/141 | 4/20 | Unavailable |
| V3 descriptor-seeded flow | 27/141 | 4/20 | Unavailable |

V1's spatial round-robin ordering interacted incorrectly with its global modulo partition: with nine or twelve occupied cells, whole cells could land in only one subset. V2 fixes that structural problem without changing any acceptance gate. Synthetic nine/twelve-cell, uneven-count and input-permutation regressions cover it. This is a correctness fix, not a measured real-data accuracy improvement.

V2 still rejects 101 suffix ORB attempts for insufficient spatial support and 17 for withheld residuals. Optical-flow fallback rejects 104 for residuals and 14 for spatial support. For example, frame 90's ORB residual is only 0.378 pixels but its withheld matches cover six cells and three quadrants; the broader optical-flow matches have 8.915-pixel residual. Neither qualifies.

Four source-only landmarks were annotated at the anchor and five later frames before V1 fitting, with approximate ±2-pixel coordinate uncertainty. The fitter never accessed these labels. V2 reuses them as development evidence, not a fresh holdout; their image regions were not excluded from feature fitting. Only frame 61 qualifies: its four corrected errors have 0.080-pixel RMS, within the conservative uncertainty. The remaining 16 slots stay unavailable. Rounded raw labels happen to be identical across frames, giving zero raw error; that does not prove zero camera motion or demonstrate a correction benefit.

## Verification and retained evidence

- 29 registration tests, 19 adapter tests, and 22 independent-grader tests pass. The adapter includes actual synthetic pixel registration through candidate generation and replay. Existing planar tests cover the unchanged raw path.
- V2 processes the real video in 2.092 seconds inside the worker. This single local observation is not a general performance guarantee.
- V1 study: 23 files, seal `c6cc4831f509c1632afbe8568d8171d502a876eaa752ec09bd908ef367cdb325`.
- V2 study: 27 files, seal `23c55f6de3fb2ef78bb16b4dd003afe9398a870a948d2f56a4005a459f543dd6`.
- Source-only labels and V1 grades use list-form manifests with path, byte count and SHA-256; all entries were independently verified without rewriting those manifests.

Next work must distinguish insufficient background coverage from incorrect correspondences. Do not relax limits, substitute rejected diagnostic transforms, trim the source, or claim complete reconstruction from partial alignment.

## Saved-correspondence follow-up

The [read-only support diagnosis](../artifacts/camera-support-diagnosis-01/README.md) checks all 259 suffix attempts without another pixel-processing run or label access. ORB's combined retained points fail spatial coverage in 91/141 frames; 84 of those failures occur below the correspondence cap. Partitioning therefore cannot explain most late failures. Bottom-right reference-quadrant coverage drops from 40/40 early frames to 9/61 late frames. Raw detected and pre-retention coordinates were not recorded, so the cause cannot yet be assigned to detection versus matching.

Optical flow has broad combined support in 111/118 fallback attempts, but **all 118 fail residual gates** when those gates are examined separately from the first reported failure. The 14 spatial-first failures also fail residuals. Broad matches are not necessarily consistent matches.

The proposed descriptor-seeded fixed-reference optical flow is now implemented and tested as V3 below. The five-file diagnostic seal is `4f2c376c433601f4c6401bd1d4f91c209d1121f893c0de2e7bd9f601fe51c2a3`.

## V3: seeded flow, still unavailable on real sources

V3 detects and deduplicates one fixed reference ORB pool of at most 1,000 features. Its fitting/withheld identities are assigned by cell **before** matching. Only fitting descriptor correspondences determine the seed. The seed initializes fixed-reference optical flow; withheld flow points never enter the seed. No temporal composition, interpolation, score-based method choice or acceptance-limit relaxation is introduced. ORB remains first; seeded flow replaces the unseeded fallback.

Reports retain the reference pool once, feature IDs, seed-training IDs, and fixed-size stage histograms for detection, matching, deduplication and retention. The validator checks these identities and summaries. Old V2 contracts are explicitly rejected by the current validator, not silently reinterpreted; their sealed outputs retain their original implementation snapshots.

Forty-six camera tests and nineteen adapter tests pass. Synthetic controls cover known motion, repeated textures, wrong seeds, uneven features, foreground exclusions, blur, parallax, provenance and tampering. Positive fallback tests deliberately reject otherwise successful ORB to exercise the fallback; they do not establish a naturally improved real case. A 12-pair synthetic measurement has median 53.84 ms including preparation; the 24.97 MB 500-row serialization estimate is not a measured 500-frame video run.

The frozen real study uses two concurrent bounded workers, each with the existing 30-second deadline:

| Source | Decoded frames | Registered required frames | Seeded-flow acceptances | Worker time |
|---|---:|---:|---:|---:|
| Three-ball | 181 | 27/141 | 0 | 4.415 s |
| Edited/hand-held toy van | 354 | 2/354 | 0 | 11.250 s |

The three-ball ORB pool/partition change yields four more accepted frames than V2, but the seeded fallback accepts none. Four reused manual-label slots have 0.060-pixel conditional RMS; sixteen remain unavailable. Identical rounded raw labels still prevent a demonstrated correction benefit. The toy van's 352 fallback attempts cannot obtain a qualified training seed; this edited, handled source is not a clean fixed-plane calibration example and has no independent camera labels.

No stabilized candidate is generated for either source. Both complete decodes, rejected frames and unchanged source/code hashes are retained. Study: `artifacts/camera-stabilization-study-03`, 38-file seal `f063f403c50dd1ffa486768ffe1302c675d08ee08ea2fef6d69263dcca732632`. No API calls or simulator changes.

### What the new stage counts establish

The [V3 saved-data diagnosis](../artifacts/camera-support-diagnosis-02/README.md) separates detection from matching. In all 61 late three-ball frames, the current bottom-right quadrant still contains 158–218 detected features, but only 0–5 reference points survive the forward descriptor checks and 0–1 survive mutual matching. Late matched, deduplicated and retained reference histograms are identical: retention is not the source of that loss.

All 114 seeded-flow attempts fail residual gates. Median per-attempt median error is only 0.201 pixels, but median 95th-percentile error is 39.67 pixels and 776/4,270 withheld observations exceed 6 pixels. A small median cannot hide that tail. The toy van's 352 fallback attempts stop before optical flow because the seed is unavailable; they are not measured flow failures. Five-file diagnosis seal: `72cfe8442be9a1203890cc68270bd1ac7fc22941ac20d72075e42a884625d06f`.

A separate post-result inspection of unannotated frames 40 and 180 identifies a large lower-right hand at the reference and a changing dark shadow. The ball-only exclusions do not describe either. This supplies a concrete missing foreground/illumination condition, not proof of the cause of every residual. [Source inspection](../artifacts/camera-foreground-review-01/report.json), three-file seal `e929cc8538267a1eabe15d0834b895d7cc7d3866777ac80ff6fd8ada8ccfabb8`.

The source-patch appearance hypothesis is now tested as V4 below. Do not use fitted transform residuals to select withheld correspondences, relax the gates, or increase feature counts to conceal incompatible matches.

## V4: image appearance checks before fitting

V4 checks both descriptor seed matches and optical-flow matches against immutable source patches before fitting. An 11×11 patch needs normalized correlation ≥0.80, a ≥0.10 advantage over competing locations within ±8 analysis pixels, and intensity standard deviation ≥5. Candidate scoring allows a ±1-pixel neighborhood without relocating the correspondence; the competing search excludes ±2 pixels. Patch support must stay inside the image and exclusion mask. Local uniqueness is not global identity.

The training seed may orient a patch for rotation/scale; its translation and fitted residuals do not select withheld points. Existing geometry, residual and resource limits remain unchanged. New bounded stage counts retain appearance rejections and accepted scores. Old V3 reports reject explicitly rather than being silently reinterpreted.

Fifty-six camera tests and nineteen adapter tests pass. Controls include moving foreground, changing shadows, repeated texture, wrong flow and parallax. Uniform brightness/gain changes are intentionally tolerated; this is not a universal shadow detector. Twelve forced-fallback synthetic pairs recover their known transforms, with 104.36 ms median including preparation; forced fallback is not evidence of natural real-source improvement.

| Source | Decoded frames | Registered required frames | Seeded-flow acceptances | Worker time |
|---|---:|---:|---:|---:|
| Three-ball | 181 | 43/141 | 22 | 6.565 s |
| Edited/hand-held toy van | 354 | 2/354 | 0 | 14.549 s |

The ball's usable count increases from 27 to 43, but 98 required frames remain unavailable. Only four of twenty reused manual-label slots have corrected estimates: conditional RMS 0.280 px, all four within annotation uncertainty. Sixteen are unavailable; identical rounded raw labels still prevent a demonstrated correction benefit. The van's 352 fallbacks stop before optical flow because no seed qualifies.

Neither source generates a corrected scene. Study `artifacts/camera-stabilization-study-04` retains unchanged full inputs and implementation snapshots: 38-file seal `cd6bcb4710fbc3c1f56102fcba74dd69429ad11bf3111144d8ad64a067f0f80b`. No paid call or physics change. Remaining ball failures are mostly spatial-support failures; inspect retained stage counts before proposing further changes, without blacklisting failed withheld points.
