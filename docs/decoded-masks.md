# Inspect actual object silhouettes

`tracking-masks` decodes saved Meta segmentation masks into transparent PNGs and an offline, frame-by-frame viewer. A mask is the model's estimate of which pixels belong to an object. It is not a measured physical boundary.

## Open the completed examples

- [Real sliding block and target box](../artifacts/meta-phys101-ramp-masks-01/index.html): 44 source frames, 72 decoded masks and 16 explicit pre-initialization slots. [Paired tracking comparison and remaining physics work](phys101-ramps.md).
- [Exercise person](../artifacts/meta-exercise-silhouette-masks-02/index.html): all 68 frames of the supplied squat clip.
- [Three balls](../artifacts/decoded-masks-01/three-ball/decoded/index.html): 181 frames, including 120 explicitly unreturned object slots.
- [Toy van](../artifacts/decoded-masks-01/toy-van/decoded/index.html): all 354 frames, including the uncertain edited transition.
- [Exercise joints versus silhouette](../artifacts/exercise-silhouette-agreement-01/index.html): three inspection overlays and complete per-frame comparison data.

Open these files yourself; no server or API request is needed. Use **Previous frame**, **Next frame**, or the slider, then toggle silhouettes, original boxes and silhouette centers. Each preview comes from the selected decoded source frame, not approximate video seeking. Previews are resized, quality-90 JPEGs for inspection; use the original MP4 for pixel grading.

The silhouette viewer does not control a simulation. Use the [3D body/foot sandbox](foot-sandbox.md#control-and-record-a-new-trial) to apply actions and record a new trial. Those bodies remain experimental; these masks do not silently replace their geometry or clear their floor failures.

## Decode another saved result

Use a new output directory and the exact MP4 submitted to Meta:

```sh
.venv-lab/bin/python -m weft tracking-masks \
  --run artifacts/meta-exercise-silhouette-evidence-01 \
  --video data/exercise/bodyweight-squats-danielflefil-500.mp4 \
  --out artifacts/my-exercise-masks-01
```

The input must be an original [`tracking-evidence` export](segmentation-evidence.md), not stabilized coordinates. For a new video, first use the [explicit tracking CLI](meta-tracking-cli.md), then export its evidence. Only `segment --execute` makes a paid request; evidence export, mask decoding and viewing do not.

The current lab environment includes the pinned `meta-sam-parser==0.1.0` dependency. Fresh environments use the [lab setup](local-lab.md) and `requirements-lab.lock`. The optional `segmentation` extra also lists the codec. Its [official package](https://github.com/meta-models/meta-sam/tree/main/python) is dependency-free and distributed under the SAM License. Source/license copies and hashes accompany each decoded run; the wheel SHA-256 is `eca156907911c56cef3a350ffc81c34b7e70655e5fa3ccdddb77042c541a0152`.

## What is preserved

`original-evidence.json` retains the original proposals. `mask-evidence.json` adds raster hashes, foreground area, centroid, component count and a projected principal axis. Transparent PNGs preserve the exact binary raster; their palette changes only display color. Boxes, raw IDs, identity uncertainty, timestamps, missing observations and any original-source mapping stay unchanged. An absent optional source mapping remains absent.

Mask dimensions need not equal box dimensions. Raster cell edges span the original half-open source box; centroids use pixel-cell centers. The principal axis has no direction and is unavailable for near-isotropic masks. Neither this axis nor a silhouette centroid establishes anatomical orientation, rigid-body rotation or center of mass. Invalid codec payloads remain explicit decode failures, not empty masks or invented positions.

One spawned worker has a 60-second deadline and up to two seconds of cleanup. Limits include 500 frames, eight objects, 8,847,360 combined mask pixels per frame, 50 million total mask pixels, 32 million encoded characters and 128 MB of image exports. The viewer loads one frame's images at a time, coalesces rapid navigation to the latest selection, and shows no stale frame after a failed load. It makes no paid calls or recurring polls.

## Verified results

| Source | Frames | Decoded masks | Unreturned slots | Decode/export worker time | Preview + mask bytes |
|---|---:|---:|---:|---:|---:|
| Three balls | 181 | 423 | 120 | 0.224 s | 4,201,369 |
| Toy van | 354 | 354 | 0 | 4.690 s | 41,543,739 |
| Exercise | 68 | 68 | 0 | 6.372 s | 2,322,720 |

Every exported mask was checked for binary pixel identity, dimensions, hash and foreground count. All 603 source frames and 965 object/frame slots remain represented. These timings describe these runs, not sustained performance or browser latency.

The original ball identity disagreements remain: only 8/17 eligible silhouette centroids fall within the prior five-pixel center radius. The van has 4/4 within its prior 16-pixel radius. Those labels describe approximate bounding-envelope centers, not independent mask annotations; these comparisons do not measure silhouette accuracy or prove an improvement over box centers.

The exercise request used one first-frame person box, no evaluation labels, and no automatic retry. Meta returned 68 billed frames in 8.027 seconds, costing **$0.0136**. At that exercise observation the ledger recorded six settled calls, $0.166152 used and $4.833848 remaining. The later [ramp request](phys101-ramps.md) brings the observed total to **seven settled calls, $0.176152 used and $4.823848 remaining** at October 29, 2026, 2:11 AM PDT. This is the application ledger, not a provider-enforced cap or invoice. The first local exercise mask export failed because it incorrectly required an optional source-conversion record. That failure remains saved; the corrected export reused the same paid response.

The exercise comparison checks 16 raw 2D landmarks on each of 34 sampled pose frames against the exact-time mask. Of 544 sampled joint slots, 442 fall inside, 57 outside and 45 remain low-confidence. All 57 outside points concern ankles/feet; the right foot-index point falls outside in 31/34 frames. Another 544 slots retain `pose_not_sampled` for the 34 unsampled video frames. No interpolation, identity rematching, snapping to the silhouette or body/floor adjustment occurs. This is agreement between MediaPipe and Meta, not independent anatomical or 3D validation.

## Reproducibility and checks

- Saved ball/van study: `artifacts/decoded-masks-01`, 1,568-file seal `039f9fd56b43cdcd79005333a7c669fb7be0042fac345be85c2895ddd5375bc4`.
- Exercise paid result: `artifacts/meta-exercise-silhouette-live-01`, nine-file seal `3c76f46c78f528dcaeb876f9149c66f3c4eda42a4c6cf167350ae0896ddf7fdd`.
- Exercise decoded result: `artifacts/meta-exercise-silhouette-masks-02`, 245-file seal `5365a1b7b7e4c5257ebe8889f1ea5befbc6ca005f95c4d13979887921950b91f`.
- Exercise comparison: `artifacts/exercise-silhouette-agreement-01`, 16-file seal `a35b8b514ca8b6d15db76132636ad44d21fc00a391ccdfa3a161e7ff1bda419c`, 0.188 s, no new request.

There are 172 distinct passing affected Python tests, ten deterministic viewer checks and ten analytic comparison-mapping checks. Coverage includes strict malformed-codec rejection, anisotropic coordinate mapping, pixel identity, missing observations, actual worker timeout/cleanup, source mutation, absent optional mapping, stale image rejection and coalesced navigation. The ball/van study snapshot predates only the final optional-mapping fix and its added regression test; its codec, geometry and viewer are unchanged. Native browser/device interaction has not been performed.

A separate scripted capacity check uses 500 frames/eight objects, 4,000 observation rows and 5,124,885 metadata bytes. Across three trials, 500 rapid selections require only the initial and latest image batches; at most nine images load concurrently. Each trial also checks 100 completed selections, 200 cached layer changes and 100 ms idle with no new image requests or draws. Startup is 7.29–14.00 ms and scripted navigation p95 is .069–.085 ms. These Node/stub-image measurements exclude image decoding, browser layout/paint and sustained resource use. Evidence: `artifacts/mask-viewer-capacity-01`, five-file seal `eacb2356ec95c16da4f309b1b7f09076ce6da20bb0fe0c4efb1cbe7124282b0b`.

The public squat source is Danielflefil's [Bodyweight Squats](https://commons.wikimedia.org/wiki/File:Bodyweight_Squats.gif), CC BY-SA 4.0, converted to a lossless MP4. [Source provenance](exercise-fullbody-source.md) distinguishes that derivative from raw-camera timing or calibrated physical observations. Mask generation does not establish muscles, anatomical dimensions, contact forces or a physically accurate environment.
