# Meta three-ball comparison input

Prepared October 28, 2026, at 4:23:53 PM PDT. No API call, credential read, ledger reservation, or model run occurred in this lane.

## Ready input

Use `artifacts/meta-three-ball-input-01/attempt-05/three-ball-crf18.mp4`: **100,131 bytes**, SHA-256 `d963bb8e3c4941921fb2dba03be34c5969e073ce6fd4c6dedfd8d719e1e3ddcb`. The complete proposed compact inline request is **133,954 bytes**, below the 50 MB envelope. The top-level standard manifest verifies **105 files**, SHA-256 `1152ff5ec9ac049ca888c3a4428eaf321c2824a5eb7c9f41f3d849550a255a18`.

The source is Rhetos's CC BY-SA 4.0 [Dreifachstoß (Pilotversuch)](https://commons.wikimedia.org/wiki/File:Dreifachsto%C3%9F_(Pilotversuch).webm), the complete official Wikimedia 240p derivative already acquired locally. Its SHA-256 remains `f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887`. This study uses the previously frozen **2580–2760 inclusive** real-time interval, not the entire 212-second source or its later annotated slow replay. Every one of the selected **181 frames** remains present, including initial hand occlusion and final departures.

Native dimensions are **426×240**. Current appearance analysis uses the same pixel scale, so coordinates require no resizing or normalization. The original sample aspect ratio, 427:426, is retained; prompt coordinates address the 426×240 stored raster. Actual API handling remains untested.

## Exact initialization and request

Both original `.scene.json` and `.initialization.json` put **all three objects at source frame 2620**, corresponding to **relative frame 40**. Do not invent earlier passive-object prompts using evaluation labels. Preserve input frames 0–39 as explicit pre-initialization/untracked slots. Use forward propagation only for the causal comparison; the provider's internal inference implementation is not independently established as causal.

| SAM ID | Frozen object ID | Original xywh | Inclusive SAM box |
| --- | --- | --- | --- |
| 1 | upper-passive | 187,104,8,8 | 187,104,194,111 |
| 2 | lower-passive | 189,111,8,8 | 189,111,196,118 |
| 3 | incoming-active | 270,113,8,8 | 270,113,277,120 |

The final request is frozen in `request-template-final.json`; attach the accepted MP4 bytes as base64. The earlier `request-template.json` is a preserved, **unexecuted** draft with `both` propagation; do not use it.

```json
{
  "model": "sam-3.1",
  "stream": true,
  "store": false,
  "input": [{
    "type": "message",
    "role": "user",
    "content": [
      {"type": "input_text", "text": "<40f>1<|box;x1=187;y1=104;x2=194;y2=111;w=426;h=240|>,2<|box;x1=189;y1=111;x2=196;y2=118;w=426;h=240|>,3<|box;x1=270;y1=113;x2=277;y2=120;w=426;h=240|>"},
      {"type": "input_video", "video_url": "data:video/mp4;base64,<ACCEPTED_MP4_BYTES>"}
    ]
  }],
  "metadata": {"propagation_direction": "forward", "include_confidence": "true", "mask_encoding": "one_bit"}
}
```

Meta documents multiple box-prompted object IDs in one request, inclusive box corners, and forward propagation from the earliest prompt frame. Video supports up to 16 prompted objects. Frames without a returned object may be omitted, so interpret explicit frame markers instead of output line numbers. [Official prompt documentation](https://dev.meta.ai/docs/sam/segmenting), [output grammar](https://dev.meta.ai/docs/sam/reading-segmentation).

No other manual centers, visibility labels, inferred contact intervals, or previous tracker predictions enter the request. The original manual file is preserved unchanged, hash `25bad603d0d36b58e2fc1185495ec093f8ae4eabbc7153757cd67a1c4b3f9953`.

## Encoding and time verification

The derivative is explicitly **lossy H.264 CRF18**, medium preset, two encoder threads, with a 120-second per-command bound. Audio is omitted. No image crop, resize, interpolation, duplicate, or favorable subinterval was introduced.

`frame-mapping-and-pixels.json` records all 181 decoded source and derivative frame hashes and timestamps. Native YUV420p aggregate differences: MAE **0.749635**, RMSE **1.172192**, maximum absolute error **23**, 95th-percentile absolute error **2**, PSNR **46.750825 dB**. Per-frame measurements are retained. These quantify compression changes, not segmentation accuracy.

Source presentation timestamps run from **43,017 to 46,017 ms**. Subtracting exactly 43,017 ms gives relative timestamps **0 to 3,000 ms**, with every original 16/17-ms gap preserved. Nominal source rate is 60 fps, but timestamps were not reconstructed as frame index divided by 60. The final sample lasts 16 ms, ending at 3,016 ms. Intermediate MP4 sample-duration metadata follows adjacent timestamps, whereas source decoding reports nominal 16-ms durations; duration metadata is not identical.

The first encoding contained 181 samples but decoded only 180 because its final sample had zero duration after trimming. Rejected attempts and their outputs remain saved: an incompatible nominal-rate flag combination, and timestamp-seek attempts that began a frame too late. The accepted candidate is a **stream-copy remux of the original encoded samples**, using `setts=duration=16` to retain the terminal sample. It does not encode a second time or add a frame. All 181 exact presentation timestamps and decoded output hashes were then verified. See `remux-provenance.json`; no failing candidate was silently replaced.

## Bounded proposed spend

Meta publishes $0.20 per 1,000 video frames, a 50-frame/request minimum, and billing through `usage.video_frames_processed`. The reviewed pricing and usage-schema pages describe frames per request and **do not specify an object-count multiplier**. Accordingly, 181 input frames estimate **$0.0362**; forward tracking of frames 40–180 may bill 141 frames, **$0.0282**. Do not assume the lower figure before receiving usage. [Official pricing](https://dev.meta.ai/docs/pricing-rate-limits), [usage schema](https://dev.meta.ai/docs/api-reference/responses/schemas).

For conservative budgeting, three times all 181 frames is **$0.1086**; a proposed **$0.15 shared-ledger reservation** covers that uncertainty envelope, not a documented surcharge or provider-enforced cap. The parent reported $4.88969 remaining before this preparation; this lane did not read or alter the ledger. Plan one inference POST, no separate upload, no polling, no cleanup calls, and no automatic POST retries. Preserve unknown charges on interruption. The request remains unexecuted.

## Scoring limitations

Retain every frozen visibility slot: 24 total, including the initial occluded incoming ball and final out-of-frame incoming ball. Do not score initialization as independent accuracy. The ±5-pixel manual-center tolerance is large relative to the approximately six-pixel balls. Post-contact sphere identities represent inferred continuity, not uniquely known physical identity. There are no independent mask annotations or measured contact forces.

Report coverage and accepted-center errors together, including nulls, losses, ambiguities, and possible switches. Never pick the returned identity nearest each evaluation label. A segmentation result cannot establish fixed-camera physics, recover physical forces, or turn this source into a validated simulator. No default or promotion is proposed.
