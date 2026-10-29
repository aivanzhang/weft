# Bounded Meta SAM toy-van comparison

Research date: October 28, 2026 (America/Los_Angeles). Planning only: no credentials read, authenticated API calls, uploads, installations, or account changes. The parent reported $0.039510 already spent against the shared $5 project budget; that balance was not independently read here.

## Recommendation and cost

Propose one complete 354-frame request, not six disconnected samples. Meta prices SAM video at $0.20/1,000 frames, with a 50-frame minimum per request; images cost $2.50/1,000. Streaming has the same price. `usage.video_frames_processed` reports billed video frames. SAM has 3,000 requests/minute per team and no token-per-minute limit. Background submissions additionally have a 600/minute team cap. [Official pricing](https://dev.meta.ai/docs/pricing-rate-limits).

| Planned work | Count | Estimated API cost |
| --- | ---: | ---: |
| Local full-length format conversion and verification | 1 | $0; local compute only |
| Inline segmentation POST, complete clip | 1 × 354 frames | $0.070800 |
| Separate file uploads, storage, polling, deletion | 0 | $0; avoided operations |
| Automatic inference retries or additional concepts | 0 | $0 |
| **Expected additional spend** | | **$0.070800** |

Expected project cumulative spend becomes $0.110310, leaving $4.889690. Reserve a conservative $0.15 in the existing shared ledger before sending; do not reset history or release an unknown charge after a timeout. This is a local reservation, **not a provider-enforced spending limit**. The official pages reviewed list no separate upload/storage/read/delete tariff, so an upload-based total cannot be guaranteed from those pages. The recommended inline path avoids those operations. No recurring work is proposed. Actual billing must be reconciled from returned usage; retain an unresolved reservation if usage is unavailable.

The six-image alternative costs an estimated $0.015. It loses temporal propagation and full-clip coverage; it cannot establish tracking continuity. Reusing the first-frame ROI on later images would be invalid, while prompting later images using frozen evaluation centers would leak labels. A separately declared concept-only image experiment would therefore be a different comparison, not a cheap substitute for this one.

## Request and media limits

SAM uses `POST https://api.meta.ai/v1/responses`, model `sam-3.1`. One user message combines `input_text` with `input_video` or `input_image`; URLs or data URLs are supported. Concept phrases and geometric prompts cannot be combined. Boxes use inclusive corners. Multi-frame point/box prompts and repeated object IDs are supported, but this study uses only initialization. Video permits MP4/MOV with H.264/H.265/MPEG-4, not WebM/VP9; maximum 15,000 frames and 16 prompted objects. `forward` propagation starts at the earliest prompt. Box prompts produce `one_bit` masks. Confidence metadata is a string. [Prompt specification](https://dev.meta.ai/docs/sam/segmenting).

Proposed payload (data omitted; not executed):

```json
{
  "model": "sam-3.1",
  "stream": true,
  "store": false,
  "input": [{
    "type": "message",
    "role": "user",
    "content": [
      {"type": "input_text", "text": "<0f>1<|box;x1=970;y1=538;x2=1077;y2=605;w=1920;h=1080|>"},
      {"type": "input_video", "video_url": "data:video/mp4;base64,<PINNED_COMPLETE_DERIVATIVE>"}
    ]
  }],
  "metadata": {"propagation_direction": "forward", "include_confidence": "true", "mask_encoding": "one_bit"}
}
```

The original frozen ROI is `[970,538,108,68]`; inclusive endpoint conversion gives `(1077,605)`. No labels beyond frame 0 enter this request. Do not add sampled-frame coordinates, earlier tracker predictions, repair prompts, or a concept prompt after inspecting the result.

Keep the original WebM SHA-256 `caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8`. Prepare a separate full-length MP4 derivative, preserving all 354 frames, 1920×1080 dimensions, 59 fps timing, and source crossfades; do not crop, resize, interpolate, or trim. Record conversion command/tool version, derivative checksum, decoded frame count and per-frame mapping. Prefer a verified lossless encode if compatible and within the envelope; otherwise explicitly report pixel changes from encoding. No format conversion was performed during this research. Abort before API submission if mapping cannot be verified or the complete encoded JSON exceeds 50 MB; do not silently reduce the video to fit.

## Upload and retention

Inline media avoids separately managed Files storage. Inline limit: 50,000,000 bytes; Files upload limit: 1 GiB; team stored-file limit: 100 GiB. An alternative upload uses multipart `POST /v1/files`, `purpose=user_data`, `expires_after[anchor]=created_at`, `expires_after[seconds]=3600`, then `input_video.file_id`. Send metadata fields before the final file part. Default uploads do not expire; optional expiry spans 3,600–2,592,000 seconds. Metadata retrieval is `GET /v1/files/{file_id}`; cleanup is `DELETE /v1/files/{file_id}`. Do not create persistent uploads for this one-off plan. [File handling](https://dev.meta.ai/docs/file-handling), [delete endpoint](https://dev.meta.ai/docs/api-reference/files/delete-file).

Standard services do not use submitted content to train Meta models. That is not zero retention: Meta's terms allow retention for service operation, legal obligations, and safety/security review without a single fixed period. `store:false` is not a promise that all logs or safety records disappear. [Terms §§4.4, 5.1](https://dev.meta.ai/legal/terms-of-service).

Zero Data Retention requires separate Meta enablement, varies by model, and is not standard. It disables file uploads and background/stateful features; exceptions remain for law and misuse. This account's ZDR status is unknown, and no sales contact or account change is proposed. [Official ZDR help](https://dev.meta.ai/help/policies-and-privacy/zero-data-retention).

## Response and failure handling

The response emits special-token text in `response.output_text.delta`, ending with `response.completed`. Boxes use source-pixel coordinates. Each emitted frame has an explicit `<Nf>` marker and integer object IDs, followed by box/mask pairs. Frames with no visible object may emit no line: reconstruct all 354 slots explicitly, rather than treating line position as frame number. Masks contain `H,W,encoding,payload` information, with a base85 payload; `!` denotes one-bit and `~` lossless encoding. Decode into an H×W binary raster using the official parser contract; preserve original payloads and confidence. Do not assume this is COCO RLE or a frame-sized mask. [Output grammar](https://dev.meta.ai/docs/sam/reading-segmentation).

Implementation detail: this is custom base85 plus arithmetic coding, **not RLE**. After the encoding marker comes a five-character length prefix and packed arithmetic-coded bytes; base85 decoding alone does not produce a mask. The official decoder checks lengths, finalization, and canonical one-bit round trips. Masks occupy the accompanying box bounds, normalized to half-open corners. Bound decoding area and payload size before allocation. SSE text chunks can split records: preserve one lane keyed by `item_id`, `output_index`, and `content_index`; finalize using `response.output_text.done.text` or `response.content_part.done.part.text`, requiring agreement with accumulated deltas. A completed response's aggregate output alone does not finalize that lane. Reject conflicting finalizers and incomplete EOF. [Official parser protocol](https://github.com/meta-models/meta-sam/blob/main/protocol/sam3.md).

Use bounded streaming with a local 600-second total deadline, bounded output storage, one in-flight request, and no automatic SDK retries. These are proposed client bounds, not documented service guarantees. Preserve response ID, headers, terminal event, raw usage, partial records, and the request hash. On a disconnect or missing terminal event, mark incomplete/unknown charge and do not repeat the POST. An empty result is valid, not a trigger to try another prompt. No mask/identity-quality claim follows from an HTTP success.

General Responses defaults to `store:true`; stored streaming requests continue processing after disconnect. Stored results can be retrieved with `GET /v1/responses/{id}`. Background requests use `background:true` without `stream:true`, then GET polling or stream retrieval; cancellation is `POST /v1/responses/{id}/cancel`. Response deletion is soft deletion via DELETE; repeated deletion returns 404. This plan instead uses `store:false` and synchronous streaming, so it does not depend on polling or recovery storage. No documented SAM-specific polling limit, creation-idempotency guarantee, maximum request duration, or reliable cancellation of a foreground request was found. [Responses protocol](https://dev.meta.ai/docs/protocols/responses).

## Independent comparison protocol

Before submission, freeze the request, derivative, input manifest, comparison code, and current local trace hash in a new study directory. Preserve the existing `artifacts/general-object-e2e-01` unchanged. Label evidence remains the original six slots: frames 0, 70, 140, 210, 280, 353. Five approximate body-envelope centers have ±16 source-pixel uncertainty; frame 140 is unknown because the source crossfade blends two times.

Map SAM object 1 to `toy-van-body` by the frozen initialization, not by choosing whichever returned object is closest to later labels. Report returned coverage over all 354 frames, missing/invalid slots, identity changes, box geometry, masks, and confidence. Score box-envelope centers against frozen approximate envelope-center annotations; report mask centroids separately because they measure a different quantity. Initialization gets no accuracy credit. Preserve the null crossfade slot and any accepted result there as ambiguity, not a false positive against asserted physical absence. Report conditional error alongside missing detections; never replace missing error with zero.

There are no independent mask annotations, so mask overlap accuracy cannot be scored. Sparse centers cannot prove physical identity through crossfades. Segmentation does not repair camera motion, establish contact forces, recover 3D physics, or justify enabling the previously rejected simulator. This is a development comparison only; no default changes or promotion are proposed.

## Local preparation status

The requested lossless derivative was not created: no existing `ffmpeg`/`ffprobe` was found in PATH or standard Homebrew/local locations, and `.venv-lab` has neither `imageio_ffmpeg` nor PyAV. System `avconvert` exists, but its required lossless/timestamp preservation was not established; it was not used as a silent substitute. No tool was installed. `artifacts/meta-toy-van-input-01` seals a preparation-blocker report and source identity, not an executable input video. An existing suitable encoder path is required before this plan can run.

### Explicit subsequent preparation plans

The preceding blocker remains historical evidence. The parent subsequently installed pinned `imageio-ffmpeg==0.7.0` using `requirements-video.lock` and authorized local conversion; this research lane did not install it.

1. **Input 02: lossless, inline rejected.** `artifacts/meta-toy-van-input-02` preserves H.264 CRF0 at original dimensions, all 354 decoded frames, identical native-YUV pixel hashes, and identical presentation timestamps. Its 163,744,503-byte video requires 218,326,354 bytes as the planned compact inline JSON, exceeding 50 MB. No shrinking or submission occurred. The 14-file manifest hash is `e0173cc3f443644ff629c2b57f09e8f218a44cdda8f78322d4abf375bb1a0333`.

2. **Input 03: separately authorized lossy comparison.** The parent explicitly authorized H.264 CRF18, medium preset, two threads, and a 120-second encode bound. This is a new plan, not a silent substitute for lossless input. `artifacts/meta-toy-van-input-03/toy-van-crf18.mp4` is 10,605,962 bytes, SHA-256 `68940b3e4f4207f4a48c3e9355c148d34d901ba14b5b219aab6cb32ae82dfcd2`. No video frame was cropped, trimmed, resized, dropped, duplicated, or retimed. Audio is omitted. API codec acceptance remains untested.

All 354 original/derived frames have individually verified mapping and timestamps. Native 8-bit YUV420p sample differences have aggregate MAE 0.914881, RMSE 1.516117, maximum 39, 95th-percentile absolute error 3, and PSNR 44.516151 dB. Per-frame metrics and hashes are retained; these are encoding-distortion measurements, not perception or segmentation accuracy.

Both derivatives preserve the source's 1/1,000-second presentation timestamps from 0 through 5,983 ms. MP4 sample durations follow the 16/17-ms timestamp gaps, unlike the source decoder's nominal 16-ms durations. Last timestamp plus final sample duration is 5,999 ms in both decoded traces; the original video duration tag says 6,000 ms. MP4 reports average 59.01 fps versus nominal 59. Do not claim duration metadata is identical. No suffix frames are hidden. The input-03 report records exact request size and manifest verification; no Meta API request was made during either preparation.

### Subsequent live verification

The later [single full-clip request](meta-tracking-results.md) verified API acceptance and returned all 354 frames for $0.070800. The implemented study uses a 160-second process deadline, 120-second between-read checks, and a 30-second socket inactivity limit, superseding the initial proposed 600-second envelope. Encoded masks are preserved but not decoded or claimed accurate; box-center grading is separate. No retry, persistent Files upload, default promotion, or physics-gate change occurred. Earlier preparation reports remain unchanged historical evidence.
