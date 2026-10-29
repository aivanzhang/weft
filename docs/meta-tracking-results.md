# Meta tracking: first full-clip result

One native `sam-3.1` request completed on October 28, 2026, at 4:12 PM PDT. It returned object-1 boxes for **354/354 frames**. All four noninitialization source-center labels were within their frozen 16-pixel uncertainty; conditional center RMSE was **3.16 pixels**. This improves coverage on the saved toy-van example but does not validate physical identity through edits or enable its simulator.

Evidence: [six-frame visual comparison](../artifacts/meta-toy-van-live-01/comparison.png), [report](../artifacts/meta-toy-van-live-01/report.json), [full parsed output](../artifacts/meta-toy-van-live-01/call/segmentation.json). Nineteen files verified against manifest `72f1233ef48691fadceb74e2b842187ff50643f0dd85e921542ea837e1cb4012`. Exact request, raw SSE, finalizers, usage, frozen code/tests, source labels, and conversion provenance remain saved. The source is OrangeLark64's CC BY-SA 4.0 toy-van clip; attribution accompanies the input bundle.

## Fixed comparison

Only the original frame-0 box was supplied. Later labels were excluded. All original frames, presentation timestamps, crossfades, camera changes, and hand occlusions remained. The MP4 was explicitly lossy CRF18; native-YUV RMSE against the original was 1.516, PSNR 44.52 dB. The earlier lossless conversion exceeded the inline limit and is preserved separately. [Preparation and API protocol](meta-segmentation-plan.md).

| Check | Local appearance tracker | Meta SAM |
|---|---:|---:|
| Returned noninitialization labeled centers | 1/4 | 4/4 |
| Those centers within 16 px uncertainty | 1/4 | 4/4 |
| Conditional center RMSE | 2.00 px | 3.16 px |
| Frame 140, unknown crossfade | Accepted | Returned |

These are sparse assistant-estimated development annotations, not expert truth. Missing predictions are not zero error. Frame 0 receives no accuracy credit; frame 140 remains unknown, not physical absence. The two inputs differ in compression and the two methods have different acceptance rules; this is not an isolated algorithm-only benchmark. No mask annotations exist, and masks remain opaque encoded payloads rather than decoded or verified shapes.

## Spend and bounds

The single call took **29.05 seconds** and reported 354 billed frames: **$0.070800** at the [verified video rate](https://dev.meta.ai/docs/pricing-rate-limits). Combined with two earlier Muse Spark calls, the shared ledger commits **$0.110310 of $5**, leaving **$4.889690**, with zero unsettled reservations. No automatic retries, separate Files uploads, recurring jobs, or account changes occurred.

The client reserved $0.15 before transport and settled only after returned usage. This is application-side accounting, not a provider-enforced hard cap. Socket inactivity is bounded to 30 seconds; a 120-second deadline is checked between reads. The one-shot study adds a 160-second process alarm, because socket inactivity alone cannot bound a slow trickle. Responses are capped at 64 MB, individual SSE events at 1 MB, and parsed text at 8 MB.

## What remains

The parser passed 77 synthetic checks before execution; 15 transport/comparison checks cover usage settlement, uncertain charges, exact initialization, and retained missing/unknown slots. Live execution confirms this specific request/response path, not arbitrary video support.

Subsequent work adds a [multi-object CLI and offline viewer](meta-tracking-cli.md) and tests the preserved three-ball collision. That case returns all three IDs after initialization but matches only 8/17 eligible labeled centers, versus the local tracker's 9/17. Complete output coverage is not reliable identity. Its 141 processed frames cost $0.0282; cumulative ledger commitment after four total requests is $0.138511, leaving $4.861489. The preceding cost/latency figures remain this toy-van study's snapshot.

Do not promote these boxes directly into fixed-camera physics. Edited time, depth, support, contact forces, and changing viewpoint remain unidentified. Next: preserve SAM as an explicit optional perception backend, retain source edit/camera rejection, and compare additional frozen diverse clips before changing defaults. The current script is an explicit one-shot study, not an automatic paid stage in the sandbox.
