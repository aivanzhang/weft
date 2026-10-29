# Selected sections of longer videos

The appearance profile can process an explicitly selected section of a longer local video. The complete original file and hash remain preserved; frames and timestamps are never renumbered to make the section look like a separate recording. This is interval ingestion, not automatic object discovery, camera calibration or physical validation.

## Use it

In the existing appearance scene configuration, add inclusive frame bounds and the frame where the manually selected object rectangles apply:

```json
{
  "tracking": {
    "start_frame": 2580,
    "end_frame": 2760,
    "decode_interval": [2580, 2760],
    "initialization_frame": 2620
  }
}
```

This snippet shows only the additional selection fields. Keep the required source hash, object rectangles/roles and explicit world assumptions. `decode_interval` must exactly match `start_frame`/`end_frame`; initialization must be inside that interval. The source-only [three-ball configuration](../data/balls/triple-collision-240p.scene.json) is a complete example, **not a validated physical model**.

```sh
.venv-lab/bin/python -m weft lab \
  --video data/balls/triple-collision-240p.webm --profile appearance \
  --config data/balls/triple-collision-240p.scene.json \
  --out artifacts/my-three-ball-interval --no-serve
```

The current three-ball scene is expected to fail its reconstruction prerequisites. Failure retains decoded evidence and tracking previews; it does not create an invented environment. Open a saved lab with `lab --resume PATH` to inspect its stages and artifacts. The viewer labels the requested original-frame interval before processing finishes; that label is not a completeness result.

## Evidence boundaries

- The default whole-video path retains its existing 60-second/1,800-frame limits and frame-count audit.
- Interval mode hashes the complete source, advances from the beginning without trusting a random seek, and streams only selected frames into perception. Prefix frames are advanced, not retrieved as image observations; the suffix is not decoded.
- Frames before `initialization_frame` are retained as `pre_initialization`, with no object position. They remain in the full interval coverage denominator. The initialized rectangles are authored input, not measured detections.
- Original frame IDs, nominal media timestamps and raw decoder timestamps are preserved. Nominal time is still frame index / declared FPS, not measured physical event timing. The repeated tenfold-slowed demonstration in this source is excluded by the independently chosen real-time interval, not used as extra physical evidence.
- `appearance/decode-audit.json` records prefix advancement, selected-frame reads, timestamps and failures. Successful ordinal reads do not independently prove that every encoded frame was decoded. The untouched suffix and full source are never labeled verified by a selected-window run.
- Shared scene observations use explicit schema 2 for intervals: full metadata frame count, selected decoded count, absolute inclusive bounds, prefix advances and `source_complete: false`. Existing schema 1 remains supported. Out-of-window frames, rebased IDs/times and relationships outside the selected interval are rejected.

## Work bounds

The source remains limited to 100 MB, advertised duration at most one hour, at most 216,000 advertised frames, 1–240 fps and 64-pixel minimum / 4096×2160 maximum dimensions. Selection is at most 1,800 frames and 60 nominal seconds. At most 20,000 preceding frames may be advanced; the source metadata ceiling does not permit selecting an arbitrarily late frame.

The reader checks a 120-second deadline around native decoder calls and always releases the capture. That cooperative deadline cannot interrupt a blocked native codec call; the lab's existing subprocess CPU/wall limits remain the outer protection. Frames stream one at a time; no full selected-video image buffer is retained. A truncated selected section, changing dimensions, source mutation, impossible decoder position or incomplete consumption fails explicitly. No paid/model calls or recurring service is required.

## Timestamp and completeness checks

Available decoder timestamps must follow the declared constant-frame-rate cadence within `max(2 ms, 10% of a frame period)`. Forward gaps, backward timestamps and repeated usable timestamps fail. Missing timestamps leave timing unverified; an all-zero backend clock is unavailable, not verified. Comparisons spanning missing timestamp entries use the actual ordinal distance, but do not establish complete timestamp coverage.

`selected_frames_complete` explicitly means the requested **decoder ordinals** were consumed. The audit separately records `decoder_ordinals_complete`, `selected_ordinal_skips`, `encoded_frame_skips: null` and `encoded_frame_completeness: not_independently_verified`. It never calls decoder read count independent encoded-frame proof. The full derivative's existing EBML audit still exceeds its 20,000-element bound; that guard was not widened to make this source pass.

## Final verification

Verified **October 28, 2026, 2:22 PM PDT**. [Pinned three-case benchmark](../benchmarks/source-interval-v7.json), `artifacts/source-interval-final-08/`: **4.435 seconds**, at most two workers. Two cases pass all declared reconstruction/action/recording gates; one desired physical collision case remains a failure.

| Case | Selected reads / prefix advances | Outcome |
| --- | --- | --- |
| Analytic 65-second source | 40 / 80 | Working interval environment; four pre-initialization frames retained; 35/39 tracking coverage |
| Real-source grayscale rolling | 22 / 39 | Working environment; 3.120-px reconstruction error; every retained observation exactly matches the earlier whole-file path |
| Real three-ball collision | 181 / 2,580 | Decoded successfully; reconstruction rejected for insufficient joint tracking coverage |

Both successful cases record 12 transitions and fully decode 13 video frames; nested run manifests remain unchanged. The final suite seal covers 225 files, SHA-256 `df26b7311eb85e9c3745f9c2930793c72ee56de0fe36a19bb88a2973f29f9c09`. The earlier `source-interval-e2e-07` remains preserved; final results use the subsequent timestamp guard and failed-run export implementation.

All three final cases have complete available timestamp coverage and pass the cadence check. For the real three-ball interval, the maximum step discrepancy is 0.667 ms against the predeclared 2-ms tolerance. This checks reported media cadence, not physical calibration or unseen encoded-frame completeness.

Frozen real-source labels retain 24 visibility observations, 22 visible centers and two unavailable centers. Only **6/22 visible centers** are tracked; five are within the source annotator's ±5-pixel uncertainty, and one has **23.54-pixel error**. Accepted-center RMSE is **9.64 pixels**. Five visible slots are before initialization or at the authored initialization and remain unmeasured; they are not hidden from coverage or misreported as detector mistakes. The camera screen remains `not_verified_fixed` despite visible source drift. These failures prevent a physical-accuracy claim.

`artifacts/interval-source-grades-01/` preserves normalized copies of unchanged source labels, all three observation exports, input hashes and the grayscale comparison. Its nine-file seal is `98e83094a6d82a2f1636cfe297ab26e4b76a4aabb5a4e0dd2d94f1d95f56e6e0`. The failed real case exports all **543 frame/object observations** and their raw evidence; no scene, contacts or physics are fabricated.

```sh
# Interact with and record a verified interval-based environment.
.venv-lab/bin/python -m weft lab --resume artifacts/source-interval-final-08/rolling-grayscale-interval

# Export observations from the failed real case without rebuilding or altering it.
.venv-lab/bin/python -m weft observations \
  --run artifacts/source-interval-final-08/triple-collision-interval \
  --out artifacts/my-collision-observations.json
```

**458 Python tests passed**, no failures/errors/skips, 16.067 seconds (`.local/source-interval-final-tests.xml`); 16 deterministic UI checks passed (`.local/interval-ui-tests.log`). The interval reader has 50 focused checks; shared observations have 51. No browser/device interaction was performed.

## Pixel identity and local decode cost

An independent full sequential read retrieved 12,758 frames. The interval reader retrieved 181 after advancing 2,580 prefix frames; all 181 selected image hashes match the full-read images at the same absolute indices. The source metadata advertises 12,759 frames; the one-frame discrepancy remains recorded, not silently fixed.

One same-backend local comparison took **0.834 seconds** for the full read versus **0.142 seconds** for the interval read, including source hashing and selected-pixel hashing. This excludes tracking, models, rendering and network; it is not sustained-performance evidence or an independent encoded-frame audit. Evidence: `artifacts/interval-pixel-verification-01/`, three-file seal `311dfa488e73ba8a8772518fe3abe9f66dd9be46419e33f813a5b072bd471e19`; script `scripts/studies/profile_interval_decode.py`.

The final implementation/design freeze is preserved in `artifacts/interval-design-freeze-final-02/`. No paid API/cloud calls or recurring resources were used. Meta's $5 project budget remains configured with $0 committed; no project key was present at this observation.
