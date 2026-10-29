# Fresh full-body squat source

Ready for a separate local diagnostic: `data/exercise/bodyweight-squats-danielflefil-500.mp4`, **68 frames, 500×500, 25 fps, 2.72 seconds**. Its SHA-256 is `ec636e28b4711887ecd9ff7014b64801d16055faf2f2b0bf6ed9221ea8280118`; size is 6,635,884 bytes. No pose inference or paid API was run during source selection or annotation. The exercise registry remains unchanged.

## Provenance and scope

Danielflefil uploaded [Bodyweight Squats.gif](https://commons.wikimedia.org/wiki/File:Bodyweight_Squats.gif) as their own work under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This study acquired the complete official **500-pixel animated derivative**, not the 37,164,484-byte full-resolution original. Credit the creator, link the license, and identify the adaptation; no endorsement is implied. The creator credit does not establish the performer's identity.

The downloaded GIF is 5,063,066 bytes, SHA-256 `14f1febdd9827951c0099ae2df6288fa5afb7b83c1fb99882f7505486982af85`. All 68 frame durations are exactly 40 ms. One full animation cycle was converted to lossless RGB H.264 using the existing pinned FFmpeg, without resizing, cropping, trimming, repeating, or interpolating frames. Every decoded MP4 RGB frame hash and timestamp equals the GIF counterpart. This preserves the animation's timing, **not independently calibrated physical or original-camera timing**.

## Source-only inspection and labels

Eleven sample indices were selected before pixels were viewed. Their contact sheet shows the head, body, and both feet inside the image throughout those samples. The background appears stationary, but this is not an independent fixed-camera verification. Far-side hip/limb occlusion remains.

Five annotation slots were frozen at frames **0, 13, 33, 53, 67** on October 28, 2026, at **4:41:47 PM PDT**, before model inference. They record approximate projected pelvis and shoulder midpoints, visible sole-to-mat boundary proxies, coarse image-space motion directions, and explicit unknowns. Pelvis uncertainty is ±20 pixels, shoulders ±15 pixels, and foot proxies ±7 pixels. These are assistant visual estimates, not expert ground truth. Anatomical foot-side identity remains unknown; a visible sole does not measure contact force. Root directions describe image motion, not calibrated world axes.

The original annotations are `data/exercise/bodyweight-squats-danielflefil-500.manual.json`, SHA-256 `decdfdd7988afa0f6d6d8c3402d26a6491ffbff37ce2951386d3b8005913a93f`. Do not fit camera/scale parameters to these labels or replace difficult slots after seeing model outputs. They are development evidence, not a final holdout.

## Rejected candidate retained

The first download, CRiles23's [Squats Wikipedia.webm](https://commons.wikimedia.org/wiki/File:Squats_Wikipedia.webm), is licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). It was rejected after source-only inspection because it repeatedly crops the body and feet and contains handheld close-ups and background people. Its original 14,763,768 bytes remain unchanged at `data/exercise/bodyweight-squats-criles23.webm`; SHA-256 `d37c07d2ae87af305a7e74ad0dc41f71281e20a0eac5a15247e792d1d56a4b32`.

Decoding yielded 1,579 frames versus 1,609 metadata frames. That mismatch remains unresolved. The preselected frame 1608 is explicitly unavailable; it was not silently replaced. No favorable tail interval was substituted.

## Bounds and verification

Exactly two media candidates were downloaded, totaling **19,826,834 bytes**, within the 30 MB limit. An initial sandbox DNS failure downloaded no bytes; the authorized bounded download subsequently succeeded. No accounts, browser interaction, model calls, registry edits, or paid services were used.

`artifacts/exercise-fullbody-source-01` preserves both candidates' provenance, the rejected source and samples, the accepted GIF/MP4, original sample PNGs, timing and RGB hashes, frozen annotations, conversion commands, and preparation scripts. Its **44-file** standard manifest verifies with SHA-256 `a13a19ed60bd46a507bd3a680776965b9764f95e8c1a54d8ed1fcc5eec8f4d51`.

The short cycle does not fit an assumed default 60-step training/evaluation split. A separately declared 20-step diagnostic can use all 68 source frames without trimming the input. No evaluation outcome is claimed here. Full foot visibility alone does not establish metric floor height, gravity, balance, contact forces, recovered anatomy, or valid biomechanics.
