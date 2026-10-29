# Image-directed foot candidate

The optional correction rotates each available foot toward its heel-to-toe direction in the source image, by at most **15°**. It resolved 16 single-leg-squat floor failures without introducing new ones, but **every tested model still fails the complete floor check**. It is not applied to default exercise models.

## Method and limits

`foot_image_refinement.py` accepts only an original `kinematic-foot-reference-v1` and produces `image-directed-foot-reference-v1`. Repeated corrections are rejected. The input remains unchanged; the output preserves its hash, original orientations, raw pose estimates, availability and per-foot correction.

- Convert normalized heel/index image coordinates to pixels before comparing directions; width and height can differ.
- Require finite, in-image endpoints with presence and visibility at least .5 and a projected image axis of at least 2 pixels.
- Rotate the fixed foot triangle about the camera-depth axis, capped at ±15°. Triangle dimensions and all depth components stay unchanged.
- Leave missing source feet unchanged. Abstain when image evidence is unavailable or the model projection is shorter than .01 model units.
- Never fit body geometry, root position, floor placement, contact forces or evaluation limits. No floor score or manual label enters the correction.

Agreement with the pose estimator's image direction is an internal consistency check, not independent accuracy. Single-view depth, anatomical roll, sole shape, load and physical scale remain unverified.

## Four-source result

All 388 raw source-pose frames remain: 526 available foot slots receive a bounded correction; 250 unavailable slots remain unavailable. Original body audits are identical before and after. Candidate floor checks require both feet at the exact existing body timestamp; no foot gaps are interpolated.

| Source | Available foot slots | Floor failures before → after | Checked body poses | Maximum penetration before → after |
|---|---:|---:|---:|---:|
| Single-leg squat | 233 | 82 → 66 | 115 | .1870 → .1810 |
| Kettlebell push-up | 236 | 25 → 25 | 107 | .0665 → .0665 |
| Full-body squat | 57 | 23 → 23 | 23 | .1770 → .1683 |
| Cropped squat | 0 | Unavailable | 0 | Unavailable |

Penetration uses uncalibrated model units and the unchanged .025 limit. No new floor failures occurred, but no candidate qualifies for promotion. The push-up still lacks a starting foot; the cropped squat still lacks fixed foot shapes. This study performs no dynamic simulation and does not establish continuous-time contact feasibility.

Seven pre-existing approximate silhouette-axis labels were inspected only after fitting; nine unavailable label slots remain unscored. The original anonymous side assignments are fixed, not rematched to improve results. Six of seven direction errors and four of seven endpoint distances improve. The full-body frame-34 endpoint error worsens from 24.10 to 25.75 pixels. These uncertain silhouettes are not anatomical endpoints or contact measurements.

Inspect the [comparison sheet](../artifacts/foot-image-refinement-01/comparison-sheet.png) and [full report](../artifacts/foot-image-refinement-01/report.json). The 118-file study seal is `efb5f6bd84734885db18d9ecc95bbc77102e64ddd893b701659ed2279b7d4823`; runtime was .890 seconds, with zero paid requests.

## Build and interact

Follow the [foot sandbox guide](foot-sandbox.md#build-from-your-own-video). `feet-build` uses a verified completed mobile exercise run. Without `--image-directions`, it builds the original foot orientation. With the flag, it saves both original and corrected references in a separate directory.

The public CLI was exercised on a full-body source and a cropped-foot source. The former produced an interactive candidate; the latter saved an unavailable report without inventing feet. A subsequent terminal recording contains 14 motor actions, one reset and 16 video frames. Independent replay reproduces every saved state, reward and termination flag exactly. [Open that recording](../artifacts/feet-build-cli-01/session/replay.html). This verifies execution and recording, not physical accuracy.

The 215-file CLI evidence seal is `1bec563038a47d18c157b15bb77e43ee7d92c3c0d2a7c5162e957c65e51559e6`. Its snapshot predates only the builder's final source-change guard: the current builder rechecks manifest/input hashes after verification and copies those exact bytes. Seven builder tests cover that guard; the saved geometry and correction runtime are unchanged.

There are **74 distinct passing affected checks** across the latest logs: 67 foot correction/reference/body checks and seven builder checks. The first log retains one invalid-NaN test-fixture failure; its JSON-safe replacement and a separate NaN-rejection test pass. Logs: `.local/foot-image-tests.xml`, `.local/foot-image-final-tests.xml`, `.local/foot-candidate-final-tests.xml`. No browser interaction or new model inference was used.

Reproduce the comparison only when relevant inputs change, in a new directory:

```sh
.venv-lab/bin/python scripts/studies/check_foot_image_refinement.py \
  --out artifacts/foot-image-comparison-new
```
