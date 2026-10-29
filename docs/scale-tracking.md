# Opt-in scale-aware appearance tracking

The tracker can now test a bounded set of image sizes while retaining its original source appearance. This is an observation experiment, not recovery of depth or a changing physical body. Fixed-size tracking remains the default.

## Try it locally

Use a new output directory; the saved configuration below already selects `scale_mode: "pyramid"` and the complete declared cylinder segment:

```sh
.venv-lab/bin/python -m weft.generic_vision \
  --video data/general/bicone-cylinder-240p.webm \
  --config artifacts/scale-tracking-01/cylinder/tracking/tracking-config.json \
  --out artifacts/cylinder-scale-new
```

For another source, supply its exact source hash, interval and initial object rectangles using the [appearance configuration](general-tracking.md), then add `"scale_mode": "pyramid"`. Only independent association is supported. The full lab retains these observations but deliberately refuses to create physical geometry from changing image extent.

## Contract

- Seventeen immutable source templates cover 0.60–1.40 times the initial size in 0.05 steps; predictions never replace source templates.
- Source-template storage is bounded to 4 MB per object and search output to one million score cells per object/frame. An exceeded search budget returns no position.
- Existing score, uniqueness, color, identity and missing-frame checks remain unchanged. Competing locations are compared across all scales; accepted center motion, not size-induced corner motion, updates velocity.
- All missing, ambiguous and pre-initialization rows remain. Each estimate carries scale/search evidence and an explicit image-extent-only geometry status.

## Frozen real-source comparison

Original intervals, identities, thresholds and manual labels were reused without tuning. These are development labels, not a fresh holdout.

| Source | Fixed-size baseline | Scale candidate | Interpretation |
|---|---|---|---|
| Cylinder, 176 rows | 26 tracked; 1/4 scored centers returned; 0/4 within 4 px | 67 tracked; 2/4 returned; 1/4 within 4 px | More coverage, mixed accuracy |
| Edited toy van, 354 rows | 144 tracked; 1/4 visible centers returned | 144 tracked; 1/4 returned | No coverage improvement; ambiguous crossfade still returned |

For the cylinder, frame 2720 worsens from 5.00 to **14.04 px** error. Newly recovered frame 2745 is **2.06 px** from its label. Frames 2770 and 2805 remain unavailable. Conditional RMS over the two returned centers is 10.03 px. More accepted rectangles do not establish better object boundaries or depth recovery.

The toy van's one returned visible label improves from 2 to 1 px; its crossfade prediction is ambiguous evidence, not verified physical presence or absence. Neither source produces a physical environment. Background status remains unverified for the cylinder and moving for the van.

## Verification and cost

Fifteen scale tests pass, covering size changes, center motion, immutable templates, distractors, occlusion, rotation, bounded work, actual encoded frames and rejection of changing collision geometry. The unchanged generic suite has 27 passes. Two initial test failures used a search canvas too small to exceed the declared budget; corrected tests pass, with both reports retained.

An eight-object, 960×540, 30-frame paired synthetic measurement returns 240/240 exact rectangles for both methods. Median/p95 processing time per eight-object frame is **2.44/2.55 ms fixed** versus **41.77/42.35 ms pyramid**. Scaling is slower; this is one offline measurement, not a 30-fps or sustained browser-performance claim.

The shared-observation adapter now converts literal `unknown` role/shape values to null unknowns while retaining raw scale evidence. Its 52 tests pass; corrected exports preserve all 530 source slots. The initial failed export is retained separately.

Saved evidence:

- [Real-source report](../artifacts/scale-tracking-01/report.json): 32 files, seal `4ac65b02d045cb807468005ced02b2396574321c241d0c5a04f5e5b6158dde33`.
- [Performance report](../artifacts/scale-tracking-performance-01/report.json): four files, seal `778da8c30bbb7e5af5f88c48a2f5404489607fe59276fb8901105ca119df5d16`.
- Shared exports: `artifacts/scale-tracking-observations-02`, five files, seal `03530f549bcbd25c2d570f5941431dd808a4a9f33a6124776cf4278a42742f8f`.

No API calls or training ran. Next work needs independent extent/boundary evidence: a uniform cylinder can match a partial image region at the wrong scale. Do not relax identity gates or equate acceptance count with reconstruction accuracy.
