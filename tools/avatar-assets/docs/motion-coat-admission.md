# Motion evidence and coat admission

The latest completed hands-off run remains **unsupported**, not accepted.
Studio `45757c9565b545119553baca` finished three real builds, dance media and
three valid typed reviews. The final review passed identity, framing and sampled
deformation, but rejected exported coat quality and clearly visible hops.
See [the current handoff](../TODO.md) for subsequent run ownership and status.

## What changed locally

- **Motion samples:** four delivered-video frames now use authenticated action
  controls and measured audit cues to cover baseline, relevant extremum, recovery
  and end. Positive measured vertical rise preserves a vertical candidate even
  when horizontal travel is larger. Degrees and metres are not compared directly.
  Missing/nonlinear controls use an explicit uniform fallback; omitted candidates
  and the four-frame limit are recorded. Original native/exported pose sheets stay
  unchanged. The selector does not decide whether the requested action succeeded.
- **Coat facts:** bounded manifest-bound summaries distinguish native-only strands,
  exported tuft cards, existing normal textures and optional authored undercoat.
  Historical reviews retain these facts. No feature is declared mandatory and no
  old review, quality gate or generated recipe is rewritten.
- **Continuation images:** the collector reproduces the authenticated paid-review
  image route rather than requiring obsolete contact sheets. It preserves all
  three previous reviews and source/model bindings without executing old code.
- **Automatic admission:** before a paid visual repair, a bounded local probe can
  measure an eligible saved model's UV/mask/export capacity. Its receipt is cached
  by model/source/config evidence and reused without rerunning native work. A
  refusal is feedback, not permission to lower counts or accept unsupported UVs.

## Actual saved-asset checks

The old four delivered frames (0, 32, 63, 95) missed all authored vertical peaks.
The new selection is **0, 24, 36, 95**, corresponding to 0, 1, 1.5 and 3.958 seconds.
Frame 24 covers an authored **14 mm** rise. The exported action audit independently
measured **14.0000004 mm root rise**; it did not measure foot-ground clearance or
prove visually readable hopping. Existing video decoding took 0.186 s and wrote
four new evidence PNGs; no original animation was rerendered or changed.

The v4 undercoat probe completed in **2.55 s**, sampling **348,585,984 bytes** peak
RSS for the direct child, under limits of 120 s, 2 GiB RSS and 64 MiB output.
It **refused** the current recipe: 50,004 source vertices + 45,900 portable-card
corners + 4,096 reserve already reaches 100,000. The selected 39,761 triangles add
an **854-corner material-boundary reserve**. The resulting **3,753-card maximum is
only an upper bound before additional UV-atlas cost**, not an admitted recipe.
No bake, render, pixel, tangent-frame, shader or appearance success is claimed.
The v3 sandbox startup SIGSEGV occurred before input loading and remains separate
from this model-specific refusal.

The real automatic continuation consumer reproduced the refusal in a separate
bounded probe (2.556 s; 348,454,912 bytes sampled direct-child RSS), then verified
its cache in 0.094 s without native replay. It routed eight trusted images and a
**103,595-byte author prompt + 1,024-byte allowance**, below the 128 KiB cap.
The ledger was unchanged and no new Studio job or provider call was made by that
preflight. An initial JSON recording failed after the checks because it contained
Path objects; the partial file is retained. `completed-result.json` is authoritative.

## Verification and provenance

[The bounded bundle](../evidence/panda-motion-coat-admission-v1/manifest.json)
preserves original safe receipts, source snapshots, exact commands and hashes:

| Evidence | Scope |
|---|---|
| `motion-evidence-sampling-tests-v1` | 18 distinct focused checks; actual saved-video decode |
| `coat-representation-feedback-tests-v1` | 45 distinct checks; actual read-only coat facts |
| `visual-continuation-image-route-tests-v1` | 34 distinct checks; actual terminal collection |
| `undercoat-admission-tests-v3` + capacity tests | 17 prior unique checks + 5 new capacity checks |
| `automatic-coat-probe-tests-v1` | 45 distinct checks, including 13 new; native calls mocked |
| `45757-undercoat-admission-native-v4` | Actual local refusal, not an analytic fixture |
| `third-continuation-preflight-v2` | Actual source-bound automatic consumer; no paid call |

These groups overlap in affected tests; do not sum them as unique project coverage.
Large models, scenes, video and private-shaped runtime receipts stay local with
hash references. Credentials, local databases, worker tokens and runtime homes are
excluded. Earlier immutable bundles and all rejected attempts remain intact.

At this milestone, settled API cost is **$6.024623**, retained unknown holds are
**$6.548016**, and conservative exposure is **$12.572639**. The remaining allowance
is **$2.427361** under the unchanged $15 cap; holds are not confirmed bills.
No new Meshy charge, Meta spend or deployment is involved. Any subsequent website
run must preserve full per-call reservations, existing holds and final quality gates.
