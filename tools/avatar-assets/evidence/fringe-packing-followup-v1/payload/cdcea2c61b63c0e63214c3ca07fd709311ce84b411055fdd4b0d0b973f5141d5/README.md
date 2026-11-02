# Independent exported packing audit

**Passed against the actual exported GLB**, not just the producer's in-memory arrays.

- Source SHA256: `52096c6a29dd3bf40f59ab76c63f9b5b5307d6efb4441fde2fff5f7ee5461c85`.
- Candidate SHA256: `502b92030c037e8e97a95befc16a0809473e4dbb6f567a8a1b13d3e788c9e304`.
- Exact original BIN prefix: **36,897,372 bytes**, including its padding. Original geometry, source-surface fin normals, indices, skin, inverse binds, hierarchy and actions are byte/descriptor-preserved.
- All **4,800 RGBA cells** match their original cells byte-for-byte after the recorded permutation. Embedded packed PNG matches the published PNG; PNG color-space metadata is unchanged.
- Exported UV bytes match an independently recomputed float32 mapping. Maximum rounding from float64 is2.960364e−8.
- The source combined material remains single-sided (`doubleSided=false`), MASK/0.18 with unchanged roughness, metalness and sampler. This is not the earlier double-sided color-only source.

## Nominal-tip gate

Using an independent CLAMP/linear-light sampler on the exported atlas and UVs, all9,600 nominal tips pass the fixed p95≤0.02, p99≤0.05 and at-most48 errors>0.1 gate:

| Metric | Observed |
|---|---:|
|p95 maximum-component error|0.0035721287|
|p99 maximum-component error|0.0042598637|
|Maximum error|0.0193071673|
|Errors above0.1|0|
|Maximum color change from original atlas at LOD0|2.598e−14|

Source RGB reference values are reused from the pinned prior actual4K-source audit; the original4K image was not decoded again. `tip-check.npz` retains source/original-atlas/exported-atlas RGB and errors.

## Preserved checker failure

V1 failed its strict JSON allowlist because it omitted the intentional `asset.generator` provenance-label change. A read-only JSON diff found that exact field and no other unanticipated mutation. V2 permits **only the exact observed old/new labels**, records the transition and retains every other preservation/color check unchanged. V1's code, failure, traceback and job receipt remain unchanged; their hashes are pinned in `prepared.json`. `checker-only.diff` records the narrow correction. No model was repaired or modified for this check.

V1 took0.257s; v2 took0.254s, for0.511s combined within the original15-second envelope. Both owned process exits were verified. Both completed before an RSS sample was captured: **actual peak memory was not measured**. V2's configured monitor cap was512MiB, with14.743s remaining and16MiB output. No extra run was performed just to obtain a memory sample.

This audit does not rerun motion, native rendering, intersections or GPU mip/alpha/visibility behavior. Preserved rig/actions establish unchanged data, not new visual-quality acceptance. Parent owns the matched native preview and its separate visual review. No production/docs/Git/model edits or native/provider/AI calls occurred here; no live child remains.
