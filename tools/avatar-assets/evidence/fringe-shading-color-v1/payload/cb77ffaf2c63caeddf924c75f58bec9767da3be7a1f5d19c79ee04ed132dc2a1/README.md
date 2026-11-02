# Actual color-only fringe candidate — technical gate passed

Model: `model.glb`, SHA256 `8086940637ab6ec2e601448326e5be2845341d6b2933b00ad5dba4ddd40cb8ec`. Source: `../panda-edge-fringe-area-v1/model.glb`, SHA256 `8faa6bc2db80be19b69a62ce214e2e80db32c98fd74c5cb37299c728ad00492d`.

This is a **color-only derivative pending native/visual review**, not accepted avatar quality. It uses the actual original 4,096×4,096 RGB texture and a new 4,096×2,400 per-fin RGBA atlas for the existing 4,800 fins. Every source BIN byte, including padding, and all original accessor/material/image descriptions are preserved. The fin's old vertex-color multiplier is unbound; its new texture already contains the source colors and factor. The source texture and original mesh colors are not whitened or suppressed.

## Gate declared before execution

All9,600 nominal tips at±2.5mm physical offsets were tested with LOD0 maximum absolute linear-RGB component error. The original194-above0.1 baseline reproduced. The fixed thresholds were p95≤0.02, p99≤0.05 and at most48 tips above0.1.

| Metric | Original endpoint colors | New source-color atlas |
|---|---:|---:|
| p95 error | 0.0265525570 | 0.0035721287 |
| p99 error | 0.2195690877 | 0.0042598637 |
| Maximum error | 0.6493915008 | 0.0193071673 |
| Tips above0.05 | 277 | 0 |
| Tips above0.1 | 194 | 0 |

All nominal-tip alphas are effectively1; none was removed from the calculation or hidden below the unchanged0.18 mask cutoff. `nominal-tip-errors.npz` preserves all sampled values; `color-error-audit.json` keeps the worst residuals and fixed gate.

## Exact preservation

- The entire34,939,732-byte original BIN chunk is an unchanged prefix, including its two padding bytes.
- Original base geometry, rig, hierarchy, materials, images and animation descriptions/bytes remain unchanged. Node19, mesh0, skin0 and all19 joint IDs are retained.
- Fin primitive0/7 retains POSITION127, NORMAL128, JOINTS_0131, WEIGHTS_0132 and indices133 unchanged. Its old COLOR_0 accessor130 remains in the file but is unbound. UV0 alone is replaced by appended accessor134.
- Fin geometric normals, doubleSided=true, MASK/0.18, roughness0.92, metalness0 and the original sampler are unchanged. This is separate from the source-normal-only lighting control.
- The exact JSON allowlist and binary prefix passed. Standard geometry/embedded-image preflight and surface-attribute checks passed. CPU linear-blend skinning observed the unchanged `dance_flight` motion at33 samples using124 probe vertices. This is not a new97-frame collision/deformation audit or native render.

## Bound and remaining limits

One CPU build completed in1.278 seconds with sampled peak RSS368,738,304B under the declared30s/512MiB/64MiB envelope. The actual4K source was decoded. Exit0 and owned-process exit were verified; no retry or live child remains. No native, browser, provider or AI-role call occurred.

LOD0 sampling is not exact texture equivalence, GPU/mip proof, lighting correction or visual acceptance. Unique-cell alpha resampling differs from the original silhouette; the analytic predecessor measured small width changes. Deep mip isolation, actual antialiasing and appearance remain unverified. Original geometric-fin lighting, phase-dependent tip merging, intersections and inherited topology limits remain. No source-normal changes or original markings were hidden to pass the color gate.

Editable/reproducible sources are pinned in `../fringe-color-candidate-preparation-v1/`: exact helper, sampler, builder, validators, configuration and immutable input hashes. Geometry/rights/provenance remain inherited through the pinned original manifest; this derivative performs no new rights verification. This model is not newly declared Weft-import-compatible.

The parent session owns the next independent native preview. Do not treat this numeric result as appearance approval.
