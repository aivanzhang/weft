# One actual color-only fringe candidate

Source: `outputs/panda-edge-fringe-area-v1/model.glb`, SHA256 `8faa6bc2db80be19b69a62ce214e2e80db32c98fd74c5cb37299c728ad00492d`. The exact model, correspondence, original two-tip atlas, build report, rig contract and source manifest are hash-pinned. Output: `outputs/panda-fringe-color-atlas-v1`.

This job uses the unchanged verified color-atlas helper. It reads actual mesh0 primitive7 fins and the original opaque UV0 source texture on mesh0 primitive5. All source binary bytes (including original BIN padding), original arrays/descriptors, base geometry, rig, animation and existing fin positions/normals/indices/JOINTS/WEIGHTS are retained exactly. It appends one atlas, one UV0 accessor and an equivalent MASK material. The old fin COLOR_0 becomes unbound to avoid double tint. Double-sided rendering, alpha cutoff0.18, roughness0.92, metalness0, sampler and geometric normals remain unchanged. This is not the separate source-normal lighting experiment.

## Fixed technical gate — declared before execution

Use all9,600 nominal tips at physical offsets±2.5mm. Error is the maximum absolute linear-RGB component relative to true source-texture LOD0 sampling. Reproduce the existing194 tips above0.1 baseline, then require:

- p95 error≤0.02;
- p99 error≤0.05;
- at most48 tips with error>0.1.

Keep all residuals, worst examples and alpha statistics. The gate does not imply exact texture/GPU/mip/appearance equivalence. Unique-cell alpha resampling is not byte-identical to the old silhouette. Source colors are neither whitened nor suppressed.

## Bounded execution

```sh
.venv/bin/python -B outputs/fringe-color-candidate-preparation-v1/execute.py
```

One authorized CPU process:30 seconds,512MiB sampled RSS,64MiB output, one numerical-library thread; no retry or native/provider/AI-role call. Stream the input with a read-only mmap and write its BIN prefix in1MiB chunks. Decode one RGB source image at≤4096²/16MP, retain uint8 pixels, and use128-fin batches. Free source/atlas/image arrays before the standard preflight, surface-frame and actual skin/motion validators. Check exact serialized JSON allowlist and BIN-prefix equality.

A numeric gate failure still retains the derived model and audit, explicitly rejected; a structural/process failure retains its partial evidence without promotion. The parent owns any native preview and visual acceptance. No production or Git edits occur here.
