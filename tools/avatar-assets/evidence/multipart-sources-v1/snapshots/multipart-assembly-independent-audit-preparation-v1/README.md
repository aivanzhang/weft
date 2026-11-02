# Independent multipart assembly audit — prepared, not run

This checker does not import the assembler or trust its success flag. It independently checks the frozen robot/dragon config against the future exact output GLB. No actual assembled model has been inspected here. Source inventory is not rerun.

## Frozen checks

- Complete source primitive/row/triangle and namespaced-material coverage; no extra mesh, node, skin, clip or rendered material.
- Output positions follow `t + (source_meters_per_unit * uniform_scale) R(q) p` within 1e-6 m, without implicit recentering. Quaternions are explicitly normalized within the producer's 1e-6 input norm bound.
- Normal/tangent XYZ rotate only (2e-7 component bound); identity-rotation frames, tangent W, UV/index rows and embedded images are byte-exact. No source normal normalization is permitted by the contract.
- Full PBR/UV0/normal/MR/alpha/culling/sampler semantics survive index remapping; no added unused textures or images.
- Output and aggregate source sizes remain at most 64 MiB; 100k vertices/triangles; 16 Mi-pixels per image and 96 Mi-pixels in aggregate. Image headers only, not full pixel decode.
- Canonical rig receipt is recomputed from actual output positions: `[x,-z,y]`, subtract `[midX,midY,minZ]`, then height-scale. It must say **not applied**. This is a predicted future transform, not a created rig or native verification.

The source/config snapshots, scalar gates and script hash are in `prepared.json`. Only `ast.parse` ran. Candidate and receipt hashes must come from the released actual output; they are intentionally absent now. Root owns the output/native lane and releases the CPU audit. Use a fresh results directory and the existing bounded runner: **30 s / 512 MiB / 16 MiB**.

Prepared argv (not executed):

```text
.venv/bin/python outputs/multipart-assembly-independent-audit-preparation-v1/audit_assembly.py --model outputs/robot-dragon-source-assembly-v1/model.glb --expected-sha256 RELEASED_MODEL_SHA256 --receipt outputs/robot-dragon-source-assembly-v1/worker-result.json --receipt-sha256 RELEASED_RECEIPT_SHA256 --out NEW_RESULTS_DIRECTORY
```

No automatic retry or data repair is included. A failure retains measured partial evidence and its traceback. This exact two-source fixture does not establish arbitrary shared-material primitive correspondence, appearance, watertightness, rigging, motion, shoulder-dragon identity or rights verification.
