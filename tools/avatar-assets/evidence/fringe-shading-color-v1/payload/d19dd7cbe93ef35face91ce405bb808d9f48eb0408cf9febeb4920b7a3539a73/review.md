# Fur ownership review

## Result

No exported fur assignment leak was found on the explicitly declared `shoulder-dragon` part. This does **not** explain or approve the visible light flecks.

The authoritative finish config excludes that part by its exact `companion_surface`/`companion_cap` materials. Its declared `CompanionRoot` subtree contains `DragonWingRight` and `DragonWingLeft`; the exported joint parent graph matches the authored graph exactly. Actual skin slots are 16–18. Ownership below uses **every stored weight greater than zero**, with no color, location, or majority-weight guess.

| Measured surface | Result |
| --- | --- |
| 4,800 fin roots, 9,600 endpoints | Zero positive companion weight |
| All 19,200 fin vertices | Zero positive companion weight; exact normalized source-weight inheritance |
| Both authored undercoat images | Exact embedded PNG hashes match the recorded bakes; only body primitives 5/6, 31,721 triangles and 23,023 exported vertices |
| Undercoat vertices | Zero positive companion weight |
| Companion surface/cap | 27,518 triangles; all 25,644 exported vertices exclusively companion-weighted, no mixed body weights |
| Companion materials | No authored undercoat image; surface retains inherited normal image 0, cap has no normal image |

Fin root positions and source UV correspondence were also verified against primitive 5. The derivative and preview are bound to model SHA `1204ae9d53b12bdc34ea74288cd2b63fa61df45d80665d5c6987d2072b4fac75`. Recorded part material face counts remain exact.

## Interpretation and next decision

A wider fur exclusion or a part-ownership repair is **not justified** by this evidence. The existing exact-material exclusion and descendant-only part weights agree with the exported result. Keep these ownership invariants in any later representation change; do not replace them with species coordinates or a green-pixel mask.

The remaining flecks could involve inherited surface appearance, shading, or projected/intersecting geometry; those causes were not measured here. No texture was decoded or recolored, and no lighting, collision, motion, or visual-quality result is claimed. Authored part ownership is not independent proof of anatomical segmentation. The native groom report records 25,642 excluded native vertices, but native particle density was not reopened. Exported vertices include UV/normal seam copies and must not be equated to native vertex counts.

## Reproduction and limits

From `tools/avatar-assets`, the preserved one-run command was:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python outputs/fringe-part-ownership-review-v1/execute_cpu.py
```

Do not rerun into this sealed folder. `prepared.json` pins 17 inputs and six source files. `ownership.json` contains the complete per-primitive counts and actual normal-image hashes; `manifest.json` seals the measured run. This note is separately sealed by `review-receipt.json`.

One CPU child completed in 0.263 s under 15 s / 256 MiB / 8 MiB bounds; PID 19845 exited successfully. The worker's measurement took 0.026677 s. RSS was **not sampled** before exit (`null`, not zero); 53,936 output bytes were present at job completion. No native, browser, provider, model, or texture-decoding calls occurred. Existing inputs and all previous outputs remained unchanged. No owned job is running. No quality promotion or production change was made.
