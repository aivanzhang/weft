# Imported glTF culling control

**Cycles did not honor single-sided culling in this fixture; EEVEE did.** Imported material flags were correct and were not edited.

| Probe | Front single | Reversed single | Double-sided | Opposed duplicates |
| --- | ---: | ---: | ---: | ---: |
| Expected front | 81 | 0 | 81 | 81 |
| Cycles front | 81 | 81 | 81 | 81 |
| EEVEE front | 81 | 0 | 81 | 81 |
| Expected back | 0 | 81 | 81 | 81 |
| Cycles back | 81 | 81 | 81 | 81 |
| EEVEE back | 0 | 81 | 81 | 81 |

Counts are opaque pixels in each9x9 interior region, not whole-object coverage.

The duplicate case uses **the same** tilted glTF normal `[.8,0,.6]` on both opposite windings, transformed to approximately `[.8,-.6,0]` in Blender. Imported corner normals preserve that direction. EEVEE Normal-pass values are approximately `[.7998,-.5996,0]` in both views. Cycles instead reports `[-.9958,.09165,0]` in front and its opposite at the back; the region is near black in both Cycles views. The cause inside Cycles is not isolated by this control.

EEVEE's duplicate front is lit; its back is near black with the camera-following sun because the shared authored normal remains fixed toward the other side. The ordinary double-sided case reverses Normal-pass direction and is lit on both sides. These observations do not approve the duplicate-normal avatar workaround.

One native invocation completed: **6.240s**, sampled peak **340,951,040B**, job-end output **134,621B**. Blender4.5.13LTS,4threads,128px,4samples for each engine. PID17098 exited. All four beauty PNGs, float EXR Normal passes, imported properties and per-region checkpoints are retained. Original unexecuted fixture retained; only `fixture-v2.glb` ran.

No production edits, avatar renders, new provider/AI calls, or quality promotion. The separate avatar derivative remains root-owned. This execution manifest is preserved unchanged; `findings-receipt.json` binds these append-only findings.
