# Equipment-feedback feature timing

## Result

**The failing object was selected, and the nonzero controller was active from the first action.** This diagnostic rejects wrong-object selection and late feature activation as explanations for these two nominal traces. It does not establish that pelvis-center features describe limb clearance adequately, identify physical contact onset, or prove which coefficient caused a failure.

Root ran one saved-data diagnostic on November 1, 2026, finishing at **4:25:07 PM PST**. PID `91492` exited 0 and was reaped; no job remains live. The published policy/training files at `566258724d20938c138067e174f2a8670769ca2d` are unchanged.

## Exact scope

The predeclared inputs were candidate **0** (unchanged v5) and **2** (best nonzero full-task rank), both original nominal seed `1263521311`. Selection preceded journal inspection. Each journal contains 254 action/result pairs. Their 2,497,605 bytes were read once with streaming hash verification; total artifact input including plan/script was **2,518,725 bytes**, below 3 MiB. No evaluation/holdout, source/runtime import, native simulation, replay, source proof, API or tuning occurred.

Root's external process took **0.072052 seconds**, 0.040372 CPU seconds and **22,642,688 bytes** peak RSS. The report is 59,100 bytes, below 128 KiB. This interpretation read only the report and terminal process receipt (**59,972 bytes**); it did not read either journal again.

## Findings

- **Both traces selected `sam-2` on all 254 intents.** `sam-3` was never selected. Proximity and nonzero features were present on every intent, starting at simulation time 0: center distance 0.790122 model units, proximity 0.473252.
- Candidate 0 correctly applied no residual. Candidate 2 requested and applied both joint and root residuals from **control 1**, including every one of the **49 completed controls before its first failed result**. Before failure, maximum applied joint/root corrections were **0.098486 / 0.063445** normalized action units.
- Candidate 2's first requested vertical-root correction was 0.066214 but its applied correction was zero; horizontal root corrections did apply. At its first failed interval, all three requested root corrections applied. This is evidence of a particular bounded-action clipping event, **not** persistent lack of root authority or permission to widen it.

| First accumulated gate crossing | Unchanged v5 | Candidate 2 |
| --- | --- | --- |
| Equipment speed, limit 0.15 | Control 49: 0.413569 | Control 50: 0.601720 |
| Equipment spin, limit 0.75 rad/s | Control 49: 3.642698 | Control 50: 5.387794 |
| Equipment orientation, limit 0.15 rad | Control 50: 0.184274 | Control 50: 0.150563 |
| Equipment displacement, limit 0.03 | Control 51: 0.040012 | Control 51: 0.035271 |

The anchor's first failed returned boundary brackets the crossing in **1.600000–1.633333 simulation seconds**; candidate 2's brackets it in **1.633333–1.666667 seconds**. The intent refers to the interval's input state, while the returned value is an accumulated substep peak. A one-control later threshold crossing is not an exact contact-time measurement or a task improvement: candidate 2's initial speed/spin crossing was larger, and its full-task worst excess remained worse than the anchor (9.126146 vs 6.499293).

## Evidence identities

Scratch and immutable outputs: `.local/equipment-feedback-feature-timing-01/`.

| File | SHA256 |
| --- | --- |
| `plan-01.json` | `c8805b34a0b7912196765e4e235cf8f0302aa35c5bc4b7ea2c925ae6d6dc0356` |
| `diagnose.py` | `5e894248e968b7dcb39fe310321279196176cb78e4c01da8bc1920222243be7e` |
| `report-01.json` | `3109835ed2987711294566de679981750c39be860629afcc47e1026252a66e6e` |
| `process-01.json` | `2afdb44a5ff6f264ce0d6ea25cd5a669236492916876543af8adfc37da7c518d` |
| Anchor nominal journal | `622f75b47584ca6c769f876beabc2ffee0a7008206e127bfb0892873d2ec7a2c` |
| Candidate-2 nominal journal | `f721fa4a206548ada8629bdef5903ef244cbbb481e8dc520200ba306ed66806a` |

Original task outcomes remain in the [training-native handoff](equipment-feedback-training-native.md). This diagnostic neither changes nor independently replays those outcomes.

## Next bounded hypothesis — preparation only

Root authorized preparing a **fixed signed root-only intervention**: tie the three root proximity/away gains to +0.08 for one candidate and −0.08 for the other; set every joint and closing-speed gain to zero. Test each over the same two original complete starts, at most **4 episodes / 1,016 controls / 16,256 substeps**, against the already saved anchor metrics and matched initial hashes. No extra baseline or parameter sweep is needed.

This isolates a reusable root-only direction from the previous simultaneous joint/root changes after ruling out wrong-object and late-activation explanations. It is not CEM training, a selected learned policy or proof of contact avoidance. Preparation must reuse the frozen runtime and lossless journal machinery without editing the CEM sampler, task, source, scorer or public defaults. Root must review the exact resource/verification plan before execution; no intervention has run as part of this diagnostic.
