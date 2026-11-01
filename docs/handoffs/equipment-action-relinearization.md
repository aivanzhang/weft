# Bounded two-round action refinement

Updated **2026-11-01 (PST)**. Implementation owner: `/root/contact_reconstruction`; independent reviewer: `/root/equipment_control_review`; Git/execution owner: `/root`. **Implementation/review and the bounded native experiment are complete. The native run made 45 predictions, reproduced the exact 49-control prefix, then abstained before control 50; it did not establish controller success.** See the [native receipt](equipment-action-relinearization-native.md) and [saved diagnosis](equipment-action-relinearization-diagnosis.md). Earlier stopped rollouts and all original pure-check receipts remain unchanged.

## Evidence and hypothesis

[Rollout02](equipment-action-rollout-native.md) stopped before control 50 because no measured candidate passed the unchanged equipment gates. Its best measured half-scale candidate reduced worst normalized violation from **2.82345147 to 1.12338066**; the best finite difference was **1.16716968**. That is a real local improvement, not a passing action or proof of eventual feasibility. The half-scale action differs from the original base by at most **.07263760**, within the original `.25` joint bound.

The [signed-velocity diagnostic](equipment-action-velocity-probe-native.md) and earlier saved secant check did not support adopting either simple response-model replacement. This new hypothesis instead spends bounded additional measurements to build a **fresh complete local finite-difference model around an actually improved action**. It does not reuse a rank-one secant update, fit signed velocity vectors, choose clip-specific joints, change gates, or claim general reconstruction accuracy.

## Explicit opt-in contract

```python
config = ProjectionConfig(
    candidate_strategy="two_round_relinearization",
    maximum_predictions=48,
    maximum_seconds=5.,
    difference=.1,
    trust_radius=.25,
    maximum_sweeps=64,
)
projector = prepare_action_projector(env, config=config)
action, diagnostic = projector.project(base_action)
```

Default `candidate_strategy="single_round"` keeps the existing 24-prediction maximum and search/report behavior. Default reports gain no refinement/configuration/round fields. Legacy configurations without the selector still use the default; explicit new configurations round-trip through the dataclass fields.

Round 1 retains the original search and passing-action selection. If it already finds any measured passing action, it repeats that action under the original acceptance rule and never spends work on round 2. This preserves the original iteration-limit proposal path, including a passing first-round result after all 64 solver sweeps have been spent. The native experiment reconstructed control 49 from its saved selected action; it did not rerun the projector at control 49.

If round 1 has no passing action, rank **actually measured** failing evaluations by:

1. `V = max(0, max(values / original_limits) - 1)` across every original constraint.
2. Squared joint-action distance from the **original base**.
3. Encounter order.

Round 2 requires a strict reduction in `V`, at least **23 remaining prediction slots**, and at least one remaining solver sweep. Its center reuses the selected measured action/values; it does not predict the center again. All 19 coordinates receive fresh one-sided differences. Choose the side with more room in the original base's feasible box; ties use the legacy toward-zero sign. Clip the probe size to available room, round inward in the original dtype, and divide by the **actual nonzero representable change**. Every other coordinate remains byte-identical to the center. A zero/unavailable difference aborts honestly.

The new local solve and its fixed `1 / .5 / .25` candidates remain inside **original-base ±.25 intersected with [-1,1]**, not a newly centered trust region. Original root bytes and dtype remain unchanged. Actual nonlinear measurement plus exact full endpoint/substep-contact repeat remain mandatory before returning an action. Passing finite differences survive a later local model conflict. There is no fallback, live-state advance, changed task/scorer/source partition, or automatic public/controller routing.

## Aggregate bounds and diagnostics

- One decision, one predictor lifetime and one scratch data object. The same five-second outer deadline spans both rounds, measurement, solve, repeat and final encoding.
- At most **48 predictions/copies/forwards / 768 detached substeps**. The algorithm's maximum is **46**: failed round 1 uses `1 + 19 + 3 = 23`; round 2 reuses its center and uses `19 + 3 + 1 repeat = 23`. Both rounds failing use at most 45. Spare cap slots authorize no extra probes.
- **64 actual aggregate solver sweeps**, counted when each sweep begins; round 2 receives only the remaining budget. Sweep exhaustion suppresses refinement before spending its finite differences.
- Original **2 MiB report cap**, original physical limits/tolerances, normalized action bounds and caller ownership requirements remain.

Opt-in diagnostics retain all ordinary evaluation rows and the original first-round `linear_model`. A `refinement` summary records center source indices/actions/violations, strict improvement, per-round predictions/sweeps, completion and the actual stop reason. Each evaluation has a round tag. An unavailable prediction retains a bounded attempted action/kind/round/details record and whether prediction work was attempted. Partial earlier measurements remain visible.

On opt-in output-cap failure, no action is returned. Drop the unaccepted full endpoint first and retain measured diagnostics if they fit. If even those exceed the hard cap, explicitly report omitted evaluation/round counts; do not silently imply complete evidence. The default strategy retains its previous byte-cap failure report exactly. Time/state failures also replace stale refinement stop reasons with the actual failure.

## Changed files and verified release pins

| File | SHA256 |
| --- | --- |
| `src/weft/shared_exercise_action_projection.py` | `1a5182e83c6a4c8981a5ef30e64c7154878e8ae46fec78060054507098894338` |
| `tests/test_shared_exercise_action_projection.py` | `4497aa21d33d7d840e9000ddbf99cc034eb4497fd7c9f059dd3c074ece357af6` |

This handoff is the third owned file. Predictor `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd`, task/scorer, source evidence, policy closure and prior frozen runtimes are unchanged. No Git operations occurred in this lane; root owns release integration and preserves unrelated work.

**47 distinct scoped checks pass:** 27 new-strategy cases, 12 exact comparisons against the immutable `04510c0…` default helper, and eight affected legacy configuration/byte-cap cases. Unaffected prior 44-case evidence was reused, not counted again. The direct comparator checks complete reports, actions, byte identity, call order and budgets under identical mocked clocks/predictions, including first-round iteration-limit passing/failing paths and output-cap failure.

Coverage includes fresh 19-coordinate relinearization, deterministic measured-center ordering, strict improvement, 23-versus-22-slot admission, exhausted/remaining sweep budgets, passing-difference retention, float32 inward rounding/actual deltas, original-base candidate clipping, exact repeat of endpoint/contact data, timeout/unavailable/live-state failure evidence, configuration round-trip and bounded diagnostics retention/explicit omission.

All checks use authored/fake predictions under a guarded analytical runner. Native physics, media, network and subprocess execution are denied. The first runner attempt stopped **before test collection** because its audit exception type was not caught by MuJoCo's optional import-time OS probe. That failure and original runner are preserved. The corrected guard still blocks that exact probe with `PermissionError`; the installed library handles it. No child or simulation ran, and no guard was broadened to allow the probe.

Scratch: `.local/equipment-action-relinearization-01/`:

- `process-02.json`: 39 passes, **.653245 s**, **117,768,192 bytes RSS**, zero native attempts; SHA `32a7e0fc57daafc9bc99fa273f23dab244176770c9eaa76c4f80a150e11b063c`.
- `tests-02.xml`: `c6702844727bf9b394c9ad8e293e901c07d4376c7008b1a3fbd787e2ae6755f6`.
- After the narrow opt-in output-cap change, `process-03.json`: six affected/new passes, **.695546 s**, **118,046,720 bytes RSS**, zero native attempts; SHA `475a4ea869cff6e0b3c1be8f9f2a7f63d2bef48567996b17cbb0bc3fd07c8573`.
- `tests-03.xml`: `595b633805f9d79df61968157a2d72fad2f40c54b0f21686ed216f5d407c31fc`.
- Final narrow float32-boundary change, `process-04.json`: two new passes, **.562333 s**, **117,833,728 bytes RSS**, zero native attempts; SHA `2d22e39f6d8337aa9fff3a69013ab24c49c2d445d017478f0bd9efc18edb2682`.
- `tests-04.xml`: `196f534902ffd22f7a85811ee833e6c1ea9c948767525cf2f9ddfcc8adbc7347`.

The first receipt covers unchanged search paths; the second covers the output-cap delta and added bound/config cases. A final self-check found that a NumPy float32 scalar comparison could round a Python bound down at nondefault difference `.25`, causing a safe but unnecessary abstention. The two-line fix explicitly compares Python floats before rounding inward; the third receipt checks that exact face for both dtypes. The default strategy and current `.1` native proposal are unchanged. The harness owner preserved its superseded `e8342cb0…` preparation as unexecuted and will use a distinct runtime for the final pin.

These are scoped pure checks, not native performance or full-episode controller evidence. The reviewer inspected but did not rerun them, closed both failure-evidence findings, and verified that reversing the final two-line rounding fix reproduces the earlier reviewed source hash exactly. Review is closed on the final release pins above.

## Historical one-decision proposal — now executed, not a full rollout

`/root/reconstruction_next_step` owns `.local/equipment-action-relinearization-native-01/` preparation. Reuse Native02's 293-file runtime and replace **only this reviewed projector**; preserve the predictor and all 25 original policy dependency pins. Reconstruct the exact pre-50 state with the 49 saved Rollout02 selected actions; this prefix prepares the input and is not another task qualification. Invoke one opt-in decision. If it returns a verified action, perform at most one actual control 50 and require exact endpoint/contact equality, unchanged base roots and original actual task metrics. Abstention means no selected step or fallback.

Proposed outer maximums: **1,568 substeps** (`784 prefix + 768 scratch + 16 selected`), **100 forwards**, 11 models, two data objects, one reset, 48 copies, 50 actual controls; retain 40-second CPU / 60-second external / 512 MiB RSS, 2 MiB result / 4 MiB total and existing output guards. The algorithm should use fewer than that ceiling. This is at most twice the bounded local prediction allowance, with estimated external and recurring cost **$0**; it does **not** authorize a full episode or new tuning sweep.

Root reviewed/froze the concrete harness and executed the changed-input experiment. The [native handoff](equipment-action-relinearization-native.md) preserves the initial observer-permission stop and completed result; neither is overwritten. The completed run found no verified candidate after two fresh models, performed no actual control 50, and kept the live state/task unchanged. The historical proposal and limits above remain provenance, not a new execution request.

Next work is the fixed-center, smaller-difference diagnostic in the [saved diagnosis](equipment-action-relinearization-diagnosis.md). Root has separately assigned its preparation; execution still requires reviewed frozen bounds. This lane has no live handle or queued native run. Do not repeat failed inputs, promote the helper, or claim independent reconstruction/controller validation from the current evidence.
