# Saved multiscale action proposal

## Result — November 1, 2026, 1:02 PM PST

**Completed negative, saved data only.** Combining the earlier improved center
from the `.1` first round with the measured `.025` local model produces no
bounded candidate. The unchanged solver returns `row_box_conflict` for
`sam-2:speed_model_units_s` before any sweep. No action proposal or native trial
was executed.

Owner: `/root/equipment_control_review`. Root owns integration, publication and
any future experiment. Scratch: `.local/equipment-action-multiscale-proposal-01/`.
No active process remains. Production code, prior evidence and task/scorer
settings were not edited; Git was not inspected or changed by this lane.

## Fixed question and computation

The preceding fixed-center diagnostic improved all three saved candidate
prediction errors. A subsequent `.025` two-round trial changed its first-round
center and abstained. This check asks only whether the already measured smaller
model works at the retained earlier center; it does not repeat either trial.

The guarded worker read exactly three files once: the old sealed projection,
the sealed smaller-step diagnostic, and the frozen helper module. It verified
their hashes, identical center action/values and all 12 constraints, all 19
single-joint probe changes, actual dtype deltas, original-base bounds and root bytes.
Reconstructing the stored normalized derivative matrix from those measurements
matched it exactly.

Only `_linear_projection` and `_bounded_trial` were AST-extracted from the
frozen module. No Weft or MuJoCo module was imported. The one solver call used:

- Original 19 joint variables and original-base `[-.25, .25]` correction box,
  intersected with action limits `[-1, 1]`; unchanged float64 roots.
- Old first-round center at evaluation 21, with normalized violation
  `1.1233806575431395`.
- The existing `.025` measured derivative matrix and original constraint limits.
- The existing normalized target margin `1e-6`.
- At most **62 remaining sweeps**, preserving the old first round's two sweeps
  within the original total of 64.
- At most three fixed candidate scales `1, .5, .25`, only if the solver returned
  a proposal. It did not, so zero candidates were constructed.

The conflicting speed row has center value `0.30479411029967923`. Its analytical
minimum over the complete original action box is **0.16876499023083216** model
units/s, above the unchanged limit **0.15** and margin target **0.14999985**.
The normalized row/box conflict excess is `0.12510093487221452`. This is a
property of the local linear model, **not physical infeasibility**.

## Exact evidence

Input artifacts:

- `artifacts/equipment-action-relinearization-native-02/projection.json`:
  `04ea0163c6de37c719c51fb89b2b8578501610781d3ab3a6522c82739448374c`.
- `artifacts/equipment-action-local-scale-native-01/diagnostic.json`:
  `ba92c592a3a571d052a488b8b71fb98b9bfce492eaa1e882d91b5bbb40cd721f`.
- Frozen `shared_exercise_action_projection.py` in
  `.local/equipment-action-relinearization-native-01/runtime-02/src/weft/`:
  `1a5182e83c6a4c8981a5ef30e64c7154878e8ae46fec78060054507098894338`.

Derived evidence under this lane's scratch directory:

| File | SHA256 |
| --- | --- |
| `plan-01.json` | `0ab93143073b63906940d22d757399005b10ec04f6ea423199b048d9d4096df6` |
| `worker-01.py` | `384e7c1b400c1df1724cf191a972fa30c8e2b3dfd020b1362cbc9dd257be0b30` |
| `inputs-01.json` | `b1a897b1437b6662f5e48205bdae530c3faa0ba0e8ce63ce6a6bb157cb4bb0f8` |
| `result-01.json` | `02afbd19a88bee40195abaa267b8ab432e7606ef42a760cb9e072f6bbad5abc0` |
| `process-01.json` | `7d15f2129fb2ddb09c4c563b588cea29bcc8e0a2cf36a91155aabc4e6cc97a37` |
| `constraint-conflict-01.json` | `e2a256746fd768400cf1f7e031ca495dfa5a9ff505431e5bb3cf195ed63b5533` |

Action/model identities below use SHA256 of sorted compact JSON with
`allow_nan=False`, explicitly distinct from the project's spaced JSON hash:

- Original base action:
  `0d5525ea48c42598ed77522e023f6f993520e605e80f688ca6c93e8e7672006e`.
- Retained center action:
  `0db82b33c38ffa1eff9d7017185aa2844d0a387ccc8c48437860350b30bc7f9f`.
- Complete local model, bounds and settings:
  `29e44165061bbb6f2c818968b296803425cc27d74bad281f64c8cec7e579415c`.
- New action/model-prediction hashes: **none**, because no proposal exists.
  Do not manufacture predicted endpoints or measured outcomes.

## Work bounds and scope

The process exited zero: one solver call, zero sweeps, zero proposals; three
input reads totaling 160,985 bytes. Its guarded analysis section took
0.028240 s, process CPU was 0.091048 s, and peak RSS was 37,945,344 bytes.
Bounds were five CPU/cooperative seconds, a 256 MiB observed-RSS ceiling and
1 MiB per output file. CPU/file limits were enforced; memory/wall observations
are not claimed as hard isolation. No blocked-operation attempts, network,
physics, API calls, new measurements or external cost occurred.

The separate conflict detail only evaluates the minimum of that one saved
linear row over its box. It is not a second solve or parameter search. The
worker/plan preserve the exact command; do not rerun unchanged evidence.

## Next action and completion gate

This saved-only question is complete. **A new native trial of this proposed
combination is not justified:** there is no returned candidate to verify.
Root can checkpoint this result and continue independent product work. Any
different modeling hypothesis needs its own evidence and bounded review; this
result does not authorize larger action bounds, changed gates, more sweeps,
another scale search or a native run.

No whole-task, generalized reconstruction, controller promotion, contact
recovery or physical-accuracy claim follows from this analysis.
