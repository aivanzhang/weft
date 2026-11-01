# Detached fixed-action sequence prediction

## Current state

Implemented a bounded open-loop sequence predictor; **56 guarded mocked checks
pass**. Root and independent review are closed on the pins below. A separate
[native two-action qualification](equipment-sequence-prediction-native.md) now
passes exact endpoint/contact comparison and preserves the actual equipment
failure at control 50. This is a prerequisite for pre-contact look-ahead, not a controller, policy
callback, optimizer, task-success result or public/default sandbox mode.

Owned files: `src/weft/shared_exercise_sequence_prediction.py`,
`tests/test_shared_exercise_sequence_prediction.py`, this note and
`.local/equipment-sequence-prediction-01/`. Root owns Git integration and native
qualification. Other policy, task, predictor, projector and avatar files remain
untouched. No live process is owned by this lane; external spend is $0.

## Contract

```python
from weft.shared_exercise_sequence_prediction import prepare_action_sequence_predictor

predictor = prepare_action_sequence_predictor(
    env, maximum_sequences=4, maximum_controls=8,
    maximum_sequence_controls=4, maximum_seconds=2.,
)
report = predictor.predict([first_fixed_action, second_fixed_action])
```

- Accept only a materialized list/tuple or 2-D numeric array containing 1–4 fixed
  action vectors. Copy and validate every vector before native work. No generator,
  callback or policy is invoked. Preserve the original human-only action contract.
- Hard limits: 64 sequence admissions, 64 total reserved controls / at most
  1,024 native substeps, four controls per sequence and the original helper's
  10-second active-compute ceiling. Defaults above are narrower. Reject a request
  exceeding either remaining aggregate controls or the shorter live engine/task
  horizon. Never silently shorten it.
- Reuse the frozen `ActionPredictor` constructor for model/action-contract/motor/
  timing validation, its full-state identity and callback helpers, and the same
  simulator contact/instability methods. Its embedded per-action numerical check
  is reproduced without calling its one-control integration path; explicit lower/
  upper comparisons avoid signed-integer absolute-value overflow.
- One lazily allocated scratch data object per helper; copy the live state once
  per admitted sequence. Retain the complete boundary-forward state between
  controls. Each control clears applied forces, applies the original motor map,
  advances the fixed substeps and performs one boundary forward.
- Reserve all requested controls before any native work. Partial failure does not
  refund that reservation. Count attempted and returned native calls separately.
  Recheck engine/model/data, timing, motor map, callbacks and horizon throughout.
  The helper lock prevents overlapping predictions; callers must still serialize
  live environment commands and model changes.
- Do not advance live state, task accumulators, reward or termination. The report
  does not forecast task completion; future fixed actions are not recomputed from
  hypothetical observations.

## Result and failure evidence

Successful reports have `status="predicted"`, `sequence_complete=True` and a final
`endpoint`. `controls` contains one record per requested control: index, fixed
human action, initial time, completion flag, attempted/returned substep and forward
counts, full `endpoint` and `substep_contacts`. Endpoint fields and contact-clock
semantics exactly match the existing one-control predictor. The top endpoint is
the final control's endpoint.

Reports include total attempted/returned allocations, copies, substeps and forwards;
current-call `controls_reserved/attempted/completed`; lifetime `sequences_used` and
`controls_used`; elapsed active time and full live-state identity evidence.
On failure, the top result is unavailable, `sequence_complete=False` and the final
endpoint is absent. Completed prefix and partial-control records remain evidence,
not an accepted shorter forecast. The **entire report** has a 2 MiB bound. If
details exceed it or cannot be serialized, the unavailable result explicitly counts
omitted endpoints/contact samples and retains compact action/call evidence; it
never silently presents a truncated sequence as complete.

## Verification and pins

First focused run: **56/56** pass, 0.416651 seconds, 111,476,736-byte peak RSS.
The reused guard blocks native physics/media functions, network and child execution;
zero forbidden native attempts. One import-time platform probe was blocked before
execution. Tests use the existing single-action fixture; they do not claim native
parity. Coverage includes continued warm-start/boundary state, first-control parity
with the old helper under mocks, copied inputs, detached outputs, hard aggregate
limits, insufficient engine/task horizons, mid-sequence engine/model changes,
callbacks, time including encoding, nonrefundable partial failures and whole-report
overflow. Old passing suites were not rerun.

- Source: `ad77e00ee1b2f8c8cd5dd91540d5a93afb08892ba103d818483b0f92e7f3f103`.
- Tests: `124d77c4d3506303e1f04ac843313014102ef70fa36a8e3e31ec83f0c59be16c`.
- `process-01.json`: `80aea5181ba7e7f07226093fa9599b95dd126bae2fc6211252f1efa4b452548c`.
- The 25 policy/task source files plus old predictor/projector are byte-identical;
  `unchanged-source-after-01.json`:
  `fcefcf858b3514b0b43dfa455b05a67146ac69c0cdfd1cf6c26cdc142620eb51`.

Independent review closed both findings: use the shorter engine/task horizon and
reject replacement of the environment's engine. Focused tests cover both. The
reviewer inspected the final source, tests and receipt without duplicate execution;
no remaining implementation finding was identified. Root separately reviewed the
same source and meaningful failure/continuation checks.

Command already completed; do not rerun unchanged passing evidence:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable \
  .venv-lab/bin/python -B -P .local/equipment-sequence-prediction-01/run_checks.py \
  .local/equipment-sequence-prediction-01/tests-01.xml \
  .local/equipment-sequence-prediction-01/process-01.json \
  tests/test_shared_exercise_sequence_prediction.py
```

## Completed bounded qualification

Root completed a separate frozen experiment: reconstruct the exact 48 saved
Rollout02 prefix controls; forecast exactly `[saved selected49, saved base50]`
with one sequence/two reserved controls. Compare both full endpoints and substep
contacts with the existing single-step evidence, then apply exactly those two
fixed actions in the isolated validation environment and compare actual results.
Keep the original task failure at control 50; no success/controller inference.
Forecast work alone: one scratch allocation/copy, 32 substeps, two forwards.
The full trial used 832 substeps in 0.880425 seconds /137,707,520-byte high-water
RSS. See the linked result for hashes, retained preparation correction and exact
scope. This does not authorize an unbounded rollout, new acquisition or repeated
run. Root owns native execution; the old helper and scorer remain unchanged.
