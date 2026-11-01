# Detached action prediction for equipment control

## State — November 1, 2026 (PST)

The prior clearance residual is rejected for promotion: the saved trial reduced
rotation but increased translation/speed, with 32 of 34 active corrections at
their action cap. This lane implements only the next prerequisite: a bounded,
detached **one-control prediction**. No optimizer, new policy family, controller,
public mode, scorer change or new defaults are included.

Root's native qualification now passes on two saved-development states; the
exact result is below. Root owns further experiments and publication. Do not alter the prior thirteen-
file checkpoint or its saved evidence while Git approval is pending. This lane
owns only the new predictor, focused test and this handoff; scratch preparation
lives in `.local/equipment-action-projection-01/`. No native process was started
by this lane.

## Why predict the actual control interval

A closest-point distance Jacobian maps joint velocity to gap velocity, not torque
to future clearance. The current engine uses `implicitfast`, joint damping and
armature, root motors, floor/equipment contacts and bounded control forces.
`M^-1 B` alone omits parts of that transition; adding stale constraint forces
does not fix changing contact response.

Read-only inspection of installed MuJoCo **3.17.0** headers and binding tests
confirms `mj_copyData(dest, model, source)` copies into existing data, `mj_step`
advances time and `mj_forward` does not integrate. The binding's own test checks
copied `qpos` and `geom_xpos`. No library import/native run or new dependency was
needed for that API inspection. Sources:

- `.venv-lab/lib/python3.14/site-packages/mujoco/include/mujoco/mujoco.h`
- `.venv-lab/lib/python3.14/site-packages/mujoco/include/mujoco/mjdata.h`
- `.venv-lab/lib/python3.14/site-packages/mujoco/bindings_test.py`
- `src/weft/shared_world.py`: motor XML, actuation and `step`/contact-clock code.

The existing step sets physical controls to normalized actions × exact motor
limits, clears applied forces, performs sixteen native substeps and one final
boundary forward. The predictor mirrors that sequence on full copied data,
including warm-start state. No intermediate forward is added.

## Prepared API

```python
predictor = prepare_action_predictor(
    env, maximum_predictions=4, maximum_seconds=2.0,
)
result = predictor.predict(action)
```

`src/weft/shared_exercise_action_prediction.py` accepts existing equipment
`shared-world-v1/v2` tasks with nineteen joint controls and optional three root
controls. A single scratch `MjData` is allocated lazily and reused. Each call
copies the live data afresh, so predictions do not inherit each other's altered
state. Disabled root channels and all object/applied-force channels are zero;
enabled roots retain the supplied base action. No task accumulator, reward,
success metric, controller or live `Env.step` is called.

Hard bounds: **64 predictions**, **16 substeps/prediction**, **10 seconds total
active compute**, **2 MiB/result**, bounded model dimensions, no plugins/sensors
or active MuJoCo callbacks. Callers can choose tighter budgets; defaults are
four predictions/two seconds. Allocation/copy/substep/boundary-forward attempts
and completions are separate. Final encoding is charged to the time budget.

Results have `status="predicted"` or an explicit unavailable reason. Failed or
unavailable results have **no usable endpoint**. Completed predictions include:

- `human_action`: all twenty-two physical human channels before motor scaling.
- `substep_contacts`: each solver-start and integrated-end timestamp with the
  existing `_current_contacts(data=scratch)` result. Contacts belong to the
  solver-start cache, not newly integrated end geometry.
- `endpoint`: final time, exact detached arrays listed in exported `STATE_FIELDS`
  (`qpos`, `qvel`, `act`, `ctrl`, acceleration/warm-start/applied-force and derived
  arrays), plus complete boundary contacts after the one final forward.
- Count/time budgets and `native_parity_claim=False`; a saved verification, not
  this flag or a successful function return, establishes scoped native parity.

Unexpected callbacks, instability, nonfinite output, altered motor/timing maps,
budget exhaustion and replaced engine data abstain. Existing instability checks
run through a detached field view; the live engine's data pointer is never
swapped. Native errors retain attempted counts, even if no call completed.

**Caller serialization is required.** The predictor lock prevents overlapping
predictions only. It does not synchronize another thread's `Env.step`, callback
installation or model mutation. A final live-state identity check detects some
misuse but cannot make concurrent native use safe. The caller must own the live
environment/model while predicting; the helper does not restore over a changed
live state.

## Analytical verification

**26 distinct mocked cases** cover exact invocation order, motor map/scaling,
root-disabled zeros, applied-force clearing, warm-start recopy, repeatability,
detached outputs, live isolation, callbacks before/during/after native work,
instability, native exceptions, budgets, encoding time and unsupported inputs.
All native work is mocked; these tests do not establish actual engine parity.

The first receipt passed 23 cases and failed one fixture whose deliberately
no-op copy still matched identical fresh/live data. Giving that fixture a
distinct live position made the intended mismatch observable; the targeted
correction passes. A later full 25-case run passed; the final exception-path
change adds one late-callback case and passes four affected tests. Do not add
overlapping counts. Every receipt is retained, with zero native attempts and
one denied known MuJoCo startup subprocess probe per process.

| File | SHA-256 |
| --- | --- |
| `src/weft/shared_exercise_action_prediction.py` | `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd` |
| `tests/test_shared_exercise_action_prediction.py` | `c46f04dd55ef3e3faa0f48ed07cd8654b61143c215b7d799e28d985f67eacf81` |
| `predictor-tests-03.xml` | `12c27c62c5e22bd70c3d400b43e7fff4a7f06966dce3a3486c8bdd9486cac1a8` |
| `predictor-tests-04.xml` | `1dcee7d0470f9c209a58bc2da7507844fb0b3d455266886545c9aee0d5ddeca0` |

Receipts are under `.local/equipment-action-projection-01/`. Run only the
affected tests after later changes; unchanged passed cases need not repeat.

## Native parity qualification — completed by root

Root used the pinned original scene/task/v5 policy and nominal seed 29. One
original prefix through control 49 matched every action and boundary snapshot
to the saved original recording. At pre-control **1** and **49**, two repeated
scratch predictions preceded the single actual step. This provides a nominal
and contact-transition check without new learning or a full baseline rerun.

Budget: **49 actual controls / 784 live substeps + four predictions / 64 scratch
substeps = 848 total native substeps**, one scratch data object, no per-substep
forwards. Actual counts match: **11 models, two data objects, 55 forwards, one
reset, 848 substeps, four copies, four predictions**. Zero forbidden operations,
media decode, replay or paid request; the known startup subprocess probe was
denied once. The worker used a five-second predictor budget, with **0.0092192
seconds total active prediction compute** across the four calls. This is a
two-state measurement, not a sustained-use or whole-interface latency claim.

All 49 original actions/full task snapshots match. Both checkpoints preserve the
live state/task, their repeated predictions match exactly, and **all 22 exported
state arrays, time and complete endpoint contacts match the subsequent actual
step with zero numerical error**. The original equipment failure still appears
at control 49, exactly as recorded; this qualifies prediction behavior, not a
successful controller or a completed interval.

The process completed/reaped at **November 1, 2026, 10:40 AM PST**, exit 0,
**0.7969075 seconds / 138,952,704 B peak child RSS**. Three output files total
475,276 B. Scope is two exposed-development states in the exact pinned runtime;
arbitrary states, changed actions and physical accuracy remain unqualified.

| Proof | SHA-256 |
| --- | --- |
| `.local/equipment-action-prediction-native-01/plan-frozen-01.json` | `1d785529eecd260b6fa3551a1f1086bb0a337168b69def5fd980554a7461a8e8` |
| Root native worker | `35a71cd4407e7c7435a8db18c6f31634397cb475732baba4038d1c99ae093c99` |
| `artifacts/equipment-action-prediction-native-01/manifest.json` | `69ba457646533ff358d9bbfb2d099807ca620235d2995cf474255e130d7dd13c` |
| Native `result.json` | `dc62c0d9a1df529ac77466a13c56dbf79501d238a2cb5f2fb1deaaba85a7f5ab` |
| Native `predictions.json` | `5430b135295bac226ae49f5be56718952a2eceadd3fac710db31da1a669d959b` |
| `.local/equipment-action-prediction-native-01/process.json` | `fb544fb93c283b7d97233910c90a089cae9ecd5106658255c5cb788b6eeaf975` |

The separate `.local/equipment-action-projection-01/design-plan-01.json` remains
the unexecuted preparation proposal, not the executed worker plan. Do not
overwrite either or rerun this passing qualification without changed inputs.

## Later projection design — proposal only

Only after parity qualification, a separate reviewed step may use explicit
finite-difference one-control predictions over **nineteen joint-action variables**
to approximate action→future-gap response. Root commands, object forces, scene,
task/scorer/gates and total action bounds remain fixed. A local minimum-change
box-constrained projection would target that sampled response, not pretend a
velocity Jacobian is a torque map. The existing generic QP helpers allow only
eight variables; do not silently widen their contract or replace nineteen
variables with unreviewed groups.

Every candidate must receive a separate nonlinear detached prediction. Contact
regime changes, inadequate prediction fit, local infeasibility and solver/time
failure are explicit unavailable outcomes; no silent relaxed margin or action
bound. A proved impossible linearized box constraint is not proof that the
physical task is impossible. Solver limits are not infeasibility certificates.

A central finite-difference model can cost **40 one-control predictions / 640
native substeps per decision** (base +38 perturbations +candidate). That cost is
not accepted as a real-time UI budget. Measure it on bounded state samples and
establish a smaller justified schedule before any controller or public routing.
No optimizer, new experiment or gain sweep has been implemented by this lane.
