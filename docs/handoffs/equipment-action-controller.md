# Bounded joint-action projection

## State — November 1, 2026 (PST)

Implemented an **experimental, authored, privileged one-decision search** on top
of the [qualified detached predictor](equipment-action-projection.md). It is not
a learned policy, public mode, default controller, successful exercise rollout
or physical validation. **No native pilot has run in this lane.** Root's first
pilot stopped before any projected nonlinear candidate was tested. The narrow
revision below then passed a second pilot: its selected action exactly matched
the actual endpoint and kept protected accumulated gates passing through
control 49. **The full task remains incomplete.** Root owns native execution;
`contact_reconstruction` prepares its separate worker.

This lane owns only:

- `src/weft/shared_exercise_action_projection.py`
- `tests/test_shared_exercise_action_projection.py`
- This handoff and `.local/equipment-action-controller-01/`

The predictor/module/tests/handoff, previous rejected clearance correction,
scene/geometry, task/scorer/gates, root references, avatars and active defaults
remain unchanged. No Git, paid request, source acquisition or browser action was
performed by this lane. Predictor parity covers two original-action states and,
in native-02 below, one changed-action state. Other states remain unqualified.

## API and exact bounds

```python
from weft.shared_exercise_action_projection import (
    ProjectionConfig, prepare_action_projector,
)

projector = prepare_action_projector(env, config=ProjectionConfig(
    maximum_predictions=24, maximum_seconds=5.0,
    difference=0.1, trust_radius=0.25, maximum_sweeps=64,
))
action, diagnostic = projector.project(pinned_base_action)
```

Prepare after reset. The base action is supplied, not recomputed inside the
search. The configuration is immutable. Exactly one decision is allowed over
the projector's lifetime; repeated calls cannot obtain another prediction
budget. A reset invalidates the captured equipment reference. Creating another
projector is a new explicit experiment, not permission to bypass the pilot cap.

The original task must have passive dynamic equipment, the frozen gate hash,
and **no required-contact windows**. Unsupported tasks and measurements reject
explicitly. The qualified predictor retains its bounded original MuJoCo model
and actuator contract. The caller must serialize live steps, resets, model and
callback changes; the projector lock excludes only concurrent projection calls.

- Nineteen joint variables; enabled root commands remain **byte-identical to
  the same-state base action**, not to a later baseline trajectory.
- Original normalized action bounds `[-1,1]` and a stricter per-joint change
  bound `±0.25`. Motor scaling, inertia, bias, damping, contact response and root
  torque remain inside the existing detached predictor; no guessed torque map.
- Float32/float64 action dtype is retained. Casting that crosses a trust/action
  bound rounds inward with `nextafter`. Integer and other action dtypes reject.
- At most **24 predictions / 384 scratch substeps / 24 copies / 24 boundary
  forwards / one scratch data object**, with sixteen substeps per prediction.
- At most **64 linear solver sweeps**, **five seconds cumulative active
  compute**, and **2 MiB/result**. Encoding is charged. Time checks are
  cooperative; the pilot supervisor supplies the hard process deadline.
- Attempted/completed native counts aggregate across the lifetime. If the
  predictor unexpectedly raises without its report, `native_counts_complete`
  is false; a zero reported count is then not proof of zero native work. Root's
  independent native wrappers remain authoritative.

`status="unchanged"` returns the base copy after one passing prediction.
`status="selected"` returns a tested correction after exact repeat verification.
`no_verified_candidate` and `unavailable` return **`action=None`**, never a
fallback presented as verified. `verified_endpoint` is present only for the
first two statuses and has the predictor's full endpoint shape: all 22
`STATE_FIELDS`, `time_s` and complete boundary `contacts`.

## Objective and measurement limits

The local problem minimizes squared change in normalized joint actions, subject
to finite-difference approximations of the following **measured** constraints.
The actual nonlinear prediction must pass before an action is returned. Each
diagnostic constraint includes `name`, `limit`, `sampling` and
`true_substep_equivalent`; each evaluation supplies its ordered physical-unit
`values` and `passes_measured_constraints`.

| Measurement | Frozen limit | Sampling / limitation |
| --- | --- | --- |
| Every object's reset-relative displacement | 0.03 model units | Integrated endpoint only |
| Every object's reset-relative quaternion angle | 0.15 rad | Integrated endpoint only |
| Every object's linear speed | 0.15 model units/s | Integrated endpoint only |
| Every object's angular speed | 0.75 rad/s | Integrated endpoint only; free-joint angular norm is rotation-invariant |
| Human joint RMSE against endpoint reference | 0.70 rad | Endpoint use of the peak gate; interval mean and substep peak are unobserved |
| Human root RMSE against endpoint reference | 0.30 model units | Same endpoint-only limitation; root actions are unchanged |
| Maximum returned floor-contact penetration | 0.025 model units | Sixteen **solver-start cached contact samples**, the same field passed to the actual substep observer |
| Final floor-contact penetration | 0.025 model units | Separate boundary forward, not a substep-equivalent measurement |

World translation/speed are divided by the original body-height scale. Equipment
poses are compared with the actual reset accumulator's copied initial states.
The authored endpoint reference uses `_target_at(initial_time + dt)`. The
existing scorer's **`+1e-12` physical-unit comparison tolerance** is preserved;
see `_ExerciseMetrics.metrics` in `shared_exercise_task.py`. No task limit is
increased. Missing floor fields or incorrect solver/integration clocks reject
rather than infer a value from boundary geometry.

Equipment velocities/poses and human tracking have **no returned intermediate
state sequence** here. Passing their endpoint tests does not prove their
substep maxima, interval means, cumulative success, future braking or absence
of contact. A passing base returns unchanged, so this is a reactive prototype;
it does not look ahead to the next control interval.

## Deterministic search

1. Predict the base once. Return unchanged if all measured constraints pass.
2. Reserve the complete remaining search budget before starting. Sample all
   nineteen axes once at a normalized action change of `0.1`, toward zero (at
   zero, positive). Use each actual dtype-rounded change in the derivative.
   Retain any already-passing finite-difference action as a candidate.
3. Form the normalized constraint Jacobian and project zero action change onto
   its halfspaces and original action/trust box using bounded Dykstra sweeps.
   A fixed `1e-6` stricter normalized target handles numerical rounding; it
   never relaxes a task limit. At the sweep limit, retain the bounded last
   iterate as `iteration_limit` / `proposal_only=True`; it is not a converged or
   feasible result. No eight-variable helper contract is widened.
4. If the linear solver produces a candidate, simulate correction scales
   `1`, `1/2`, `1/4`. Compare actual squared action changes with any passing
   finite-difference candidates. Nonlinear response errors are recorded; the
   linear prediction alone cannot accept an action.
5. Repeat the smallest passing candidate. Require exact equality of its **full
   endpoint and substep contact reports**, not just optimized metrics. Only
   then return `selected`.

Maximum work is **1 base + 19 differences + 3 scales + 1 repeat = 24**. A passing
finite-difference action survives a local solver failure and can be repeated
without the three scale calls. A local row/box conflict or sweep limit is not a
global nonlinear/physical infeasibility certificate. No verified sample means
an explicit failure result; the search does not expand its radius, gains, gates,
iterations or prediction count. Minimum change is only over the tested actions,
not a claim of the global nonlinear optimum.

## Analytical verification

The existing guarded runner denied native calls, subprocesses and network.
**42 distinct mocked/analytical cases passed**, zero native attempts, exit 0,
0.241568 seconds. The one known MuJoCo startup subprocess probe was denied.
These checks establish software behavior, not physical responsiveness or task
success. Earlier `tests-01.xml` contains 37 passing cases; final `tests-02.xml`
contains 42 and supersedes that coverage. Do not sum the overlapping cases.

Covered: all nineteen dimensions (including a first/nineteenth-joint minimum-
change case), bounded conflicting-halfspace iterations, preserved finite-
difference candidates, exact selected repeat, real float32 rounding at a trust
bound, root bytes, source-scale/reset pose metrics, endpoint human peak gates,
solver-start floor timing, unsupported measurements/tasks/dtypes, live-state and
reset checks, partial/unknown counters, lifetime caps, active timeout and final
encoding/output budgets. No unchanged passing checks need another run.

Command:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable .venv-lab/bin/python -B \
  .local/split-limb-feedback-actor-01/run_pure.py \
  .local/equipment-action-controller-01/tests-02.xml \
  .local/equipment-action-controller-01/process-02.json \
  tests/test_shared_exercise_action_projection.py
```

| File / evidence | SHA-256 |
| --- | --- |
| New projection module | `ad8d2a537fe12563ed938e1d6a67ae5d7671367530bd88a58d0c92d633169e50` |
| New focused tests | `dcd7fadfe060e8468f61b9c7d7f9e348eb7ee22dbeba87c1de477549f2781765` |
| `tests-02.xml` | `d1b8ef27801b086ab433cbdbc857da9be4515ffe927d7b6e700b02984f44642b` |
| `process-02.json` | `813d8ebc0f6352317371a0f220de8fd0ddd2c9e73a40e4015b17082bae475d30` |
| Unchanged predictor module | `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd` |
| Unchanged predictor tests | `c46f04dd55ef3e3faa0f48ed07cd8654b61143c215b7d799e28d985f67eacf81` |
| Unchanged predictor handoff | `ff316bb677536465b31867e746c35fcbe5f3f0640fef585ff0878a02d09c481f` |

## Native pilot protocol — original proposal

Preparation metadata: `.local/equipment-action-controller-01/pilot-proposal-01.json`.
The separate worker directory `.local/equipment-action-controller-native-01/`
is owned by `contact_reconstruction`; root reviews and freezes its bytes before
execution. The preparation metadata remains unchanged; the actual frozen
native-01 plan/result are recorded below. A separately numbered native-02 plan
must pin the changed implementation and receive root review. This handoff does
not authorize new experiments or retries.

Reuse the original episode-1 nominal seed-29 scene/task/v5 policy, preserving the
saved original controls and full task snapshots through **48** controls. Call the
projector once on the original control-49 base action. Only if it returns an
action (`selected` or `unchanged`), perform one actual control and compare its
full endpoint against `verified_endpoint`. Check same-state root bytes, passive
objects, live isolation before application and the actual protected substep task
metrics afterward. If no candidate is verified, save that result and stop; do
not widen or retune the search.

Maximum envelope, inclusive of preparation/reset and both branches:

- **11 model constructions, two data objects, one reset, 75 forwards, 24 copies**.
- **49 live controls / 784 live substeps** plus **24 predictions / 384 scratch
  substeps** = **1,168 total native substeps**.
- One projector/decision, zero optimizer retries, zero extra per-substep
  forwards, zero source decoding, training, replay, acquisition or paid calls.
- Proposed supervisor: **20 seconds wall time, 512 MiB peak memory, 4 MiB output**;
  projector's own five-second active cap and 2 MiB result cap still apply.
- External spend **$0**, no recurring work. Prior four-prediction timing was
  0.0092192 seconds but is not a benchmark for this new search; measure actual
  pilot latency and memory once. No UI/sustained-use claim follows from it.

Original evidence pins (exposed development evidence, not an untouched holdout):

| Input | SHA-256 |
| --- | --- |
| Canonical scene | `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d` |
| Task | `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da` |
| v5 base policy | `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015` |
| Original recording seal | `94ee100c5093ad51880578972aa123cf01b11f75a791362379ea23a053db91cd` |
| Protected scorer | `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb` |
| Shared-world runtime | `19c0f07fc9174bd23d086250bb9d5cc3e2c6ded7c1e84612e120d82772d309df` |

## Native-01 result — preserved failure

Root completed and reaped the first pilot on **November 1, 2026, 11:05 AM PST**:
exit 0, **1.106289 seconds**, **139,640,832 B peak child RSS**, five artifact files
totaling **101,445 B**. The experiment completed; the controller did not succeed.
All 48 original actions/full task snapshots match and live state/task metrics
were unchanged during projection. No action was returned or applied, so the
live state remains at control 48.

The only decision reason was `no_verified_candidate` /
`linearized_iteration_limit`: **64 sweeps**, normalized maximum linear violation
**1.0240089176605238e-6**, **20 predictions** (base + nineteen differences).
**No nonlinear projected candidate was tested.** None of the sampled
finite-difference actions passed. The unused four-call budget therefore did not
establish that no feasible nonlinear candidate existed.

Actual counts: **11 models, two data objects, 70 forwards, one reset, 1,088
substeps, 20 copies**; 48 live controls / 768 substeps plus 320 scratch substeps.
Projection active time was 0.071620625 seconds. No forbidden operations, media
decode, replay, paid request or retry occurred. These are one-state experiment
measurements, not sustained-use or UI latency evidence.

The actual root-frozen plan used a **60-second external deadline** (including a
five-second cleanup reserve), 40-second worker CPU cap, 512 MiB monitored RSS and
4 MiB output cap. This differs from the original 20-second supervision proposal;
the executed plan is authoritative. RSS/output monitoring is not hard isolation.

| Native-01 evidence | SHA-256 |
| --- | --- |
| `artifacts/equipment-action-controller-native-01/manifest.json` | `596c1614a831a23e2f17352c0793411389a94afcad93d453cf4135571855fbd1` |
| `result.json` | `61faf5ccd8424621c563e250ba62b9cdf774ab558b0593097023dca97153f29b` |
| `projection.json` | `cc0082369fc96d7f0bc156655c558d79b66c68a38c5cc08a4e222c7956cbad0f` |
| Frozen `plan.json` | `2df8d16338ce531aa88ac3b46a2ad63122f1705335296e2fc9b2044b04edcf48` |
| Saved `prefix.json` | `3c16d077281a8667ccbf6177663c2878d74da7d23482c811e9cfa2518cf6807e` |
| `.local/equipment-action-controller-native-01/process.json` | `694684f08e81e247c5fdb59683ca71db72dba43df09f01fc3a4090a670e514d9` |

The module/test hashes in the original verification table above belong to this
first pilot. Do not replace the native-01 artifacts or original proposal with
new implementation identities.

## Narrow revision after native-01

The fixed 64-sweep solver now returns its already box-bounded last iterate as an
**explicitly unconverged proposal**. The existing three nonlinear scale checks
and final repeat determine acceptance exactly as before. Sweeps, prediction/time
caps, actions, trust radius, gate values, tolerance, scorer, geometry and root
commands are unchanged. Linear convergence never replaces nonlinear validation.

Root's saved-only reconstruction (`linear-analysis-01.json`, SHA-256
`5a6db9d90e17ce5783c7e721ef4a654a3e8cb727624c66a82d4138aa58ea5c92` in
`.local/equipment-action-controller-01/`) found the same residual to 4.4e-16.
Its local scalar model predicts impossible negative speed/displacement for the
proposal. That is evidence against trusting the local scalar model, not evidence
that the actual candidate passes. The change allows the actual nonlinear checks
to answer that limited question; it does not increase the search envelope.

Only **five affected/new checks** ran in `tests-03.xml`: updated bounded
iteration-limit behavior, retained finite-difference candidates under a local
box conflict, nonlinear pass/fail cases for unconverged proposals, and a
scratch-only regression using the exact pinned twenty native-01 evaluation
vectors. All passed, exit 0, **0.503294 seconds**, zero native attempts. The full
42-case suite was not rerun. The committed test set now contains 44 distinct
cases; the saved-vector scratch check is additional and deliberately not a
runtime/CI dependency on ignored artifacts. Its synthetic predictor verifies
control flow only, not native or physical feasibility.

| Revised evidence | SHA-256 |
| --- | --- |
| Current projection module | `04510c0a46ca13ce397e0d73ff6a5edcebd325d3df3ccd6092e2643b1d90acab` |
| Current focused tests | `4991c6a0ef1c78132636923735e4002442c0bd51b2d41a7890327124f2d75c57` |
| `.local/equipment-action-controller-01/tests-03.xml` | `3095336d461f4ac41437a865082246cd6d43264186ba26afe9c4b78a466f8c94` |
| `process-03.json` | `5f477343b872c5fa8fabc49398bad54bdf30cec98dd3f923f92b5089be57509c` |
| `test_saved_iteration_limit.py` | `f33907fc77db6d8e351ca397ce8822b9e2e8781e5eb88abbd993ec499e8b51b1` |

## Native-02 result — one protected contact boundary passes

Root completed and reaped the separately frozen second pilot on **November 1,
2026, 11:10 AM PST**: exit 0, **1.0741705 seconds**, **139,935,744 B peak child
RSS**, five artifact files totaling **169,863 B**. It uses the revised module
`04510c0…` and tests `4991c6a0…` above, with the same physical/scoring limits and
24-call envelope. It is a changed-implementation experiment, not a retry of
unchanged native-01.

All 48 original prefix actions/full task snapshots match. The bounded
unconverged full-scale proposal passed the actual nonlinear prediction; scales
`1/2` and `1/4` failed. The full-scale action was selected only after its exact
repeat. Its squared joint-action change is **0.18995645892616722**, maximum
per-joint change **0.25**, with original action bounds retained.

After that one selected actual control:

- All 22 exported state arrays, time and complete boundary contacts exactly
  match the verified endpoint, with **zero numerical error**.
- Root command bytes equal the same-state base; applied object-force channels
  stay zero. Projection did not change live native state or task metrics.
- Every protected accumulated gate remains passing through **49 controls /
  784 observed substeps / 1.6333333333 simulated seconds**.
- The saved same-state original base action crosses the equipment speed/spin
  limits at control 49. The selected action avoids that observed failure.
- `completed_interval=false` and `success=false`: **49 of 254 controls is not
  full-task success**, generalization, sustained closed-loop robustness or
  physical transfer.

For sam-2, actual accumulated peak speed remains **0.0204375 model units/s**
and spin **1.0411545102750766e-5 rad/s**, versus the original control-49 peaks
**0.4135693241043099** and **3.642698006943431**. These are actual accumulated
substep metrics, not endpoint estimates or claims about future controls.

Actual counts: **11 models, two data objects, 75 forwards, one reset, 1,168
substeps, 24 copies**; 49 live controls / 784 substeps plus 24 predictions / 384
scratch substeps. Projection active time was **0.08582858298905194 seconds**.
The original aggregate supervisor limits still apply; no forbidden operation,
decode, replay, paid request, solver expansion or extra per-substep forward ran.
This one-decision timing does not establish a whole-trajectory or UI budget.

| Native-02 evidence | SHA-256 |
| --- | --- |
| `artifacts/equipment-action-controller-native-02/manifest.json` | `71afa27c64072d7fbb142331b42c2e6b0b2750802e1af4a14a9c93bf514cec58` |
| `result.json` | `9dac5e6d9321982fa62e5e13ddda6314832da45c9b719686306ef3a8c798af1e` |
| `projection.json` | `26902044ed86759550b1ba21d2c11ca0033dfaabab2e2d3097f4c795b36f4696` |
| Frozen `plan.json` | `8a007cfef0b44545b4bb385a2f449676525b2815657b3b671a70b15288994a34` |
| Saved `prefix.json` | `3c16d077281a8667ccbf6177663c2878d74da7d23482c811e9cfa2518cf6807e` |
| `.local/equipment-action-controller-native-02/process.json` | `ac51e9c9377dcae55e7a62dfeb1eb4154923c1afcd0a134a10307a2bdece2126` |

This closes the one-decision checkpoint. No full nominal trajectory has run.
Any further pilot requires a separate explicit aggregate plan; its preparation
belongs in a separate note/run directory, not this frozen checkpoint. Keep the
hybrid authored correction unrouted and preserve all rejected proposals.
