# Equipment-conditioned feedback actor

## Current state and ownership

Implemented a **pure, experimental actor**, not a trained policy or public
route. All **73 focused authored/mocked cases pass** in one guarded run:
0.550655 seconds, 117,243,904-byte peak RSS, zero native or forbidden operations.
One import-time OS probe was blocked before execution. Independent source/test
review is closed with no blocking finding; the reviewer read the receipt and
cases without repeating execution. No simulation, training, full-task evaluation
or new holdout query has run. These are correctness checks, not a
controller-performance measurement.

This lane owns only `src/weft/shared_exercise_equipment_feedback.py`,
`tests/test_shared_exercise_equipment_feedback.py`, this note and
`.local/equipment-feedback-01/`. Root owns integration, native execution and Git.
No live handles; $0 external or recurring spend. Existing v5, predictors,
projectors, root search, task/scorer, source partitions and defaults are unchanged.

## Motivation and exact scope

The [late root search](equipment-root-search-native.md) tested 24 nonlinear
two-action pairs before control49; all failed first-boundary equipment speed or
spin. Its best overall normalized excess improved but remained failing. That is
not proof of insufficient authority or task infeasibility. More identical late
polls are not the next hypothesis.

The distinct hypothesis is **equipment-conditioned feedback throughout the
episode**, including before tracking error appears. Existing v5 has ten tied
position/velocity-*error* weights; its residual vanishes at zero tracking error
and its roots stay authored. The new basis can modulate moving joints near an
equipment center even when current motion exactly follows the reference, and
can redistribute the already-permitted root assistance. This differs from the
previous current-clearance joint-only correction, but neither center proximity
nor this pure implementation demonstrates collision avoidance.

The actor requires an already admitted `shared-world-v1/v2` scene,
`shared-exercise-task-v1`, nineteen original joints plus explicit three-axis root
assistance, 1–12 equipment objects, and **no required-contact windows**. Unsupported
layouts reject rather than silently ignore contact obligations. No equipment
force, body, task, reset, scorer, gate or evaluator changes are permitted.

## API, features and bounds

```python
from weft.shared_exercise_equipment_feedback import prepare_equipment_feedback

actor = prepare_equipment_feedback(env, coefficients)  # finite 8×2 in [-.25,.25]
action, diagnostic = actor.sample(base_action, observation)
```

The caller validates/prepares the pinned v5 policy separately, computes its base
action once, and supplies the **same-current-state** reset/step observation from
the same frozen environment. Preparation reads only existing metadata; sampling
retains no environment and performs no live/native/policy query. It cannot
authenticate base provenance, freshness, early engine termination, or pairing;
those remain caller responsibilities. It rejects the completed elapsed signal
actually present in the observation, malformed shape/types and nonfinite values.

The exact existing observation schema is checked before preparation:

- Human prefix: `q22, velocity22, target22, target_velocity22, PD_features66,
  reference_phase1` = 155 values.
- Each equipment row: position3, quaternion4, velocity3, angular_velocity3,
  reset-displacement3, orientation-error1 = 17 values.
- Each required contact would add six values, but this version rejects those
  tasks. The final value is **elapsed fraction**, not a contact summary.
- Total is `156 + 17E`: **190 for two equipment objects**. Tests use the existing
  task's real pure observation builder and contract, not a separate producer.

All sample inputs are explicit finite bounded float32/float64 arrays. Base shape
is 22 and bounds are `[-1,1]`; observation shape follows the admitted schema and
its existing `±1e6` bound. Original array/dtype bytes are copied, not mutated.

Pelvis position in body-model units is
`origin_world / scale + (0,0,base_height) + root_q / scale`; pelvis velocity is
`root_velocity / scale`. Equipment position/velocity are already normalized by
the task. Nearest **center** wins, with original task order resolving exact ties.
For center distance `d`, direction `away`, and relative pelvis/equipment velocity:

- `p = clip(1 − d/1.5, 0, 1)`.
- `c = clip(max(0, −away·relative_velocity)/2, 0, 1)`.
- Joint row: `[p·clip(current_velocity/4,−1,1),
  p·clip(current_target_velocity/4,−1,1)]`.
- Root-axis row: `[p·away_axis, p·c·away_axis]`.

Coincident centers report unknown direction and use zero root-direction
features. No unobserved direction or geometry is fabricated. The constants are
**authored priors**: 1.5 body-model units, 4 rad/s, and 2 body-model units/s. They
are not fit to a failure frame, task threshold, measured contact or physical
calibration. Current target velocity is from the existing observation; the
actor never queries a future reference or replay frame.

Sixteen coefficients use the existing five v5 joint groups (torso, proximal/distal
arms, proximal/distal legs) plus separate x/y/z root groups, two weights each.
Residuals are clipped to `±.25` and legal action `[-1,1]`; inward representable
rounding preserves the original-base box in the supplied dtype. **All-zero
coefficients return the exact supplied v5 action bytes**, including signed zero.
No root budget is added: the existing physical motor limits remain unchanged.

Per sample, work is bounded by at most 12 equipment rows, 22 output channels and
44 feature/weight products. The complete diagnostic is capped at **16 KiB** and
records coefficients, nearest center/direction certainty, normalized features,
requested/applied residuals, actual changed channels, phase, elapsed time and
limitations. Metadata and weights are detached and immutable; there is no
dependency hashing or full-scene validation in the sample path.

## Verification and release evidence

The 73 cases cover exact disabled-v5 bytes and signed zero; nonzero motion at
zero tracking error; near/far behavior; all eight tied groups; reordered original
joint names; relative closing signs; task-order ties and coincident centers;
scale/origin invariance; float32/64 legal/inward residual bounds; metadata and
array rejection; final elapsed semantics; immutable preparation; no environment
queries; and bounded reports at the 12-object limit.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-lab/bin/python \
  .local/equipment-feedback-01/run_checks.py \
  .local/equipment-feedback-01/tests-01.xml \
  .local/equipment-feedback-01/process-01.json \
  tests/test_shared_exercise_equipment_feedback.py
```

This exact completed command must not be rerun into its preserved outputs.

| Evidence | SHA256 |
| --- | --- |
| New actor source | `500c199baadf281f406dbff92e6b5a33b447b03e9ecf98e51c5d7d85cd1431a5` |
| New tests | `0cb3b8eb9624cb276665360eb249871208c6e4acaa19ab66ab3851876c0aca4d` |
| `process-01.json` | `56a8a500b9640ac6448a89e044f5d0225436bf847d3a67f376a97cf1e94c5844` |
| `tests-01.xml` | `827fcbecca10a02960b811bd053c0c9d8904388e8d25aafe1a1a8d44e485aa3a` |
| Guarded runner | `d27f0f7f491a3532b3a9bbd2c52dfb73b1414598cf7240156bf1a3efeca63c55` |
| `unchanged-sources-01.json` | `5c7fe3ec313cfc1dbfd1ab770bcfa164ec618703d82e04b0e6bfe10283f158d4` |

One bounded hash check confirms **30 existing frozen sources unchanged**,
including v5, the task/scorer, old learning/policy dispatch and prior control
helpers. Their successful checks were reused, not repeated. No saved source
archive was reread for these pure checks.

## Next safe action

The reviewed source/tests are frozen for root integration. Release metadata is
`.local/equipment-feedback-01/release-01.json`. There is intentionally **no trainer
or public dispatcher integration**. The
existing v4/v5 trainer hardcodes its old coefficient shape; this new component
must not be mislabeled as an existing policy artifact. Any later versioned
driver should reuse the current full-interval feasibility-first CEM/rank and
frozen v5 base, retain all failed trials, and check original accumulated task
metrics independently and measure the integrated hot path. Native/training
resources and source/runtime identities must be frozen separately before
execution. No final holdout, full-task success,
new policy improvement or unfamiliar-video claim follows from this release.
