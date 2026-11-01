# Experimental equipment-clearance controller

## State — November 1, 2026 (PST)

Implemented and analytically tested. Root completed one matched native comparison:
the candidate still fails equipment stability and is **not promoted**. Public
lab integration is deferred; do not retune this candidate. This is an **authored,
privileged joint-action correction**, not a trained policy or a collision-safety
guarantee. No actor, trainer, simulator, protected task/scorer, source, avatar or
saved recording was changed.

The saved v5 trial first reports `sam-2` speed/spin failure at control 49,
elapsed 1.600–1.633333 seconds, alongside a right-forearm boundary contact.
The declared source-reference bracket is frames 48–52, missing 49–51; it is not
native-video synchronization or causal substep evidence. See the sealed
`artifacts/split-limb-contact-diagnosis-02/timeline.json`. Unlike the old static
arm-reference fitting, this candidate responds to current simulated separation
and relative closing motion without rewriting the reference or geometry.

## API and exact scope

`src/weft/shared_exercise_clearance_controller.py` exports:

```python
controller = prepare_clearance_controller(
    env,
    base_action=prepared_policy,  # Optional; defaults to env.authored_action.
    config=ClearanceConfig(),
)
action, diagnostic = controller.sample()
action = controller()  # Same action, without detailed pair rows.
description = controller.describe()
```

The caller must authenticate and record the base controller/policy, module bytes,
settings and scene/task identities. A callable is not authenticated provenance.
The new module does not create a learned-policy artifact or reuse a misleading
legacy controller tag. Root owns later recording/lab routing if comparison
justifies promotion.

Only existing `shared-world-v1/v2` equipment tasks with nineteen joint channels
and optional three root channels are admitted. Nonempty required-contact tasks,
static supports, camera/feet/human-only variants and more than 512 geometry pairs
reject before geometry queries. This avoids repelling a declared required grasp
or silently skipping pairs. The existing physical model/data is reused; no
constructor, forward solve, step, applied force or live-state mutation occurs in
action selection.

For each human-geometry/equipment group, all declared compound parts receive a
bounded distance query; only the nearest signed-distance part supplies response.
Duplicating identical compound parts therefore does not multiply the force.
At most 32 detailed rows are returned; omitted-row counts remain explicit.

Let `s` be world units per body model unit, `d` signed world separation, `n`
the normal pointing away from equipment, and `J_h/J_o` the current closest-point
Jacobians. The controller uses:

```text
relative_speed = n · (J_h − J_o) · qvel / s
deficit = max(0, margin − d/s + lookahead × max(0, −relative_speed))
force_fraction = min(force_cap, stiffness × deficit)
joint_torque += (n · J_h[:, joint_dofs]) × mass × |model_gravity| × force_fraction
joint_residual = clip(joint_torque / exact_motor_limits, −residual_cap, residual_cap)
```

Defaults: margin **0.025 model units**, lookahead **0.10 s**, stiffness
**8 body weights per model unit**, pair force cap **0.25 body weights**, joint
residual cap **0.25 normalized action**, query radius **0.20 model units**.
Mass and geometry remain model priors. The model's actual gravity magnitude
sets the force scale; zero gravity returns the base action unchanged.

The original root channels never receive correction; passive objects still
evolve only under the engine. Floating-action dtype and root bytes are preserved.
An active integer base is promoted to float64, retaining exact root values.
Root equality is with the base controller at the **same candidate state**, not
with the original baseline trajectory: changed joints change subsequent root
feedback. An unchanged correction returns a detached bit-exact base action, including
signed zero. Float32 rounding cannot enlarge the residual cap.

Degenerate zero-distance normals, missing closest points and inconsistent
signed points skip that group with an explicit unavailable reason. Nonfinite
state/Jacobian/arithmetic rejects. Saturation, finite query radius, control-step
sampling and missing normals mean this is **not guaranteed collision avoidance**.
Nearness is not evidence of real contact; desired support is not recovered.

## Verification and custody

Twenty-nine focused analytical/mocked cases pass, covering outward response on
both sides of zero distance, relative equipment/root velocity, world rotations,
motor/scale units, compound invariance, no state mutation, unchanged/disabled
anchors, float/integer roots, saturation, unavailable geometry, bounds and
contract rejection. Native APIs remain blocked except the two mocked query APIs.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable \
  .venv-lab/bin/python -B .local/split-limb-feedback-actor-01/run_pure.py \
  .local/equipment-clearance-controller-01/tests-02.xml \
  .local/equipment-clearance-controller-01/process-02.json \
  tests/test_shared_exercise_clearance_controller.py
```

Receipt: **0.176003 seconds**, exit 0, zero native attempts; the known MuJoCo
startup subprocess probe was denied once. Tests 01 passed 24 cases before the
five dtype/overflow cases and corresponding implementation checks were added;
do not add the overlapping counts. Both receipts remain saved. No native run,
training, decoding, API call, deployment, server/browser work or Git mutation
was performed by this lane. No process remains live in the analytical-check
lane. The later native process was owned and reaped by root, as recorded below.

| File | SHA-256 |
| --- | --- |
| Controller | `7755f7489ebbba838076f2599af99a03d1a7871334c19fe0af715695bcb66195` |
| Focused test | `e3ea24b59834c2b871b43987e2e58a3b82311d09cc79875b66247a25caa9c05c` |
| `tests-02.xml` | `2d35865bdecfb366569a773137164bde00dd00336be74d886b662c2494c2a9b9` |
| `process-02.json` | `921d12bacb69f8fba57e719b56432087abfe44e502ea3dd36c09071ad2c92314` |

## Matched native result — November 1, 2026, 10:19 AM PST

Root's `artifacts/equipment-clearance-native-01` completed **254 controls / 4,064
substeps** on the same initial state and pinned v5 policy as saved episode 1.
The original baseline was reused, not rerun. Thirty-four joint actions changed;
all 254 root actions match the same-state base bit-for-bit. A detached-data
positive-gap derivative check agrees within **1.69434e−10**; that one derivative
does not prove effective dynamics or avoidance.

| `sam-2` whole-interval metric | Saved v5 | Clearance correction | Limit |
| --- | ---: | ---: | ---: |
| Displacement, model units | 0.1057493 | 0.2858288 | 0.03 |
| Orientation, radians | 0.6503403 | 0.0645998 | 0.15 |
| Speed, model units/s | 0.7220860 | 1.3410945 | 0.15 |
| Spin, radians/s | 5.6244696 | 3.7264454 | 0.75 |
| Mean reward | 0.7453870 | 0.7434594 | Not success |

Only `equipment_stability:sam-2` fails at completion. Body/floor/other-equipment
gates still pass, but floor penetration increases from 0.0035708 to 0.0103867
model units. Lower spin/orientation did not offset worse translation/speed.
This single exposed-development result is not independent generalization.

Process receipt: **1.704560 seconds**, **134,938,624 B** peak child RSS, exit 0,
reaped. Fourteen model compilations, two data objects, 258 forwards (including
the two detached gradient-check forwards), 27,571 distance queries and 608
Jacobian calls; zero forbidden operation, media decode, replay or paid request.
These are whole worker measurements, not a per-control latency benchmark.

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/equipment-clearance-native-01/manifest.json` | `648f22e6c1ad485a58d41027311f3a86baa22241936456721ce10dd3acdef9b1` |
| Native `result.json` | `818b75dbf611468dc1fc5fe854cc40c9e76b3b76133d6477351d5f5fbfe5a7fb` |
| Native `trace.jsonl` | `3d5aff57f981604c40469620a4df4c3d3e6f672903710e46c2cb0cee0345dc90` |
| `.local/equipment-clearance-native-01/process.json` | `b1106ce6d14c0b8ed38ea66f56e275a7c0e6094eadfab3c8c0ac49074416a0c5` |
| `.local/equipment-clearance-controller-01/trace-analysis-01.json` | `a414cb540ea42683f3838170cc91894bcae7221d7ef821e9006016c630b71604` |

### Saved-only failure analysis

The full **254-row / 1,049,432-byte** trace and result match their manifest. No
new physics or decoding was run for this analysis.

- First speed/spin failure is control **44**, elapsed **1.433333–1.466667 s**,
  five controls earlier than baseline. Speed first reaches 0.841170 and spin
  1.457094; displacement crosses at control 45 (0.046105 model units).
- At pre-control 44, right-forearm gap is **0.0072013** model units and normal
  speed **−2.021923** model units/s. Both forearms request the capped force;
  the action correction reaches its 0.25 bound. These are pre-control geometry
  and cumulative post-control maxima, not an exact causal contact substep.
- **32 of 34** active controls reach the joint-residual cap; **21** controls
  reach a pair force cap. There are **zero unavailable/unactuated pairs, omitted
  pair details or reported compound-part switches**. Normal abstention and
  compound gain multiplication do not explain this failure.
- Corrections alternate: control 40 active, 41 unchanged, 42 active, 43 unchanged,
  44–45 active, 46 unchanged, then 47–54 active. Unchanged steps have no active
  predicted deficits, not a processing error. Corrections already occur at
  controls 3–5; this is not simply late activation at the first impact.
- Speed worsens again at control 49, after pre-control right-forearm penetration
  of −0.001396 model units. By control 57 displacement is about 0.285827; it
  subsequently changes little. Saving orientation mainly coincides with a
  larger translation, not preserved equipment stability.

## Next safe action

Keep this candidate rejected; do not increase its gains or repeat its run.
The trace supports investigating **dynamics-aware minimum-change joint actions**:
account for base-controller torque, inertia and available braking acceleration,
instead of extrapolating velocity and saturating a geometric repulsive torque.
This is a new hypothesis, not a proven explanation of causality or a validated
fix. A bounded analytical/detached-state check should establish the acceleration
mapping and constrained-action feasibility before any new rollout. No such code
or run was created in this lane. Preserve fixed root/object channels and all
source, task and evaluation boundaries. Root owns the next decision and overall
comparison/current handoff; no public mode should present this failed candidate
as successful avoidance.

Only the controller, focused test and this handoff are owned by this lane.
Root owns publication and the current integration handoff; preserve all parallel
surface-reconstruction and avatar work.
