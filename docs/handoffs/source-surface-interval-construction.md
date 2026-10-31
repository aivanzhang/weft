# Source-surface interval construction

## Current milestone — 2026-10-31 4:34 PM PDT

Implemented, checks complete, code/test frozen for root review; owned by
`/root/exercise_reconstruction`. Root approved only new
`src/weft/source_surface_interval_construction.py`, its focused new test file,
and this handoff. Released envelope/earlier source files remain frozen. No live
job, native run, source study, network request or paid call.

## Intended contract

Prepare one construction objective from the existing `SourceSurfaceEnvelopes`
object, exact original body bytes, and an explicit binding document. Reuse the
canonical source registry and `world-alignment-v1` validator for source/entity,
construction partition, coordinate and clock joins. Keep every original source
slot and channel, including missing, unsampled and excluded slots. Body geometry
and the caller's candidate kinematics are not independently authenticated by a
copied hash; their execution/admission remains the trusted construction worker's
responsibility.

Map candidate **absolute body-model** endpoint positions into the unchanged
supplied world box. Per axis, residual is `p - clamp(p, minimum, maximum)`;
the squared-error term is zero throughout the box, not only at a chosen center.
This is deviation from a conditional outer enclosure, not contact, load, grasp,
measured calibration or a guarantee that a point inside the box is attainable.
The residual derivative is zero at both boundaries (explicit one-sided
convention), zero inside, and one outside before the supplied coordinate map.

Existing `body_refinement._kinematics` returns pelvis-relative coordinates and
derivatives. A small adapter must restore the live pelvis position **and its
derivative** before evaluating the term. Tests will call that unchanged producer
using analytic model/data doubles and stub native functions, never real physics.
No existing fitter has an objective callback; this milestone makes the term
callable but does not install it in optimization or change independent scoring.

## Bounds and next action

Body <=8 MiB; explicit binding <=64 KiB; existing observation-registry bounds
(<=8 documents /64 MiB); <=1,800 original slots, <=900 original body references,
<=12 endpoint channels. Preparation <=5 seconds cooperatively, retained join
cache <=8 MiB. Each query handles only one source frame and <=12 points, with
bounded named joint derivatives. No hard process isolation is claimed.

Queries additionally require the exact original raw-body SHA, named body frame
including units/axes, and original reference timestamp. At most 19 named joint
columns are accepted (the existing articulated inventory, no unbound root or
floor variables). Extra body/interval evidence is never reparsed per query.
Preparation retains the original envelope object/raw inputs once plus a compact
join cache; callers receive defensive per-frame or metadata copies.

## Public API and binding example

```python
from hashlib import sha256
import json
from weft.source_surface_interval_construction import prepare_surface_interval_objective

meta = envelopes.describe()  # Existing prepare_source_surface_envelopes output.
binding = {
    "schema_version": 1,
    "profile": "source-surface-interval-body-binding-v1",
    "source_sha256": meta["source_sha256"],
    "body_sha256": sha256(original_body_bytes).hexdigest(),
    "envelope_bindings": meta["bindings"],
    "human_entity_ref": exact_person_entity_ref,
    "body_to_world": explicit_body_to_surface_world_alignment,
    "channels": [{
        "channel_id": "left_ankle_surface",
        "body_point": "left_ankle",
        "status": "assumed",
        "method": "User identifies this authored annotation as the model ankle",
        "uncertainty": "Correspondence is authored, not measured anatomy or contact",
    }],
}
objective = prepare_surface_interval_objective(
    envelopes, body_bytes=original_body_bytes,
    binding_bytes=json.dumps(binding, allow_nan=False).encode(),
)
term = objective.evaluate(
    original_frame, body_sha256=sha256(original_body_bytes).hexdigest(),
    body_time_s=original_body_time_s, coordinate_frame=objective.coordinate_frame,
    positions={"left_ankle": candidate_absolute_model_ankle},
    jacobians={"left_ankle": candidate_ankle_joint_jacobian},
    parameter_names=exact_joint_column_names,
)
```

The named channel must already exist in the prepared envelope. No entity or
annotation is created by this example. The body source references must resolve
to that exact entity's construction observations/evidence. The explicit
`world-alignment-v1` maps **from**
`{id: "body:" + canonical_hash(body), length_unit: "human_model_unit",
axes: "x_right_y_away_z_up"}` **to** `meta["world_frame"]`. Supply the actual
calibration/placement hypothesis and its status/uncertainty; do not use identity
scale/placement as a default for missing calibration. Each original body time
must already equal its source time, and the declared clock must preserve it.
Unknown mapping or missing binding gives complete unavailable results.

An unavailable term has `objective_m2: null`, not a misleading zero error.
Available terms preserve signed XYZ outside deviations, active axes, the
conditional squared error, and optional derivative/gradient/Gauss–Newton matrix.
Unavailable channel counts remain visible even if other channels contribute.
No numeric confidence or likelihood is inferred from an authored range.

For example, a supported X range `[0.1, 0.2]` gives zero X residual at both `0.11`
and `0.19`; `0.26` gives `+0.06`. Nothing is pulled toward `0.15`. The range is an
outer box: zero residual does **not** prove the simultaneous XYZ point is
attainable by the original image/height intervals or physically supported.

### Existing kinematic producer join

`evaluate_kinematics(...)` consumes the unchanged
`body_refinement._kinematics(env, data, angles, True)` result directly:

- `relative_positions`: its 12×3 output in the existing `POINTS` order.
- `relative_jacobian`: its 36×19 matrix in the exact existing joint order.
- `pelvis_position`: current `data.site_xpos[env._site_ids['pelvis']]`.
- `pelvis_jacobian`: current `mj_jacSite` pelvis result restricted to the same
  `env._v_array` columns. **Required**, even when the caller establishes it is
  zero; the adapter never silently assumes fixed pelvis derivatives.
- Explicit `point_names`, `joint_names`, exact body SHA, coordinate frame and
  original reference time bind the numerical payload to its intended model.

The analytical integration calls that actual unchanged producer code under a
private import name, with deterministic model/data doubles and stub native
functions: one stub forward, 13 stub site-Jacobian calls, **zero real native
calls**. A nonzero pelvis position and derivative verify the relative→absolute
conversion. The test does not validate a real model or calibrate a physical body.

## Verification and frozen identities

| Owned path | Bytes | SHA256 |
| --- | ---: | --- |
| `src/weft/source_surface_interval_construction.py` | 26,618 | `143853e9d12beccd0e90b32d44094e33bf1e278c2e1c4ba8c812b77911697100` |
| `tests/test_source_surface_interval_construction.py` | 21,708 | `c83d05a059b093f44598d647a64cc1011170b59b5764a9a4dc586ae16357a190` |

Commands (existing `.venv-lab`; no installs or separate runtime):

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_source_surface_interval_construction.py -o junit_family=legacy --junitxml=.local/source-surface-interval-construction-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_source_surface_interval_construction.py -o junit_family=legacy --junitxml=.local/source-surface-interval-construction-tests-02.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_source_surface_interval_construction.py::test_same_video_hash_does_not_authorize_another_source_entity -o junit_family=legacy --junitxml=.local/source-surface-interval-construction-tests-03.xml
```

**35 distinct cases pass**, using receipt02's 34 passes plus receipt03's one
corrected case. The autouse test guard blocks real native/model/media/network and
subprocess work. No old envelope suite or real source study was rerun.

| Receipt | Outcome / duration | SHA256 |
| --- | --- | --- |
| `.local/source-surface-interval-construction-tests-01.xml` | 33 pass, one malformed analytical two-entity fixture failure /0.462s | `a63620a2f88f673478f000057345a98e5c5974775812dea418e6e4d50f92c21a` |
| `.local/source-surface-interval-construction-tests-02.xml` | 34 pass, same fixture failed on its next invalid identity /0.461s | `801012dd425b26be6a935ccfad2011b31e135b058ceed8e3b1a0f551134ccd63` |
| `.local/source-surface-interval-construction-tests-03.xml` | One corrected fixture/guard case passes /0.044s | `0b890f1f61a30083514db85c713db63fd2f0f3fa19743b90c9241aff21f9c472` |

Both earlier failures remain saved. The first fixture appended out-of-order
observations; after sorting, its invented observation ID still violated the
existing source/object convention. Correcting those fixture-only mistakes made
the exact intended cross-entity guard reachable. No source evidence or admission
threshold changed. Receipt02 also covers the new required candidate coordinate
identity (hence the affected suite rerun); after its fixture-only correction,
only the one failing case was rerun.

Receipt02 representative analytical workload: 600 original frames, 1,200 retained
channel slots, 685,766B cached joins; preparation **0.054927125s**, 60 cached full
19-column Jacobian queries **0.018537000s**. This measures this bounded local
fixture, not sustained UI/service responsiveness, real model speed or accuracy.
Read-only scoped whitespace checks reported no errors.

## Remaining boundary / next safe action

Root reviews and publishes the new three-file lane; no Git mutation by this
agent. Existing released envelope files and all prior failure artifacts are
untouched. No live handles, native/source jobs or paid cost.

**Callable construction term, not installed optimization.** A later reviewed,
versioned opt-in fitter must pass its verified body/source snapshot and exact
native kinematic/Jacobian payload to this term, declare any weighting/units
before fitting, and add its residual/gradient/matrix without replacing image
evidence, regularizers, original root/floor, update bounds or independent scoring.
It must preserve incomplete-channel denominators, every rejected alternative,
and original-start limitations. No existing callback currently installs this
term; no fitted candidate or reconstruction gain is claimed here.

Real-source use still needs an explicit compatible pixel annotation, static
surface interval, undistorted calibration and body/world map. The existing
nonzero-radial abstention remains. Authored scale/height/correspondence stays
authored; unknown contact, grasp, load and independent physical accuracy remain
unresolved. Do not rerun the jogging heel rejection or move its floor to obtain
a favorable result.
