# Source-surface interval construction

Root integration follow-up, October 31, 2026, 5:00 PM PDT: the CLI/worker/report
are connected. One authored native worker completed; the outer acceptance failed
on a blocked subprocess-attempt assertion and remains failed. A separately
verified saved-only inspection used the actual output without rerunning the fit.
The independent saved audit passed. [Commands, exact artifacts, failure and next
action](surface-interval-workflow.md). No compatible real-video interval fit or
physical-contact validation is claimed; no live native jobs remain.

## Opt-in fitter installed — frozen, 2026-10-31 4:54 PM PDT

Root approved the next narrow lane after publishing the callable boundary at
`f1bfe83`: edit only `whole_body_reference_refinement.py`, new
`tests/test_whole_body_surface_interval_refinement.py`, and this handoff. The
other two fitter files were inspected but do not need edits. The released
envelope/objective modules remain unchanged. Root owns CLI/config/custody/worker;
`support_ui` owns report rendering/validation and its new tests.

API: existing `propose_whole_body_reference_refinement` gains explicit
`proposal_objective='surface_interval_v1'`, `surface_interval_objective` (prepared
released object), required authored `surface_interval_weight` in `(0,1000]`, and
optional tighter `surface_interval_settings` for <=20,000 evaluations /4 MiB
trial journal. New method `blocked-whole-body-surface-interval-reference-v1`.
Supplying interval arguments in old modes rejects; absent mode retains their
existing behavior/output. Fit/final budgets remain 30s+8s, <=120,000 geometry
evaluations and <=16 MiB complete output. Additional source and before/final
interval audits each reserve <=4 MiB before geometry; no slot truncation.

The term augments the existing image/regularization objective and exact QP path
only at original source knots with existing usable image evidence. No interval
target is invented on the interpolated control grid. Original first pose,
root/floor/nonreference fields and all existing hard gates stay unchanged. Each
available channel must additionally keep its outside-box distance nonincreasing.
The opt-in geometry path restores current pelvis position **and derivative**.

All source slots and actual ranges, exact body/source/binding pins, baseline/fresh
final outcomes, actual weight and every QP/backtrack alternative are retained.
Pending backtracks are recorded before evaluation so budget interruption cannot
erase an attempted correction. The evaluator limit reserves a complete final
audit; journal capacity reserves a complete 16-KiB record before each attempt.
Independent fresh final geometry is still mandatory; a rejected final result is
reportable, not forced to match a cached fit.

No native/source job is authorized in this lane. Root separately prepared one
fixed authored 3-reference/10-source-slot/1-channel native fixture; no favorable
outcome is required. Saved-input review found no compatible real interval
bundle: jogging uses unsupported nonzero radial calibration and a camera body
this fitter rejects. Neither its heel rejection nor its floor is changed.

**Implemented in the existing optimizer**, not a parallel fit/demo. The default
image and floor-first routes retain their old behavior; in particular invalid
legacy backtracks do not acquire a new `_loss` call. The existing image QP now
adds the opt-in weighted interval gradient/matrix and accepts a combined-loss
improvement only when every old nonlinear check plus each interval channel's
exact nonregression succeeds. The released consumer's historical
`inputs.optimizer_installation` is retained, while the new fit report says
`optimizer_installation: installed_in_existing_whole_body_fitter` separately.

### Exact public call / schema

```python
result = propose_whole_body_reference_refinement(
    exact_human_bytes, exact_derived_pose_evidence_bytes,
    proposal_objective="surface_interval_v1",
    surface_interval_objective=prepared_surface_objective,
    surface_interval_weight=2.0,  # Explicit authored choice, fixed before fitting.
    surface_interval_settings={
        "maximum_evaluations": 20000,
        "maximum_journal_bytes": 4 << 20,
    },
)
```

The ordinary optional `joint_mask` / tighter `settings` remain available. The
interval evaluation bound must cover twice the original reference count; its
fit portion reserves the other half for fresh-final evaluation. Journal bounds
must allow one complete 16-KiB record. The weight is not a confidence estimate;
its units convert supplied-world-metre-squared error into the existing model
objective. No automatic weight search is installed.

`report.surface_intervals` profile `whole-body-surface-interval-fit-v1` contains:

- Complete `inputs` inspection/pins/assumptions, actual `weight`, authored status,
  work limits, installation status and unknown contact/physical validation.
- `source_slots`: every original frame, source status, original body-reference
  link, every channel's original status/reason/partition and actual unchosen
  `world_envelope_m` bounds.
- `references`: original reference index, frame/evidence/time plus complete
  consumer `before` and independently recomputed `after` results. Neither audit
  call requests derivatives. Missing errors remain null, never zero.
- `summary`: complete frame/channel counts, available/unavailable reference
  terms, evaluated channel counts, before/after summed squared outside error,
  authored weighted sums and peak outside distance in supplied world metres.
- `evaluations`, `trial_journal`, and `journal_bytes`. Sequences start at zero;
  bytes equal the sum of compact `json.dumps(record, allow_nan=False,
  separators=(',', ':')).encode()` lengths, with default ASCII escaping and no
  newline/array-punctuation allowance. Every record is <=16 KiB.

QP journal records retain source/reference/block, original correction, solver
status/iterations, attempted step, feasibility residual and disposition. Rejected
nonfinite direction elements are tagged `{"nonfinite":"nan"}` / `"inf"` /
`"-inf"`, never invalid JSON. Backtracks retain the exact proposed 19-angle
correction in the runtime-pinned `NAMES` order, trial index, before/after costs,
every old and new gate outcome and quantified floor/depth/image margins.
`evaluation_incomplete` retains the attempted vector and unknown/null outcomes
when a deadline/work bound interrupts it; no trial silently disappears.

The candidate marker carries profile, exact input bindings, actual weight/status,
work limits and the same summary, not another duplicate of the full slot ledger.
The new gate spelling is **`surface_interval_nonregression`**. A fresh final
geometry failure stays a rejected result even if the original first angular
reference is unchanged and its recomputed interval error differs.

### Focused verification / exact frozen pins

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `src/weft/whole_body_reference_refinement.py` | 57,162 | `1637d8c0b1c03343a2e5977855038c904481c71133867a990d705d6a112a7619` |
| `tests/test_whole_body_surface_interval_refinement.py` | 18,818 | `0da1669569bafc59b635fee62960fae2a9f1e00528a5bde3e58672246efb394c` |
| `.local/whole-body-surface-interval-tests-01.xml` | 6,510 | `949b019dc85dc79cc62e10f2b59198fff22fa5bfb57a49ab419a9a30985c708e` |
| `.local/whole-body-surface-interval-tests-02.xml` | 1,395 | `fbb63fa77c6f622d8b251710875efd967e5be2a77e987726769548b3360adbe3` |

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_whole_body_surface_interval_refinement.py -o junit_family=legacy --junitxml=.local/whole-body-surface-interval-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_whole_body_surface_interval_refinement.py::test_opt_in_native_geometry_wrapper_captures_current_pelvis_without_a_second_forward tests/test_whole_body_reference_refinement.py::test_coupled_leg_fit_improves_source_without_new_floor_conflict tests/test_whole_body_reference_refinement.py::test_floor_first_image_neutral_direction_can_clear_violations_without_native_work tests/test_whole_body_reference_refinement.py::test_floor_first_default_and_explicit_image_v1_keep_identical_output_and_calls tests/test_whole_body_reference_refinement.py::test_independent_final_geometry_cannot_be_replaced_by_optimizer_caches -o junit_family=legacy --junitxml=.local/whole-body-surface-interval-tests-02.xml
```

**25 new pure cases + four affected legacy controls = 29 distinct passes.**
Receipt01: 24 passes /0.199s; receipt02: one new changed-wrapper test plus four
affected legacy controls /0.076s. No test failures in this fitter milestone; no
unchanged 35-case consumer suite rerun. Scoped whitespace checks passed.

The actual existing `_prepare` and constrained QP reduced the authored
three-reference fixture's summed interval loss from **0.003137960034013614** to
**0.0010459875272815368** supplied-world-m², with first pose frozen and every
original final gate true. There were **44 analytical geometry evaluations and
19 interval evaluations**, zero native engine/model/media/network operations.
This is a deterministic optimizer integration result, not real reconstruction.

Negative checks retain nonlinear floor/image/depth rejections, unchanged floor
infeasibility, fresh-final geometry rejection, unknown/null observations,
nonfinite rejected directions, evaluation/journal exhaustion with complete final
audits, and interrupted pending trials. A focused test calls the modified actual
`_NativeGeometry.evaluate` method with native stubs: no second forward is added;
live pelvis position/19-column derivative are included only for opt-in calls.

Temporary analytical schema fixture (not a source/native study; **no longer available**):
`/private/var/folders/l9/yglj_5vs151cvc2fw14h6p3c0000gn/T/pytest-of-ivan/pytest-635/test_interval_term_improves_ex0/`.
The test wrote result, exact original human/evidence, body binding, observations,
camera, intervals and annotations. Its `result.json` SHA was
`0154cedb7cf132add0986b11710ca129ab9086fde444268ec3af9c3f7a65eaac`.
The path/hash/counters and passing assertions remain JUnit properties, but later
parallel pytest retention removed the temporary directory before the report
agent's saved-payload check could run. No preserved eight-file copy is known.
That additional schema-alignment check is therefore **unavailable**, not passed;
no fit/test was rerun to recreate it. Root's already-planned immutable native/
report acceptance will cover the actual path. Future cross-agent payloads must
be copied into a unique owned workspace artifact before more pytest runs.
Do not reinterpret this analytical test as a captured-video or accuracy result.

### Next safe action / live ownership

Fitter/test bytes are released to root for final delta review and its **single
separately prepared authored native acceptance**. Root owns CLI/config/custody/
worker publication; report agent owns saved-only inspection. No live job or Git
mutation in this lane. The parent prepared
`.local/surface-interval-native-inputs-01/fixture.json`, reported SHA
`b4083087e495cea174eb92efde1bef795007b8ce4b3ff9707d96c5e1453f0942`,
before any fit: three original references, ten source slots, one channel, fixed
authored map/ranges/weight. No favorable outcome or tuning is required.

Real unfamiliar-video interval qualification remains open. The saved-input
review found no compatible package; jogging camera radial coefficient
`0.00047528394168603385` is unsupported by the released envelope contract, and
this fitter deliberately rejects its `camera_reference` profile. No recovered
heel, contact/load/grasp, floor shift, anatomy claim or physical validation is
added. Continue only from a separately source-qualified or explicitly authored
bounded input plan; do not rerun unchanged source failures.

## Published callable boundary — 2026-10-31 4:34 PM PDT

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
