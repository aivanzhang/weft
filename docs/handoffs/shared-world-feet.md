# Shared-world feet integration

Owner: `/root/support_learning`. Workspace: `/Users/ivan/kintsugi/weft`. Recorded **2026-10-31 12:32:54 PDT**. Pure implementation is tested, frozen and released to root. Root owns shared-world engine, task, lab, recording, native acceptance and Git. This lane owns only `foot_body.py`, `shared_world_feet.py`, its new contract tests and this handoff. No live study or native job exists in this lane.

Git at release: `foot_body.py` is modified; the new module, test and this handoff are untracked. No Git mutation or other product-file edit was performed by this lane. Root publishes these changes with its coordinated integration; do not overwrite concurrent engine or avatar work.

## Contract and integration interface

The new `shared-world-feet-v1` extension is intended only for root's explicit `shared-world-v5` scene. Existing v1–v4 profiles, task gates, source body/root/floor and original nineteen-joint score must retain their meanings. This extension does not itself admit a scene or qualify learning or ground forces.

Exports from `weft.shared_world_feet`:

- `ANKLE_NAMES`: left ankle Z/Y/X, then right ankle Z/Y/X. `ANKLE_LIMITS`: six authored `12.0` torque limits. The engine preserves its original 22 channels and appends these six; it must not adopt the separate `FootEnv` ordering.
- `build_foot_extension(spec, reference_bytes)`: bounded exact UTF-8 JSON input, maximum 8 MiB and the existing reference validator's 900-frame limit. Uses `validate_reference`, both fixed source-inferred shapes and both available feet at the body's exact original starting time. No missing-pose interpolation or later-start substitution.
- `validate_foot_extension(spec, extension)`: validates all source/body/reference/start bindings and returns an independent copy.
- `append_shared_feet(root, spec, extension, *, scale)`: appends two capsules and six motors to an existing parsed MJCF tree and returns that same tree. It requires the exact unchanged nineteen-body-plus-three-root actuator ordering before mutation. Shared-world limbs are already scaled, so pass the scene's scale; there is no later common scaling pass.
- `initial_ankle_angles(spec, extension, knee_rotations)`: returns six local intrinsic ZYX angles. Supply `left`/`right` nominal-original-start knee world matrices from the same native model, before reset jitter. Those matrices already include the source `body_frame`; the helper uses `knee.T @ foot_world` without applying it twice. Freeze these values as authored initial-angle hold targets, not video ankle imitation.
- `added_foot_mass(scale)`: both feet total `1.0 * scale**3` kilograms-prior. Include this added mass when computing complete human-subtree mass, root-force limits and support-load normalization.

The extension's exact fields are `schema_version=1`, `profile`, `contract`, `body_model_sha256`, `source_sha256`, `reference_json`, `reference_sha256`, `reference_canonical_sha256`, `initial_reference={time_s,frame,evidence_id}`, and `target_mode=authored_initial_angle_hold`. `reference_json` preserves the original UTF-8 text; its byte hash is verifiable offline rather than an uncheckable reference to an external file. The embedded reference remains unchanged, including later missing-foot rows. The target uses only its exact original-start orientations.

`foot_body.append_foot_capsules(root, spec, reference, *, scale=1.0, name_prefix="")` is the reusable pure attachment helper. It requires both unique existing knee bodies and one actuator element; duplicate names or invalid structure are rejected before mutation. Added geometry scales linearly, each foot mass is `0.5 * scale**3`, and authored torque limits stay ±12 like existing body torque limits. Existing body/floor/gravity/collision setup is untouched. `foot_xml` delegates at unit scale, preserving its legacy output.

## Verification and limits

- A bounded pure XML comparison against the committed pre-extraction `foot_xml` passed with native model/data/forward/step/render entry points blocked: 8,895 bytes, SHA-256 `0475f885ab5dba0adbe7a3e44818925471c514315813660e401ca0251f8bf286`, byte-identical before/after.
- **19 distinct pure contract cases passed.** Initial full-file run: 18 passed, zero failures/errors/skips, 0.108 seconds. A final explicit old-channel-order guard then passed its four affected cases (three scale cases and one new rejection case), zero failures/errors/skips, 0.095 seconds; unchanged cases were not rerun. Native model/data/forward/step/render entry points were blocked. Do not infer a passing native integration from these pure checks.
- Receipt `.local/shared-world-feet-contract-tests-01.xml`: SHA-256 `9fdd4f93db94bae2c5ce48b2b32b7145b95c1c21f81c75f155407c67d951a217`. Command: `PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest tests/test_shared_world_feet_contract.py -q --junitxml=.local/shared-world-feet-contract-tests-01.xml`.
- Receipt `.local/shared-world-feet-contract-tests-02.xml`: SHA-256 `cbf97e6632cb048792218986736ed4ba2d0ebf3064a14dde7e09dd35d37b996b`. Command: `PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest tests/test_shared_world_feet_contract.py -q -k 'append_preserves_original_tree or shared_attachment_rejects_reordered' --junitxml=.local/shared-world-feet-contract-tests-02.xml`.
- Capsule radius, mass, torque and angular limits remain authored priors. Source-inferred foot axes are not measured sole normals, anatomy or physical contact. Body and floor are not refitted to make added geometry pass.
- Ground-force qualification and learning remain explicitly unsupported. Added foot collisions must participate in existing floor limits; ankle tracking must not dilute the nineteen-joint body error.
- Caller responsibility: knee matrices must come from the same model at the nominal original start, not from a later reference or a jittered reset. No native model is constructed by these helpers.

Frozen source SHA-256 identities:

- `src/weft/foot_body.py`: `a0595dbb3e0a84f2a598fcecb2c06ef6e28cbb7661cf56334e6a8ac932c944d7`.
- `src/weft/shared_world_feet.py`: `6a9f98eab7dbc2b7424a54eb01e0be3f4babf20c06b88e9ca4c6fe0f7a1d1575`.
- `tests/test_shared_world_feet_contract.py`: `74b48abb9518599ce26fd4166f6ff690f7f53323a810636e18b8011a05bf88c4`.

The new test's `inputs()`/`raw()` helpers are stable analytical fixtures for root's native-test owner. Importing those functions does not authorize a source study. Rebind the source and canonical body hash after changing the fixture body. Native execution remains coordinated by root after integration freeze.

## Remaining work

- [x] Complete focused pure contract checks and record immutable receipts/hashes here.
- [x] Root integrates explicit v5/base-profile admission, 28-channel engine layout, fixed initial targets, complete mass accounting and separate ankle telemetry in the one shared model/clock; see the checkpoint below.
- [ ] Root preserves original task gates/source interval, disabled root-force defaults, manual interaction and exact recording/replay; any unsupported learned mode must reject clearly.
- [ ] Root coordinates one bounded native acceptance after freezing the full integration. Preserve original source/root/floor and every failed admission or floor check; do not run a new source reconstruction study.

Completion requires the focused pure checks plus root's separately recorded coupled-runtime acceptance. Neither completion of the helper nor adding foot geometry establishes recovered ground/contact or video ankle imitation.

## Coupled engine checkpoint — October 31, 2026, 12:48 PM PDT

Root's engine integration is frozen/reviewed; this is not yet task/lab admission.
Engine `src/weft/shared_world.py` SHA-256:
`cd0af3d569c40b96adba971801d75397901e2ee455d33feb1ec8b7de1ed5c7a5`.
Native test `tests/test_shared_world_feet.py`:
`ba4094aa2e032884146424d3a50702236f418a92357d86acbe4895cd8a5487fc`.
The helper/foot XML/contract-test pins above remain current.

`add_feet_to_shared_world(base_scene, exact_reference_bytes)` constructs the new
versioned scene. `SharedWorldEnv` then exposes Gym reset/step, 28 human motors,
197 human observation values, fixed original-start ankle targets, foot collisions
and separate ankle errors. Base task/source/root/floor values remain unchanged;
object actions follow index28. Six ankle torque limits are ±12 model-torque units.
No existing policy may be relabeled for the new shape. A substep callback must
explicitly opt into `substep_observer_motor_count=28`; old scorers reject it.

Prepared snapshot validation derives nominal foot targets once from its own
zero-step model. Prepared rendering reuses that validator; live environment
rendering reuses its own model-derived targets. Every state is still validated.
Neither render path recompiles a native model per frame or advances simulation.

### Verification receipts

- `.local/shared-world-feet-native-tests-01.xml`, SHA
  `5010d3b8fc1bac078427c4da4a0ab93adefc22c849d064feb02936d408ffd412`:
  ten pass, one test-only wrong-argument call to `rotation_zyx`; 0.694364 s,
  126,844,928 B observed waited-child peak RSS. Product runtime unchanged.
- Corrected only that call and reran only its node in02: one pass /0.476527 s,
  125,599,744 B. Receipt SHA
  `34789a3512cf7051ae8423b821efc9b85c79d85cb438535105ae56d4d742d016`.
  Frozen product runtime01 manifest
  `bd87b756670285041c4bdc1992585ec9fd82aca428af279a64a38d6251657839`.
- `.local/shared-world-feet-native-tests-03.xml`, SHA
  `d12be77bd18931e3e6dcedabcc406136f3e5855993bf96a8d78226bd26046b11`:
  one new zero-step PIL-render regression plus eighteen affected legacy checks,
  all pass. 1.239413 s /143,507,456 B observed waited-child peak RSS. Runtime03
  manifest `a8cf0be3c9bf29dffeea7217f27c4d0f7ee2b9e7a9bbc2bb048b3d0bdd3c57fc`.
  Plan `.local/shared-world-feet-native-plan-03.json` preserves exact nodes/command;
  supervisor allowed60 s wall/45 s CPU/1 MiB per log. No memory-cap claim.
- Runtime exports use committed342a3bb plus only the three product overrides and
  focused tests. Independent avatar and source-surface WIP are excluded.
- Twelve distinct new native/render cases now pass, in addition to nineteen pure
  helper cases and eighteen legacy cases. Earlier unchanged passing tests were not
  rerun. Native action/substep counts in plan01 were estimates, not instrumented
  execution totals. These results are not a whole-repository or physical benchmark.
- Independent read-only engine review found no additional blocker in motor order,
  mass, collision masks, original scoring, reset/start targets, telemetry or legacy
  compatibility. No protected evaluator file changed.

### Next safe action

Integrate the new controls through the existing human-controlled exercise task,
lab and recording, with root assistance still explicit and default-off. Preserve
original nineteen-joint scoring and independently account for ankle effort/telemetry;
never silently slice or enlarge old policies. Ground-load qualification requires
new explicit telemetry/task semantics; current generic floor contacts do not supply
measured ground forces. Keep all existing failures and source clocks. Run one
frozen bounded workflow only after these adapters are complete. All current native
processes are terminal; no live handle or paid work remains in this milestone.
