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

## Task projection checkpoint

The existing scorer is unchanged. New `shared_exercise_feet_projection.py` maps
25 task rotations to engine indices `[0:19] + [22:28]`, optionally followed by
three task-root channels mapping to `[19:22]`. All object-force channels stay
zero. The full v5 human observation block has 197 values; a legacy 155-value prefix
is not a valid projection. Original 22 substep arrays are projected explicitly,
with clocks/failures/floor penetration/equipment/wrists preserved. New foot
contacts remain separately inspectable and receive no legacy support credit.
Only exact canonical engine geometry/segment pairs are accepted. The complete
mapping is part of the immutable projection contract hash.

The one-time base-scene preparation reuses existing world validation, which can
compile mass-check models. It is not a native-free hot-path API. Its result is a
derived scoring identity, not original file-byte recovery; actual v5 world and
derived base scene/task require separate hash bindings. Ankle effort is separate
unscored telemetry; learned policies and ground-load requirements remain unsupported.

Frozen pins:

- Module: `c6004ad05628864bf1dc7d057d47b29ebac45aa647fbad91d7af906b5e0b2a69`.
- Test: `2f951017d8e0b3b3ecf7e39dd641458ca9f1eff1723f7ad960388469ffe701da`.
- Projection contract: `09bf31a0067692db77fcbe8ee36a0e6b8465eaccb82bb0339132072389eb1aab`.

The initial 32 analytical checks passed with only human mass replaced by a
`70 * scale**3` fixture; all native/model/network/process entry points were
blocked. That original command emitted 32 progress dots but no JUnit summary.
Its exact tool transcript is retained in
`.local/shared-exercise-feet-projection-tests-01.json`, SHA
`bd58e20e4470bedccd6a4040d06daf3960c7ed6a00327a95d89aa7aaaa24c9bd`;
no JUnit receipt was fabricated and those checks were not rerun for logging.

Independent review found an inverse alias gap: unknown geometry claiming a known
forearm segment could receive old support credit. Preserve the two expected
failing regressions in
`.local/shared-exercise-feet-projection-contact-before-01.xml`, SHA
`30d8f19676f3e0fc1a7aa9364b52c37ba7f7e4924e2047a483f1a693053e5009`.
The canonical mapping fix passes 16 new/affected cases; 21 unchanged cases retain
the earlier pass, 37 distinct checks total. After receipt
`.local/shared-exercise-feet-projection-contact-after-01.xml`, SHA
`f886ba59acc38d27376f31e508ec878d9fe9041dd6afc90f18271eebabde75a1`.
Root reviewed the correction and exact result bytes. A bounded 128×256-contact
projection measurement took 60.2 ms total, without native execution; it establishes
only adapter timing, not full simulation or UI responsiveness.

Next owner: `/root/support_learning` is implementing only new
`shared_exercise_feet_task.py` and its focused test file. It must reuse unchanged
legacy metric accumulators, create just one actual v5 engine, expose explicit
root-off controls, retain 197 human observation values, and bind actual/base/task/
projection/reference identities. Pure tests only until root freezes a bounded
native acceptance. Root owns strict snapshot/recording/lab integration; no existing
scorer/gate/UI file may change in that lane. No public v5 task/lab claim yet.

## Feet task and recording preparation

Recorded **October 31, 2026, 1:27 PM PDT**. Published base is
`4b0bad9940cda425d1813bae323d8285acc72424`. New task/state/recording files and
the generic recorder dispatch are reviewed, tested at their stated boundaries,
and uncommitted. Root owns the next frozen native acceptance and lab integration.
Independent avatar work is excluded. No native job is running.

The versioned `shared-exercise-feet-task-v1` uses one actual v5 world and the
unchanged base task accumulator. Public actions are 25 rotations plus optional
three root translations; root assistance defaults off and object forces stay zero.
Observations retain all 197 human values. Snapshot identities distinguish the
actual engine from derived base scene/task scoring. Ankle hold/effort and excluded
foot contacts are unscored diagnostics: latest substep plus bounded per-pair
aggregates, not a full force history or recovered physical contact.

The dedicated state validator preserves original metric-validation algebra while
validating the actual v5 engine separately. It prepares models/hashes once, checks
all reset/failure/clock/action/null semantics, and rejects snapshots above 480 KiB.
Independent task and state reviews found no core blocker. The original scorer
`shared_exercise_task.py` remains SHA
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.

New `shared-exercise-feet-recording-v1` allows only manual/coast/authored controls,
exact event schemas and full-state/base-score replay. Learned policies, reference
proposals and avatars remain unsupported. Its per-event limit is 512 KiB; the
128 MiB journal and all legacy limits stay unchanged. Direct task rendering caches
the dedicated PIL renderer, still validating each current snapshot.

### Preparation evidence

- Task: 23 distinct analytical cases; initial maximum-contact fixture failure and
  its correction are retained. Final receipt
  `.local/shared-exercise-feet-task-checks-01.json`, SHA
  `8e2c83a9ae9234533b9aca651e7680d82d8fb621e8b2529c9a70c7f77a64ce6e`.
  The cached-renderer replacement test separately passes once, 0.158 s:
  `.local/shared-exercise-feet-task-render-checks-01.json`, SHA
  `59f3622b5759de9c2b37adb2407f0b7078d4ad5b65b17e9d1f89b5dfd097e655`.
- State: 22 analytical cases, 0.585 s,
  `.local/shared-exercise-feet-state-tests-01.xml`. Maximum analytical state is
  71,353 B with 256 latest contacts/24 pairs. A 32-validation measurement took
  0.127 s with the native engine guard replaced; not native/UI performance.
- Recording: original 115 mocked cases pass,
  `.local/shared-exercise-feet-recording-tests-01.xml`, SHA
  `b0dc9517347bdabeed78cf4bd32bc9dcf87041252cf851839d94defe4f928d5d`.
  Independent review then found arbitrary controller/avatar event metadata could
  pass through to exports. Nine expected failing regressions are preserved in
  `.local/feet-recording-event-contract-before-01.xml`, SHA
  `192b8c9917371029bd7a4c9f26cd620385dd71f3d38b99fc4a0adf7c30569f34`.
  The exact feet-only event schema passes all 76 new/affected cases, 0.133 s,
  `.local/feet-recording-event-contract-after-01.xml`, SHA
  `ebb19915569d0c520c1b04ffa1892d1dcbef7d42f908a125451420521de78afd`.
  There are 136 distinct recording cases after 21 additions, not 191.
- Legacy dispatch: all 65 `test_recording.py` / `test_shared_exercise_recording.py`
  checks pass, including short authored exports/exact replay/recovery. Frozen
  `.local/feet-recording-legacy-runtime-01` contains committed 4b0bad9 plus only
  `recording.py`; manifest SHA
  `4e8ecbe8b92fbbdae77d5b1f71c2c377300f10135dd8e6e838e957ad0e2a925c`.
  `.local/feet-recording-legacy-plan-01.json` SHA
  `940261808ebc62f524000b8f973a835f48ae4620c414e015a58d888896688a04`;
  `.local/feet-recording-legacy-process-01.json` records exit 0 /2.358554 s /
  164,397,056 B waited-child peak RSS. Bounds: 60 s wall, 45 s CPU, 1 MiB per
  log; memory observed, not hard-capped. No real source decode/inference/training.

Frozen product SHA-256 identities before native acceptance:

- Task `a6de8094d36a77e2f9c7fef472475aa2007f6e4f29376bf7dad5b07191c8eecf`.
- State `e80c1b9fe3f25924c8f274dd8c3f9669764a2f8916f3284446b07182f635ba86`.
- Task/projection/state combined adapter
  `d007e0d021a927e183acdb88522c45261103ddb1986a8e463613dbb4b483b87f`.
- Feet recorder `0d2c719fb60b0743781336abed0b6a8bd20e34511ea3c1b9f95c3591e70a7a51`.
- Generic recorder `3db0f025815a56cbaa7448841e89197d8dad653f72656fc16e35975ff1f46477`.

Next: freeze the six-case test with these implementations, then root executes
once. Bases v2/v3/v4 × roots off/on, two manual steps/reset/two authored steps
and exact replay per case (48 controls maximum across six cases). Only v3/root-off
exports six frames through ordinary PIL/MP4/GIF/HTML and checks direct render.
Preserve physical failures/early termination without fixture or gate tuning.
No source inference, training, network, browser/native GL, paid calls or avatars.
Lab/CLI/UI integration and grounded source recovery remain unfinished.

## Native task and recording checkpoint

Completed **October 31, 2026, 1:30:09 PM PDT**. All six authored cases pass:
v2/v3/v4 bases × roots off/on. Each records two manual ankle actions, reset29,
two authored actions, then replays the full state/base scores/diagnostics exactly.
Across six cases: 24 recorded +24 replayed controls, six recorded resets, 36 events.
The v3/root-off case also checks two identical live renders without physics advance
and exports six frames through ordinary PIL/MP4/GIF/HTML. No full-interval task
success or source reconstruction claim is made by these short fixtures.

Preserved failures, with all product bytes unchanged:

- Failed01 incorrectly expected the derived scoring scene to equal original
  bytes; the derived scene explicitly adds learned_policy_available=false. Six
  cases stopped before task construction/actions. 0.650367 s /126,337,024 B RSS.
  Receipt `.local/feet-task-native-tests-01.xml`, SHA
  `e706b77230119faf7188e01e8229aec4928606c9a181afd23a7c25c7fe031325`.
- Failed02 then expected nonexistent engine scene_profile/scene_sha256 fields.
  Actual snapshots use profile; the outer task pins the actual scene. Six cases
  stopped at initial snapshots before controls. 0.766905 s /131,170,304 B RSS.
  Receipt `.local/feet-task-native-tests-02.xml`, SHA
  `55e232b4394f399ba978e7c2b0565e110d2f1c2ad0df6e55f26eac39e3cd18a7`.
- Corrected only those test assertions, statically inspected the remaining
  assertions against actual producers, then ran03. All six pass in 1.776699 s /
  155,746,304 B waited-child peak RSS. Receipt `.local/feet-task-native-tests-03.xml`,
  SHA `4b5ea3017b3b2b8083285b8acd45e79bfceb2dacd39a58586d6b55c9eca60910`.
  Final native test SHA
  `f93793993a8612924c85be311aba8abc9f388aef586f2daa9c28afdb71fd00aa`.

Runtime03 is committed4b0bad9 plus only the four explicit task/state/recording
product overrides and native test; no dirty avatars or later lab/UI/receipt work.
`.local/feet-task-native-runtime-03/runtime-manifest.json` SHA
`22164ea6269adc04cd233b1d4d58aa1a71575598cf19bde5351972acaebd5a9b`.
Plan `.local/feet-task-native-plan-03.json` SHA
`c1dfec420cc521478cc41fa796ec90a069a1d0a64d4ae36022e8e97b4ab6c7cf`.
Process `.local/feet-task-native-process-03.json` SHA
`7a7b719adb1c6b97851f86718dfaa597e2d64de9e738f61376c47484a8a80515`
pins all 28 output files /1,643,411 B. Bound60 s wall/45 s CPU/1 MiB logs/
64 MiB128files aggregate; each case enforces eight controls including replay.
Memory is observed, not hard-capped. All jobs are terminal.

Saved outcomes live under `.local/feet-task-native-output-03/test_native_feet_task_record_l0`
through `...l5`; ignore pytest's duplicate current symlink when counting cases.
All retain four actions and no early simulation failure. At saved boundaries,
v2/v3 roots-off/on record16/8 foot-floor contact details; v4 records16/8
foot-static contact details. These are snapshot counts, not unique physical events
or measured forces. Ankle diagnostics remain unscored and v4 requirements still
refer only to the original left forearm. Every object-force channel is zero.

The exported recording is `.../test_native_feet_task_record_l2/recording/`,
six-member manifest SHA
`895ef2626eb80515a2e2948ff2a5aafa711b352ff5d06799431e308c1024a7dc`.
After this run, only the replay receipt return gained foot_reference_sha256; its
one affected mocked replay test passes, receipt
`.local/feet-recording-reference-receipt-01.xml`, SHA
`1efd6f96e0775fd88ca5f744bfa44afd6f741b4237b3d3705c47b2a5b01808dc`.
No physics/state/scorer/adapter identity changed and the native study was not rerun.

Next ownership: root owns shared_exercise_lab.py/lab.py/cli.py and sealed candidate
construction. support_ui owns capabilities.py/lab.js/replay HTML/lab_console.py
and focused mocked tests. support_learning owns recording_receipt.py and feet
receipt tests. exercise_reconstruction resumes the source-backed unfamiliar-video
calibration/contact lane; no new source study is authorized until its bounded
hypothesis is reviewed. This native checkpoint does not establish lab/UI readiness,
learned policies, recovered ground/equipment contact or physical accuracy.
