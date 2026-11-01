# Split-limb feedback actor

## Released scope

The optional **v5 actor/policy contract is implemented and analytically tested**. It separates arm and leg gain corrections without adding observations, forces or a new action calculation. No training, simulation, source reconstruction or physical validation ran in this lane. The previous v4 coupled result remains infeasible and unchanged; this implementation is not a claim of better control.

Owned files:

- `src/weft/shared_exercise_feedback_residual.py` — `3aefc6ef0e5be09fc3e710ffd277dc200c44bbc0b6051f852a13795f6a89268c`.
- `tests/test_split_limb_feedback_actor.py` — `1fbc25daa26e5d86931c88b93e49f1888d10e9ddd52ee14dfb8237cc7bff6d30`.
- This handoff. Root owns trainer/CLI/workflow integration; the UI lane owns common policy/lab/JS routing. No shared ownership or native job is active in this lane.

## Explicit contract

`POLICY_PROFILE` remains `shared-exercise-policy-v4`. Opt in with `SPLIT_POLICY_PROFILE = "shared-exercise-policy-v5"`; coefficient shape never chooses a profile.

| Field | v4 default | v5 opt-in |
| --- | --- | --- |
| Algorithm | `shared_exercise_feedback_residual_feasibility_cem_v1` | `shared_exercise_split_limb_feedback_residual_feasibility_cem_v1` |
| Actor profile | `shared-exercise-bounded-feedback-residual-v1` | `shared-exercise-bounded-split-limb-feedback-residual-v1` |
| Groups | `torso, proximal, distal` | `torso, arm_proximal, arm_distal, leg_proximal, leg_distal` |
| Coefficients | 3×2 | 5×2 |

The two columns remain normalized position and velocity feedback. Every coefficient and realized joint residual stays within ±0.25. Features retain their ±1 clipping, actions their ±1 limit. No learned bias, root coefficient, object force, equipment-state feature, phase feature or new privileged input is introduced. Joint names, not object IDs or clip-specific information, determine groups.

The torso's three original axes use group 0. Both shoulders' three axes use group 1; elbows group 2; both hips' three axes group 3; knees group 4. The v5 mapping requires all 19 exact original joint names and, if supplied, the exact three root channels. Reordering joint names changes only the corresponding output order. Unknown, duplicate, root-as-joint and object-force channels are rejected.

APIs:

```python
contract(profile=POLICY_PROFILE)
coefficients(value, *, profile=POLICY_PROFILE)
joint_groups(names, *, profile=POLICY_PROFILE)
make_policy(env, weights, seed, score, rank, hashes, *, profile=POLICY_PROFILE)
validate_shared_exercise_feedback_policy(policy, scene, task)
prepare_shared_exercise_feedback_policy(policy, env)
```

`contract()` returns a fresh dictionary with exactly `profile`, `algorithm`, `actor_profile`, `groups` (tuple), `coefficient_shape` (tuple), and `scope`. Validation/preparation infer only the exact document profile. Existing policy keys, scene/task/evaluator/gate/runtime pins, privileged-input declaration and feasibility-rank checks remain mandatory. `make_policy` now rejects a shape inconsistent with the requested profile instead of creating a document that later fails validation.

The inherited scene/task admission boundary is unchanged: this does not admit camera, feet or static-support tasks through the legacy actor. Coupled and person-only metadata routing was checked with the admission provider explicitly mocked, not with native scene compilation.

## Numerical and runtime invariants

`action_sample` has an **identical AST** to the original actor (`e6605b718a2208e7d50315aa0cc3165e6bf593f9e3b5f395d9cac93a32ca67d1`); AST hash `9a09adeac983f12a02a9e270985eb3cfe676fc941e2043de8cdcced6f319c793`. Only validated row routing changes. Zero coefficients reproduce the exact authored action bytes. Tying v5 arm/leg weights to v4 weights reproduces v4 action, feature, authored-action and residual bytes. Root actions always copy the existing authored result.

Preparation freezes detached, read-only weights/groups. Each action reads the existing feature matrix once, with no scene admission, source-file hashing or model construction on the action path.

The runtime file list is unchanged; `mobile_human.py` and `spatial_pose.py`, now direct constant imports, already belong to that closure. Actual runtime hashes truthfully change with actor/integration source bytes. Historical policies and training artifacts are **not repinned or reinterpreted**. The v4 output golden is exact only when the supplied scene/task/runtime/version identities are identical; it does not exempt an old policy from current-runtime equality checks.

## Verification and preserved evidence

**47 distinct analytical/mock cases passed once** in 0.350024 seconds including imports. XML: `.local/split-limb-feedback-actor-01/tests-01.xml`, SHA `be8f0d5587cea8f0d745182b167a53b7fd854dd120bf2fc36230569445e4b173`. Process receipt: `.local/split-limb-feedback-actor-01/process-01.json`.

Coverage includes explicit profile selection, malformed coefficients, exact named routing, separate arm/leg effects, 19/22-action zero anchors, unchanged v4 numerics, bounds/root invariance, policy tampering, detached prepared state, one feature read per action and no hot-path hashes. A pre-change v4 policy golden (`748bb1a3c3ff831a3f373b66be475de0a9646c7ae086618f31421ce3b7d41d4e`) was derived from the original pure maker with fixed authored metadata and runtime identities.

Native constructors/steps, network and child-process calls were denied. One import-time subprocess attempt was blocked and recorded; its caller was not captured by this small test harness, so no exact caller is inferred. No model/physics/source work or external spend occurred. These tests mock scene admission and runtime providers; root's separately frozen end-to-end acceptance remains required.

The first isolated golden setup omitted the imported `RULE` constant and failed before producing a golden. `.local/split-limb-feedback-actor-01/prechange-golden-failure-01.json` preserves that checker failure. The corrected setup reads the exact unchanged rule literal; no producer/native experiment was repeated. Original actor source, golden and AST receipts remain in that local directory.

Root requested the reusable guard at `.local/split-limb-feedback-actor-01/run_pure.py` (SHA `c8d10113ef3344428cebf51157adf7f6cb3964917a1b7de96569761bb5c7d08d`). It is extracted from the executed audit/test guards, syntax-checked and **not rerun in this lane**. Arguments are XML path, exclusive process-receipt path, then explicit test-file paths. It sets 10-second CPU and 4 MiB per-file limits and installs native denial before test imports. It is not an OS sandbox or a new production runner.

## Next safe action

- [ ] Root completes profile-aware trainer/CLI/workflow integration and the UI lane completes exact policy/lab routing, preserving v4 defaults and old policy artifacts.
- [ ] Independently review the complete changed runtime and freeze inputs/settings before any distinct bounded native comparison. Reuse the existing training/evaluation pipeline; keep every original task/scorer/reset/source gate and retain failed candidates.
- [ ] Report whether the new policy actually passes body and equipment gates and improves matched outcomes. Nonzero weights or a better feasibility rank alone do not complete the learning goal.

No additional actor edits, test reruns or native execution are planned after this release without a concrete finding.
