# Separate arm/leg feedback — integration in progress

## Outcome and ownership

November 1, 2026, PST. The prior matched v4 comparison completed with zero
successful trials; equipment `sam-2` remained unstable. This change tests a
distinct, opt-in controller, not more seeds for that unchanged search. The full
video-to-world goal remains active; physical contact accuracy and unfamiliar
video reconstruction remain unresolved.

Root owns `shared_exercise_feedback_learning.py`, `exercise_learning_pipeline.py`,
`cli.py`, `scripts/build_sandbox.py`, trainer/workflow tests and common handoffs.
The actor lane owns `shared_exercise_feedback_residual.py` and its focused tests;
the UI lane owns `shared_exercise_policies.py`, `shared_exercise_lab.py`, `lab.js`
and its focused tests. The independent reconstruction lane reviews integration
and has completed the [bounded source screen](exercise-source-readiness-02.md).
Do not edit or stage the independently owned avatar files.

## Exact contract

- Public opt-in: `feedback-residual-split-limbs`; policy `shared-exercise-policy-v5`.
- Five position/velocity groups: torso, arm proximal/distal, leg proximal/distal.
  Ten coefficients, each bounded by 0.25; the action correction remains 0.25.
- Existing features, authored bias/root actions, task, full interval, training
  reset distribution, independent scorer and equipment-force prohibition stay
  unchanged. The zero-coefficient anchor is the unchanged authored controller.
- Existing v4 remains the default feedback-residual path with its original
  six coefficients. Saved v4 bytes remain immutable; new runtime hashes are real.
- One trainer/selector/evaluator handles both explicit profiles, including
  regenerated search dimensions and per-action saved diagnostic verification.
  Array dimensions never infer the policy version.
- Existing static-support/camera policies remain separate and unsupported by this
  option; no fallback or shorter task is permitted.

## Verification and next safe action

Implementation and independent review are complete: **108 Python +9 JavaScript
checks pass**, including default-v4 compatibility, explicit v5 search/proof/
evaluation routing, malformed artifacts, unsupported modes and lab controls.
These are authored/mocked checks, not native physics or learning evidence.
Root receipt `.local/split-limb-feedback-integration-checks-01.json`, SHA
`ba972ab304bf84444f8213287dc5b22b04ff62af3847d915ee1ed1850cc98c92`,
binds exact implementations/checks. Preserve the first test failure: blocked
identity-quaternion native utility in an old fixture and an obsolete mock
environment. Fixture-only corrections passed with zero native attempts; the
guard was not weakened. Actor/lab lanes retain their own exact receipts.

No native job is running. Preparation freezes the committed `52a2559` archive plus exact reviewed
overrides, sources, algorithm, seeds, bounds and acceptance criteria before any
native comparison. Reuse the corrected current-02 harness; use new study/output
directories and preserve all old failures. Do not rerun the old study.

The [last coupled comparison](current-coupled-learning.md) is authoritative for
current observed learning performance. No new policy has been promoted. No API
call, ledger/key operation, deployment, browser, server or source acquisition has
run in this continuation. The global $5 Meta budget is unchanged.

Fresh private-remote fetch still matches `52a2559`; the stash/index were empty.
Preserve two avatar edits and all saved artifacts. Publication of this new scope
is unfinished. Before handoff, record actual checks, study identities,
live handles, publication result and the next supported action here.
