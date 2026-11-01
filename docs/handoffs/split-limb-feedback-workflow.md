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

The reviewed native comparison is **terminal, reaped and complete** in
61.261301 seconds; former root session `41540` is closed. Output is
`artifacts/split-limb-feedback-native-01`, seal
`cb6249e97ec4208507ab5d11685113cb925a41a7b0b8e822c44e1bbcc0cb4e1e`,
1,046 files/120,782,513 bytes. Do not duplicate it. Runtime is the
committed `52a2559` archive plus seven exact reviewed/published overrides; manifest
`e1dcc006c9c17b4926d0a01e3c3de1e93e8dd0abade5ab32e4f7716a91daf79e`.
Input plan `e1734213f1ba222c968459d9ffd90505f942adcc25fa0041bf9b9f011a871e20`;
execution plan `478fa5aa7252226023df82dd3928991484bc9a318af804fe69a09c30027a1ce5`;
root review `711a52b31611fdbcdd02d6c9f778caf10ced43ea2e6d5be3c6ded9c573f7fb33`.
It reuses the corrected current-02 guard/supervisor, the exact source/task and
exposed development seeds, two generations/eight candidates/two training starts,
then four controllers/three evaluation starts. Limits: 11,176 controls, 178,816
physics substeps, 180 seconds, 400 MiB/1,900 files; no paid calls. No shorter task,
extra baseline run, media decode or policy promotion. Preserve all old failures.

Actual work: 7,112 training +2,413 evaluation controls, 152,394 physics substeps,
40 resets, 34 model compilations and two simulation states. Peak measured RSS
was 245,235,712 bytes. All 762 learned evaluation actions changed joint commands;
no learned action changed the authored root command. Saved selection used zero
new control steps. All source/task/policy bytes and 192 source stat identities
were retained. Root custody audit read only 12 metadata files/294,999 bytes and
joined every stage entry against the producer's full-read outer seal; receipt
`.local/split-limb-feedback-native-01/terminal-custody-audit.json`, SHA
`cd553ecb66364af2355172e0eed9a3c37cdf45576d8ef7e92656df1b4dbef53e`.

Training selected generation 0/candidate 6; none of 14 unique candidates was
feasible. Worst normalized training violation fell from 16.187768 to 6.499293,
but eight limits still failed. **Every controller scored 0/3 evaluation
successes**. Learned and authored both completed all three full intervals;
learned reward 0.745110 remains below authored 0.747771. Learned still fails
only the `sam-2` equipment-stability gate; all body/floor and `sam-3` gates pass.
The new controller is available for inspection, not promoted as a successful
policy. [Saved detailed interpretation](split-limb-feedback-results.md) is complete.
The exact lab opened and completed 266 live actions plus exporter-owned exact
replay and 268 video frames in attempt05; its second same-process open then
exceeded the unchanged 512 MiB limit. The failed overall attempt is preserved.
Separate fresh-process resume passes in 2.885010 seconds at 399,474,688 B peak
RSS; zero steps/replay/decode were repeated. The complete run and all six saved
recording members validate. [Actual saved-UI verification](split-limb-feedback-ui.md)
passes; the broad zero-write checker failure and two legacy status-label writes
remain explicit. No native job is live. [Exact preparation, attempts and scoped evidence](split-limb-feedback-play.md).

The [prior v4 comparison](current-coupled-learning.md) remains preserved and
unchanged. No new policy has been promoted. No API
call, ledger/key operation, deployment, browser, server or source acquisition has
run in this continuation.
The global $5 Meta budget is unchanged.

The 22-path implementation is published and independently remote-verified at
`dfa060f9566f8b4fab5bab96c60412c5c578d987`; receipt
`.local/split-limb-feedback-publication-01.json`, SHA
`e00fc6a24a52cb1f38b58ee9acb8d15004aa47374d99f79e826cf201ac128f75`.
Index/stash were empty; only two independent avatar edits remained. Preserve
them and all saved artifacts. Study-result documentation will require a later
checkpoint. Next: publish the reviewed capacity fix/results and launch guides. Reuse the successful fresh-resume proof; do not repeat completed
controls, export or replay. The [pinned launch command](../local-lab.md#try-separate-arm-and-leg-feedback)
keeps this example usable while development continues.
