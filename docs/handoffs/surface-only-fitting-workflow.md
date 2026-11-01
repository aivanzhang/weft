# Missing-image fitting and usable fitted controls

## Current outcome and ownership — November 1, 2026

The previous goal turn was **progress**: camera/floor calibration was published
as `046261b6922f1b42318a6cfe48ff888d041d754b`, with 130 focused checks, four
genuine public numerical cases and independent saved audits. Private remote
verification and scoped clean state are recorded in
`.local/floor-target-calibration-publication-01.json`, SHA
`3a2d32ad275b1869e4f4569dea8fbb01a833c7927b267b4a1aff10e25d6f14e5`.
Two separate avatar edits remain preserved; stash/index were empty at this turn's
start. No current profile completes the full video→world→RL goal.

This turn prioritizes existing reconstruction/control gaps over another optional
calibration worksheet. Parallel ownership:

- Reconstruction: `whole_body_reference_refinement.py`, new
  `test_surface_only_reference_fitting.py`, and the
  [core handoff](surface-only-reference-fitting.md). Extend the existing QP
  scheduling only when a missing-image pose has independently qualified support
  intervals. Root owns any actual native acceptance.
- Root: optional config/worker/report binding in `source_surface_fit_inputs.py`,
  `whole_body_reference_worker.py`, `whole_body_reference_report.py`, new
  `test_surface_only_fit_workflow.py`, this note/common docs, independent UI
  review, frozen acceptance and publication. Existing CLI/pipeline are reused.
- UI: `assets/lab.js`, new `camera_reference_lab_ui_smoke.mjs`, and
  [GUI handoff](fitted-control-lab-ui.md). Expose the already supported fitted
  joint-target controller and its distinct recording proof. Do not change the
  backend, scorer, avatars or physics. Independently review root's report path.
- Learning: prepare one current-runtime full-span feedback-residual study using
  existing learning code and matched baselines. No new policy family yet. Older
  full-span failures use different runtime/scorer pins and cannot establish
  current performance; see [learning handoff](current-coupled-learning.md).

No paid call, ledger/key operation, deployment, browser, server or external
message is part of this slice. Scientific work requires a separately frozen
committed runtime, exact reviewed overrides and explicit bounds before launch.

## Optional reconstruction contract

The existing `source-surface-fit-config-v1` accepts one optional field:

```json
{"missing_image_policy": "qualified_surfaces"}
```

This is an addition to the complete existing config, not a standalone config.
Omitting it, or supplying `preserve`, keeps the original fitter behavior and
serialized output. The exact config bytes and optional field remain bound into
the worker descriptor. Other values are rejected. Existing
`body-reference-propose --objective surface-intervals --surface-config FILE`
and camera surface objectives carry the option; there is no second pipeline or
new CLI flag.

Core keyword: `missing_image_policy='preserve'|'qualified_surfaces'`. The latter
requires a surface objective. A noninitial pose with zero usable image weights
is eligible only when its exact bound current interval term has evaluated
channels and finite positive outside-envelope loss. Being inside the envelope,
missing/excluded support, mismatched source identities and the original first
pose do not authorize movement. Current interval availability/loss is checked
again before each block; no midpoint target is manufactured.

Only the explicit opt-in adds its policy to the candidate/angle-overlay marker
and report objective, plus `surface_only_eligible` and `surface_only_attempted`
booleans on reference rows. Attempted means the QP call was entered, not merely
that a row was visited. Completed journals prove the corresponding calls;
partial-budget results may lack a completed QP record. Changed image-unavailable
rows say `changed_surface_only_proposal`, never unchanged or observed.

Image weights/errors stay zero/null. Original start, root, floor, units, source
slots, all image/depth/floor/rate/channel gates, independent scorer, final-audit
reserves and existing budgets remain unchanged. This extends the earlier
image-only scheduling contract; it does not retroactively label that deliberate
restriction a bug or claim old physical reconstruction failures are resolved.

## Verification and pending work

Root's **20 new +21 affected checks** pass in
`.local/surface-only-fit-workflow-tests-01.xml`. They exercise exact config and
descriptor binding, old defaults, public package/worker routing, null/invalid
policy rejection, saved interval/QP provenance, forged eligibility and partial
call evidence. The numerical worker is mocked; no native work ran.

UI's four new integrated checks and seven affected checks pass, with two
pre-fix failures retained. Fitted-control node/label writes remain zero across
200 repeated maximum-shape mock updates (113.733 ms). This is scripted DOM
evidence, not browser/device QA or new physics. Receipt
`.local/fitted-control-lab-ui-release-01.json`, SHA
`84575b5402ddae955cc021d15085b825e843c2051add8602603607f9f9c705f3`.
Independent source review is closed; actual saved-state composition is pending.

Core has **31 distinct passing cases**, including exact old-output goldens and ordinary/camera/two-view interval descent. The two report review gaps were reproduced in `.local/surface-only-fit-report-red-01.xml` and fixed; all twelve affected cases pass in `surface-only-fit-report-green-01.xml` (SHA `e6af53f248e12524c8f24936d7207fb6459961b01c919e9f2bb00034be904e88`). Preserve-mode missing rows cannot change, and the new unchanged status must match actual eligibility/joints. Independent delta review is closed.

Independently authored native-input preparation uses five ordinary references/ten slots: existing root admission requires at least three observed scales and 80% observed coverage. Four observed rows and one explicitly interpolated root preserve those gates; no new interpolation rule is introduced. No solver/native study is authorized merely by a finished
input fixture. Root must freeze the current committed runtime, unchanged scorer,
input/candidate hypothesis, numerical limits and complete output budget, then
run the distinct public case once. Do not repeat exposed heel/Interval/jogging
failures without changed evidence or a specific new algorithm hypothesis.

## Actual integration results — November 1, 2026, PST

The single public native case passed. Root process exited zero and was reaped in
**0.608321 seconds**; acceptance checks took 0.556819 seconds. Output:
`artifacts/surface-only-reference-native-01`, **67 files / 1,184,687 bytes**,
outer seal `b94613614a407b83452ef2336f9e1065bb73a27b46725309b80ca2b462ed7182`.
[Inspect the authored missing-frame result](../../artifacts/surface-only-reference-native-01/inspection/index.html).
Frame 4 is eligible, attempted and changed; interval loss falls from `0.0163805`
to `0.001038643912756187` supplied-world m². All twelve gates pass. Image weights
remain zero/errors null, the first pose/root/floor are exact, and all ten source
slots remain. This is synthetic fixture evidence, not a new-video accuracy claim.

Actual counts: two models, four data allocations, 204 native forwards, 23 accepted
updates and 69 interval evaluations; zero physics steps, resets, media or
inference calls. Worker 0.241893 seconds /78,774,272-byte observed peak RSS;
parent 94,420,992 bytes. The one predeclared MuJoCo startup subprocess attempt
was **blocked** with its exact pinned stack, not executed or silently forgiven.
No guard exception or old-failure reclassification was used. The overall
interval loss remains nonzero and the peak distance is still 0.181 m: the frozen
first pose cannot be corrected, and passing gates does not prove contact.

Root receipts: `.local/surface-only-reference-native-process-01.json` and
`.local/surface-only-reference-native-acceptance-01.json`; frozen plan
`9850642b94b150a4c09d5b505cd5614dbc6971988a20870580a36be893c86d35` and explicit
root execution receipt `.local/surface-only-reference-native-execution-01.json`.
Frozen runtime is `.local/surface-only-reference-native-runtime-01`, manifest
`196d0414c79ca4fab6b3f95cb099afdb389b5e88cc20786afc0c57caf8db2c72`: committed
046261b plus only the four reviewed core/config/worker/report overrides. The
caller verified all terminal seals/members. Independent saved UI audit passes
all ten frames, five references, nine grid views, twelve missing image endpoints
and fifteen local links (five selected reads/336,460 bytes). Preserve its first
auditor counter-label error and metadata-only amendment; no DOM/native rerun.
Do not rerun the fitter to inspect the report.

The real saved GUI composition also passes: two exact summaries, 22 controls,
correct fitted step request and the recipe-bound 246-action/240-fitted-decision
receipt. A wrong recipe is rejected. Its fifteen reads total 2,293,245 bytes;
check 87.480917 ms /root process 0.207195 seconds, terminal and reaped. Two hundred
unchanged updates took 54.460625 ms with zero fitted-control text/attribute writes.
This is scripted DOM evidence, not browser/device or new physics evidence.
Check `.local/fitted-control-saved-ui-check-01.json`, SHA
`d2f6ea80ed59b672ab66d0743cf3ac9c10c611057e6d1db87d5456332073b417`.

The new `.local/camera-reference-gui-runtime-01` contains 270 files/5,687,400 bytes;
manifest `a50a5ecc5f3179fee080561668eb8cd5c09d083222a1a0e06770d0448804b1a2`.
It is a verified copy of the old compatible runtime with **only `assets/lab.js`
changed**; all Python execution files, source, task and controller recipe remain
unchanged. The [local guide](../local-lab.md#compare-a-fitted-reference-without-changing-the-task)
gives the GUI/console commands. Neither the old runtime nor saved lab was mutated;
no server was started on the user's behalf.

## Fresh-source screen: no acquisition

A bounded new Fit3D metadata screen did not qualify a recording. Official-domain
search text described noncommercial/access restrictions; both direct official
page requests timed out. Live rights, exact floor measurements and a small
complete source unit remain unverified. No registration, agreement, media
download or third-party mirror was used. Source references:
[dataset](https://fit3d.imar.ro/fit3d), [terms](https://fit3d.imar.ro/legal).
Retained receipt `.local/fit3d-source-readiness-01/receipt.json`, SHA
`80ee12efaa02d70db1129a9c8745a64a5fa5f260eb24a3398ad079296bce0c09`;
three-member seal `d03bb498af8883859ab7d3d50d999f1ca958b15ac92defe32f2b53f77f4e7c50`.
One search/two timed-out opens, 36,541 returned bytes, zero paid work. This
particular source route is stopped; useful local implementation continues.

Independent unfamiliar-video ground/equipment/contact accuracy remains an
unresolved workstream, not something synthetic fitting or simulator learning
can certify. Keep the complete goal active.

## Learning job ownership

The first current-commit learning attempt is **terminal, failed before training**.
Handle `73650` is reaped; root duration 1.144349 seconds. The unchanged production
loader compiled two models, then the study's file-write guard mistook descriptor-
relative package paths for current-directory paths. There were zero data
allocations, forwards, resets, controls or physics substeps; no policy or learning
result exists. No source/task/scorer or physical gate failed or was changed.

Preserve `artifacts/fullspan-feedback-current-01`, 14 files/178,008 bytes, seal
`50af1a9f644a67c5b98fe39389be33dc5a84fa5d9be660d2b8d7418ad4bfbbd4`, and all
original preparation under `.local/fullspan-feedback-current-01`. All 192 named
source files match their before/after metadata. Only the diagnosed startup probe
and erroneous package-path attempt were blocked; neither executed outside scope.

The learning lane is preparing a narrow descriptor-aware helper correction and
new attempt02 paths. It must retain the same runtime, source, task, seeds, control
budgets and success criteria; no native execution before review. Do not rerun
unchanged attempt01 or remove its failed evidence. The independent helper review
also fixed seal byte/file reservation and post-write terminal timing before any
learning invocation; unused support-agent `supervise.py` stays unexecuted.

## Publication and next action

The scoped fitting/UI checkpoint includes the user guide and durable handoffs;
actual commit/private-remote result belongs in
`.local/surface-only-fitting-publication-01.json`. Do not claim publication if
that receipt is absent. Preserve both separate avatar edits and the empty stash
list. Complete reviewed learning-helper02 integration next; the entire generalized
video→3D→RL goal remains active, especially unfamiliar-source physical accuracy.
