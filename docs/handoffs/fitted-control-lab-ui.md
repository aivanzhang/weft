# Fitted controls in the graphical lab

## Result and ownership

The camera lab now exposes its already-attached fitted controller in the existing sandbox. This is presentation/control routing only: no new simulator, fitting, training, source admission or task change.

Owned files:

- `src/weft/assets/lab.js` — SHA256 `cf61db1d95b8216d7bd7c6df8a0de331b66a23b6deab4c5610f4b1dc6a30907b`.
- `tests/camera_reference_lab_ui_smoke.mjs` — SHA256 `d1b91d20c67bfd436315896e1522d64cef5998a703f029d811f4555e28ffc5d6`.
- This handoff.

No HTML template, avatar, backend, scorer, CLI or common guide was edited. The parent owns independent review, any actual saved-payload acceptance and publication. No native worker, server, browser or paid job was started; no live handle remains. Git was not mutated in this lane.

## User behavior

The existing camera task card contains **Fitted joint targets**. The original programmed control is labeled **Original joint targets**. Manual, coast and learned controls retain their separate meanings. A lab without a valid fitted attachment shows a disabled fitted button and explains why. This does not attach a controller automatically.

Choose fitted targets, then use the existing Start, single step, Record & start, reset and Stop & save controls. Starting a recording still resets the trial. Fitted targets are not learned and are not necessarily better than original targets; body-position assistance stays programmed when enabled. The saved comparison shows original and fitted outcomes separately and is labeled as a previous trial, not the live result.

The recording-evidence panel recognizes the fitted replay protocol and shows its verified decision count, including **zero** for an attached lab recorded entirely with manual/original controls. A missing replay check remains unavailable, not zero. The receipt view does not itself run replay or prove physical accuracy.

## Exact existing contracts

- Actual scene/task remain `shared-world-v6` / `camera-shared-task-v1`; all existing camera task, source-clock, state and metric joins still apply.
- Capabilities must declare `controllers.fitted_reference_controller: true`. Their `fitted_reference` summary must match top-level `exercise_reference` field by field, independent of JSON key order.
- Summary profile is `camera-reference-lab-attachment-v1`: exact controller/trial/proposal SHA256 values; `kind: source_fit_authored_controller`; `learned: false`; original root-assistance mode; bounded original/fitted saved trial summaries; unverified physical scope.
- Missing/mismatched attachment summaries disable fitted control without inventing another controller. Contradictory learned-policy and fitted capability authority invalidates the camera control contract. Learned control is never inferred from a fitted proposal.
- A fitted step sends only `{steps, controller: 'reference_proposal'}`. The server resolves the cached controller and writes the exact recipe identity into its journal. The UI neither sends action guesses nor rereads/parses the recipe on each step.
- Receipt protocol is `camera-shared-reference-full-state-replay-v1`. Existing full camera identities must match; `reference_controller_sha256` must match the attachment and the exact `run/reference-controller.json` metadata. The selected trial manifest and attachment metadata must be present. `reference_actions_verified` is an integer from zero through the total actions; learned count remains null. Unperformed replay requires null proof counts.
- A plain camera or learned-policy receipt cannot masquerade as fitted evidence, and a fitted receipt is rejected when no matching fitted attachment is active.

Existing controls serialize pending steps, recording starts/stops and resets. The new mode uses that same queue; it adds no timer, source fetch, target force or actor-selection route. Its small attachment summary is checked once at ready-state setup. The complete recipe stays outside the UI hot path.

## Evidence and failures

Four new integrated scripted checks cover:

1. Both 19-joint and 22-control layouts; original/fitted/manual/coast switching; pending-step → record/reset serialization; no client-generated recipe provenance.
2. Absent or malformed attachment fields and contradictory learned/fitted authority; valid ordinary controls remain available where appropriate.
3. Exact fitted receipts, changed/missing recipe/trial pointers, wrong protocol, forged counts, zero fitted actions and unperformed replay.
4. Repeated updates with 32 body segments, 38 illustrative muscles and 256 contact records; unchanged fitted controls and receipt labels are reused.

The two primary regressions first failed on the old product: no fitted button and rejection of the exact fitted receipt. Their full output is preserved in `.local/fitted-control-lab-ui-red-01.log` and its JSON receipt. No failed evidence was replaced.

```sh
# Preserved red run, before the implementation:
node tests/camera_reference_lab_ui_smoke.mjs \
  fittedControlsRecordDistinctModesAndSerializeReset \
  fittedReceiptsRequireRecipeAndDistinctCounts

# Four new integrated checks:
node tests/camera_reference_lab_ui_smoke.mjs

# Only the affected repeated-update case after adding unchanged-text reuse:
node tests/camera_reference_lab_ui_smoke.mjs \
  repeatedFittedUpdatesDoNotRewriteStaticControlsOrEvidence

# Seven affected existing camera/learned controls and receipt checks:
node tests/camera_shared_ui_smoke.mjs \
  manualCoastAuthoredUseOnlyExplicitHumanChannels \
  malformedIdentitiesAndMetricsPauseWithoutRelabeling \
  recordingWaitsForActionsAndResets receiptRequiresCameraPinsAndNoLearnedProof
node tests/camera_shared_policy_ui_smoke.mjs \
  admittedControlsAndRecordingStaySerialized \
  mismatchedPolicyDisablesLearnedAndMalformedAdmissionPauses \
  policyReceiptRequiresDistinctProtocolAndBothIdentities
```

Passing evidence: `.local/fitted-control-lab-ui-green-01.{json,log}` and `.local/fitted-control-lab-ui-affected-02.{json,log}`. Coverage is **four new plus seven existing checks**, not the sum of repeated executions. The final repeated-update case measured 200 updates in **113.733 ms** with zero fitted-label/attribute writes, zero added control nodes and no control requests. This is Node fake-DOM behavior, not browser responsiveness or a native control trial.

The initial read-only audit also inspected the actual preserved receipt at `artifacts/camera-reference-play-01/lab/sessions/session-06fb54b1d283/evidence/receipt.json` (SHA256 `3e20089f15ba1cfd8a13837ec894eac27a2bfa231f81fca23e448b930e83796a`). It reports 246 exact actions, one reset and 240 fitted decisions. That existing proof motivated the missing-protocol regression; this lane did not replay it or overwrite its old exported UI. Parent acceptance must select a compatible frozen runtime; no old lab/policy runtime was repinned here.

## Next safe action

Review the frozen JS delta and use its existing exported analytical fixture/harness for one exact saved-payload presentation check if needed. Do not refit, retrain or rerun physics merely to check this UI change. The separate ordinary-video entry-point ambiguity remains parent-owned: the older README `weft lab --video` example still selects the legacy single-joint workflow, while `build_sandbox.py --person-only` selects the current full-reference exercise workflow. No automatic no-equipment declaration or legacy behavior change was introduced here.

## Saved-payload acceptance

Parent reviewed the GUI change and prepared `.local/camera-reference-gui-runtime-01` from the original fitted-lab runtime, changing only `assets/lab.js`. Its manifest is `a50a5ecc5f3179fee080561668eb8cd5c09d083222a1a0e06770d0448804b1a2`; all Python backend files and the original lab/recipe remain unchanged. Parent then executed the frozen saved-data check once successfully.

- Plan: `.local/fitted-control-saved-ui-plan-01.json`, SHA256 `9d53fed8619a7c4f121a5208a23bbf1ddc7dac57d7e8b72b1e0be701bfd18431`.
- Harness: `.local/check_fitted_control_saved_ui_01.mjs`, SHA256 `7318511a5fe55264b336a4317ec8cccd865188488c1763101d2ca22a50361d81`.
- Selected inputs: the exact initial/final summaries and saved receipt view from `artifacts/camera-reference-play-01`, outer/run manifests, lab contract/attachment and original session receipt; frozen JS/HTML and two pinned existing DOM helper sources. The plan lists every raw hash/length (2,277,145 input bytes).
- Ceiling: 16 reads / 8 MiB / 10 seconds, Node heap 128 MiB. No producer, browser, server, source/media read, model, physics action or replay. Off-scope artifact routes return unavailable, not fabricated source geometry.
- Expected presentation: 22 controls; fitted request exactly `{steps:1,controller:'reference_proposal'}` with no client-supplied recipe; actual saved receipt accepts 246 actions, one reset and 240 fitted decisions; terminal state disables stepping; 200 unchanged updates do not rewrite fitted labels or add controls. No request is answered with a new simulated state.

Completed check command:

```sh
node --max-old-space-size=128 .local/check_fitted_control_saved_ui_01.mjs \
  .local/fitted-control-saved-ui-plan-01.json
```

The original prepared plan remains unchanged. Passing receipt `.local/fitted-control-saved-ui-check-01.json` is SHA256 `d2f6ea80ed59b672ab66d0743cf3ac9c10c611057e6d1db87d5456332073b417`; process receipt `.local/fitted-control-saved-ui-process-01.json` is `474f1e13a5a5ad5a6e45d259d1511356ea27b58b5537d3fbecbd719d01e097f7`. The child exited zero and was reaped in **0.207195 s**. Check work was **87.480917 ms**, **15 reads / 2,293,245 bytes**. All expected controls, exact fitted request, actual 246-action/240-fitted proof, terminal snapshot and wrong-recipe rejection passed. The **200 unchanged updates took 54.460625 ms**, with zero fitted text/attribute writes or added controls. End-of-check RSS was 87,425,024 bytes, not a peak measurement.

Six unselected artifact requests were explicitly unavailable. The saved final snapshot predates recording stop; the separately saved receipt supplies completed recording evidence. This verifies actual saved-payload presentation in the existing scripted DOM, not browser layout/CSP, source imagery, new physics or a new replay. No rerun was performed for this handoff update.

The compatible graphical runtime is available from the repository root:

```sh
PYTHONPATH="$PWD/.local/camera-reference-gui-runtime-01/src" \
  .venv-lab/bin/python -B -P -m weft lab \
  --resume artifacts/camera-reference-play-01/lab
```

The user starts this command and opens the printed local address. No server or browser was launched by this check. The earlier frozen runtime, lab and recordings remain untouched.

## Independent missing-image report review and saved audit

The separate `missing_image_policy` workflow review closed at report SHA256 `da118a9a1672c2eaec878f9235cb34d466c96d4d7a5989ff05af7c801aa20780`. The two found gaps were corrected: preserve mode cannot accept relabeled missing-image joint corrections, and the new unchanged-surface status must match eligibility plus unchanged joints. Root retained three red regressions and twelve affected green checks; this lane did not rerun them or the fitter.

Parent subsequently completed one native authored fit at `artifacts/surface-only-reference-native-01` (outer seal `b94613614a407b83452ef2336f9e1065bb73a27b46725309b80ca2b462ed7182`). Our independent saved presentation audit read **five selected artifact files / 336,460 bytes**, not the full 67-member tree: outer and inspection manifests, inspection JSON/HTML, and the original saved proposal report. Exact bytes were cached once; the existing report DOM helper was reused. Parent's full seals/runtime/native checks were not repeated.

The saved script exposes all **10 original frames, five body references and nine control-grid views**. Frame 4 alone is eligible/attempted/changed under the explicit surface-only option; all twelve source image endpoints remain unavailable, with null confidence/error and no target coordinates. Saved conditional interval loss is `0.0163805 → 0.001038643912756187 m²`. All twelve saved gates remain true, the first reference remains labeled frozen, and the report remains an unadmitted, physically unverified proposal with unknown contact. All fifteen local links resolve in the exact parent inventory, including the unchanged raw proposal-report download.

- Auditor: `.local/audit_surface_only_reference_report_01.py`, SHA256 `b2effdadfc338b5cd86173c5087f69d40c259deb31d11296a1d1833169643be0`.
- Original passing receipt: `.local/surface-only-reference-saved-report-audit-01.json`, SHA256 `59a1227c60465ccd4be7c84a6f16779a1884bdd3df8ef88222f7f151dfb1e8bb`.
- Metric-label amendment: `.local/surface-only-reference-saved-report-audit-02.json`, SHA256 `34053d3f0bf35ea4f7fdf7ae73c6fcf787a550ce3049b637246ac4cf2c090340`.

Audit checks took **0.163173 s** after preparation reads; the actual saved script ran for **4.720917 ms** in Node. Its 200 unchanged interval selections asserted zero added nodes (0.068166 ms). The original final counter was mislabeled because it included 1,596 nodes from subsequent grid navigation; the amendment records that checker-only error and the already-passed assertion. No artifact or DOM rerun was needed. These are scripted-DOM observations, not browser responsiveness or physical accuracy. No product code, source data, saved artifact, model or simulation was changed by this audit.
