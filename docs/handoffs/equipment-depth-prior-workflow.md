# Guided equipment depth priors

## Status and ownership

**Implemented and analytically verified — November 1, 2026, 11:50 AM PST.**
Owner: `/root/equipment_control_review`. Root owns integration and Git work;
`/root/reconstruction_next_step` owns core geometry and separate native-acceptance
preparation. No job is active in this lane. Native admission is not established
by these tests.

Changed product file: `src/weft/shared_exercise_config_report.py`. New tests:
`tests/test_shared_exercise_depth_prior_report.py` and
`tests/test_shared_exercise_depth_prior_workflow.py`. Exact product, test and
receipt pins are in `.local/equipment-depth-prior-workflow-01/release-01.json`.
Prior action-prediction/controller code, evidence and handoffs remain untouched.

`shared_exercise_config.py` needs no change: the existing compiler forwards the
complete equipment rows into `validate_proposal_config`, exposes the emitted
`config.json` in `build_inputs`, and archives loaded Weft dependencies. Its
unchanged SHA256 is
`856e8532fa9be0164ebab82aa0fea0c08afe43e5e49683ee21d522b7ac95dc91`.
The core release is `.local/equipment-depth-prior-core-01/release-01.json`;
acceptance runtimes must include its updated files and this updated report.

## User-visible behavior

The ordinary guided equipment editor offers an explicit optional mode,
`floor_box_explicit_depth_ratio`. Every selected object then needs its own
`depth_prior.depth_to_width_ratio` and `depth_prior.provenance`. Inputs start
blank; no ratio is inferred or filled in. Ratio means **model Y depth divided
by model X width**, not the image-box aspect ratio. The UI describes it as a
supplied, unverified assumption, not recovered physical geometry or contact.

The report uses the core validator and unchanged bounds: finite numeric ratio
`[1/120, 120]`, excluding booleans; nonblank provenance up to 2,000 Python
characters. Text is preserved exactly, including whitespace and non-BMP
characters. The editor counts codepoints, allows the corresponding UTF-16 input
length, and conservatively bounds ASCII-escaped request/config bytes without
dropping objects or truncating text. Switching back to the existing equal-depth
mode omits prior fields; switching back again retains unsaved local drafts.

Opted-in `report.json` adds the exact `equipment_depth_priors` and
`depth_prior_scope: supplied_unverified_not_recovered`. Default/discovery
requests remain unselected; legacy output lacks these new metadata fields.
Contact, grasp and load stay unknown; world admission remains not evaluated.
No body, floor, source-clock, task/scorer, controller, avatar or default-policy
change is introduced.

## Verification

**25 distinct passing cases: 22 new and 3 affected legacy cases.** Passing
receipts and earlier failed fixture checks are retained under
`.local/equipment-depth-prior-workflow-01/`; no unchanged passing suite was
rerun. The core lane's 101 checks were reused, not repeated or counted here.

- `report-tests-01.xml` / `report-process-01.json`: 16 new and 3 affected legacy
  cases; 0.786247 s, four bounded Node fake-DOM children.
- `report-tests-02.xml` / `report-process-02.json`: one overlapping source-copy
  check after the textarea style and ratio explanation changed; not an extra
  distinct case.
- `report-tests-03.xml` / `report-process-03.json`: the new maximum-size editor
  interaction; 0.228406 s including one Node child.
- `workflow-tests-03.xml` / `workflow-process-03.json`: the corrected real
  producer composition case; 0.247123 s.
- `workflow-tests-04.xml` / `workflow-process-04.json`: four invalid-choice
  cases; 0.302785 s.
- `workflow-tests-05.xml` / `workflow-process-05.json`: that same integration
  case with a direct generated Y-depth/X-width assertion; 0.364673 s.

The composition uses discovery → edited request → resume → exact emitted
config → real `build_equipment_proposal`, with two analytical objects at frames
2 and 4 and respective ratios `.5` and `2`. It checks generated dimensions,
exact provenance, new proposer method, unchanged body/source bindings,
snapshot and parent bytes, and unknown contact/grasp/load. The analytical
fixture supplies complete source metadata; it is not recorded physical evidence.
Model/native schema and geometry-admission boundaries alone are mocked. The
workflow fixture also stubs report rendering; the real report is independently
covered by the report tests.

Invalid choices retain the original request and explicit saved issues with
`needs_input`, no `config.json` and no build inputs. The report separately
rejects invalid ready-state priors, config/request disagreement, missing or
out-of-range values, blank/overlong explanations and oversized downloads.

The focused responsiveness check uses all 12 selected objects with 2,000-character
provenance each. **240 changed ratio/text events took 20.474 ms** in the local
fake DOM, created **zero nodes**, kept selected-row identities, and retained
exactly **60 row handlers**. Existing unchanged-event coverage also records
2,000 no-op events without writes or nodes. This is bounded editor evidence,
not browser rendering, sustained memory or whole-application throughput proof.
JUnit retains the measured properties despite a benign xunit2
`record_property` warning.

The report guard is `run_report_tests.py` in this lane's scratch directory;
it permits only bounded Node fake-DOM children and blocks native/model/network
work. Workflow checks use the existing
`.local/split-limb-feedback-actor-01/run_pure.py` guard. Each receipt records zero
native calls. Its known MuJoCo import-time subprocess probe was denied, not
executed. CPU and per-file output limits stayed 10 s and 4 MiB. No browser,
dependency installation, source acquisition, Git operation or paid call occurred.

### Retained fixture failures

`workflow-tests-01` failed because the analytical fixture renamed `person` to
`person-0` without updating canonical observation IDs. The targeted
`workflow-tests-02` then exposed an unrelated equipment document in the base
human-only binding. The fixture now constructs consistent exact identities
before sealing and keeps only referenced documents in each binding. Both
failure receipts remain intact; source or native validation was not weakened.

## Next safe action and completion gate

- [ ] Root reviews the frozen release and integrates it with the core changes.
- [ ] The native-preparation owner completes the separately bounded real-source
  acceptance plan using existing copied inputs, unchanged XY-translation
  checks, and explicit **ratio 1** for each originally equal-depth object.
  Root alone runs the reviewed plan. Include exact report/config dependencies
  in its runtime archive and preserve any admission failure.

This workflow is complete when its opt-in choices reach the ordinary producer
and reports preserve supplied provenance and unknowns without changing legacy
behavior. Native source acceptance remains separately pending; neither it nor
these analytical checks demonstrates unfamiliar-video reconstruction, measured
depth, equipment contact, physical accuracy or whole-task RL success. Do not
choose a non-unit real-object ratio to tune against recorded task failures.
