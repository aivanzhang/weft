# Camera fitted-reference recording

Updated October 31, 2026, 8:04 PM PDT. Owner: root. Native trial 03 verified
238 fitted decisions and all full states exactly; the matched authored recording
also passed. Interactive attachment is still in progress.
[Exact artifacts, comparison and limits](camera-reference-real-acceptance.md).

## Contract

`camera_reference_recording.py` adds `camera-shared-reference-recording-v1` and
`camera-shared-reference-full-state-replay-v1`, separate from manual camera and
learned-policy recordings. The exact current runtime closure and recipe are
checked before model admission. The original camera/body/scene/task/scorer stay
unchanged. A recipe supplies fitted authored joint controls; it does not install
a body, authenticate archived source fitting or recover physical contact.

Only a `reference_proposal` step carries its exact controller SHA. Preparation
binds the controller once; per-action calls do no file reads or JSON parsing.
Replay recomputes proposed and authored actions, checks each outcome and full
state, preserves reset options and closes the environment on failure. Rendering
prepares once and exposes only a bounded eight-string controller summary.

`recording.py` dispatches this distinct profile and reads both scene (16 MiB)
and metadata (1 MiB) through bounded no-follow file reads before adapter/model
validation. `shared_exercise_lab.py` adds replay dispatch only; interactive mode
selection/attachment remains pending. Existing camera recording contracts are
unchanged.

## Verification

Receipt `.local/camera-reference-recording-checks-01.json`, SHA
`261864335f5f2ae08c551677c46681d1aaeedbee31ce8061369f8ad5efcb77fe`,
pins all five source/test files and four XML receipts. **61 distinct final cases
pass:** 26 reference-recording, six input-bounds and 29 affected camera/policy
cases. The actual recorder, event validation and full-state replay use an
analytical engine, not native physics. No source/video processing or paid calls.

Primary command (individual already-passed cases were excluded when unchanged):

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q \
  tests/test_camera_reference_recording.py \
  tests/test_recording_input_bounds.py \
  tests/test_camera_shared_recording.py \
  tests/test_camera_shared_policy_recording.py
```

Preserve `.local/camera-reference-recording-tests-01.xml` through `04.xml` and
matching logs. Check 01 failed because the analytical fixture omitted required
gravity. Check 03 had one roots-on assertion querying authored action after the
horizon; moving that assertion before each step fixed the test. Neither fixture
correction weakened product checks. Checks 02/04 only reran their corrected case.

The independent replay UI work has six new and three affected scripted groups;
see [its handoff](camera-reference-replay-ui.md). No browser was opened.

## Next safe action

1. Independent source/trial review is complete, including the remaining-deadline
   fix and 14 new pure checks. Three affected existing trial cases pass. Frozen
   runtime is `.local/camera-reference-real-runtime-01`, seal
   `55a87422a77a964cd141532b0a570f7dbb507e179fd234ccd0e2571571fb4010`.
2. Use a genuine matching source/candidate/task for one bounded fit and matched
   authored/proposed trial; preserve all failures and original scoring. The old
   authored fit's incomplete source package is not public-builder admission.
3. Audit saved output and run the released UI fixture against the actual bundle,
   without rerunning simulation merely for inspection.
4. Attach fitted controls to the existing interactive lab and verify mixed
   controls, reset, recording/export and exact replay before claiming usability.

Protect scorer SHA
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
Native work must use a committed archive plus exact reviewed overrides, not the
dirty full tree. Keep reconstruction and unrelated avatar ownership separate.

## Genuine-source attempt 01

The actual saved jogging source has 240 slots/112 references and a matching
238-control task. Metadata preparation and independent saved audit pass, but
the first public fit entry failed before any model/geometry/physics/media work:
`Camera body fit: surface objective needs a qualified explicit body/world binding`.
Image-source readiness is not that admission. Receipt
`.local/camera-reference-real-acceptance-01.json`, SHA
`6d3a4dc3de96d17bc4956ed2b6567d1661935b2af07dca855a848177e964b6cf`.
No trial or lab attachment has been verified. Preserve the failed output at
`artifacts/camera-reference-real-01/fit`; use a new explicit-input plan/output if
the existing camera-world declaration can satisfy the unchanged contract without
inventing surface height, kind or contact. Do not rerun unchanged failed inputs.
