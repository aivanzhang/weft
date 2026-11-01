# Surface-interval body fitting workflow

## Integrated — October 31, 2026, 5:00 PM PDT

Root owns `source_surface_fit_inputs.py`, `whole_body_reference_pipeline.py`,
`whole_body_reference_worker.py`, `cli.py`, workflow tests and this note.
`/root/exercise_reconstruction` owns the opt-in existing-fitter integration;
`/root/support_ui` owns its saved report; `/root/support_learning` independently
reviews input readiness and custody. All code lanes are frozen; no native job is
running. Base: `f1bfe83`. Publication outcome will be recorded in
`.local/surface-interval-publication-01.json`.
Unrelated avatar changes stay excluded. No API, model inference, source download,
server, browser or deployment is authorized by this milestone.

The existing command gains `--objective surface-intervals --surface-config FILE`.
Other objectives reject the config and retain their original worker closure.
The frozen worker verifies exact config/input bytes, builds the published interval
envelopes/objective and passes them to the existing fitter. Inputs are copied once
into the new proposal; source bodies, floors, original starts and scoring are not
rewritten. A candidate is never installed automatically.

## Exact configuration

`source-surface-fit-config-v1` has these strict fields:

- `schema_version: 1`, `profile`, original `source_sha256` and explicit `weight`
  greater than zero and at most 1000. Weight is an authored objective tradeoff,
  not a calibrated physical quantity.
- `observations`: one to eight `{path, sha256}` entries.
- `camera`, `surface_intervals`, `annotations`, `body_binding`: each an exact
  `{path, sha256}` entry or explicit `null`. Null preserves unavailable results;
  it does not invent a camera, support, correspondence or transform.

Paths are relative to the config directory, without traversal or symlinks.
The original config is `inputs/surface-config.json`; observations become
`inputs/surface-observation-00.json` through `07`. Optional members are
`inputs/surface-camera.json`, `surface-intervals.json`,
`surface-annotations.json` and `surface-body-binding.json`.
`plan.json.surface_constraints` binds the config, source, weight and exact optional
inventory; `input_files` pins every original byte sequence.

Config <=64 KiB; each input <=8 MiB; all additional inputs <=16 MiB;
camera/surface/annotation inputs together <=4 MiB; body binding <=64 KiB.
Existing aggregate workflow reservations remain enforced. Fitting remains at
most 30 seconds plus eight seconds finalization, 120,000 geometry calls, and a
45-second parent worker deadline. New interval limits are 20,000 evaluations and
4 MiB of decision journal. No hard RSS guarantee is claimed. The reviewed child
disallows network, inference, media, rendering, physics steps and nested jobs.

## Verification so far

87 distinct scoped workflow/worker checks pass. Initial receipt
`.local/surface-fit-workflow-tests-01.xml` has 85 passes and two test-only failures:
the new unavailable assertion used a nonexistent frame-level status, and an old
worker guard test inherited a previously imported CLI module instead of the
actual worker's cold start. Both were corrected; only those two cases reran in
`.local/surface-fit-workflow-tests-02.xml`. Neither failure required a product fix.

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q \
  tests/test_surface_fit_workflow.py tests/test_whole_body_reference_pipeline.py \
  tests/test_whole_body_reference_worker.py \
  --junitxml=.local/surface-fit-workflow-tests-01.xml
```

The six changed fitter/workflow/report modules are outside the current camera
lab's 54-file execution closure; no old lab was repinned or rerun.

## Actual native worker and preserved acceptance failure

One **authored-fixture** invocation completed: three original references, ten
source slots, one channel, 97 native forward calls, two models, four data objects,
36 interval evaluations and 30 journal records/26,264 bytes. It made no physics
step, reset, media/inference call, training or paid request. The worker took
0.174 seconds (fit 0.048 seconds); observed peak RSS was 76,234,752 bytes, not a
hard memory cap. Eleven updates reduced the conditional summed loss from
0.0491415 to 0.02921165789092061 m². All twelve construction gates stayed true;
the worst distance remained 0.181 m because the first pose was frozen.
These are authored test coordinates, not measured human or equipment accuracy.

The **outer acceptance remains failed**. Its additional assertion required zero
denied attempts; the guard blocked one nested-process attempt. The caller was
not recorded and is unknown. No guard was relaxed and the fit was not repeated.
The completed worker output and failed acceptance are separate facts.

- Original: `artifacts/surface-interval-fit-native-01`, 40 files/527,940 bytes,
  seal `3d58524e3cf6f87cb93aa51a2536f9af8b84eee0e30e5c69dbb1ebb8bf661233`.
- Frozen 26-file runtime: `.local/surface-interval-native-runtime-01`, committed
  `f1bfe83` plus five reviewed overrides; runtime manifest
  `02bf110572ea70efa86dce5c7521abe1f700d3f154cb8df2d22160aeb52ecc45`.
- Fixed-before-fit inputs: `.local/surface-interval-native-inputs-01/fixture.json`,
  `b4083087e495cea174eb92efde1bef795007b8ce4b3ff9707d96c5e1453f0942`.
- Runner: `.local/run_surface_interval_native_acceptance_01.py`. Its failed
  assertion and original source/study manifests remain unchanged.

A new **saved-only inspection** verifies/copies the completed worker bytes,
retains the original failed manifest, and runs the frozen public reporter on the
actual native payload. No physics or fitting is rerun. The new manifest's
completion scope explicitly covers saved worker outputs, not a passed original
acceptance. Its first page explains the failure before linking the result.

[Inspect the result](../../artifacts/surface-interval-fit-inspection-01/index.html):
59 files/807,471 bytes, 0.014 seconds, seal
`a182b4a1fec74ace8e2b1eb631c9368e8cf299596da6ecca7bf23a5d4e7fc227`.
Command: `.venv-lab/bin/python -I -B .local/inspect_surface_interval_saved_native_01.py`
(already completed; output is exclusive, do not rerun).

## Fitter and report verification

The fitter has 25 new analytical cases and four affected legacy cases passing;
the report has 21 new cases and four affected legacy cases passing. Together
with root's 87 workflow/worker cases, this is **141 distinct passing checks**,
not a full repository or browser suite. The report's two reproduced gate
consistency failures are preserved, followed by targeted passing fixes. A
temporary analytical payload expired through pytest retention; its extra saved
schema check could not start. It was not recreated; the native saved payload
above supplies the actual producer/report integration check.

The report checks exact input pins, complete source/channel joins, clocks,
residual arithmetic, unknowns and the fitter's exact nonregression gate. It shows
pixel/height ranges, saved world bounds, before/after metre distances, separate
m² loss and authored weight, and the full linked rejected/interrupted ledger.
It archives the current input-validator code separately from historical study
inputs. Legacy selected reads and eight-file report output stay unchanged.
Bounds remain 16 MiB selected input/output and 2 MiB HTML, with explicit failure
rather than dropped slots. A small mocked DOM check retained nine nodes over
200 alternating selections; this is not browser performance evidence.

Report receipt: `.local/whole-body-reference-interval-report-checks-01.json`,
SHA `e07c42c2fc1351a792c024d87afc6dd94e71879256679631080654380d2282d8`.
Final fitter SHA `1637d8c0b1c03343a2e5977855038c904481c71133867a990d705d6a112a7619`;
report SHA `ca49cfa48aeb22cdcd073d55563e9f705d760a67d5052ae6f675a50ae4b37ff0`.

## Next safe action and remaining limits

The independent saved audit passed: `.local/surface-interval-native-saved-audit-01.json`,
SHA `0dfdfda45f3504689062da154255557534fa05d2d7f8f8fd6940c6595b73cbdc`.
It checks five seals, 21 consumed runtime files, exact inputs and copied outputs,
fixed fields, three-reference interval arithmetic, all twelve saved gates,
ten slots/seven unavailable joins, 30 journal records and 17 resolving offline
links. The original acceptance remains failed. It used 103 unique cached reads/
1,353,724 bytes (plus two initial metadata reads); no native or source rerun.
An auditor-only syntax failure occurred before execution and is preserved;
the correction reused its cached bytes. No product change was required.

Finish scoped publication. Do not repeat the
fit to erase the wrapper's failure. Future work must make the blocked-attempt
caller observable before another justified native run, and improve real-source
camera/body/support evidence rather than treating this fixture as reconstruction.
Saved jogging is not eligible:
nonzero radial distortion, a supplied-camera body unsupported by this fitter,
and missing authored interval/correspondence/body-world inputs. Do not remove
those guards, change the floor, crop the start or fabricate calibration to run it.
