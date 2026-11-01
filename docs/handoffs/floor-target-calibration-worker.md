# Floor-target calibration worker

Implemented and analytically checked on November 1, 2026 (America/Los_Angeles). No real child process, OpenCV solve, source read, media operation, model, physics or paid call ran in this lane. Root owns the first frozen numerical acceptance and the outer package seal. This lane changed only the new worker, its tests and this handoff; no Git operations or shared-worker edits.

## API and exact files

`run_floor_target_worker(package_dir, *, expected_plan_sha256, max_seconds=10)` reads the existing `worker-plan.json`, starts the exact frozen `runtime/weft/floor_target_calibration_worker.py` using Python `-I -B`, and creates a new exclusive `worker/`. It never resumes, retries, overwrites or seals the outer package.

The exact plan uses schema version 1, profile `floor-target-calibration-worker-plan-v1`, status `frozen`, and these additional keys:

- `source_sha256`.
- `input_files`: relative name → raw SHA256. Required `inputs/request.json`; one to eight `inputs/observations/<sha>.json`; zero to eight `inputs/measurements/<sha>.json`; optional `inputs/camera.json` and `inputs/ground.json`. Digest filenames equal their raw pins. Core document maps use those digests, independent of map order; raw request bytes remain unchanged.
- `runtime_files`: the exact ten-file child closure below, with raw SHA256 values. Extra parent/report modules are not child imports.
- `runtime_versions`: `python`, `numpy`, `opencv`; the last is the installed `opencv-contrib-python` distribution version, not a relabeled `cv2.__version__`.
- `settings`: only `max_seconds`, strictly positive and at most five.

The child closure exported as `RUNTIME_MODULES`/`RUNTIME_FILES` is `__init__`, `floor_target_calibration_worker`, `floor_target_calibration`, `camera_pose`, `camera_ground`, `source_surface_anchors`, `world_evidence`, `observations`, `world_alignment`, `io`, each under `runtime/weft/<name>.py`. It excludes uncalled human/native adaptation and source-equivalence readers. The import finder executes cached, verified code bytes and rejects any other Weft module; it does not execute arbitrary archived source.

Output names are fixed:

- `worker/calibration.json`, always on successful core completion.
- `worker/camera.json` and `worker/ground.json`, together only for `qualified_nominal`.
- `worker/worker.json`, child status/counts/limits/loaded-code receipt.
- `worker/stdout.log`, `worker/stderr.log`, `worker/process.json`, parent capture/supervision evidence.

The return value has `status`, `calibration_status`, `output_files`, `worker_receipt`, `process`. On completed execution, `output_files` pins **all** five or seven worker members, including empty logs. `completed` means the core completed; `unavailable`, `rejected` and `ambiguous` remain valid non-qualified calibration outcomes. Root may copy only the three core artifacts to the package root and must count both copies. Neither receipt is the package terminal manifest.

## Bounds, failure and trust boundary

- Inputs: four MiB total, including request; request 64 KiB; at most 19 members. Runtime: four MiB total/two MiB per source file. Plan: 128 KiB.
- One core call, at most one exact `solvePnPGeneric(..., flags=SOLVEPNP_IPPE)` and two Rodrigues conversions. The supplied route explicitly forbids OpenCV imports without eagerly loading it. Complete input/runtime bytes and plan hash are checked before computation and again afterward.
- Core: five-second cooperative limit. Child CPU: at most eight seconds hard limit. Parent child wall limit: at most ten seconds, followed by up to two seconds for owned-process kill/reap. Bounded saved readback follows supervision; the outer workflow owns its remaining deadline.
- Four MiB combined core output; `calibration.json` at most one MiB; each worker/process receipt 64 KiB; combined logs 256 KiB. Complete output is under 4.375 MiB/seven files. No slots are dropped to satisfy bounds.
- Peak process RSS must be at most 512 MiB for successful acceptance; this is an observed check, **not** an OS memory cap. Hard CPU/file limits and trusted-code guards are not an OS security sandbox.
- Minimal environment has no inherited credentials, Python hooks, proxies or user configuration. Native model/physics, decoder/provider imports, socket operations and nested processes are denied. OpenCV media functions are blocked; only the reviewed numerical path is permitted.
- All paths use no-follow directory/file reads. Logs use exclusive no-follow descriptors. Source/runtime changes, oversized output, forbidden attempts or capture failures preserve partial artifacts and produce failure evidence. A valid outer terminal seal is never written or reversed here.
- Cleanup runs for normal completion, timeouts and capture errors. Group-kill failure falls back to the owned child. Unverified reaping is saved as failed with its PID and raises; it is never reported as a terminal successful child. The caller must inspect that exceptional cleanup receipt rather than assume zero live work.

## Verification and next action

`tests/test_floor_target_calibration_worker.py` uses authored byte fixtures, stubbed core calls, fake OpenCV functions and fake child pipes. Actual process creation, resource-limit mutation and numerical loading are blocked in the tests.

Thirty-one distinct cases pass:

- `.local/floor-target-calibration-worker-tests-01.xml`: 27 passes; command wall 0.1982 seconds.
- `.local/floor-target-calibration-worker-tests-02.xml`: three new cleanup/no-follow cases plus three affected orchestration checks, six passes; command wall 0.1743 seconds. Unchanged passing checks were not rerun.
- `.local/floor-target-calibration-worker-tests-03.xml`: root's supplied-mode import-guard finding is closed by one new fake-module case plus one affected numerical-guard case, two passes; command wall 0.1190 seconds. No real OpenCV import occurred.

The exact source/test/document hashes and commands are in `.local/floor-target-calibration-worker-checks-01.json`. These checks prove routing/custody/bounded failure behavior, not genuine IPPE accuracy or startup behavior. Next: root reviews the frozen worker, combines it with the frozen core/report/pipeline, then owns one bounded authored public acceptance. No live jobs remain in this lane.

## Root-owned numerical acceptance and saved process audit

November 1, 2026, 1:54 AM PST: root completed `artifacts/floor-target-numerical-02` once through four public `floor-calibrate` calls, at most two concurrent. Its outer seal is `4af92dec9c72b9b83fa08ea39d382478c3f8c18cc35dad1d7d32679bc06baf56`; acceptance receipt is `f14c69aacba722316de399580f38bfc63b2a04e8c6b87fe833f1c4e36a3d186b`. The earlier `floor-target-numerical-01` remains unexecuted: review found a cleanup path that could skip the known CLI after descendant lookup failed. The corrected runner independently kills and waits for that CLI while retaining lookup/kill errors.

All four expected outcomes passed: metre and millimetre planar fixtures were `qualified_nominal`; the ambiguous planar fixture stayed `ambiguous` with CLI exit 1 and no camera/ground artifacts; supplied calibration remained `qualified_nominal` with exact input sidecar bytes. The worker counts were one core call each, IPPE calls **1/1/1/0**, and Rodrigues calls **2/2/2/0**. All four CLI processes and four numerical children were reaped without cleanup errors or forbidden attempts. Maximum observed worker RSS was 74,612,736 bytes; total caller time was 1.003870 seconds. Root's inventory recorded 189 files / 2,431,160 bytes. Cost remained $0.

Independent saved-only process audit: `.local/floor-target-numerical-process-saved-audit-01.json`, SHA `51a982a0a898f28df015348abfe38d0587311b6505551287a53467f67ba24e7c`. It read 15 selected metadata files / 62,751 bytes once, joining raw pins, exact commands, versions, child plans, copied outputs, status/count summaries, limits and reaping. A display-only checker error (`TextEncoder` unavailable) is preserved in the receipt; validation continued from cached documents without rereading artifacts. The audit did not repeat numerical math, report checks, tests, process execution or a full output scan; it reuses root's reviewed numerical checker and leaves the separate report audit to UI.

This is authored numerical/integration evidence, not measured-floor accuracy, source authentication, propagated uncertainty, body admission, contact evidence or an installation into an existing world. No live jobs remain. Further real-data admission needs explicit measured declarations and the unchanged downstream gates; no automatic rerun is pending.
