# Camera fitted-reference lab

## State — October 31, 2026, 8:27:31 PM PDT

The console-first attachment passed root's native interactive acceptance in 11.931 seconds. The new lab is ready to resume with its exact frozen runtime. This lane performed only analytical checks and a small process-receipt read; it did not repeat native work, decode media, run a browser/server, retrain or call an API.

The lab preserves one original camera scene/task and offers manual, coast, original authored feedback (`mode reference`), and fitted joint targets (`mode fitted`). Fitted controls are **not learned** and are not promised to improve the simulator score. The existing completed comparison passed both modes, but fitted mean reward was about 0.80217 versus 0.82621 authored; angular error was about 0.15789 versus 0.14574. Contact and physical accuracy remain unverified.

## Interfaces and authority

- `prepare_shared_exercise_lab(..., reference_trial=..., reference_manifest_sha256=...)` requires the exact original task, either sealed in the source or explicitly supplied. Root overrides, policy/evaluation attachments and non-camera scenes are rejected.
- `camera_reference_lab.prepare_reference_attachment` authenticates a completed matched trial, its exact original proposal aliases, source/task bytes, all existing fit gates, original recipe/current core runtime and saved comparison proofs. It reuses the actual camera input loader; it does not invent ancestry or replace the body with an angle overlay.
- `load_reference_attachment` repeats bounded admission from retained selected bytes. Fresh and resumed full attachment checks precede native world/task admission. Source verification receives the existing absolute lab deadline.
- The original recipe runtime map is immutable. Additional lab/CLI/receipt adapter hashes are stored separately in the lab execution closure; no compatibility allowlist or silent repinning was added.
- The selected trial copy is bounded to 64 files/32 MiB, with existing per-document limits and no copied trial journals/videos. The full original trial seal and exact selected-member hashes remain. Saved trial action proofs are custody evidence, not a new replay.
- `SharedExerciseSession` prepares the unchanged controller once. Fitted steps use the cached action closure, preserve original root actions, and record only `controller=reference_proposal` plus the exact recipe hash. Manual/authored/coast events receive no fitted metadata. Reset clears the cached decision; failed preparation closes the environment. Learned mode remains unavailable.
- The recording protocol remains `camera-shared-reference-full-state-replay-v1`. New receipt dispatch includes `reference_controller_sha256` and `reference_actions_verified`, including **zero** for attached manual/authored-only recordings. Saved receipt inspection reads metadata/hashes only; it never claims a fresh action/state replay.

## Owned files

Product: `camera_reference_lab.py`, `shared_exercise_lab.py`, `lab.py`, `cli.py`, `lab_console.py`, `camera_reference_receipt.py`, `recording_receipt.py`, all under `src/weft/`.

New tests: `tests/test_camera_reference_lab.py`, `tests/test_camera_reference_receipt.py`. Exact frozen hashes and check commands are in `.local/camera-reference-lab-checks-01.json`.

The controller, recording adapter, original scene/task/body/camera/root/floor and protected scorer were not edited. No UI/brand/avatar changes were made. The existing offline reference replay already handles the distinct fitted summary.

## Verification and retained failures

- 24 distinct new analytical cases pass: actual saved metadata loader joins with native entrypoints blocked; stale/missing source/recipe/runtime/matched-start/proof rejection; fresh/resume pre-native ordering and deadline forwarding; cached controls/reset/cleanup; explicit CLI routing; receipt custody/replay distinction and zero fitted decisions; legacy integer-to-float identity normalization.
- 21 affected legacy checks pass: camera lab prepare/resume, old console modes/failures, old shared/spatial/policy receipt paths and old CLI argument routing.
- The cached action test makes 1,000 analytical decisions after one preparation, with disk reads/re-preparation forbidden. Its recorded timing is not native or browser performance.
- Preserved failures: lab XML01 had a fixture missing action-contract dimension; receipt XML01 exposed a real nested-member reader mismatch, fixed using a bounded nested root without widening the flat-name guard. Preflight XML04 caught a mis-targeted ordering edit and a fixture-only undefined encoder. Only the affected failed cases were rerun; XML02/03/05 and legacy06 retain passing evidence. Full terminal logs are not separately archived; JUnit results and commands are retained.
- Independent review found metadata after model admission and an omitted caller deadline. Both are corrected. Its final check also caught raw identity checks affecting legacy normalization; that path is now fitted-only, with an isolated regression passing in XML07. The reviewer closed all three findings without rerunning tests or native work.

## Completed interactive acceptance

Root's one bounded run completed with exit code 0 on October 31, 2026, at 8:27:31 PM PDT. It prepared a new lab, exercised two steps each of manual/coast/authored/fitted control, reset once, then completed 238 fitted steps. The recording has **246 actions, 240 fitted actions, one reset and 248 events**. Export, exact action/state replay, saved receipt inspection without native work, and resume all passed. No native job remains active.

These results come from root's preserved acceptance, not a second audit or replay by this lane. The process receipt reports 387 output files/90,131,979 bytes and 530,661,376 bytes observed child peak RSS. This is one bounded acceptance, not a sustained-performance claim.

- Lab: `artifacts/camera-reference-play-01/lab`
- Lab run seal: `b4375d1fdc321a99ed6098546213a9a90dbfff0728463192b0a3152d4aa9f6f5`
- Session: `session-06fb54b1d283`; its `replay.html`, `video.mp4`, journal and `evidence/receipt.json` are retained under `lab/sessions/`.
- Outer artifact seal: `61298ff790531bf99486c6f340214c732b489595594bc9b27938c9451873f8a8`
- Process receipt: `.local/camera-reference-play-process-01.json`, SHA-256 `9edaf787cdd48daca261e71e9e9287c72fa566f39daf69bf018edcb6e387146d` (48,172 bytes; independently read/hash-checked for this handoff).
- Frozen runtime manifest: `.local/camera-reference-play-runtime-01/runtime-manifest.json`, SHA-256 `b65e5f56b20ee169d110853876dda5ec3e3f4d93f0a4a2f1b637031aeb1bc98d` (root-reported pin).

The runtime is committed `70b9793` plus the reviewed attachment overrides, excluding active `observations.py`/precursor edits. The original recipe correctly rejects those changing worktree bytes. Do not modify the old policy lab, repin its policy, or replace this frozen runtime with an arbitrary current checkout.

Original completed trial: `artifacts/camera-reference-real-01/trial-03`.

- Trial seal: `ca80907ca85c741b7c1dcc46d1ad096df64132cb1b7248ff40043737408cbec3`
- Recipe: `run/reference-controller.json`, SHA-256 `aaec5420f3d0e6f0fa2bc98aa9d9ee31419bd80b3fdcf6c8df614adce1eea962`

From the repository root, resume the completed lab with:

```sh
PYTHONPATH="$PWD/.local/camera-reference-play-runtime-01/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/camera-reference-play-01/lab
```

Console: `record start`, `mode manual` / `mode coast` / `mode reference` / `mode fitted`, `step 1` through `step 6`, `reset`, `record stop`, `record evidence`, `quit`. Starting a recording resets the trial; each new recording has its own session directory. Nothing moves until `step`. The frozen-runtime selection is required while current source files differ.

Fitted control still does not mean learned balance, better task performance, recovered anatomy or verified ground/equipment contact. The original source/task/root assistance and their limitations remain unchanged. Saved receipt inspection does not execute a new replay. No browser QA was performed.

No Git operations or cleanup occurred in this lane. Root owns common documentation and publication. The analytical receipt remains an immutable pre-native checkpoint; the separate native process/artifact receipts above record the completed acceptance.
