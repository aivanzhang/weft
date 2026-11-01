# Camera reference trial integration review

## Scope and result

Reviewed the released camera proposal loader and existing matched-trial path against camera_reference_recording.py. The 48 passing analytical cases in .local/camera-reference-trial-checks-01.json were read first and reused, not rerun.

One concrete defect was found: the trial checked its deadline before source admission, but the nested source verifier always started a fresh 30-second budget. A short or nearly exhausted trial could exceed its declared cooperative time budget during pure source processing.

The parent assigned a minimal fix in two source helpers. The optional deadline API is now frozen. Root added deadline=work_deadline at the trial call; that exact call was reread after the helper release. No other concrete admission, identity, runtime-closure, or matched-loop blocker was found in this review. This does not establish native acceptance.

## Reviewed boundaries

- Camera overlays remain separate from the original human specification. The loader checks the actual saved report model, all fourteen unchanged gates, full saved geometry schedule, exact raw original human/evidence/camera/ground bytes, full source slots, and task/coupling/source-clock joins.
- Selected metadata is not called independent source admission. The real trial separately admits source evidence before native scene/task validation and checks that validation did not rewrite their identities.
- The camera runtime includes the recorder's required source closure and controller files. Exact current source bytes are rechecked before model validation and after the trial. Historical legacy feedback incompatibility remains rejected, not silently repinned.
- The real loop uses the existing camera task environment, two identical seeded resets, original authored actions versus the same prepared controller's public action method, unchanged scored targets, retained early failures, and exact recipe action/full-state replay. Root actions remain those of the original authored feedback; object-force channels remain absent.
- Completed recordings are not relabeled cancelled during cleanup. Reports distinguish delivery completion from task success and physical accuracy.

## Optional deadline contract

    collect_shared_source_files(scene, source_dir, *, expected_manifest_sha256=None, deadline=None)
    verify_shared_world_sources(scene, directory, *, include_files=False, deadline=None)

An explicit deadline is a finite absolute monotonic timestamp. Invalid or expired values fail before manifest/source work. Packaging checks it again after manifest parsing, during source hash joins, and before returning. Source verification takes the earlier of the caller deadline and its existing 30-second budget, passing that same bound into the admission loop and nested mapping/construction checks.

Omitting the deadline preserves the old source budget and packaging call shape. A distant caller deadline cannot extend the old budget. These are cooperative checks, not preemption of an individual filesystem, hash, or native operation. No source validation or byte limit was removed.

Frozen files:

- src/weft/shared_world_packaging.py — 615b7da52226c6dfce86c2042f79c582c1a861cc6d61f3263ccbdc091d4c2a62
- src/weft/shared_world_evidence.py — 2ac770a8105ff0f4cb3b968382f864d11c6ea8523b4774adfe942fbc9cbe9341
- tests/test_shared_source_deadline.py — a9311eb9c95ab2ec0aa6701c9e38fb75b19a6c8e78f60cfd5a12c8a5a5320e22

## Verification and limits

    .venv-lab/bin/python -m pytest -q tests/test_shared_source_deadline.py --junitxml=.local/shared-source-deadline-tests-01.xml > .local/shared-source-deadline-tests-01.log 2>&1

Fourteen focused cases passed in 0.034 seconds; no failed check. They cover expired entry, expiry after manifest read and before directory/payload work, default/tighter/looser source budgets, unchanged legacy forwarding, invalid deadline types, and expiry before publishing a source result. Source loaders and filesystem/native paths are blocked or explicitly stubbed; no actual source processing, model, media, network, browser, API, or job ran. No unchanged broad suite was repeated.

JUnit SHA256: 2504d4c5eac9523df0088b3676c70591f616d713b19dc8d7686dfd3ccf57e2ca.

## Ownership and next action

Root owns the completed single trial call change, its integration checks, common documentation, native acceptance, and Git. This lane changed only the two helper modules, one new test file, this handoff, and its local receipt. The controller and replay asset remain frozen. No live handles or jobs remain.

New helper hashes change the saved runtime closure. Freeze the new trial runtime with the caller update; do not silently repin a historical controller or policy recording.

## Camera metadata-budget follow-up

Read-only review of the three released input/trial files against `.local/camera-reference-real-runtime-01` found no new blocker. The original frozen runtime manifest remains `55a87422a77a964cd141532b0a570f7dbb507e179fd234ccd0e2571571fb4010`.

The generic loader retains its 32 MiB default and cap. `allow_camera=True` alone does not increase it. A separately supplied camera budget, at most 64 MiB, becomes active only after the completed proposal seal selects the distinct angle-overlay route. Mixed replacement-human packages reject before report reads; the exact camera method and all fourteen gates are required before the remaining camera inputs are read. The dedicated camera wrapper explicitly requests that larger budget.

The 8 MiB member limit, 128 KiB manifest limit, 32 MiB preserved-copy limit, exact raw alias joins, two loader reads and terminal hash/path recheck remain. The trial reserves `2 * floor(family_budget / 3)` for loading plus its first recheck, then checks the remaining family allowance **before** each terminal read. Legacy and camera budgets therefore bound all three reads at 32 and 64 MiB respectively. The measured selected volume of 18,825,425 bytes needs 56,476,275 bytes across three reads, below 67,108,864. These are recorded size calculations, not a fresh source admission.

Only repeated JSON parsing was removed from the camera loader. Original scene/task/body/camera/root/floor/source identities, all gates, source admission, current runtime validation, native entry ordering and matched-loop behavior remain unchanged in the reviewed diff. Receipt `.local/camera-reference-input-budget-review-01.json` records exact released/frozen pins and reuses the owner's 23 distinct passing checks; no tests, source loading or native work were repeated.

## Offline report recovery entrypoint

Use the completed proposal, not its failed parent inspection, with the released reporter:

```python
prepare_whole_body_reference_report(
    "artifacts/camera-reference-real-01/fit-02/proposal",
    NEW_REPORT_DIRECTORY,
    expected_manifest_sha256="12ed22060e2dc699d26139121b73605d95c31a76625c54817395014eae3df8b0",
    max_seconds=10,
    max_input_bytes=16 << 20,
    max_output_bytes=16 << 20,
)
```

Import from `weft.whole_body_reference_report` in the reviewed runtime containing reporter SHA `5688e2cbee1029bbcabe4b35d604f9b207671f63af68853434582caca605b64a`. The output directory must be new and outside the proposal. Equivalent existing CLI: `body-reference-report --run artifacts/camera-reference-real-01/fit-02/proposal --out NEW_REPORT_DIRECTORY --manifest-sha256 12ed22060e2dc699d26139121b73605d95c31a76625c54817395014eae3df8b0 --max-seconds 10`. Root owns that saved-only recovery and the separate matched trial. No refit is needed or authorized by this review.
