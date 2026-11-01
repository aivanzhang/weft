# Floor-target calibration inspector

## Status and ownership

Implemented, independently reviewed and frozen, November 1, 2026 (America/Los_Angeles). Nineteen distinct analytical cases pass, plus one actual saved core-produced payload composition. No calibration, source, native, browser or paid work ran in this lane. No live job remains.

- Report SHA256: `0a560793a6b659a99200508f717372d4071e266fce45955623a3e17d819b6670`.
- Test SHA256: `7a1d3c2702466090e4ddcab481b4a788b3dc5663a90dabc10a918249971bd949`.
- Latest scoped release receipt: `.local/floor-target-calibration-report-checks-02.json`; initial receipt remains unchanged.

Owned paths: `src/weft/floor_target_calibration_report.py`, `tests/test_floor_target_calibration_report.py`, and this note. The reconstruction lane owns calibration math; the parent owns packaging and CLI. Unrelated configuration reports and avatars remain frozen.

## Agreed interface

`build_floor_target_calibration_report(files, *, max_seconds=5) -> dict[str, bytes]` consumes `calibration.json`, `inputs/request.json`, `input-files.json`, and qualified-only root `camera.json` / `ground.json`. The caller writes returned members under `inspection/` and authenticates/seals the saved package. The reporter does not load original measurements, observations, images or videos, nor invoke calibration.

The inventory is `{schema_version:1, profile:'floor-target-calibration-input-files-v1', files:[{role,path,sha256,bytes}]}`. Roles are observation, measurement, request, camera and ground. Paths identify exact original copies under `inputs/`; admitted camera and ground are separate root artifacts. Local links use `../` and preserve this distinction.

Bounds: four MiB aggregate input/output, one MiB prepared calibration result, 64 KiB request, 32 KiB input inventory, 64 KiB ground; sixteen returned files, two MiB per HTML, 1,800 complete source slots and at most 200 frames per page. Selected-frame comparisons show at most 64 points from one of at most two alternatives. No all-frame DOM or event-time parsing. The five-second deadline is cooperative. Oversized output fails explicitly; no frames or alternatives are trimmed.

Output members are `index.html`, `brand.css`, `report.json` and `frames-NNN.html`. Raw input/result downloads remain in the parent package, not duplicated in the report. The only external asset read is unchanged `src/weft/assets/brand.css`, at most 32 KiB; archive its exact bytes with the generator. The sole Weft dependency is pure `weft.io.canonical_hash`, not the solver/core. Root's generator closure already includes `io.py`.

## Meaning and limits

Supplied camera/floor support is distinct from a target-derived pose. The supplied route needs no target or target-world relation. Measurement quality is the supplier's claim; declared bounds or standard deviations are retained with their units and are not propagated into certified intervals. Target position uncertainty uses the explicit `target_units` header (`m`, `mm`, or unknown), never inferred metres.

Every alternative, duplicate and rejection remains inspectable. Only `qualified_nominal` can expose admitted root camera/ground downloads. Raw hashes and camera canonical identity are distinct; ground binds the canonical camera. Supplied artifacts retain exact original bytes and quality. Reconstructed artifacts retain the chosen fixed pose and nonzero ground height. The viewer validates saved identities, counts, residual/depth arithmetic and nominal selection relationships; it does not reproject geometry, authenticate measurements or independently re-admit calibration.

Complete source slots and separate nominal/observation clocks survive. A native-timestamp-only source keeps `fps` and nominal times null with `unavailable` status; no nominal clock is manufactured. Only prepared construction frames can carry point comparisons. Evaluation/excluded points remain absent, with original files linked; missing values never become zero errors. All supplied measurement headers remain in complete `calibration.json`; selected global records and selected-frame records are shown in the UI.

An early stop may retain named point headers on an eligible decoded construction frame without admitting its pixel geometry. Those headers remain visible but do not count as prepared frames or used corners. Raw request hash-list order is retained; only the result's canonical inventory comparison uses sorted identities. Pose comparisons follow explicit anchor/check request order, while original frame coverage and navigation stay chronological.

No body, floor or geometry is installed, and contact, grasp and load remain unknown.

## Verification and preserved failure

Initial focused run:

```sh
.venv-lab/bin/python -B -m pytest -q tests/test_floor_target_calibration_report.py \
  --junitxml=.local/floor-target-calibration-report-tests-01.xml
```

Fifteen cases passed in 0.171 seconds. The only warning is pytest's `record_property`/xunit2 compatibility warning; measurement properties are retained. Matching complete `.log` files exist for every run.

The finalized core added native timestamp metadata and nullable nominal clocks. One new case and three affected controls passed in `.local/floor-target-calibration-report-tests-02.xml` (0.149 seconds), bringing coverage to sixteen distinct cases. Unchanged passing cases were not repeated.

The first actual saved core composition found a genuine report defect: compact JSON was incorrectly used for camera canonical hashing. Weft's existing canonical helper uses sorted JSON with default spacing. Failure `.local/floor-target-calibration-report-composition-01/receipt.json` remains unchanged, SHA256 `ea6b72bccc9a90c063bb72dc53d38f9eeaed59a0e8f546336dc3b6deddaa52c8`. Its eight original fixture reads total 34,871 bytes; all exact copied inputs are retained.

The fix uses `weft.io.canonical_hash` directly and keeps raw SHA separately. Three affected controls passed in `.local/floor-target-calibration-report-tests-03.xml` (0.151 seconds). The final pure-helper import was then exercised by the saved composition below; no solver or native dependency was imported. Last code-only polish changes overview wording/pixel-error text, not validation or frame event behavior.

Independent review then found three valid producer outputs the report rejected: unsorted explicit request hash lists, named point headers retained after early abstention, and projections in nonchronological anchor/check request order. Three new regressions reproduced the exact failures in `.local/floor-target-calibration-report-compatibility-red-04.xml` (SHA256 `2bc76415db86245e426c3f17ac3e3f4e4ddfd04096af35107e4b6000eb8ac370`). The same three cases plus the affected qualified-result/evaluation-exclusion controls passed in `.local/floor-target-calibration-report-compatibility-green-04.xml` (five passes, 0.051 seconds; SHA256 `da8cdfbdf4a25b2b04b8620fd180ef8fd32fec8ddade91a973a22fcab47665c8`). Both full logs and the pre-fix module are preserved. Total distinct passing cases are nineteen, not the sum of repeated executions. The unchanged saved composition and prior passing cases were reused; no producer/solver work was repeated.

The independent reviewer verified the final product/test hashes and closed all three findings by reading only the narrow delta. Review closure: `.local/floor-target-calibration-report-review-closed-01.json`. No tests or source work were repeated for review.

### Actual producer-schema composition

Core fixture `.local/floor-target-core-analytic-01` has fixture SHA256 `766ab478fe12205d028c0fff9ca8511f5f297a89f7d48c8fa5fe45ea6672d0db`. It uses an explicitly stubbed IPPE pose result with oblique geometry, skew, radial distortion and fractional clocks. It is analytical schema evidence, not a real numerical solve or physical measurements.

The report-only retry reused the retained exact bytes, without repeating core preparation. Passing receipt `.local/floor-target-calibration-report-composition-02/receipt.json`, SHA256 `8a54cebc026e9720431e492b14acb2f526b81e2b338646ca593a75c6c622b8e6`: nine selected cached-input/receipt reads, 37,058 bytes, 0.01041 seconds; 29,632 output bytes and ten resolving local links. Four original slots, two prepared construction frames, one unused evaluation frame and eight corner observations remain. Both failure and passing packages are retained.

### Bounded UI measurements

The final affected test run prepared an authored 1,800-frame package with 807,958 input bytes and 786,636 output bytes in 0.02777 seconds. Nine frame pages preserve every slot. A 200-frame scripted page visit checked 512 saved point comparisons, at most 128 SVG circles (64 supplied plus 64 projected), in 6.147 ms. One thousand unchanged events produced no writes or new nodes. Clearing the selected frame removes stale clocks, measurements and geometry.

These are Python preparation and Node fake-DOM measurements, not browser rendering, sustained-use or physical-accuracy evidence. They precede the final overview-only wording/error-summary polish; frame code and data are unchanged. No performance rerun was needed for static text.

## Remaining acceptance

- [x] Bind final core fields, saved arithmetic and qualified download identities without running the solver.
- [x] Verify malformed/admitted/unavailable/ambiguous cases, raw input links, safe markup and complete frame pagination.
- [x] Measure selected-frame navigation and unchanged-event no-ops with authored scripted DOM inputs.
- [x] Consume the preserved actual core-authored result without rerunning calibration and preserve the canonical-hash failure/fix.
- [x] Close the three corrected compatibility findings by independent read-only delta review.
- [x] Parent four-case public CLI acceptance, followed by the selected saved-page audit below; no source or solver work repeated in this lane.

The report is released to the parent for publication. No source study or simulator run is authorized by this report lane.

## Four-case saved-page acceptance — November 1, 2026, 1:53 AM PST

The parent completed `artifacts/floor-target-numerical-02` once: outer seal `4af92dec9c72b9b83fa08ea39d382478c3f8c18cc35dad1d7d32679bc06baf56`, 189 files / 2,431,160 bytes, 1.003870 seconds. Its separate checks cover all stored files, inputs, runtime, numerical projections and authored true poses. This report lane did not repeat those checks or run calibration. The earlier numerical `01` plan was never started after a cleanup-review finding; it and the unexecuted report-audit `01` plan/helper remain preserved.

One bounded selected-page audit passed with no checker failure:

```sh
.venv-lab/bin/python -B -P .local/audit_floor_target_numerical_reports_02.py \
  --manifest-sha256 4af92dec9c72b9b83fa08ea39d382478c3f8c18cc35dad1d7d32679bc06baf56 \
  --out .local/floor-target-numerical-saved-report-audit-02.json
```

- Audit plan `.local/floor-target-numerical-report-audit-plan-02.json`: SHA256 `7d40ee66fd75efaf87dfa04f9593ca0044c7813f4d0ec5fbb4084cf966c08eb5`.
- Helper `.local/audit_floor_target_numerical_reports_02.py`: SHA256 `335d4f5c36f8b4583569217c6e67fe9395358f3c8e73dae53e93225d83bb1eea`.
- Passing receipt `.local/floor-target-numerical-saved-report-audit-02.json`: SHA256 `aa81d8ef983166ad7acf1e8841562695f1a7e814c4589b4a4011b5c7ae3b72be`.
- Exact selected reads: 25 files / 334,751 bytes, each read once and cached; 0.156523 seconds. Bounds were 40 files / 4 MiB / 10 seconds, with at most 5 seconds for Node.

Cases 1 and 2 retain one passing and one rejected pose alternative; the rejection reason remains visible. Case 3 retains both passing alternatives, remains ambiguous, and offers no admitted camera/floor downloads. Case 4 retains the supplied result, offers its admitted downloads, and states that no target pose was solved. All four retain twelve original frame slots.

The audit checked all 48 frame option labels and views, six alternatives, 144 displayed saved point comparisons, and 48 local links against the sealed inventory. The three retained evaluation frames expose neither point headers nor projections. Complete result/request downloads and original-input links remain distinct from admitted outputs. No original measurement, camera, ground, runtime or media file was reread.

The existing Node fake-DOM harness exercised the exact saved frame scripts: 4,000 unchanged events caused zero writes or new nodes; clearing selection removed stale data. Maximum visible SVG circles were 24. Node wall time was 0.151458 seconds; summed navigation time was 3.389125 ms. These are saved-script checks, not browser rendering or sustained responsiveness evidence. The numerical fixtures are authored measurements, not physical calibration validation. Product/test files stayed frozen; no live handles remain.
