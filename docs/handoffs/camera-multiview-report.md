# Two-camera angle-proposal inspection

## Released scope

Owner `/root/support_ui`; product and tests frozen. Public saved-report recovery and the actual saved-page scripted audit passed. This adds one explicit mode to the existing offline body-reference report, not a new simulator or controller admission route. No live jobs. Parent owns CLI/pipeline/worker, source preparation, source/native acceptance and publication. Independent reviews found no remaining blocker.

Owned files and final pins:

| File | SHA256 |
|---|---|
| `src/weft/whole_body_reference_report.py` | `7e60de58a186b59e04ae8ba1703beeba250d1e454b231bf9af1ea43480aef1e1` |
| `src/weft/camera_multiview_report.py` | `cc72b973d95a694849eb557207444c043a786bb9992195e2c47b650742432729` |
| `tests/test_camera_multiview_report.py` | `f37d8d50d8e844f9b6d67d9681a4910f09f98b57c82b73cf8b2c0420ce2f0422` |

Checks receipt: `.local/camera-multiview-report-checks-01.json`, SHA `f035f9a5d4f88f5cb8afe65a513fe506484016bd327a9d918f307671fe386440`. Independent review: `.local/camera-multiview-report-review-01.json`, SHA `c376168b1935cc6e58b6c3489ee9593d8d23f9b68f772a0ab5c706050a5c4a64`.

## Corrective report-only recovery after completed source fit

The actual two-camera source fit completed once with all fourteen gates true and status **unadmitted**, but the outer workflow failed only while generating the offline page: `Offline report HTML exceeds its 2 MiB bound`. Keep `artifacts/camera-multiview-real-01/fit-01` untouched. Completed proposal seal: `27c1922017155f2160b805af3fdccf4d3620933944dfab56911d921e9b797adc`; failed outer seal: `7454d904787761c163a52328cb7df17f62d951efced7c4bf4572f8ed43f7806a`. No refit or native rerun is needed or authorized in this lane.

One bounded saved inspection read 22 selected files / 19,515,767 B once, under the 24 MiB allowance. The source report is 6,924,187 B; overlay 117,144 B. HTML was 3,032,873 B, including 2,993,260 B of inline data; endpoint rows alone were 2,050,000 B. Existing exact aliasing plus source-report gzip did not alone meet output limits. This failure and measurements remain in `.local/camera-multiview-report-payload-measurement-01.json`.

The final **overflow-only, explicit multiview** correction keeps the full plain `report.json`, using compact whitespace. Only when the normal page exceeds 2 MiB, its display data uses shared point-field layouts and repeated-string tables. The main script expands only the selected camera's twelve points; absent fields remain distinct from null. Complete source, grid, surface and both-camera slot inventories remain; all other logical fields stay in the full plain JSON and exact source downloads. Inline display bytes are pinned by `report.json.display_payload` (`encoding: inline_shared_point_tables`, exact hash/length) and by the ordinary HTML manifest pin. No new external script, fetch, backend, file-origin policy or browser feature is introduced. Missing/invalid JSON, required dictionaries, layouts or indices disable interactive details visibly while preserving saved static results/downloads.

A provisional external-script/SRI design passed scripted checks but was abandoned before public recovery because local-file origin policy was unverified. Its measurements and tests are retained as provisional evidence, not final browser acceptance. The final inline route uses the existing CSP hash-authorized script. Ordinary small multiview output and all four legacy/default model+HTML byte goldens remain unchanged.

Final cached actual-model size check:

| Item | Bytes |
|---|---:|
| Complete plain `report.json` | 2,997,652 |
| Packed JSON within HTML | 1,847,907 |
| Complete HTML | 1,891,052 |
| Entire output including terminal manifest | 16,614,274 |
| Remaining HTML / output allowance | 206,100 / 162,942 |

This sizing reused the once-measured deterministic gzip length; its unknown gzip content hash was represented only by a same-width sizing placeholder. It was **not a custody receipt or delivered package**. The later public recovery below verified actual output bytes, hashes and manifest in a NEW directory. No sealed source reread followed the cached-model capture in this lane.

Scripted checks retained all 480 source slots, 112 references and 2,688 point records. 200 view changes took 20.300 ms; 3,000 duplicate events caused zero DOM writes; selected point tree stayed 109 nodes. The focused test instruments decoding: twelve points on initialization and no decoding on unchanged events. These are Node/scripted-DOM observations, not browser responsiveness or physical accuracy.

Seven final recovery cases plus six affected default/small checks pass. Current total: **40 distinct new report cases plus the earlier affected legacy public case**. The provisional five external-script cases are superseded, not added again. Corrective receipt `.local/camera-multiview-report-overflow-checks-04.json`, SHA `58a038360eb9df64630f7c1ad936375c6c98dbe93afdc5676a2e649a3ae64d70`; cached current-size receipt `.local/camera-multiview-report-recovery-size-check-03.json`, SHA `2da8096ada0f13068211a87a9b5bf8cc90bc1886d52b23694148272adb753a25`. Reconstruction independently reviewed the exact final code/tests without rerunning checks and reported no remaining blocker.

Independent saved arithmetic audit confirms primary mean error 12.792154 → 10.868088 px (1,241 known / 103 missing), secondary 27.036614 → 24.971869 px (704 / 640), zero invalid known projections and all fourteen gates true. All 480 surface-channel slots remain unavailable; contact remains unknown. Source-image improvement is not simulator/task success or physical validation.

## Delivered recovery and actual saved-page check

Parent ran the public `body-reference-report` path once against the completed proposal, with no refit or native work. It passed in 0.8303 s caller time. Open `artifacts/camera-multiview-real-01/inspection-recovered-01/index.html`; seal `64c6d9bbb55d869d193664924810cab84b83bf4c6c4ac1c58f8bf41b73755ade`. Parent receipt `.local/camera-multiview-report-recovery-01.json`, SHA `694aa8e556d69a437a0d7ce0d1006c9912deecfbcebcf2a1c0d0bf0fac8d0ea6`, verifies stored files, raw/gzip aliases and local links. That scan was not repeated here.

Saved-only UI receipt `.local/camera-multiview-recovered-ui-audit-02.json`, SHA `ed0e1fca63756fee770c588bd42b2780b00b6bf61a0645eff81dec08481a5977`, passed in 0.269604 s. Exact saved pins are HTML `8a5a05d5914a12a6d789701241da504c1e19bce202f1f97dc680d1cd8a519756` and plain model `ce1af3c646a5ad46c2fee35f32c800faea30a4d20673568c45e886094666fd74`. Inline payload is 1,847,907 B, SHA `6d8c653f1c0b3401e2fe32c6e288ca855d655a483fed9a1fb5f7e9492e728186`; there is **no `report-data.js` file**.

- Complete packed/plain equality covers 2,688 point records, 120 retained source-pose rows (112 fitted references plus eight other retained rows), 239 grid rows and 240 original slots per camera. All fourteen saved gates remain true.
- The actual page script visited all 480 source slots and checked 224 fitted view/reference details. Initialization took 5.826 ms, source-slot navigation 1.616 ms, and 200 view changes 52.579 ms. Three thousand unchanged events caused zero DOM writes; the selected point tree stayed at 109 nodes. These are Node/scripted-DOM measurements, not browser or physical validation.
- Auditor01 retained a checker-only failure: it confused 120 retained source rows with 112 fitted references. Complete packed/plain field checks had already passed. Auditor02 corrected only that denominator and reused those checks after all three saved hashes matched. Both attempts read exactly `manifest.json`, `index.html` and `report.json`, 4,891,771 B each; aggregate 9,783,542 B remained below the explicitly approved 10 MiB bound. No source, gzip, other artifact, report-generation, physics or browser work was repeated.
- Failed receipt `.local/camera-multiview-recovered-ui-audit-01.json` remains unchanged, SHA `462ad72b453e3c1c4279d8b938a86e9d5bbd31dcbc0d6f0f7d979350c7abe8a1`. Final auditor `.local/check_camera_multiview_recovered_ui_02.py`, SHA `bc810c6c9e259767afc135da05ecdbd7d9e4c0cb3f90712c13a8dd1062f5d60f`, uses stdin for Node payload delivery.

Next: parent publishes the reviewed report correction and continues controller admission separately. Do not refit, overwrite either failed or recovered output, alter fit01, or relabel its completed worker result. Source-image improvement remains distinct from simulator performance and physical accuracy.

## API and exact boundary

Existing entry point remains:

```python
prepare_whole_body_reference_report(source, out,
    expected_manifest_sha256=None, max_seconds=10,
    max_input_bytes=None, max_output_bytes=16 << 20)
```

```sh
.venv-lab/bin/python -B -m weft body-reference-report   --run PATH_TO_SEALED_PROPOSAL --out NEW_REPORT_DIRECTORY   --manifest-sha256 EXACT_PROPOSAL_MANIFEST_SHA256
```

- Explicit objective `camera_multiview_surface_interval_v1`, method `blocked-multiview-camera-body-surface-interval-reference-v1`, overlay `camera-body-multiview-angle-proposal-v1`. No `candidate-human.json`, body installation, source relabeling or physical admission.
- New helper `validate_multiview_report(report, overlay, body, primary_evidence, documents, hashes, check)` validates saved joins/arithmetic after unchanged shared floor/surface/primary checks. Renderer helpers select one camera at a time.
- Optional imports occur only in the new mode. Inspection archives the exact current `camera_multiview_report.py`, `camera_multiview_inputs.py` and existing `source_surface_fit_inputs.py` bytes and hashes. Default legacy report/model/render bytes remain unchanged.
- Root's `plan.secondary_view` descriptor selects `inputs/secondary-camera.json`, `inputs/secondary-evidence.json`, `inputs/camera-secondary-config.json`, optional alignment, and one to eight observation files. Exact raw pins join the outer seal, original source proof and `plan.source.copied_files`. Original camera/evidence aliases are compared byte-for-byte.
- Primary input bindings remain exact. `additional_view` joins overlay and static camera metadata; its canonical binding pins both views, secondary raw/canonical evidence, correspondence, alignment, original root-source entries and complete qualification denominator.
- Every reference retains 24 named points, primary twelve then secondary twelve. Source confidence is separate from authored per-view loss scale and weight. Per-view raw-pixel means, known/missing/invalid counts and image/domain/depth gates stay separate; a better pooled mean cannot override a failed view or zero-known-target view.
- All fourteen original gates remain mandatory. Every valid saved projection must have depth within `[0.2,100]`; secondary radial coefficient must be within `[0,1]`; every camera slot keeps its original frame/fps clock. Saved arithmetic is checked without projecting points or running geometry.
- Complete original source slots survive, including excluded, unsampled and metadata-only rows. A missing secondary pose can remain explicitly unavailable at an original body reference. The shared pixel checker permits that only via a new-mode-only option; no source record or observed target is fabricated.
- Selected secondary details show secondary evidence ID/decoder/status. The unchanged primary body-reference identity is labeled separately. Source, decoder and body clocks are not retimed.

## Bounds and performance

Input limit defaults to 16 MiB for legacy modes and 24 MiB only after explicit multiview classification under the original envelope. Explicit tighter caller limits remain effective; a caller limit above 16 MiB rejects legacy modes before reading other members. New-mode individual members stay at most 8 MiB. Output remains 16 MiB, HTML 2 MiB, body references 900, source slots 1,800 per camera. No alias read/hash is skipped. The new-mode report records the resolved input cap; source readback remains bounded at twice that cap. Exact-byte output aliasing and deterministic gzip fallback for the original source report remain unchanged; every logical member/hash/link is retained. Over-limit evidence is rejected, not truncated.

The UI holds one 24-point array per reference and renders only the selected twelve-point view. The shared floor chart is not duplicated. A numeric source-slot inspector visits complete slots without a table of every frame.

Scripted DOM measurements (not browser responsiveness):

- 1,800 renderer-only slots per view: 3,600 slots visited in 7.963 ms; HTML 1,067,555 B; initialization 2.656 ms. This is a display stress projection, not a newly qualified producer/source fixture.
- 200 camera changes: 10.227 ms; selected point tree stays 109 nodes; shared chart path preserved.
- 3,000 unchanged events: zero DOM writes or new nodes.
- Actual saved analytical producer payload with current final validator/script: 20 source slots, 200 camera changes, 3,000 duplicate events, zero duplicate writes; HTML 118,044 B; 0.074515 s total local check. No browser, network or server used.

Real-source size preflight used metadata/stat only: known selected inputs were 14,830,612 B before the new report/overlay/plan/seal. Identical primary aliases account for 2,031,429 B, but every logical alias is still read and verified. Parent losslessly re-encoded secondary observations from 5,145,496 to 2,769,293 B into NEW input02 with new explicit pins, preserving input01 and all parsed values. Even then 12,454,409 B of selected input plus the prior single-view compact report (5,208,872 B) totals 17,663,281 B before the overlay/new-view fields. This measured insufficiency justified a parent-authorized **24 MiB input allowance only for the new multiview profile**; no output/HTML/native cap changes. Additional raw allowance is bounded at 8 MiB; parsed-object/RSS cost remains to be measured by parent acceptance. Local/API cost is $0, recurring cost $0. This lane did not read source arrays. Initial arithmetic is in `.local/camera-multiview-report-size-preflight-01.json`; final boundary evidence is `.local/camera-multiview-report-budget-checks-02.json`, SHA `7cda81a7b707bc0b5e6e01e6e285ea19e2605720bc539928cfab496f6ee820d1`.

## Verification and retained failures

Before the corrective overflow work, thirty-three distinct new pure/mock test cases plus one affected legacy public-report case had passing evidence; the current forty-case total is recorded above. The budget delta adds three focused cases: automatic 24 MiB only for the explicit new family, tighter/legacy caller limits, and the unchanged 8 MiB new-mode member ceiling. The 17 MiB analytical input test stops at a mocked model after exact byte preflight; it does not claim semantic admission for padded data. They cover default byte goldens, complete source joins, view/endpoint/clock/pin corruption, excluded and missing secondary poses, per-view regression despite improved pooled mean, invalid final targets, zero known targets, original proof byte counts, archived helper custody, supported depth/radial domain, source-grid clocks, safe JSON/markup and bounded selected-view navigation.

All eleven initial failed attempts remain in JUnit receipts; two expected pre-feature failures for the new input envelope are separately retained in budget-red-10:

- Two floor-golden failures came from test-only nested maps built from sets; recursive fixture ordering produced identical published-old/current model and HTML bytes.
- One initial portable fixture lacked required secondary source width/height metadata; fixture-only correction.
- One genuine missing-secondary-pose indexing failure; explicit unavailable handling fixed it.
- Four genuine depth/radial contradiction failures and one original source-clock failure from independent review; scoped fixes and targeted green checks.
- Two DOM harness failures omitted the required `report.json` artifact link; fixture-only correction.

An overbroad `-k` in focused03 unintentionally repeated four previously passing default goldens. Receipt counts deduplicate them. No broad legacy/fitter suite was rerun. Exact available commands, XML pins and early-log limitations are in the checks receipt.

## Actual producer composition and provenance limits

Reused `.local/camera-multiview-analytic-02` without fitting: source manifest `0527213eff0296380ada56b73b1e8d6c313e28d3a52344ee3cd19b4754921cc7`, report `711233c4fa074a2b40d14bb224540a909182a24d84708d36d4749b2487231ccb`, angle overlay `f8129287b3132c4cdbb0c10271f931cfbae1bfbc2c2f5071bd0e25beeaccbdd1`.

The actual producer used authored analytical kinematics and the existing QP, not native physics or a qualified real-source ancestry. Its missing plan/config/source-proof wrapper metadata was added only in `.local/camera-multiview-report-analytic-source-01`, explicitly labeled **inspector-only authored declarations**. Every retained producer input/report/overlay byte stayed exact. This proves producer-to-inspector schema/arithmetic, not source-loader admission or recovered history.

- Public report output `.local/camera-multiview-report-analytic-inspection-01`: seal `ee88ddd1ec1ae057593c0423aca535d4c730a8c2c827ae26f9e19feb102ad5dc`, 28 files / 540,519 B; 0.016528 s report call. Both views retain ten source slots and 36 known targets; all fourteen saved gates true, separate scales `.01/.025`, weights `1/3`.
- `.local/check_camera_multiview_report_saved_01.py` and `.local/camera-multiview-report-saved-producer-01.json` retain the wrapper recipe and exact source pins. This first package predates the final narrow metadata/UI fixes and remains unchanged.
- `.local/check_camera_multiview_report_final_ui_02.py` revalidates the same saved authored bytes with the final code and tests the current script. `.local/camera-multiview-report-final-ui-checks-02.json` is the final validation/display receipt, preceding only the later input-budget wrapper change. Its HTML is a scripted test page, not another sealed standalone report package.

No native model, media decode, fitting, projection, source inference, API call, browser interaction or Git mutation occurred in this lane. Contact, load, grasp, anatomy and physical accuracy remain unknown/unverified. Budget and final inline recovery deltas are independently reviewed clear; earlier schema/display findings are also closed. Reviewers reused passing receipts and performed no tests or source/report runs. Root owns subsequent controller integration and publication; do not refit or repin old evidence merely to rerun this inspector.
