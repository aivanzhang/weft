# Guided shared-exercise configuration report

## Status and ownership

The offline editor and complete frame inspector are implemented, independently reviewed and frozen. Eighteen distinct analytical test cases and one saved actual core-produced payload composition pass. No browser, source processing, model, simulation, API or service was run. There is no active job in this lane.

- Report SHA256: `81e84996b225e5700fbacb55782f7c7043061dcc5937310020c9d09ad4f7c5e9`.
- Test SHA256: `6c033474b75506dad6d18a4fd0ab06572baf567afb1cff2028d33f4439003b40`.
- Final scoped receipt: `.local/shared-exercise-config-report-checks-03.json`.

Owned files: `src/weft/shared_exercise_config_report.py`, `tests/test_shared_exercise_config_report.py`, and this note. The reconstruction lane owns `shared_exercise_config.py`; the parent owns CLI integration, publication and any later real acceptance. Do not edit their files or repeat their source work.

## Contract

`build_shared_exercise_config_report(files, *, max_seconds=5) -> dict[str, bytes]` consumes exactly `metadata.json`, `snapshot.json`, `request.json`, and optional `config.json`. The caller writes returned members under `inspection/`, preserves the exact root inputs and publishes one final seal. The report does not create a seal or authenticate original source files.

- Returned files: `index.html`, `brand.css`, `report.json`, and `frames-NNN.html`. Raw input links use `../`; the complete snapshot is not duplicated.
- The only external asset read is `src/weft/assets/brand.css` (at most 32 KiB), returned unchanged. Include it in the generator archive.
- Limits: 16 MiB combined input, 8 MiB snapshot, 256 KiB metadata, 128 KiB request, 64 KiB config; 16 MiB output, 2 MiB per HTML, 32 output files. At most 1,800 original/decoded frames, two observation sources, 12 entities per source, 900 body references and 43,200 observation slots. Each frame page has at most 100 frame indices and 1 MiB of prepared frame data. The five-second deadline is cooperative, not process isolation.
- Each observation view retains its own frame/time. `original_frame` remains null when alignment is unavailable. The pages retain original metadata-tail slots separately; they do not invent observations or interpolate body references.
- Hashes join the exact request, snapshot and optional config. Counts, complete entity grids, body-reference availability and request-to-config projection are checked without geometry or media work.

## User behavior

Discovery makes no body-evidence, object, frame, target or camera choice. Body size `1.0` and assistance `false` are visibly authored defaults, not measurements. Saved explicit choices remain editable. Adding equipment requires a source-eligible frame; that eligibility does not admit geometry.

The current builder requires explicit saved body ID `person-0`. Other IDs remain visible in the inventory/frame inspector but cannot become accepted edited body choices. A saved incompatible choice is preserved, not renamed; its original request remains downloadable. Incompatible `config_ready` payloads are rejected. This is a current builder limitation, not an identity inferred from labels.

The request keeps the existing core schema: explicit human identity, fixed-pelvis camera-distance prior, `floor_box_depth_equals_width`, selected object/frame pairs, one selected target ID, optional positive XY frame translation, and separate build options. Explicit null translation means disabled, as in the existing core; it is preserved in saved data and unchanged downloads. Only actual vectors receive the existing nonzero/bounds checks. No target coordinates, hand-derived geometry or contact labels are added.

The page downloads a new request; it never edits the sealed package or runs commands. Incomplete choices can be saved. Invalid but parseable choices retain the core's `needs_input` status and named issues; the raw request stays downloadable, with no invented editor values. Malformed envelopes and foreign snapshot pins are rejected.

`config_ready` means **configuration ready; world validation still required**. The saved config is the unchanged equipment-proposal projection, not a body or world. Floor-resting box assumptions remain explicit, and building still checks geometry and collisions. Contact, grasp and load remain unknown.

After editing:

```sh
.venv-lab/bin/python -m weft shared-exercise-config \
  --resume PATH_TO_CONFIG_PACKAGE \
  --request PATH_TO_DOWNLOADED_REQUEST.json \
  --manifest-sha256 PRINTED_MANIFEST_SHA256 \
  --out NEW_CONFIG_DIRECTORY
```

The page uses labeled placeholders because the package's final seal does not exist while rendering. The CLI prints exact paths and the actual seal after publication. Ready pages show the existing `shared-exercise-build` command with copied source paths, explicit body size, optional assistance and mandatory `--full-reference`; it is not executed.

## Verification

First focused run (16 distinct cases, zero failures):

```sh
.venv-lab/bin/python -B -m pytest -q tests/test_shared_exercise_config_report.py \
  --junitxml=.local/shared-exercise-config-report-tests-01.xml
```

Full output is retained in `.local/shared-exercise-config-report-tests-01.log`. The XML SHA256 is `65a575ea20e78a56bf550d763e5bf17cfab69340b48d8ea180d28464a3a6925c`. Its only warning concerns pytest's `record_property`/xunit2 format; the recorded measurements are present. No unchanged suite was rerun.

The person-ID compatibility boundary adds one distinct case. Only that case and the affected ready/editor checks were subsequently run: three passes in `.local/shared-exercise-config-report-tests-02.xml`, SHA256 `ecc55852c63979ccd3c5737a21154ca77ae5c7fe49dba65b82276d56226fc067`, with full output in the matching `.log`. Those first two runs had no failed attempts.

Independent review then found a real report/core mismatch: the core accepts explicit null translation, but the report required a vector and its editor indexed null. One new regression reproduced it in `.local/shared-exercise-config-report-null-red-03.xml` (SHA256 `89f99990d739348798a67d39186081026b7cc18cf0e29a0e991e124890d8e0db`). The report-only fix passed that same case in `.local/shared-exercise-config-report-null-green-03.xml` (SHA256 `719e3646e2ddc4def5fc64021c6e95d4de115af763ff4e2e00360fb017afaef9`, 0.188 seconds). Both full logs remain. The test checks disabled controls, untouched saved bytes/config pin, an unchanged null-preserving download, and explicit vector enable/disable. Prior seventeen passing cases and the saved producer composition were reused, not rerun.

The independent reviewer verified the final product/test hashes above and closed this finding without rerunning checks. The review-closure receipt is `.local/shared-exercise-config-report-review-closed-03.json`; earlier receipts remain unchanged.

Checks cover empty discovery choices, exact downloaded requests, ready config/options/CLI joins, unavailable-source clocks, invalid options without fabricated defaults, resealed missing/duplicate slots, wrong reference/admission/config/identity, strict JSON and bounds, safe markup, unchanged-update no-ops and complete maximum-frame pagination.

Representative authored package: 1,800 original frames, 900 references and 5,400 observation slots; 2,082,802 input bytes and 2,390,072 output bytes, prepared in 0.0671 seconds. All slots survive pagination. Scripted navigation checked 300 exact selected-page slots; 1,000 changed updates took 7.790 ms with no new nodes, and 1,000 unchanged updates caused no text writes. These are Node fake-DOM measurements, not browser responsiveness or physical-validation evidence.

Those measurements belong to the first tested report (`b6393379…`). The identity and null-translation fixes add 472 fixed bytes only to the overview form/script; the maximum-frame output is therefore 2,390,544 bytes by exact static-byte accounting, not another benchmark run. Frame-page code and data are unchanged. The focused editor check also preserves its unchanged-update no-op assertion.

### Actual producer-schema composition

The core preserved `.local/shared-exercise-config-analytic-01`, producer seal `1e9fbe2a6cb661cda10d128082152b23697d167e81c4bcf91fbf2027472a81d6`. One guarded bytes-only composition verified that seal and the exact metadata/snapshot/request/config member pins, then rendered them without rerunning the core fixture. Its original report was a test stub, so this check verifies the actual core output schema against the real reporter.

Evidence: `.local/shared-exercise-config-report-composition-01/receipt.json`, SHA256 `9d8f9d2634e2c39dd7caeb7b469f20cd8aec852c8e53d4c1c1a985422052f6e2`; five selected metadata reads, 17,159 bytes, 0.00706 seconds. The inputs contain eight original slots, six decoded slots, three body references and twelve observation slots. The generated four report members total 33,142 bytes. Inputs and generated members are retained as schema-check evidence, not as a separately sealed user-ready package. The supplied identity is `person-0`, so the later focused compatibility boundary needs no producer rerun.

## Next safe steps

- [x] Consume one preserved actual core-authored metadata/request package with this reporter, without rerunning source or fitting work. No schema mismatch occurred.
- [x] Complete independent review, freeze exact hashes and record final scoped evidence.
- [ ] Parent integrates the CLI/runtime archive and performs any separately authorized public acceptance. Do not treat this analytical report lane as source or world admission.
