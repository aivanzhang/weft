# Public exercise/equipment evidence composition

## Current status — November 1, 2026, 1:04 AM PDT

The public ordinary-input path is implemented, reviewed and verified on the existing real kettlebell development source. Root ran **tracking-masks → exercise-equipment-evidence → equipment-support --hand-points** exactly once; all three stages passed in **19.166414 s**, with no paid call, new inference or physics. This lane independently audited the saved mask/composition bytes; root and UI audited the disjoint downstream custody/report. **60 focused analytical checks remain passing**; no unchanged suite was rerun. No live handles remain. Product/test paths are frozen for root's scoped publication.

The source is exposed development evidence, not an unfamiliar-video holdout. Conditional hand/equipment image consistency does not establish contact, load, grasp, ground or a runnable reconstructed body. Original-start missing human evidence, all excluded slots and original parent limitations remain visible.

Owned paths: new `src/weft/exercise_equipment_evidence.py`, `tests/test_exercise_equipment_evidence.py`, `tests/test_segmentation_masks_binary.py`, this note; optional binary retention in `segmentation_masks.py`; the root-approved narrow known-manifest guard in `pose_precursor_custody.py`. Root owns CLI, public workflow guidance, the completed real acceptance and publication. No Git mutation or unrelated avatar/scorer/lab change was made.

## Agreed callable and outcome

`prepare_exercise_equipment_evidence(pose_run, equipment_run, masks_run, out, *, human_id, equipment_ids, minimum_confidence, uncertainty_px, uncertainty_method, uncertainty_note, source_equivalence_run=None, expected_pose_manifest_sha256=None, expected_equipment_manifest_sha256=None, expected_masks_manifest_sha256=None, max_seconds=30., progress=None)`.

Compose the existing **pose-equipment-silhouette-composition-v1** archive accepted by `equipment-support --hand-points`. Return `profile`, `status=completed`, `sealed=True`, `output_path`, `manifest_sha256` and bounded diagnostics. No fixed-camera assumption is adopted; the next support command is a separate explicit user action. Confidence and pixel ranges remain authored, not calibrated.

Reuse precursor custody → partition-safe `adapt_exercise` → pose annotations → existing wrist silhouette evidence. First floor(60% of original sampled pose rows) is fixed before point queries. Retain all original decoded/container/sample slots, raw/context/audit/model identities, original clocks, excluded metadata and unknown contact. Explicit original human/equipment IDs are required. A differing equipment source requires the complete verified existing SourceEquivalence bundle, including exact saved media bytes without decoding.

## Parent failure and binary custody

Complete pose extraction can precede a later body failure. Preserve the exact parent seal and sealed failure marker; never relabel the parent, construct a body, or claim simulation readiness. The only precursor extension accepts the explicit `exercise-body-source-v1` schema_version=1 manifest with status `construction_only` or `failed`, retaining its bytes and exact three member hashes. Legacy schema=1 serialization remains unchanged. Unknown manifests are rejected; no synthetic schema-1 ancestor.

`segmentation_masks.export_masks(..., retain_binary=False)` now optionally retains the **already decoded** binary cells plus original PNG hashes/dimensions/IDs in `binary-retention.json` and `binary/<original-mask-path-without-.png>.bin`. The old three-positional public call and stable default mask document/result fields remain byte-exact; runtime hashes and real timings correctly change when implementation changes. Retention does not reduce the existing full mask decode scope or its 128,000,000-byte / 50,000,000-pixel limits.

The compositor copies every eligible **construction** mask once, independently of wrist confidence, so later qualified fingers are not blocked by absent wrist queries. It reuses the existing equipment/clock/mask qualification helpers after the existing wrist producer. `retained_binary_loads` counts copied masks; `raster_loads` counts actual wrist geometry queries. PNG bytes are copied and hashed, never decoded. Excluded, unsampled, unavailable-human, preinitialization and unavailable-mask slots supply no raster. There is no hull, hand-contact claim or changed point algorithm.

Known ordinary parents require exact source/config metadata and the sealed `failure.json` when status is failed. `model.json` is optional on failed parents: pose quality can fail after all original pose/audit files are written but before the wrapper writes that separate model file. Embedded raw/context model metadata still must agree; absent files are not synthesized. No `human.json`, `observations.json` or `source-clock.json` is required. Legacy schema-1 parent bytes remain supported without an assumed body-success status.

Complete scheduling requires both raw/context rows to match the original inference stride, including unusable rows. A container/decoded difference larger than one needs the existing encoded-frame audit **or** a real loaded whole-video SourceEquivalence with exact original source, dimensions, frame count and every sampled decoder clock. It does not authorize retiming or reinterpret nominal frame/fps timestamps. The exact mask report must bind the equipment parent seal/file map, not merely assert a shared source hash.

## Bounds and verification plan

- Selected input bytes ≤32 MiB; ≤900 pose rows / 1800 source slots; ≤512 selected raster loads / 4 MiB binary cells.
- New output ≤64 MiB / 384 files; bounded selected manifests, no overwrite, path traversal or symlinks. Record failed/unsealed work honestly.
- The inherited decoded-mask producer still supports at most 500 decoded frames / eight objects. The larger original slot bound does not bypass that limitation or permit truncation.
- ≤30 seconds cooperative orchestration; reserve existing 5-second custody / 10-second annotation / 10-second silhouette budgets before each stage. This is not hard process or memory isolation.
- One bounded ZIP stores loaded local Python source bytes/pins rather than many loose generator files. This is not a standalone interpreter/dependency archive. No current shared lab runtime was repinned.
- `result.json` is a preparation receipt (`prepared_pending_terminal_seal`, `sealed=false`). All deadline/source checks precede the final verified manifest write. That manifest is terminal and cannot later be relabelled by caller timing or cleanup. A caller exception after an exact completed seal raises and saves only a new exclusive **sibling** `<out>.caller-failure.json` (≤8 KiB), leaving the entire completed directory unchanged. An existing sibling is never overwritten. Only an invalid partial manifest may become `attempted-manifest.json` under a failed artifact. Failure cleanup is best-effort and bounded cooperatively to two seconds; it is not success within the original time budget.
- The complete SourceEquivalence is verified once with before/after and final member identities. Its small manifest is additionally read for the preflight inventory; these duplicate bytes/read operations are explicitly counted within 32 MiB. Proof bodies are not read twice. `input_files` counts unique selected members and `input_read_operations` includes that manifest reread.

## Verified implementation and exact checks

- `.local/exercise-equipment-evidence-tests-01.xml`: 37/37, 1.026 s, SHA `2f6ee7e074f0836c66a565e6a071ce966e8ec683ff71ebadc2e8ce898f6df31a`.
- `.local/exercise-equipment-evidence-tests-02.xml`: 24 affected/new cases, 0.890 s, SHA `47c530cc9de0a9d701183abd430c06ddb4952bdf6ab430343263ef7f07cd5538`.
- `.local/exercise-equipment-evidence-tests-03.xml`: nine new/extended cases, 0.193 s, SHA `2dfb88e56a928f0da4fda1dc88a3eeefc468648f5bc6227bc7348d029abca0b2`.
- `.local/exercise-equipment-evidence-tests-04.xml`: seven affected budget/deadline/count-qualification cases, SHA `79628a5b4dd98c84fde5b30948fb09dca2e8dc4c7c16eea8ad6d948ff571b142`.
- `.local/exercise-equipment-evidence-tests-05.xml`: six focused terminal-publication/proof cases, SHA `06591fac068d68e79688b0a76f09a8b9316760dea807c2e005eae53d7bac0ab8`. Includes completed-seal preservation on late write and later caller error, no sibling overwrite, invalid partial-write retention, full proof admission and a late proof-member mutation rejection.
- Initial implementation checks had no test failure. The later capacity regression deliberately reproduced the ordinary-report rejection; its red receipt is retained below. Review identified two real issues despite the first passing suite: revoking a published seal and omitting proof files from final identity checks. Both were fixed and independently cleared. The old rollback assertion and unparameterized proof test are superseded, not counted alongside their replacements. Coverage at that milestone was 57 cases, not the sum of run totals; the three capacity cases below bring current deduplicated coverage to 60.
- `.local/exercise-equipment-default-goldens-01.json`, SHA `6a281b6aff8825b9b98226d8264ab973a7827169b11a3775bb26e5caa22d8814`: compares actual function bodies against committed `3feeb1d` under the same authored/stubbed decoder harness. Default mask document/result and legacy precursor metadata/all rows are exact. No codec, video, native engine or model was run.

Commands:

```sh
.venv-lab/bin/python -m pytest -q tests/test_exercise_equipment_evidence.py --junitxml=.local/exercise-equipment-evidence-tests-01.xml
# Historical tests-02 included the now-superseded late-write rollback assertion; its original receipt is retained.
.venv-lab/bin/python -m pytest -q tests/test_exercise_equipment_evidence.py::test_recorded_derivative_proof_resolves_declared_count_discrepancy_without_retiming tests/test_exercise_equipment_evidence.py::test_committed_legacy_precursor_metadata_and_rows_remain_byte_exact tests/test_exercise_equipment_evidence.py::test_unknown_parent_manifest_is_not_a_new_legacy_alias tests/test_exercise_equipment_evidence.py::test_loaded_runtime_is_a_single_bounded_exact_source_archive tests/test_segmentation_masks_binary.py::test_optional_retention_reuses_each_decode_and_leaves_default_mask_document_exact --junitxml=.local/exercise-equipment-evidence-tests-03.xml
.venv-lab/bin/python -m pytest -q tests/test_exercise_equipment_evidence.py::test_explicit_byte_file_and_load_caps_preserve_failure_without_dropping_rows tests/test_exercise_equipment_evidence.py::test_output_path_no_overwrite_or_symlink_and_deadline_stops_before_new_stage 'tests/test_exercise_equipment_evidence.py::test_pose_parent_admission_fails_before_raster_and_preserves_failed_package[decoded]' --junitxml=.local/exercise-equipment-evidence-tests-04.xml
.venv-lab/bin/python -m pytest -q tests/test_exercise_equipment_evidence.py::test_final_verified_seal_is_not_relabelled_when_write_finishes_late tests/test_exercise_equipment_evidence.py::test_caller_failure_preserves_published_seal_and_partial_attempts tests/test_exercise_equipment_evidence.py::test_recorded_derivative_proof_resolves_declared_count_discrepancy_without_retiming --junitxml=.local/exercise-equipment-evidence-tests-05.xml
```

Frozen product pins:

| Path | SHA256 |
| --- | --- |
| `src/weft/exercise_equipment_evidence.py` | `21b32b26a0284ae9b06453474879a45705558cf02c2b797f1dca997013f7eef9` |
| `src/weft/segmentation_masks.py` | `99226ceeae0ebf5c1fec20f09f501f5024d92fc9cf9317712d8859eb1d974e6f` |
| `src/weft/pose_precursor_custody.py` | `6dd5d851fea39ef86167917f1f240cfcb522e4cd8b63268ba9b57ddb93045da1` |
| `tests/test_exercise_equipment_evidence.py` | `58ac7bf361d6ff03c4738e64c8525471bc7ec7a205a8809f1c6c19d98d54954f` |
| `tests/test_segmentation_masks_binary.py` | `2c3f65b9665f57bc5591a78e3bd343d1f166fb080a54f94851a10e87b42e954e` |

The binary/precursor delta and corrected full compositor independently passed read-only review. There is no source performance/accuracy result here. Analytical timings do not prove responsiveness for a maximum-size real package.

## Root's next integration boundary

Prepared **inputs only** for root's public CLI check: `.local/exercise-equipment-cli-inputs-01/fixture.json`, SHA `55e6666d9cacffc91773fa6046fb405534a422e8076c2ccdc94cd308a42ae631`. The fixture has 24 original slots / 23 decoded / 12 pose rows / seven construction rows; the parent is deliberately failed and has no separate model/body files. Media/PNG bytes are opaque authored placeholders, not real decoded imagery. Preparing it called no compositor. Exact source paths, all three seal hashes and settings are in the receipt.

Root completed the authored public composition invocation on its frozen runtime; its saved-only audit/publication are root-owned. Later hand support remains a separate action with an explicit fixed-camera prior. A future real input pilot requires an exact sealed binary-retention export; old PNG-only mask packages fail clearly instead of silently decoding them. Do not rerun the exposed co-motion/hand studies or infer a new camera/ground/contact prior.

This slice makes the existing evidence pipeline reachable from ordinary saved producers. It does **not** complete unfamiliar-video body reconstruction, calibrate pixel uncertainty, detect equipment automatically, recover ground/scale/contact/grasp/load, or turn a failed body into a runnable world.

## Ordinary report capacity correction

The real ordinary equipment report is **168,516 bytes** by stat, larger than the initial 128 KiB selected-member bound. Only that file limit changed to **256 KiB**; the aggregate **32 MiB** input cap, all scientific settings, source guards and other file limits remain unchanged. The real report body was not read by this correction or discovery. Frozen prior runtime `891036f9…` and release receipts 01/02 remain untouched.

- Red `.local/exercise-equipment-report-cap-red-01.xml`, SHA `326d40649c10e8500734f7187235cbfd235e8bc3471725341b963c4d0cb80e3a`: one expected `ValueError: Package input is not a bounded regular file` under the original cap.
- Green `.local/exercise-equipment-report-cap-green-01.xml`, SHA `ab1c8063e6eb465f3b87b1c9ee3bb364da358fed34f61ca0d262a535cc21495d`: three authored boundary cases pass. Exact accepted bytes at 168,516 and 262,144 are retained; 262,145 is rejected, and the total input cap remains fixed.
- Independent reviewer checked only this one-line bound change and new cases; no unchanged test, real report read or scientific work was repeated.

```sh
.venv-lab/bin/python -m pytest -q tests/test_exercise_equipment_evidence.py::test_equipment_report_size_matches_ordinary_export_without_raising_aggregate_cap --junitxml=.local/exercise-equipment-report-cap-green-01.xml
```

## Real ordinary-input preparation — metadata only

Receipt: `.local/exercise-equipment-real-discovery-01.json`, SHA `17d6751013114e6bb8f2691d2143238edab435f91263a7b244d6ca3d148ce11e`. Actual discovery read **12 metadata inputs / 225,020 bytes in 0.017703 s**, with **960 final-phase stats plus a 64-stat preliminary upper bound** (1,024 total maximum). The CLI read was explicitly only its first 49,152 bytes. The full named pose/observation arrays, PNGs, binaries and source media were not read. Metadata hashes were checked; hashes beside stat-only members are ancestor declarations, not new body verification.

| Ordinary input | Exact seal SHA256 |
| --- | --- |
| `artifacts/mobile-exercise-e2e-10/kettlebell-pushup/run` | `e6e88ded526ec582b138dfd90285a0cdde65ddfd33596d255cc23f95afe5529b` |
| `artifacts/kettlebell-shared-evidence-01/evidence` | `acc67c78a8db45d07b855273a255d6e2e77eabd01f3d37c66e1f9e82df15cc40` |
| `artifacts/kettlebell-shared-evidence-01/masks` (old PNG-only export) | `2aa2243f27bdb1ea5b6783c34694cdbcca00734caa22b37823620af8772112a5` |
| `artifacts/pose-equipment-silhouette-02/source-equivalence` | `2c3cf30c05aeddd1d0751ee054993f1c9875bf85beb0f6e7c959f411a3a9e49c` |

All required named pose/context/raw/audit, equipment, old-mask and full proof members were present as regular files. The proof contains 26 members plus its seal, **7,546,905 bytes**. Its report preserves 259 declared versus 257 actually decoded source frames, complete original→derivative frame identity, native PTS 3…8536 ms and decoder-relative PTS 0…8533 ms; the explicit 3 ms origin offset is not a shifted source clock. Future qualification must use the real complete proof loader, not this metadata summary. The original WebM SHA is `4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`; submitted MP4 SHA is `34e74d9ab854a474ba647395f232e61dfde5cfa166bb449fd3057084c6dd5416` (6,023,720 bytes).

The old masks report is `decoded_proposals`: **257 frames / 771 object slots / 747 decoded / 24 not returned**, 9,744,912 binary cells and 78,746 encoded payload characters. It reports 14.371388 s historical decoder work and 9,045,866 bytes of PNG+preview exports. It has no binary-retention inventory, so it is **not** a valid new composer mask input. Existing study binaries must not be relabelled as an ordinary export.

**Inventory limit is explicit:** the old package has 1,158 declared members plus its seal. This discovery statted 903 members (14,032,429 bytes) and stopped at its cap with **255 names unmeasured in that inventory pass: 250 PNGs and five other members**. All names are retained in the receipt. The five non-PNG sizes are already available from named metadata stats (including the earlier pre-run stat), but individual remaining PNG sizes are not known. Using the old report's complete PNG+preview byte total gives a **report-derived**, not fully re-statted, old package size of 23,043,257 bytes including seal. Adding 747 binary files / 9,744,912 cells gives **32,788,169 bytes before the new binary inventory and runtime-closure changes**, about 1,907 files at the old closure. This is an estimate, not a newly measured export. The composer’s 384-file limit does not apply to full mask acquisition.

### Proposed public sequence; not executed here

1. `tracking-masks --run artifacts/kettlebell-shared-evidence-01/evidence --video artifacts/kettlebell-shared-evidence-01/masks/source.mp4 --retain-binary --out <NEW_MASK_RUN>` — retain all 257 original frames/747 available masks, with the existing 60-second spawned-worker / 2-second cleanup, 128,000,000 export-byte / 50,000,000 decoded-cell bounds. No new model/API request; this **does decode saved masks/source video**, requiring root's frozen execution plan.
2. `exercise-equipment-evidence --pose-run artifacts/mobile-exercise-e2e-10/kettlebell-pushup/run --equipment-run artifacts/kettlebell-shared-evidence-01/evidence --masks-run <NEW_MASK_RUN> --source-equivalence-run artifacts/pose-equipment-silhouette-02/source-equivalence --human-id person-0 --equipment-id sam-2 --equipment-id sam-3 --minimum-confidence 0.5 --uncertainty-px 2 --uncertainty-method <EXPLICIT_AUTHORED_METHOD> --uncertainty-note <NOT_CALIBRATED_OR_CONTACT> --pose-manifest-sha256 e6e88ded526ec582b138dfd90285a0cdde65ddfd33596d255cc23f95afe5529b --equipment-manifest-sha256 acc67c78a8db45d07b855273a255d6e2e77eabd01f3d37c66e1f9e82df15cc40 --masks-manifest-sha256 <NEW_VERIFIED_SEAL> --max-seconds 30 --out <NEW_COMPOSITION>`.
3. `equipment-evidence-report --run <NEW_COMPOSITION> --manifest-sha256 <NEW_VERIFIED_SEAL> --max-seconds 10 --out <NEW_REPORT>` — saved-data inspection only; optional hand support is a different explicitly fixed-camera step.

The fixed pose prefix remains 77 of 129 sampled rows, with all 259 original slots (52 later poses, 128 unsampled decoded slots and two metadata-tail slots) retained. Confidence 0.5 and ±2-pixel range are authored conventions, not measured error bounds. Missing mask/pose/identity remains unavailable; contact, grasp, load, ground and human-body admission remain unknown.

Root owns the three separately checkpointed CLI stages and their outer process/RSS/aggregate-file envelope. Existing stage work bounds total 102 seconds before interpreter startup, copying/hashing and final caller receipt; an outer bound must account for those costs separately. Full mask output is estimated roughly 33 MiB at the old closure, while composition/report retain their own caps. No expanded run or scientific setting is authorized by this preparation note. No live handles remain in this lane.

## Completed real public path and independent saved audit

Root's one frozen real acceptance used plan `4abe6e9326a34cbc2e436416fba4cc963ce36153b6b9e608e414383334f70266`, helper `39dc7cf6592892ef985cfa533bb924361abdd0c02e4f0241072b8444f14bedd3`, runtime `e613b8ce23aa5fe4ff019261f3d5d86e6b488024c956e1cf5782a16126a564aa`. Process `.local/exercise-equipment-real-01/process.json`, SHA `d19a5311e6c0bf30a128ef351aa39b65c362b248b6a39e0a3516c7266b871715`, records terminal success: **19.166414 s**, observed child peak RSS **203,423,744 bytes** (measurement, not a hard memory ceiling), 2,652 output files / 95,193,908 bytes. No model inference, paid calls or physics. The source video/saved masks were decoded only by the mask stage.

| Saved stage | Parent wall time | Files / bytes | Terminal seal SHA256 |
| --- | --- | --- | --- |
| `artifacts/exercise-equipment-real-01/masks` | 15.411572 s | 2,054 / 37,126,230 | `71def14d26bf11078ca32028047ca380f629c2873778ab6c97f6e7134fb8072d` |
| `artifacts/exercise-equipment-real-01/composition` | 1.369173 s | 343 / 22,438,731 | `067cd53bd009c7f1e53ac9957d2fa198908ce4df1bcbd2f648deb3779e9efaba` |
| `artifacts/exercise-equipment-real-01/support` (root/UI audited) | 2.248697 s | 254 / 35,344,677 | `29bd5d2123acd5d9c5954194d453d44c753da2d2a17727e85c5584893277019b` |

Full outer seal: `87eae41eb9b3e9496010adc03798a92d7b6338896fd3bfa243327206db9e089d`. The public follow-up kept the two original co-motion candidates and introduced none; this is conditional four-point image support, not physical contact admission. Root's independent outer/source/generator custody receipt is `.local/exercise-equipment-real-custody-audit-01.json`, SHA `80765c2ec9f6905696b094f8d99f3dcbd159176deac8aed73e4b9f75f506e2ca`; UI's disjoint saved report audit was reported clear at `9ca4551be15347288a6aa574f9189ab5e2485d9dfe9575601fe000348c2be5ab`.

### This lane's audit scope and retained checker failure

`.local/audit_exercise_equipment_real_01.py` (SHA `8b2d5a128387a3760501d1e29e134b4fb8ca1789234da8cf53f2df94f8406029`) read each of the **2,397 mask/composition files / 59,564,961 bytes once**, cached the bytes, and completed in **0.216121 s** under the 2,500-file / 96 MiB / 20-second bounds. It verified full member seals; all 747 binary/PNG/row/dimension joins; PNG headers only (no PNG decode); every original mask and source slot; exact parent/proof/source-clock copies; the selected pixel/confidence arithmetic; and all 40 archived generator-member hashes. No producer, loader, decoder, model or physics routine was imported or rerun; no original source inventory or data was reread.

Receipt01 `.local/exercise-equipment-real-saved-audit-01.json`, SHA `2d1c0458762bb6f3bda8c5d5f7907efe5abbfe761ac0a4b0a70d603fd37cec0e`, is preserved **failed**: its last assertion searched for `/exercise_equipment_evidence.py`, but the archive correctly uses the bare package-relative key `exercise_equipment_evidence.py`. This checker-only naming error occurred after all saved-body checks and all archive-member hashes passed. It is not a product or reconstruction failure.

Root authorized a narrow amendment, not a repeat audit. `.local/amend_exercise_equipment_real_audit_02.py` verified only the exact completed manifest plus `result.json`: **two saved metadata reads / 39,188 bytes / 0.000485 s**. The exact result pin is `1be219f06de17e77ef407952d79f9e70e22fc9dcc8107c0d8931228086832cf5`; its bare generator key matches reviewed compositor `21b32b26…`. **Passing amendment02** `.local/exercise-equipment-real-saved-audit-02.json`, SHA `39a95ceaa812f198f419d1ca04350dc43fad066c4d94dea628b895bc78c59b39`, reuses receipt01's completed body checks and preserves its failure. No tree, archive, source, producer or decoder was reread/reexecuted.

Verified saved denominators:

- Masks: **257 frames / 771 object slots**, 747 decoded masks / 24 not returned. All 747 binary files total **9,744,912 bytes**; every binary is exact 0/1 cells with its pinned original PNG and dimensions. Missing masks have no invented binary.
- Composition: **259 original slots / 257 decoded / 129 sampled poses / 77 construction poses**, with 52 later poses, 128 unsampled decoded slots and two metadata-only tails preserved.
- Wrists: **518 slots = 109 copied source estimates + 409 unavailable**. The 1,036 wrist/equipment pairs retain **64 conditional image intersections / 146 image exclusions / 826 unavailable**. Original frame zero remains unavailable; all four pairs say `human_source_unavailable`.
- Exactly **146 construction binary masks / 253,054 bytes** are retained with the original descriptors and PNG bytes. Composer reports **333 unique inputs / 334 read operations / 17,686,419 bytes**; the extra read is the explicitly charged 2,633-byte proof manifest, not a second proof-body read.
- Parent is still `legacy_hash_only_run`, with no declared success status and `body_admission=not_requested_or_evaluated`. The saved original Matroska count audit is retained, not rerun. All contact/grasp/load claims remain unknown, and no body/floor is installed.

The earlier output estimate was deliberately conditional on the old runtime closure; actual full masks are 37,126,230 bytes / 2,054 files, below the separately frozen stage/outer bounds. The completed masks and compositor are new immutable artifacts. Do not repeat either stage for subsequent reporting or publication.

Next safe action: root's scoped publication and user-facing ordinary-video instructions. Future improvement must supply or qualify missing ground/calibration/contact evidence; these image checks alone do not close the unfamiliar-video reconstruction gap.
