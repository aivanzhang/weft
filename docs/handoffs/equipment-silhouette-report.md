# Equipment silhouette inspector

## Frozen implementation — October 31, 2026

The offline inspector is implemented in `src/weft/equipment_silhouette_report.py`. It reads already saved equipment-silhouette metadata and complete frame rows; it does not acquire source data, query pose/raster adapters, decode images, run native physics, fit geometry, train or make API calls. No browser/server or Git operation occurred in this lane. Root owns the public saved-run wrapper, CLI, actual saved-data acceptance and publication.

Owned paths: the module above, `tests/test_equipment_silhouette_report.py`, and this handoff. These paths were new. No shared product files, avatars, scorers, source producers or common docs were edited.

## Exact interface and package

```python
build_equipment_silhouette_report(metadata_raw: bytes, frames_raw: bytes, *, max_seconds=5) -> dict[str, bytes]
```

Inputs are the exact saved `silhouette/metadata.json` (`EquipmentSilhouetteEvidence.describe()`) and `silhouette/frames.jsonl` (one complete `frame(i)` object in original order). There is no new wrapper schema. The versioned evidence profile is declared locally; the renderer imports only the Python standard library, never an evidence-producing/native module.

The caller verifies the source seal and selected input hashes before invoking this function, then writes/seals the returned fixed-relative-name package. The report does not authenticate original source/raster evidence or claim that matching saved hashes prove physical validity.

Returned members:

- `index.html`: complete coverage, named wrist/object pairs, raw reason counts, provenance/clock details and raw downloads.
- `frames-001.html` onward: at most 100 frames or 1 MiB of projected frame data per page. Only one selected frame/pair is rendered; repeated unchanged selections do nothing.
- `equipment-metadata.json` and `equipment-frames.jsonl`: exact supplied bytes, not reformatted or truncated.
- `report.json`: input/stylesheet SHA-256s, all page/frame boundaries, pair identities and explicit inspection-only scope.
- `brand.css`: exact current `src/weft/assets/brand.css`, read once with a 32-KiB bound. Root's wrapper must archive/pin this asset in its generator closure; it already does so.

No remote URLs, dynamic source links, fetches, timers, image loads or unsafe HTML insertion are used. Input strings are escaped in static markup and set through `textContent` in scripted views. All artifact links target members of the returned package.

## Meaning and validation

Every original source frame and every selected wrist/object pair remains available, including unsampled, excluded, unknown and metadata-only slots. The inspector retains the producer's reason priority: an explicitly excluded unsampled frame stays `excluded_by_explicit_frame_selection`; its missing pose ID remains visible instead of being reclassified.

The optional metadata field `canonical_frame_selection` is now accepted only with the exact `restrict-only-original-frame-selection-v1` contract. Its one canonical-source mapping must equal the original source SHA and `construction_frames` (including an explicit empty list); selected frames must be decoded original slots. Same-source aliases must be `{}`; derivative aliases must be exactly `{equipment_source_sha256: source_sha256}`. The fixed bound is1,800frames, `partition_consistency_scope=selected_frames_only`, and `excluded_partition_labels=preserved_not_reconciled`. An absent field adds no defaults or output changes.

When present, the exact object is retained in `report.json` and displayed in the overview. The overview and every frame page explicitly say partition consistency was checked only on selected original frames and excluded labels were preserved, not reconciled. Conflicting excluded labels remain visible as supplied. An empty selection means no frame received this consistency check. This never asserts full-source partition consistency or changes source partitions. Root reported a real producer-composition partition mismatch; the reconstruction lane owns its restrict-only correction and preserved failure. This inspector change does not repair or relabel that source.

The view distinguishes conditional image overlap, conditional image separation and unavailable comparisons. Squared distance is shown in **px²**, with the saved exact rational and outward-rounded interval retained. Zero distance at an excluded half-open cell edge is **not** overlap. Pixel uncertainty remains an authored range, not a confidence-derived physical error. Confidence, model identity assumptions, source pose time, decoder time, original/derivative frame mapping and native timestamp values remain distinct.

Metadata/pair schemas, complete frame order, stable Cartesian pair identities, named wrist declarations, source identities, counts, saved distance arithmetic and applicable clock relationships are checked. Honest unavailable clock mismatches remain reportable. Unsupported physical claims, excluded source pixels/raster geometry, missing slots, changed identities, invented zero values and inconsistent summaries reject the whole report; nothing is silently dropped.

Image overlap is not touch, image separation is not verified physical non-contact, and a wrist is not a hand-contact point. Physical contact, support, load, grasp and physical accuracy remain unknown/unverified. No body/floor installation or fit/simulator admission occurs.

## Bounds

- 1–1,800 original frames, 1–16 pairs per frame (at most two named wrists × eight model objects), 900 pose slots.
- Metadata ≤256 KiB; each raw frame row ≤128 KiB plus its newline.
- Compact complete producer content ≤8 MiB; raw input ≤8 MiB +1,801 separator bytes, preserving the producer's byte limit across JSONL serialization.
- Each HTML ≤2 MiB; total output ≤16 MiB and 32 files; cooperative preparation deadline at most five seconds.
- No global cache. The frame selector has at most 100 options; pair selector at most16. Page data is parsed once. Frame/pair changes update fixed text nodes only.

These bounds are enforced before/through bounded parsing and output assembly. The scope is saved metadata; it does not rerun geometry or validate PNG bytes.

## Checks and measurements

25 distinct reporter cases have passing evidence:19 in `.local/equipment-silhouette-report-tests-01.xml`, with two affected cases rechecked in `tests-02.xml` after bounding the stylesheet read and rejecting huge numeric inputs before float conversion; one new producer/schema composition case passed in `tests-03.xml`; five new selected-only metadata cases passed in `equipment-silhouette-report-selection-tests-04.xml`. No reporter check failed. The JUnit warning about `record_property`/xunit2 was non-failing; the properties are present in the saved XML. No unchanged broad suite was rerun.

The selected-only delta checks same-source and derivative bindings, unchanged excluded partition labels, explicit empty selections, eight malformed binding/scope variants, and full byte compatibility when the optional field is absent. All six old output member hashes equal the pre-change `e8da5e8` result recorded in `.local/equipment-silhouette-report-legacy-output-01.json`. The previous20passing cases were reused, not rerun. The maximum-size measurements below remain the original frozen renderer measurements; this narrow optional metadata/notice delta did not repeat that workload.

Coverage includes exact raw copies and resolving links, all original slots/reason priority, conditional overlap/exclusion/zero-edge differences, retained unknown clock failures, count/identity/physical-claim corruption, unsafe text, strict JSON, input/output/deadline caps, 16-pair width and scripted selected-detail rendering. Normal reporter tests block evidence-producer/native/media imports and network calls. Only local Node execution is allowed for the mock DOM.

The additional composition case reuses `tests/test_equipment_silhouette_evidence.py::fixture(count=7)` once. It passes the **actual** producer's `describe()` and all7 `frame()` outputs through the public report API:14pairs,21,215inputbytes, exact raw retention and unchanged squared-distance/status fields. Its preparation uses three nine-byte **in-memory analytical** rasters (27bytes), not source files or PNG decoding; native/media/model/network/process work remains blocked. The renderer then runs under its stricter no-producer-import guard. Metadata SHA-256: `8d62719c09aeb88cb44c4dab3f632420fc867ae037cd7b50f9160078f23abea8`; frame JSONL SHA-256: `d53ebb1f7e910ef7ceeef365c44df7c65c3c616cb71071e1ccfb68f81260ae6f`. This closes schema composition, not real-source qualification.

Representative maximum-frame analytical fixture:

| Measurement | Observed |
| --- | ---: |
| Source frames / pair slots | 1,800 / 7,200 |
| Exact input bytes | 7,297,323 |
| Output bytes / files | 13,482,599 / 23 |
| Largest HTML | 392,473 bytes |
| Report preparation | 0.235664 seconds |
| 1,000 changed detail selections | 8.206 ms total; zero new nodes |
| 1,000 repeated unchanged selections | zero text writes |

The separate maximum-width case retains all16 pairs per frame. A simultaneous1,800×16 fixture does not fit the producer's8-MiB content bound and is not represented as a supported maximum. Measurements are analytical Python/Node mock-DOM results, not browser responsiveness, RSS, source qualification or native performance.

The first two immutable checkpoints remain `.local/equipment-silhouette-report-checks-01.json` and `checks-02.json`. Final selected-only delta hashes, commands and reused check identities are in `.local/equipment-silhouette-report-checks-03.json`. Root's read-only review of the optional delta is clear: exact selection/aliases, decoded bounds, absent-field compatibility and excluded-label disclosure are retained. Product/test files are frozen for root's inspector runtime02 and saved-source acceptance. No lane job is active.

## Next safe action

Root's public wrapper is `equipment-evidence-report --run PATH_TO_SEALED_COMPOSITION --out NEW_REPORT_DIRECTORY [--manifest-sha256 EXACT_SEAL]`. It verifies the selected source metadata/frame members, not all ancestor media, and preserves generator/stylesheet custody. Root will run it once on the terminal composition; this lane has not read or reprocessed that source. Append the actual acceptance result and limitations separately—do not relabel the analytical checks as source or physical validation.
