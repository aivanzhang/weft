# Restrict co-motion with named hand points

## Status and ownership

Implemented and frozen by `/root/support_learning`. Owned files:

- `src/weft/equipment_hand_constraint.py` — new explicit hand consumer.
- `src/weft/equipment_silhouette_constraint.py` — shared private custody,
  baseline and window-reduction helpers; wrist output bytes unchanged.
- `src/weft/exercise_support_evidence.py` — optional dispatch only.
- `tests/test_equipment_hand_constraint.py` and this handoff.

Reconstruction owns the prepared hand-point producer and annotation extension;
root owns saved snapshots, workflow, CLI, later bounded studies and Git;
support_ui owns reporting. No source, raster, native, model, fitting, media,
network or paid study ran here. Tests used authored analytical metadata and
in-memory binary fixtures under the existing blocked-operation fixture.
There is no live process. Do not repeat the passing checks below unchanged.

## Public boundary

```python
estimate_exercise_support(
    pose,
    camera_prior=explicit_fixed_camera_prior,
    equipment_observations=exact_equipment_bytes,
    equipment_ids=explicit_ids,
    equipment_source_equivalence=verified_proof_or_none,
    equipment_hand_points=prepared_hand_evidence,
)
```

`equipment_hand_points=None` leaves old behavior unchanged. Supplying both it
and `equipment_silhouette` fails before source preparation. A hand constraint
requires its exact equipment observation bytes; no implicit evidence lookup.
The direct helper is
`constrain_equipment_hand_support(baseline, prepared, *, deadline)`.

The accepted type is exactly `EquipmentHandEvidence`, profile
`equipment-hand-pointset-evidence-v1`. It is not cast to or disguised as wrist
evidence. One or both explicitly selected hands are accepted. An omitted hand
gets four unavailable, explicitly not-supplied records; its original candidates
cannot be kept by the missing evidence.

The top-level baseline profile remains `exercise-support-evidence-v1`.
All original source slots, supplied-pose partitions, ground fields, windows,
wrist pixels/confidence/boxes, identities and physical-unknown fields remain
exact. Only an existing `projected_co_motion_candidate` can change:

| Every original window sample | Active pair status |
| --- | --- |
| All four points qualify; at least one intersects at each sample | Original candidate kept |
| All four qualify, but at least one sample has all four disjoint | `image_hand_pointset_exclusion` |
| Any required point/sample is unavailable | `unavailable` |

Unavailable takes precedence over exclusion. No noncandidate is promoted.
This is named-point/image-mask consistency, not a filled hand, physical
contact/noncontact, grasp, depth, force, identity or support result. No scorer,
physics, body, floor, root trajectory or calibration is changed.

## Custody and selected-only access

Both consumers share the original bounded binding validation and window
reduction. The new helper supplies its explicit type, profile, named roles and
denominator; the old wrist wrapper still supplies its original contract.

- Original raw pose bytes must independently derive the baseline canonical
  pose hash; raw and canonical pins are not interchangeable.
- Equipment, mask, annotation and observation-document bytes retain their
  exact raw pins. A derivative source needs the actual verified
  `SourceEquivalence`, exact original/derivative IDs and complete recorded
  inspection; a dictionary resembling a proof is insufficient.
- Construction selection is exactly the baseline's first
  `floor(0.6 * supplied_pose_rows)` entries. Canonical selection disclosure,
  source aliases, frame counts and prepared-cache totals must agree.
- Only those selected cached rows and original selected pose landmarks are
  queried. Excluded cached rows remain opaque bytes in the length-prefixed
  custody digest. The full pose is parsed/canonical-hashed for identity only;
  excluded landmark payloads are not used for decisions.
- Each supplied point joins its exact channel, human/entity observation,
  equipment observation, partition, original frame and both clocks. Pixels,
  visibility, presence and confidence match the original named raw landmark.
  The wrist additionally matches the unchanged original co-motion baseline.
- Producer group/count/missing-point arithmetic must agree with all four
  saved point records. The consumer recomputes its own qualification and group
  decisions; an inconsistent saved group cannot conceal missing points.

Clock, partition or point-identity mismatch makes the affected point
unavailable. The original saved fields remain visible rather than being
rewritten into qualified values. Missing required slots or inconsistent
whole-document custody fail admission. No annotation qualifier, live precursor,
raster loader or geometry routine is called by this consumer.

## Exact saved schema

New top member `equipment_hand_constraint` has profile
`equipment-hand-window-constraint-v1`, status `applied_restrict_only`:

- `baseline_profile`, `baseline_canonical_sha256`,
  `baseline_equipment_status_counts`, `input_bindings`, `construction_frames`.
- `hand_rows` in original construction order. Each row preserves the raw
  prepared `source_frame`, `source_status`, `pose_evidence_id`, `pose_time_s`,
  `pose_decoder_timestamp_s`, `equipment_source_frame`, `source_mapping`, and
  `hands`. The corresponding original frame is `construction_frames[i]`;
  raw header mismatches remain inspectable and force unavailable points.
- `hands` uses baseline equipment order, then left/right. Every record has
  `object_id`, `hand_id`, `wrist_name`, `status`, `reason`, `point_counts`,
  `unavailable_points`, and four ordered `points`.
- Point order is wrist/pinky/index/thumb: left indices 15/17/19/21, right
  16/18/20/22. Supplied records retain every producer field and add only
  `producer_record_present: true`, `consumer_status`, `consumer_reason`.
  Qualified consumer values are `intersection` / `exclusion`; otherwise
  `unavailable`. Raw point names may be unavailable or mismatched; their fixed
  expected descriptor remains in `input_bindings.hand_groups`.
- Absent-hand points use the exact `_NULL_POINT_FIELDS` list in the module,
  plus explicit equipment/name/index, unknown contact, `status: unavailable`,
  `producer_record_present: false` and reason `hand_group_not_supplied`.
  These are missing-evidence slots, not fabricated producer records.

`input_bindings` retains source/equipment IDs, pose raw/canonical IDs,
equipment/mask/annotation raw hashes, observation raw hashes, actual proof
manifest pin or null, `hand_metadata_sha256`,
`hand_rows_length_prefixed_sha256`, its encoding description, and `hand_groups`.

Each original baseline equipment pair adds `co_motion_baseline_status` and
`hand_constraint` containing `status`, `reason`, `window_point_range` and
`window_hand_counts`. Noncandidates/later entries have null range/counts.
Removing these additions and the top extension, then restoring original
statuses/counts, reconstructs the baseline byte-for-byte.

Counts retain source frames, selected hand/point slots, all original pose-pair
slots, original candidates, kept/excluded/unavailable candidates, unchanged
noncandidates/later pairs, and hand/point status counters. `work` records selected
queries, full point/hand slots, original-window queries, serialized cache and
prefix-counter memory; raster and annotation queries are exactly zero.

## Bounds and verification

The original ten-second estimator deadline remains authoritative. The optional
consumer receives at most five seconds of that remaining time; a slow initial
binding/hash does not gain a fresh deadline. Existing prepared cache is at most
8 MiB, each row 128 KiB, additional serialized hand/prefix cache 8 MiB and the
complete final result 8 MiB. The old wrist consumer retains its separate 2 MiB
cache bound; historical hand outputs explicitly declaring 2 MiB remain valid
within that bound. Limits remain 540 construction poses, eight
equipment IDs, two hand slots and four named points per hand. Overflow fails;
no slot is dropped. Window prefix counters avoid repeated point scans.

Command run once:

```sh
WEFT_HAND_CONSTRAINT_FIXTURE_DIR=.local/equipment-hand-constraint-analytic-01 \
  .venv-lab/bin/python -B -m pytest -q \
  tests/test_equipment_hand_constraint.py tests/test_equipment_silhouette_constraint.py \
  --junitxml=.local/equipment-hand-constraint-tests-01.xml
```

The initial **18 new + 34 affected wrist cases passed** in 4.158 seconds. Coverage:
actual selector/producer composition, wrist exclusion versus finger intersection,
all-disjoint and missing-point windows, absent hand, negative pixel/confidence/
name/clock/partition custody, complete grouping, excluded payload nonaccess,
raw/canonical re-pinning, real analytical derivative proof, simultaneous-option
preflight, deadlines/cache failure, and exact published default/wrist goldens.

The bounded 90-selected-frame / 720-point consumer case took
0.13634874997660518 seconds, used 1,242,429 cache bytes and emitted 1,669,509
bytes. This is an analytical consumer measurement, not full-video pipeline or
native performance. No new optimizer or source science was timed.

Actual small output is preserved at
`.local/equipment-hand-constraint-analytic-01`: 31 source slots, 15 original
pose rows, nine construction rows, 18 selected hand slots and 72 point slots.
Its `support-evidence.json` SHA256 is
`ffc9250f664f9a8ff8e6abcbda920c5025d469b0ba6a133fc1846300fa0bf727`;
fixture manifest is
`e4b9c30b32438e0997c334c9d1614c068a9cebe1f29a143db9e8e1b0b37cc5f0`.
UI may consume this saved output without rerunning the producer or selector.

| Frozen file | SHA256 |
| --- | --- |
| `equipment_hand_constraint.py` | `e9c9091c2c7765a86961e5114e4ef985d7a611ea730e2bdaeef49ec546901ce0` |
| `equipment_silhouette_constraint.py` | `c76adae338b882525217255b56bd14dfbc5e1b04653d7830c6ea5eed71bc7cb5` |
| `exercise_support_evidence.py` | `b098637e38d1294f212ff66a85dd5dd793f85d2b62c42656741c5acf4ab202a3` |
| `test_equipment_hand_constraint.py` | `e0b1a5add9aa438f90c5b932e2828bfdb387f0b363f3aa2e4682023af74a218d` |
| Focused JUnit receipt | `005e0e3403e3fe0b938708db75279ec35446d0bb4466f3ccf84a13ee2279b847` |

Independent review found one selected-payload ordering defect: the first
implementation queried a raw landmark before rejecting its already-unavailable
cached point. The narrow fix moves coordinate/visibility/presence access after
cheap source, clock, partition, known-status, origin and mask-relation checks.
It changes no schema or saved point values. A new sentinel fails on the old
ordering and proves the unavailable `(frame 8, landmark 17)` is not queried;
all 71 remaining qualified fixture points are still checked against raw data.

Preserved red receipt:
`.local/equipment-hand-constraint-order-red-01.xml`, SHA256
`b8846588b704c8e99a46f73471521945064eceeebd1b9333a923e09cf188c09c`.
Only that new case and two affected qualified pixel/confidence cases reran:
three passed in 0.357 seconds, receipt
`.local/equipment-hand-constraint-order-green-01.xml`, SHA256
`f3095cbd16ed7ff300c13b5210403adf0c70870024720a88d26db9af00482b8e`.
There are **53 distinct cases with passing evidence**: 19 new and 34 affected
wrist cases, not 55 independent tests. The previously saved analytical payload
and performance evidence remain unchanged; no source/producer study was rerun.
Reconstruction's independent read-only rereview closed the ordering finding
at the preceding `7b718344…` consumer, without a test rerun.

## Source pilot failure and bounded capacity correction

Root's one public source pilot on October 31, 2026 at 11:51 p.m. PDT stopped
after 2.453572 seconds with `complete point/prefix cache exceeds bound; no slots
dropped`. It produced no output package. Preserve
`.local/equipment-hand-acceptance-01/process.json`, SHA256
`2c4a981354fa7f4a81e27694d345c3442b4155ed621791b1ce001787ca1b7c30`,
and its plan/logs. This failed acceptance is not relabeled successful.

The failing counter includes only full hand rows and small prefix arrays,
not duplicate baseline/wrist rows. The consumer had imported the wrist-only
2 MiB limit despite retaining four point records per hand. The only product
correction declares a separate hand-only 8 MiB cache bound, matching the
unchanged complete-output ceiling. No compression, field deletion, geometry,
confidence threshold, window, decision or physical claim changes.

An authored 259-source-slot fixture uses 258 supplied pose rows, 154 selected
rows and one object: 1,232 complete point records, matching the pilot's point
cardinality but **not** its two-object/129-pose source data. It reproduces the
old cache failure and now retains all producer fields and the exact recoverable
baseline. Measured consumer work: 2,125,538 cache bytes (3,720 prefix bytes),
2,852,522 output bytes, 0.28560595819726586 seconds. These are analytical
measurements, not a successful source run.

Preserved evidence:

- `.local/equipment-hand-capacity-red-01.xml`: initial test fixture exceeded
  the existing 500-frame mask limit; no product correction was inferred from it.
- `.local/equipment-hand-capacity-red-02.xml`: corrected authored fixture
  reproduces the actual 2 MiB guard, SHA256
  `916413e8d683e271897859dfedfb1bd12b3b47fbd8e814cdc584a2420b04c5e4`.
- `.local/equipment-hand-capacity-green-01.xml`: only the new capacity case,
  retained deadline/overflow rejection and old default/wrist byte-golden case
  ran; three passed in 1.915 seconds, SHA256
  `16c86aca42ab2eb8cb90ea71081be4163ab2a0a19ac41228c833f4082a81dec8`.
- `.local/equipment-hand-capacity-measurement-01.json`: exact measurement,
  SHA256 `4425886b7c9c92045f3c406eb52a65e98e2e1aee71ae2c5097219998107cb72a`.

There are now **54 distinct cases with passing evidence**, not three new
independent behaviors from the targeted rerun. No source evidence was read or
processed for this diagnosis/correction. UI owns strict admission of the
explicit historical/current hand cache bounds; old wrist remains fixed at
2 MiB. Root owns any separately reviewed changed-runtime source invocation,
intermediate preservation, terminal saved audit and Git checkpoint. Do not
rerun the failed source input with the unchanged implementation.

Reconstruction's independent read-only capacity review is clear at consumer
`e9c9091c…` / tests `e0b1a5ad…`: the hand-only constant, exact retained records,
baseline recovery, unchanged wrist/final-output limits and active overflow
rejection are coherent. No test or source work was repeated for that review.

## Changed-runtime source pilot and saved audit

Root's separately pinned pilot02 completed once in 2.480205 seconds. Its
output `artifacts/equipment-hand-support-02` contains 254 files / 35,247,312
bytes, seal
`a607f6df387c59ae4a833718d783d1a5fc9493d8f2b71847a82cdb2cd17e6c38`.
Process receipt `.local/equipment-hand-acceptance-02/process.json` has SHA256
`500629c0bc4140f5f288620884854d892000420862d53313d7c570492c51d91a`.
Pilot01 remains failed and immutable; this is changed-runtime evidence.

The independent saved-only audit passed:
`.local/equipment-hand-support-saved-audit-02.json`, SHA256
`52d4f8a6e39b126dfa262927ce38d55aaec880967b2c6a3556b58a180a046e00`.
It hashed every output file once, joined all 188 copied source files to the
original seal and plan, and all 47 generator files to the frozen runtime.
Including three external metadata files and one prior audit receipt, reads
totaled 258 files / 35,335,088 bytes; active checks took 0.211121 seconds.
Review idle time is separately reported. No source, producer, selector, proof
loader, raster, media, model, physics or API operation ran in this audit; its
process is terminal. The optional duplicate hand checkpoint was not read.

Verified source coverage remains 259 original slots, 257 decoded slots,
129 supplied poses, the exact 77-pose construction prefix, 52 excluded poses
and 516 retained equipment pairs. Producer records preserve 1,036 hand groups
and 4,144 points. Construction records preserve 308 groups / 1,232 points:
89 group intersections, 109 exclusions and 110 unavailable groups; point totals
are 274 intersections, 540 exclusions and 418 unavailable.

The original baseline is byte-for-byte recoverable and canonically identical
to the independently audited wrist-only result:
`08fdbdf4bc2e277ef552ce087e675113207fcb65cb86dcc2d4df56d8df865459`.
Only its two existing `sam-2` / right-wrist candidates occur, at frames 20 and
22. Their unchanged windows cover frames 14–26 and 16–28 respectively; each
contains seven qualified hand point-set intersections and no exclusions or
unavailable group. Both candidates are kept; none is created. Every raw
producer point field, source ID and clock survives. Contact, noncontact,
grasp, load and physical support remain unknown; no body or floor is installed.
This exposed development clip does not establish unfamiliar-video accuracy.

The audit's first count check incorrectly grouped unavailable producer points
under the generic status instead of their saved reason labels. Preserve
`.local/equipment-hand-support-saved-audit-failure-01.json`, SHA256
`f6c9b97bd9dcd85bc50af692559e8a935f8d4224c152fce2a9ec603c9623fe02`.
Only that cached counting assumption was corrected; successful checks and
artifact reads were not repeated. Independent UI audit
`.local/equipment-hand-support-saved-ui-audit-02.json` separately passed exact
page projections, windows and scripted navigation; those checks were not
duplicated here. Root owns publication and the next project milestone.
