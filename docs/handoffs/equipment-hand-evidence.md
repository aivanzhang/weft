# Named hand-point equipment evidence

## Frozen producer — October 31, 2026, 11:41 PM PDT

Owner `/root/exercise_reconstruction`: new `equipment_hand_evidence.py`, new
focused tests and this note; narrow changes only in `pose_surface_annotations.py`
and `equipment_silhouette_evidence.py`. These three new paths were absent.
The producer and 27 distinct focused checks are complete. Root owns saved-input,
workflow and CLI integration; `/root/support_learning` owns the restrict-only
selector consumer; `/root/support_ui` owns the saved-result display. The preceding
two-view milestone is published at `1c18243d5b90132afbf6f95145193c6220f23b79`;
these five hand-evidence paths are excluded pending root review/publication.
No live handle, Git mutation, source pose/raster read, native/model/media job or
paid call. All previous source failures and released handoffs remain unchanged.

The exposed equipment-support run excluded both `sam-2`/right-wrist candidates
(frames20/22): each original seven-point window had seven wrist exclusions.
A wrist is not a hand-contact point. The saved33-point model has distinct
pinky/index/thumb slots, but the published annotation map did not expose17–22.
This is new source information, not permission to enlarge a wrist range or
reinterpret the old failure. Existing hand-point availability on that source is
unverified; no source pilot has run in this lane.

## Contract and implementation boundary

`prepare_equipment_hand_evidence(pose_annotations, *, hand_channel_ids,
equipment_document, mask_document, equipment_ids, raster_loader,
source_equivalence=None, construction_frames=None)` requires explicit named
channel maps for one or both hands. Each map contains `wrist`, `pinky`, `index`,
`thumb`, referring to the existing source annotation channels; all four have
their own exact model index, confidence, authored uncertainty and status.

New profile `equipment-hand-pointset-evidence-v1` retains every original source
slot and equipment×hand pair, with all four original point records. Every point
must qualify: any missing/occluded/excluded point makes that group unavailable.
Otherwise any rectangle–mask intersection gives conditional point-set image
consistency; four exclusions give point-set exclusion. No hull, dilation,
substitute endpoint or full hand surface is built. Neither result establishes
physical contact/noncontact, grasp, load, depth or identity.

The public wrist API/profile/output remain unchanged. A private parameterized
preparation path reuses the same canonical selection, source-equivalence,
observation/pose/mask identities, both clocks, half-open raster cells and exact
rational rectangle tests. One frame/object raster is shared by all named points.
The new group retains the original wrist point unchanged and exposes all named
point/source/clock fields for root's separate restrict-only window consumer.
That consumer may only remove/abstain original co-motion candidates; it must not
fabricate a v1 wrist object or promote a new candidate.

Preparation and grouping share one ten-second cooperative deadline. Existing
900-pose/1800-source-slot bounds and the mask producer's 500-frame bound remain;
the new complete retained cache is at most 8 MiB. Aggregate binary rasters are
bounded to 4 MiB, 512 loads and 100,000 foreground runs. The two temporary encoded
row caches are bounded to 16 MiB; intermediate point rows are released after
grouping. Cached queries do no pose/raster work. These are cooperative library
bounds, not hard RSS/process isolation.

### Stable serialized schema

`EquipmentHandEvidence` retains the original `PoseSurfaceAnnotations`, equipment
and mask bytes, real verified `SourceEquivalence` when needed, one metadata byte
document and a tuple of per-frame bytes. `describe()` and `frame(original_frame)`
return detached JSON; callers cannot mutate cached decisions.

- `hand_channel_ids` maps `left_hand` and/or `right_hand` to exactly `wrist`,
  `pinky`, `index`, `thumb`. Indices are left 15/17/19/21 and right 16/18/20/22.
  All selected channels are unique and belong to one explicit human entity.
- Metadata retains source/equipment/raw/canonical/proof/clock/selection pins,
  `hand_groups`, `point_channels`, `equipment_ids`, complete counts, work and
  bounds. Explicit canonical selection is preserved, not inferred anew.
- Each row keeps original source frame/status, pose evidence/time/decoder time,
  equipment source frame and mapping. `hands` are equipment-order × canonical
  left/right hand-order; each group retains four exact `points` in the order above.
- Point fields use `point_channel_id`, `point_name`, `landmark_index`, alongside
  original source observations/partitions, equipment/mask IDs and clocks, bounded
  model identity, pixel/confidence/representation, authored uncertainty, exact
  image relation and rational distance, raster pin/dimensions and image box.
- Each group has `hand_id`, original `wrist_name`, `human_entity_ref`,
  `point_counts`, `unavailable_points`, `status`, `reason`, `intersects` and the
  four unchanged point records. Status is `unavailable`,
  `conditional_pointset_image_intersection` or
  `conditional_pointset_image_exclusion`. Unavailability uses
  `required_hand_point_unavailable`; known groups have null reason.
- Contact, physical noncontact, grasp and load remain `unknown` everywhere;
  `body_or_floor_installed` is false. A different supplied depth does not become
  evidence of physical contact merely because the image relation is unchanged.

The raster callback receives `path`, `raster_sha256`, `dimensions: [h,w]`,
`maximum_bytes: h*w`, `remaining_total_bytes`, `deadline_monotonic`,
`maximum_seconds`, `mask_document_sha256`, `mask_evidence_id`. It must return the
exact binary 0/1 bytes, not PNG bytes. Existing validation rechecks length, hash,
binary values, foreground count and image box. A missing required retained raster
is a failure, never an empty mask. Per-point evidence survives a group-level
abstention, so a qualified finger can require a raster even if another point is
missing.

## Verification and preserved failures

Focused tests block native/media/model/network/process work and use authored
binary arrays. Latest deduplicated outcomes are **27 passed**, with no unchanged
suite reruns. Coverage includes strict four-point qualification, holes and
half-open boundaries, no filled hull, exact names and indices, foreign/duplicate
channels, withheld coordinate sentinels, one-hand selection, both derivative
clocks, missing proof, preinitialization/occlusion, full slot retention, combined
deadline/cache limits, detached queries and unchanged default bytes.

Two test-only expectation failures are preserved, not hidden:

1. Receipt 01 compared whole records after changing supplied pose depth. The
   original observation hash correctly changed; image geometry did not. The
   corrected test compares image fields and separately preserves the changed pin.
2. Receipt 02 expected a whole-call exception for one wrong original decoder
   time. The existing contract correctly abstains for that row and skips its
   raster. Receipt 03 verifies that exact unavailable outcome; no guard changed.

Commands (all from the repository root):

```sh
PYTHONPATH=src WEFT_HAND_EVIDENCE_FIXTURE_DIR=.local/equipment-hand-analytic-01 PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q tests/test_equipment_hand_evidence.py --junitxml=.local/equipment-hand-evidence-tests-01.xml -o junit_family=legacy
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q tests/test_equipment_hand_evidence.py -k 'published_default or identical_image or one_explicit or unqualified_equipment or verified_derivative' --junitxml=.local/equipment-hand-evidence-tests-02.xml -o junit_family=legacy
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q 'tests/test_equipment_hand_evidence.py::test_verified_derivative_retains_both_clocks_or_rejects_before_raster[pose_clock]' --junitxml=.local/equipment-hand-evidence-tests-03.xml -o junit_family=legacy
```

| Receipt | Raw outcome | SHA-256 |
| --- | --- | --- |
| `.local/equipment-hand-evidence-tests-01.xml` | 18 passed, one test-only failure; 0.938 s | `e7c257a70a0b0948e7659f2c1bc8d812f4b7b7287eca84b7980c4cec28414dd2` |
| `.local/equipment-hand-evidence-tests-02.xml` | Eight passed, one test-only failure; 0.235 s | `87e0d6bbcf582fb53406843a53702c0d85f0e3a48d7da14d42ae80904b4f751a` |
| `.local/equipment-hand-evidence-tests-03.xml` | Corrected clock case passed; 0.088 s | `c89b7c0d9f3b3a703b5a8c450a8a5c9fca00a975221aa996fbe368a16f85092f` |

The representative 181-slot/90-pose/eight-point fixture prepared in
0.270858875 s, loaded 90 shared rasters and retained 2,079,367 row bytes.
One thousand cached queries took 0.031511542 s. This is bounded analytical
performance evidence, not hard peak-memory or browser responsiveness proof.

### Exact default compatibility

Exact pre-change modules are retained at `.local/equipment-hand-default-runtime-01`:

- `pose_surface_annotations.py`: `49da6ee96c30c943c7db07b4450c3ed35298276654428330b7cf663a720167db`.
- `equipment_silhouette_evidence.py`: `a2f3665f1910e1cd17aaa166ef440f4c4a0a07e3c1087284e98346a4154fb556`.

Old and current analytical annotation/wrist bytes matched exactly. The capture
receipt is `.local/equipment-hand-default-golden-01.json`, SHA-256
`412f74f77131e64a035a808102342d3e31ee4f1ebd60363c35573d47be915236`.
Persistent tests use an eight-byte big-endian length prefix per byte chunk:

- Annotation document + metadata + rows: `5d4d93517a8546585faa6f6295b55d576a2e31eee8da53dbd9f11299abf30f88`.
- Wrist metadata + rows: `9958e2052cb9161baa6a2f601bf92f21e90a58a613272b53e293f79ecfea474d`.

### Frozen files and reusable analytical payload

| File | SHA-256 |
| --- | --- |
| `src/weft/equipment_hand_evidence.py` | `4e4e413acb4e6523e2b3564c2425aa8d7244cf236159680a89dd45ffa238573e` |
| `src/weft/equipment_silhouette_evidence.py` | `4319b66d666fa378dd7cf2d5ff4586e6d4b6dde11a51946b3a5b81ca2f667fdb` |
| `src/weft/pose_surface_annotations.py` | `1436bb3f2797aba6dada73a363790a611ee0a05262478d6861d65fde287f66eb` |
| `tests/test_equipment_hand_evidence.py` | `1e7bcebfba4605198911910e5824ba6e9fe730f369c4b6c0507411366ad26430` |

`.local/equipment-hand-analytic-01` preserves the existing successful fixture;
do not rerun the producer to recreate it. Its manifest SHA-256 is
`d0df3dcbeaae5b11f75836317fa55107c1f6364c6071786efe5183beb6f05272`.
`hand-metadata.json` is 6,713 bytes, SHA-256
`d4ff4d72ac819be5ba160fda9a22639e63d65b5868512c863eb53e62883ae8b4`;
`hand-frames.jsonl` is 76,633 bytes, SHA-256
`b0078c75c71267ebb476236c25641d529c9feacecdd175d59502ce62abf50f82`.
All seven original slots, three poses, 14 hand pairs and 56 point slots remain:
six conditional intersections, eight unavailable hand groups. Known wrists are
inside the mask hole and excluded while the three fingers intersect. All inputs
and binary arrays are analytical/authored, not authenticated source ancestry.

## Next integration boundary

Root is preparing an explicit `equipment-support --hand-points` route on the
existing sealed composition, using its verified precursor/raw/audit custody and
already-retained binary masks. Preserve the original wrist result, fixed
77-frame construction selection, all 259 source slots, same authored confidence
threshold/pixel uncertainty and real source-equivalence proof. No new PNG/video
decode or model call is necessary if every required binary raster is retained.

The separate consumer must compare exact raw/canonical pose, every named point,
source/equipment/clocks and every original candidate-window sample. It may only
keep, exclude or abstain an existing co-motion candidate; missing opposite-hand
evidence must abstain, not be synthesized. The producer alone does not install
this constraint, create physical geometry or resolve ground/contact accuracy.

Next safe action: review the consumer and saved loader, then root freezes a
separate bounded saved-input pilot. No source pilot, physical study or additional
data acquisition is authorized by this producer release. Critical pins and the
complete latest test-case outcomes are in
`.local/equipment-hand-evidence-release-01.json`; no live job remains.
