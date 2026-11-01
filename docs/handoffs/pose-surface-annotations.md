# Source-pose surface annotations

## Raw-model contract refrozen — October 31, 2026, 6:46 PM PDT

This lane owns only `src/weft/pose_surface_annotations.py`,
`tests/test_pose_surface_annotations.py` and this note. The previous collection
API and camera workflow are published at
`25eac48fec09e7f078e6c8bfa0eadab12350dc0d`; root owns common CLI integration,
publication and native jobs. Root has authorized one bounded saved-metadata
pilot after runtime/plan freeze; no native/media/model/network/API work is
authorized. No live job or Git mutation is present.

## Approved hypothesis and boundary

Exact saved image landmarks can remove manual pixel copying while leaving every
physical assumption explicit. Join each supplied pose to canonical exercise
observations and its exact construction-authorized evidence record before using
coordinates. Retain every original frame/channel and confidence/visibility/
presence separately. Copy normalized original-image X/Y to pixels only; never
use model-world depth, infer equipment, replace a heel with an ankle, or infer
ground height/contact. The caller explicitly supplies a per-channel pixel range
and confidence threshold. Confidence is not a calibrated pixel-error bound.

Identity/crop-derived rows abstain in the first profile. An unmarked saved pose
is still only a supplied estimate: missing derivation markers do not authenticate
raw anatomy/identity. Exact raw/canonical pose pins, source identity, landmark
index policy, confidence threshold and authored uncertainty disclosure must
survive in the existing annotation document's bounded `provenance`, not just a
sidecar. Overlong provenance fails rather than dropping critical information.

Implemented bounds: eight observation documents /16 MiB, original pose ≤8 MiB,
1,800 original source frames, 900 supplied pose slots, 12 channels, annotation
document ≤4 MiB and complete cached results ≤8 MiB, ten-second cooperative
preparation. Cached queries do not revisit source payloads. Sixty-six distinct
analytical cases pass, including the actual released collector/envelope/body
source-join path; no new simulator, inference or body-geometry evaluation ran.

## Public interface and actual consumer join

```python
from weft.pose_surface_annotations import prepare_pose_surface_annotations
import json

# Use the exact same channel/entity selection in the existing collection request.
# Its point_role is the explicit landmark name, such as "left_ankle".
annotations = prepare_pose_surface_annotations(
    exact_observation_documents,
    pose_document=exact_saved_pose_bytes,
    source_sha256=original_source_sha256,
    channels=selected_channels,
    landmark_bindings=[{
        "channel_id": selected_channels[0]["id"],
        "landmark_name": "left_ankle",
        "uncertainty_px": explicitly_authored_pixel_radius,
    }],
    minimum_confidence=explicitly_authored_confidence_threshold,
    uncertainty_method=authored_range_method,
    uncertainty_note=authored_range_limitations,
)
request = json.loads(collection.request_document())
request["annotations"] = json.loads(annotations.annotation_document())
qualified = collection.qualify(json.dumps(request, allow_nan=False).encode())
print(qualified.describe()["required_explicit_inputs"])
```

This example does not fill the collector's other missing inputs. Camera,
surface-height ranges/static extents, explicit body endpoint correspondence,
body/world alignment and weight still need their own supplied evidence or
authored assumptions. No default pixel uncertainty or confidence threshold is
chosen by the adapter. `uncertainty_px` is an explicit 0–64-pixel half-width;
`minimum_confidence` is an explicit [0,1] inclusion policy. They are independent,
not probability-calibrated or inferred from one another. Each method/note is
bounded to 192 characters; complete embedded provenance must fit the existing
2,000-byte limit or preparation fails without dropping anything.

Existing channel `point_role` must equal its selected `landmark_name`. Supported
names are left/right shoulder, elbow, wrist, hip, knee, ankle, heel and foot_index
from the existing 33-image-landmark convention. Heel indices 29/30 are never
replaced by ankle indices 27/28. A wrist/ankle coordinate does not locate or
classify equipment or establish contact with a surface. Initial model left/right
assignment and physical anatomy remain unverified.

The output `source-surface-annotations-v1` is the existing schema: exact channel,
source/frame/observation/time identities; copied image pixel estimates; explicitly
authored pixel ranges. Critical provenance survives **inside that document**:
pose raw/canonical hashes, source SHA, landmark/index bindings, authored inclusion
threshold, per-channel uncertainty, method/note and no-contact/no-authenticated-
raw-identity disclosure. The collector preserves that string and annotation
content in its compiled, hash-pinned inputs. The diagnostic sidecar is not the
only location of these facts.

## Qualification, exclusions and honest unknowns

The existing world-evidence/source validators admit the original observation
documents. Each applicable exercise document must pin the supplied pose's exact
canonical identity. For a construction-authorized, visible source row, the
adapter checks the same entity/evidence/frame/time and exact saved pose-record
payload before reading its named image coordinate. Hashing the complete pose
document is a custody check, not target selection; withheld landmark values and
withheld retained-record payloads are never used as coordinates or confidence.
Only construction-authorized raw records are retained for payload checks.

Every original slot remains in cached diagnostics, including unsampled source
frames, metadata-only tails, excluded/low-confidence/unavailable/derived rows.
Nominal times and nullable original decoder timestamps remain separate. Image
dimensions must equal the original source; no implicit resizing, crop mapping or
camera correction occurs. Wrong canonical/payload/entity joins abstain. Wrong
source identity or malformed source schemas fail closed.

Eligible `tracked`/`tracked_partial` rows require an explicit exact integer
`pose_count: 1` and all 33 image
landmarks and finite X/Y, visibility and presence. The selected point must lie
inside normalized [0,1]². Low confidence, unknown confidence, ambiguity and
outside-image estimates retain unavailable diagnostics; construction rows get
`unknown` annotations with null pixels/ranges rather than filled geometry.
Missing, boolean or floating-point counts cannot establish one model hypothesis.
Even exact `pose_count: 1` is not an independently verified person count.
Original `frame_count` and `decoded_frames` must both be supplied exact integers;
container counts never substitute for missing decoded counts.
Excluded or source-unavailable rows get no usable annotation. Supplied range
choices remain retained in embedded provenance even where a pixel is unavailable.

Identity-derived and crop-derived rows always abstain in this first profile.
Their retained originals are not silently substituted, nor are changed arrays
called raw. Unmarked rows are labelled
`supplied_unmarked_image_estimate_not_authenticated_raw`: absence of a marker is
not proof of raw/anatomical identity. No source ancestry, model accuracy, camera
calibration, pixel-error coverage, equipment or physical contact is authenticated.

## Verification and exact pins

| File | SHA256 |
| --- | --- |
| `src/weft/pose_surface_annotations.py` | `a4d29c84f46f0be63566c32947ad4909a5ff2d54758b1ec75a9823cc408076e6` |
| `tests/test_pose_surface_annotations.py` | `058d700f53920687294aa53e8f6fa356921bd5c9cab1a5115a84be1da1c30b4a` |

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -o junit_family=legacy --junitxml=.local/pose-surface-annotations-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'withheld_and_unavailable or complete_large_input' -o junit_family=legacy --junitxml=.local/pose-surface-annotations-tests-02.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'withheld_and_unavailable or annotation_provenance_survives or critical_annotation_provenance' -o junit_family=legacy --junitxml=.local/pose-surface-annotations-tests-03.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/pose-surface-annotations-tests-01.xml` | 23 passed, 2 preserved fixture failures /0.595 s | `70fb344d3f692a4b8b017a6b90d322c58858c64d34e04b17d18f8c8a9b31022c` |
| `.local/pose-surface-annotations-tests-02.xml` | Only the two corrected cases passed /1.688 s | `4a5f480864a48d6a949429bb5f223cc48dfb7b45bc88558d0121b75e225c558e` |
| `.local/pose-surface-annotations-tests-03.xml` | Two changed/affected controls and one new provenance-bound case passed /0.067 s | `9b68427157769cf9c1836270d32f7db789c8fbf7ad49282d72ac68ee4aa45db3` |

Preserved fixture failures: a `not_detected` observation retained measurements
(corrected by clearing them), and the synthetic 1,800-frame observation document
was 8,420,405 B, above the unchanged 8,388,608 B per-file cap. Compact serialization
of the **identical synthetic JSON values** reduced it to 7,552,677 B with a new
explicit hash; all frames, landmarks and confidence survived. No source study,
threshold, cap or geometry was changed to obtain a favorable outcome. The bounds
failure occurred before source preparation, not during a pose/geometry result.

Measured large workload: 1,800 original frames, 900 pose slots, two channels and
3,600 retained channel slots; 1,800 copied estimates and 900 qualified record
payload checks. Exact inputs total 10,362,816 B. Preparation took **1.101569 s**;
1,000 cached row queries took **0.005642 s**. Row cache is 3,363,535 B and annotation
document 629,119 B. No body geometry, physics, model, media, network or subprocess
operation ran. This timing precedes only removal of an unused parsed-document
retention and withheld-record indexing; no output or numeric loop changed. It is
a bounded analytical measurement, not a hard RSS guarantee or video accuracy
benchmark. No unchanged passing full suite or earlier study was rerun.

The actual composition case uses copied pose coordinates → ordinary annotation
document → released collection → existing conditional envelopes → existing body
and independent image-source joins. Three explicitly authored body references
retain three conditional body terms, exact clocks and all six source slots.
No body geometry is evaluated and no contact or task success is asserted.

### Independent-review admission correction

Root identified that the original `.get("pose_count", 1) != 1` admitted missing
counts, `True` and `1.0`. Equality-only denominator checks similarly admitted
booleans/floats, and a missing decoded count defaulted to the container count.
The correction requires exact integer fields without changing known-one image
coordinates, thresholds, uncertainty, source partitions or geometry. Missing or
ambiguous multiplicity remains an unavailable row, not a filtered-out slot.

Eleven new cases cover missing/bool/integral-float/nonintegral/ambiguous pose
counts and missing/bool/float source denominators. Before the fix, eight failed
and three already rejected the input. After the fix, all eleven plus the existing
known-one pixel/slot control pass. Only these changed/affected nodes ran:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'pose_multiplicity or source_denominators' -o junit_family=legacy --junitxml=.local/pose-surface-count-guards-red-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'pose_multiplicity or source_denominators or exact_pixels_and_all_original_slots' -o junit_family=legacy --junitxml=.local/pose-surface-count-guards-green-01.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/pose-surface-count-guards-red-01.xml` | Eight expected failures /three passes, 0.119 s | `6d93c4e656770da8dbe3d15962224a5db71406c73f307947446dfa8fd8318682` |
| `.local/pose-surface-count-guards-green-01.xml` | Twelve passes, 0.070 s | `c82bc6475c3ce1c97251e2d70261519405030a6723be90906f27629bd7f612df` |

The previous 26 distinct passing cases remain retained; total distinct coverage
is 37, not 38. No source/native work, performance rerun or unrelated edit occurred.

## Handover / next safe action

These three paths are frozen and released for root's read-only review. No live
handles remain. Root may later route this optional seed path through the common
collection CLI; no common files were changed here. A real saved-source acceptance
requires a new exact-input/runtime plan before execution. Existing source failures,
ground/heel ambiguities, missing measurements and equipment-contact uncertainty
remain unchanged. This is less manual transcription, not automated calibration
or unfamiliar-exercise reconstruction completion.

## Saved-metadata pilot preparation — October 31, 2026, 6:36 PM PDT

Root authorized bounded metadata discovery only before approving a real saved
adapter invocation. Exact settings were fixed beforehand: original jogging
development source, left/right ankle only, confidence threshold 0.5, authored
±2-pixel range (not calibrated/measured), original start/all slots and canonical
partitions unchanged. No heels, surface heights, equipment or body/world mapping
are guessed. The new output `artifacts/pose-surface-source-01` has **not** been
created and the adapter has not run on this source.

Four named JSON reads verified candidate seal
`29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`,
copied original manifest
`325fd37a2a0d856949c4d63513769b318f2af79558d02a65123f6543d3201eed`,
pose and canonical exercise observations. An adaptive fifth read inspected only
the previously unrecorded identity-status metadata; total **6,573,449 B /0.052479 s**,
within 20 files /16 MiB /10 seconds. No landmark coordinate query, media decode,
adapter, model, native or paid call ran. The complete discovery helper is
`.local/discover_pose_surface_source_01.py`, SHA
`f1135e4f2b1ede555c0b236cfd08bb27b1772efc6a46a2c0cbb96dd565b4af39`.

| Member under `artifacts/camera-shared-admission-jog-01/candidate/` | Bytes | SHA256 |
| --- | ---: | --- |
| `evidence.json` | 1,903,436 | `375277287e836df1fe4e8930d6a5203857df349dbdbf8752065d14d7ab2a326b` |
| `evidence/5483442b26725f532aaa91fc2d46fa0464b1a6b5f1c5f5629d7ab4071bc997dd.json` | 2,732,749 | `5483442b26725f532aaa91fc2d46fa0464b1a6b5f1c5f5629d7ab4071bc997dd` |

Source SHA is
`b1b362fa7516879c8ae37c48c40972b0090680e6bdcb6933fcf9ccb8c2e2b0cd`.
Both decoded/container counts are explicit integer 240, dimensions 1920×1080,
FPS 30000/1001. There are 120 pose slots (112 integer-one and eight integer-zero
model hypotheses), 120 construction/120 unsampled observation slots. First
pose is frame0; last is frame238. Nominal and decoder clocks remain distinct.
The canonical observation provenance matches pose canonical SHA
`1b8d14413c25ee622f22c81557c5e69ffe195ddffeae85d2d03a9cbcdc25b824`.

**Representation compatibility must be settled before launch.** No row has
`identity_raw`, recomputation or crop markers, but all120 have `identity_status`:
112 are `single_model_hypothesis_not_verified_person_count`, eight are
`no_model_hypothesis`. Top identity metadata declares raw mode, zero repaired
frames and raw-artifact pin
`e98a2af4304fc0962362a3f73e40d3478c8d1ff5c24116120431bf726d380ea4`.
The current adapter classifies every nonempty identity-status string as derived;
therefore all112 visible rows would abstain before landmark copying. This is
identified before an unnecessary predictable zero-output study, not a failed
geometry result. No marker was stripped, count relaxed or source rewritten.

Discovery receipts:

- `.local/pose-surface-source-discovery-01.json`, SHA
  `0db9b219b76cdea9c1b59eb6f9d1436b673d856e86f5ac64cdcc87847be8a7aa`.
- `.local/pose-surface-source-identity-detail-01.json`, SHA
  `ebf73f243931576f180a60e4b1f90afebe3a06a70f2ae824454385d69db7f01d`.

The draft `.local/pose-surface-source-plan-01.json` records exact inputs/settings,
one adapter call, optional one collector draft only, 30-second outer/20-CPU-second
bounds, 16 MiB /40-file output and $0. It is explicitly pending the representation
decision and root approval. Runtime/runner have not been frozen or launched.
Next: root reviews whether declared raw-model uncertainty labels can be handled
distinctly from actual derived arrays without claiming authenticated identity;
only then prepare a reviewed final runtime/plan. All old failures remain intact.

## Representation correction and pilot authorization — October 31, 2026, 6:46 PM PDT

Producer inspection (`exercise_vision.summarize_landmarks`, `summarize_result`,
`resolve_pose_identity`) establishes that
`single_model_hypothesis_not_verified_person_count` is the normal raw-model
uncertainty label, not itself a derivation marker. The adapter now admits that
exact label only with top-level `pose_identity.mode: "raw"` and exact integer
`repaired_frames: 0`. It preserves the label as
`declared_raw_model_estimate_not_authenticated_identity`. This is consistency of
supplied declarations, not verification of the separate raw/audit files, person
identity or anatomy. The exact declaration's canonical hash and relevant
raw-mode/zero-repair fields survive in the ordinary annotation provenance.

Any presence of `identity_raw`, `identity_derivation_id` (including null), or
`identity_recompute_required` (including false), crop markers, unknown explicit
identity status, temporal/unknown mode or non-integer/nonzero repairs abstains.
Unmarked legacy rows remain unauthenticated, and a contradictory present top
identity declaration cannot be bypassed by omitting the row marker. Every prior
count, confidence, partition, custody, clock and byte/time guard remains unchanged.
The producer's zero-hypothesis/no-pose row remains source-unavailable without
reading its landmark payload. It is not reclassified as a usable human point.

Five targeted red cases failed before the correction, as expected. The green
run passed 29 new cases plus four affected controls; no unchanged full suite or
large-workload test was repeated. The new declared-raw case also passes through
the actual collector/envelope path, preserving the embedded provenance exactly.
The previous 37 distinct passes plus 29 new cases give 66 distinct passing cases.

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'known_raw_producer_label or unmarked_legacy_rows_do_not_override' -o junit_family=legacy --junitxml=.local/pose-surface-representation-red-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_pose_surface_annotations.py -k 'known_raw or raw_label or unmarked_legacy or declared_raw_label or raw_producer_no_hypothesis or exact_pixels_and_all_original_slots or derived_arrays_are_not_silently_original_pixels or withheld_and_unavailable_source or declared_raw]' -o junit_family=legacy --junitxml=.local/pose-surface-representation-green-01.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/pose-surface-representation-red-01.xml` | Five expected failures, 0.111 s | `5470ded116ff4d3f988bb8519a1db2a5f8750ea82a241aca0afd8b6a2266830a` |
| `.local/pose-surface-representation-green-01.xml` | 33 passed, 0.234 s | `bbd1dfa7df725498b5ced94a60be7c9faafe1a4fcc7199cdd73451eb4f1254cd` |

Root independently reviewed and accepted the frozen source/test hashes above.
Original plan01 remains unchanged at SHA
`acf6f3987bfc92abbfb391ad7f8235a7c6fcb0ed9badbb9a1228d44414d8f15f`.
A separate prepared draft `.local/pose-surface-source-plan-02.json`, SHA
`070d76023218465ef7a501fdf3a3e1e17b0ea4b761471b7331e07b1c4c0d73ce`,
records the new representation contract and fixes the already-permitted
collector draft count at one. Neither draft ran. Root authorized one launch
with the same source/settings/bounds after a new final plan pins the committed
runtime plus only this reviewed override and the worker/runner. No extra approval
is required unless inputs, settings or scope change. Next: freeze that runtime
and final plan03, run once, retain every unavailable result and seal the output.

### Frozen saved pilot — October 31, 2026, 6:49 PM PDT

Root accepted the correction and authorized exactly one metadata-only attempt.
The final plan is `.local/pose-surface-source-plan-03.json`, SHA
`838e8380e56cb0e4a0b69d08d67f5b71ea8f6763f95c05cda78c5829515b0da0`.
Both earlier drafts remain unchanged. Fixed input bytes, threshold 0.5, authored
±2 pixels, ankle names, original 240 slots/partitions and 30-second outer/
20-CPU-second/16-MiB/40-file limits are unchanged. One collector draft is included;
its qualification/geometry path is disabled and missing geometry stays unknown.

Runtime `.local/pose-surface-source-runtime-01.zip`, SHA
`338243b2b7df27c4c40c905aefb7c7388d28c3c9a59d88fbdb9f3a7b791b758d`,
contains 18 named files /323,187 expanded B /92,475 archive B: committed
`25eac48fec09e7f078e6c8bfa0eadab12350dc0d` plus only the reviewed adapter override.
No dirty shared or avatar files were copied. Worker/runner
`.local/pose_surface_source_01.py` is pinned at
`53bb519724ac38411eebd6c14afaa17901ce9fea95f6ff7a9e3c1eaca7e08d96`.
It blocks native/model/media/network/nested-process imports/actions and geometry/
fitting/collector qualification, enforces a 28-second worker deadline and retains
terminal failures. It tracks exclusive writes, scans aggregate output at most
once per second in the parent and performs one final member readback. The seal
proves member integrity; the separate process receipt adjudicates elapsed bounds.
RSS is measured, not hard-limited. Expected cost is $0, no provider request.

```sh
.venv-lab/bin/python .local/pose_surface_source_01.py 838e8380e56cb0e4a0b69d08d67f5b71ea8f6763f95c05cda78c5829515b0da0
```

No live handle exists at this pre-launch checkpoint. Run the command once; do
not tune or retry a failure. The output is new `artifacts/pose-surface-source-01`.


### Saved pilot completed — October 31, 2026, 6:49:47 PM PDT

The one authorized command exited 0 and the external process receipt passed.
All ten saved integrity checks passed. Original 240 frames /120 supplied pose
slots /480 ankle-channel slots and original first frame remain. Results:

| Channel outcome | Left ankle | Right ankle | Total |
| --- | ---: | ---: | ---: |
| Copied source image estimate | 99 | 101 | 200 |
| Below authored confidence threshold | 13 | 11 | 24 |
| Source entity unavailable | 8 | 8 | 16 |
| Source frame unsampled | 120 | 120 | 240 |
| All retained channel slots | 240 | 240 | 480 |

The ordinary annotation document contains 224 construction-authorized rows,
including the 24 low-confidence rows with null pixels/ranges. The 16 unavailable
and 240 unsampled slots remain in the complete diagnostic ledger; they are not
fabricated annotation targets. Exactly 112 source record payloads were checked,
once each despite two ankle channels. Both original-frame-zero ankle estimates
are available under the fixed 0.5 policy; their two-pixel half-width is authored,
not measured/calibrated. No heel was substituted and no marker/count was changed.

The actual collector draft preserves that same annotation JSON and embedded raw
identity declaration/provenance. It remains **unqualified**: camera and body are
absent; the one named surface retains unknown kind/height/extents/world frame.
No qualification, surface calculation, fit or simulator admission ran. Contact,
load, grasp and physical accuracy remain unknown. This is exposed jogging
**development** evidence, not an unfamiliar-video accuracy or holdout result.

One adapter call +one collector preparation; zero collector qualification,
forbidden-operation attempts, native geometry/physics, media decode, inference,
network or paid requests. Four named original metadata reads total 4,670,013 B.
Outer elapsed time **0.6812125 s**; adapter preparation **0.2709514 s**; all 240
cached row queries **0.0013941 s**; collector draft preparation **0.1689949 s**.
Peak child RSS **76,759,040 B** (measurement, no hard memory ceiling). Output is
14 artifact files /5,446,828 B, plus two empty process logs and the 1,109-byte
external receipt, within the aggregate 16-MiB/40-file bound. No input media,
original body, camera or unrelated artifact was read. Terminal output member
readback occurred once; root's independent saved-only audit remains separate.

| Evidence | SHA256 |
| --- | --- |
| `artifacts/pose-surface-source-01/manifest.json` | `233d97a44eefeb7b322e37dbc3b9393ce18e7b4056e98bf56e1236e20e90ebe1` |
| `artifacts/pose-surface-source-01/result.json` | `83ac51d8c71d26679a379f4e7fb9c1a9a16b7f7ef5ca0058a18c14758818d778` |
| `.local/pose-surface-source-process-01.json` | `8a84cc11281565b4b12729f63d48ae40f04326ebe7dd8da0020edcfba1cb125a` |

The source/test hashes above remain unchanged. All three owned product/test/doc
paths are frozen and released for root's review/publication; no live handles,
extra source calls or Git mutations. Earlier source failures and plan01/02 remain
intact. Root owns common CLI routing and independent saved-output review.

### Next independent source-side proposal (not implemented)

The pilot removes pixel transcription but cannot supply surface height, geometry,
contact or anatomical correspondence. One remaining reusable transcription gap
is the body-to-calibration-world transform already encoded in a qualified
`grounded-two-view-root-body-v1` camera body. The existing
`camera_shared_contract.build_camera_coupling` validates its fixed world-to-local
mapping; `camera_body_fit.validate_intervals` requires the exact inverse:
scale1, identity rotation, translation `[origin.x, origin.y, ground.height_m]`,
unchanged original clock. A thin **conditional binding builder** could expose
that existing mapping in the ordinary interval-body-binding schema instead of
asking the user to retype it. Exact body/camera pins and an explicit matching
calibration-world declaration would be required; unsupported/mismatched inputs
must stay unavailable. Camera scale and anatomy would remain supplied priors,
not measured units or contact evidence. It would not choose a height, endpoint,
first frame or fit result. Root must approve new ownership/interface before code
or a new saved-source study; none has started.
