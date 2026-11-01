# Exercise surface input collection

## Implemented and frozen — October 31, 2026, 6:14 PM PDT

Ownership: `src/weft/exercise_surface_inputs.py`,
`tests/test_exercise_surface_inputs.py`, and this note only. Root owns the frozen
camera fitter/workflow/report acceptance and publication. No live jobs or Git
mutations; no source media, model, native physics, network or paid work is planned.

The saved-input readiness review found no real complete surface-interval bundle.
Positive radial calibration now has an implemented enclosure; the new camera
fitter produces an uninstalled angle overlay. Neither supplies missing surface
height/extents, pixel uncertainty, endpoint correspondence, or body/world mapping.
The earlier jogging heel rejection, six negative gaps, Interval camera rejection,
and kettlebell contact/equipment failures remain unchanged.

## Approved implementation

Add a thin collection wrapper around existing observation, surface-interval,
annotation, body-binding and fit-config contracts. It retains exact input bytes
and every original frame/channel, exposes source-bound annotation templates only
for construction observations, and starts with null geometry/correspondence and
no static-equipment assumption. An edited request qualifies through existing
validators and emits files consumed by `body-reference-propose --surface-config`.
Only deterministic foreign keys and hashes are filled; no pixels, uncertainty,
heights, transforms, contact, load or grasp are inferred.

Limits reuse the existing fit caps: 16 MiB surface inputs, 4 MiB combined
camera/surface/annotation inputs, 64 KiB body binding/config, plus the optional
8 MiB original body. At most 1,800 source frames, 12 channels and 900 annotated
frames; bounded cached metadata/rows and a 20-second cooperative composition
deadline. This is trusted pure local preparation, not hard process isolation.

The wrapper and 30 distinct focused analytical checks are complete. No real-source
study ran; source-readiness discovery reused existing handoffs only. Root will
review and wire the common CLI separately. These three paths are released.

## Public API and collection workflow

```python
from hashlib import sha256
import json
from pathlib import Path
from weft.exercise_surface_inputs import prepare_exercise_surface_inputs
from weft.shared_world_packaging import write_package_files

# Exact bytes from an already qualified source; these names are illustrative.
observation_raw = Path("observations.json").read_bytes()
observation_sha = sha256(observation_raw).hexdigest()
source_sha = json.loads(observation_raw)["source"]["sha256"]
person = {"document_sha256": observation_sha, "object_id": selected_person_id}
collection = prepare_exercise_surface_inputs(
    {observation_sha: observation_raw}, source_sha256=source_sha,
    channels=[{"id": "ankle_surface", "entity_ref": person,
               "point_role": "Explicitly selected image endpoint", "surface_id": "surface-0"}],
    camera_document=exact_camera_bytes, body_document=exact_original_body_bytes,
    human_entity_ref=person,
)
Path("new-surface-request.json").write_bytes(collection.request_document())
template = collection.frame(original_frame)["channels"][0]["annotation_template"]
# A usable template has the exact ID/time/observation join, but null pixels/range.
# Fill only explicitly justified values in the existing documents in the request.
qualified = collection.qualify(Path("edited-surface-request.json").read_bytes())
print(qualified.describe()["required_explicit_inputs"])
out = Path("new-surface-input-package")
out.mkdir(exist_ok=False)
write_package_files(out, qualified.files())
```

No hardcoded person ID, image pixel, ground height, uncertainty, scale, transform,
static-equipment classification or authored weight is chosen by the API. The
caller selects existing channel/entity IDs; a missing original body or camera
may remain absent. Draft requests are `unqualified_editable_request`. Source
frame templates are supplied only for construction observations with detected
or partial visibility. Excluded, unsampled, unavailable and metadata-only slots
are retained without templates. Complete source-clock declarations are copied;
non-nominal clocks are never retimed to satisfy a supplied camera.

The editable wrapper contains the **existing** interval, annotation and body
binding schemas, plus an explicit weight and immutable original-input pins.
`qualify()` fills only a null calibration hash and null envelope-input hashes
from the actual bytes. Non-null conflicting pins are not repaired. The original
request bytes are retained alongside the compiled documents, so filled foreign
keys remain reviewable. Annotation rows need exact original-frame IDs/times;
their pixel/uncertainty values remain explicitly authored, not automatically
derived from model confidence or contact proximity.

Incomplete schema inputs return `incomplete_request`, named missing-field paths,
every original worksheet slot and **no** `surface-config.json`. Once the supplied
schemas are valid, existing envelope and source/body-join validators run once.
Unknown height, mapping, correspondence or annotations yield
`qualified_but_unavailable`, not a zero residual or fit-ready claim. When a
conditional body-channel term exists, status is `conditional_inputs_available`;
without an original body the same status can describe source envelopes only,
with `conditional_body_channel_terms: 0` and no body join. Every status keeps
`fit_or_simulator_admission: false`, zero body-geometry evaluations, unknown
contact/load/grasp and unverified physical accuracy.

A static support needs explicit `kind: static_support`, finite XY extents and
its authored provenance. Unknown extents block config compilation. Moving or
handheld declarations are rejected, never converted into ground. Height ranges
can remain unknown. Neither a height nor a support label establishes contact.

## Actual common-pipeline boundary

Qualified files use the unchanged `source-surface-fit-config-v1` contract and
relative fixed filenames. Both `parse_surface_fit_config` and
`validate_surface_fit_snapshot` validate the result. The real
`prepare_source_surface_envelopes` and `prepare_surface_interval_objective`
produce the retained geometry/join metadata; no substitute fitter is introduced.
`collection-frames.jsonl` contains every frame, source slot, unavailable envelope,
body join and independent image admission. The last identical request reuses one
bounded cached qualification. Queries return detached single-row JSON, not a
fresh parse of the whole source tree.

After root wires input collection, the generated `surface-config.json` feeds the
existing `body-reference-propose --surface-config` workflow with its separately
qualified original body source. It is **not** itself a body run. Camera mode still
needs explicit `--objective camera-surface-intervals --camera-image-scale` and
produces an uninstalled angle proposal. This milestone does not call that worker,
fit or install geometry, or relax any protected gate.

Returned files are a pure snapshot (at most 18 files under the current schema,
32 MiB); callers use a new directory and existing exclusive package writer. The
file-count guard permits 20 members before adding the manifest, not a general
20-file inclusive cap. The current fixed inventory is eight observations, four
optional documents, one config, one original body, three collection documents
and the manifest. `collection-manifest.json`
pins all returned members. It is not a fresh media hash, decoder proof, source
ancestry authentication or native runtime receipt. Invalid inputs/deadlines
raise without publishing a partial result; already supplied bytes and previous
completed qualification remain untouched. Limits are cooperative, not hard
process, RSS or CPU isolation.

## Verification and exact frozen pins

| File | SHA256 |
| --- | --- |
| `src/weft/exercise_surface_inputs.py` | `65660d206ac68106cba06575be12589d9a27b28c088e44a87e1a64122cce0d41` |
| `tests/test_exercise_surface_inputs.py` | `66bed1bfd2f5e5df7974a8ec5b60fafa9908d54060df2bc03d07fa86d1837c6e` |

Commands used existing `.venv-lab`, with native/model/media/network/process
operations blocked by the analytical fixture:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_exercise_surface_inputs.py -o junit_family=legacy --junitxml=.local/exercise-surface-inputs-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_exercise_surface_inputs.py -k 'excluded_unsampled or (wrong_identity and original)' -o junit_family=legacy --junitxml=.local/exercise-surface-inputs-tests-02.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_exercise_surface_inputs.py -k 'malformed_small or original_decoder or positive_radial' -o junit_family=legacy --junitxml=.local/exercise-surface-inputs-tests-03.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/exercise-surface-inputs-tests-01.xml` | 23 passed, 2 test-fixture failures; 0.503 s | `958e20a450fb648295dca9bef619451231a04ac58d0c4a03b3bc9378c9d4e0b8` |
| `.local/exercise-surface-inputs-tests-02.xml` | Only the 2 corrected cases passed; 0.048 s | `8b5379c2e2d4cfbacb7b19b95022af226fe78c13e4c2d6f763373e46384c77a2` |
| `.local/exercise-surface-inputs-tests-03.xml` | 5 additional bounded-schema/clock/radial checks passed; 0.071 s | `62024bc02e4dc1415b11acb18f006d6aa94f6882a1cc462d9c47d9df52537b10` |

The two preserved failures were an invalid fixture partition named `validation`
(corrected to existing `audit`) and a supposed changed source hash that was
identical to the fixture's original `a…a` hash (corrected to `b…b`). No product
gate was weakened. No unchanged passing suite or earlier study was rerun.

The first batch measured a 1,200-frame, one-channel analytical input with two
body references and two supplied annotations: preparation **0.097521 s**,
qualification **0.241923 s**, 1,000 cached queries **0.007438 s**; worksheet
1,241,756 B, qualified rows 3,258,198 B, 11 output files /4,085,888 B. This
measurement predates only the subsequent clock-declaration metadata and cheap
malformed-shape checks; no numeric path or full-frame loop changed. No hard RSS,
general responsiveness, unfamiliar-video accuracy or physical claim follows.

## Remaining work / handover

Root reviews these three frozen files, then routes the common CLI and chooses
any one-shot saved-metadata acceptance with its own exact input pins/bounds.
Do not run a new source study merely to obtain favorable ground/contact results.
The existing jogging camera metadata supports the radial domain, but its surface
ranges/correspondences/body-world evidence have not been supplied or recovered.
Human annotation/calibration collection is explicit assistance, not automated
unfamiliar-video reconstruction. All previous failed source alternatives remain.
