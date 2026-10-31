# Calibrated two-view foot references

Updated 2026-10-31 13:56 PDT. Owner: `/root/exercise_reconstruction`.

## Scope and ownership — implemented, tested and saved study complete

Root approved an isolated pure adapter, focused tests and this handoff:
`src/weft/multiview_foot_reference.py`, `tests/test_multiview_foot_reference.py`,
`docs/handoffs/multiview-foot-reference.md`. No shared foot/body consumers,
simulator/task/lab/recording files, root guides, avatar work or Git state are owned
by this lane. The native feet acceptance belongs to root; this lane owns no native
job or live handle.

The saved Full two-view jogging predictions already contain all 33 landmarks.
Do not repeat inference or triangulation. The missing adapter is an explicitly
versioned, body-bound reference for calibrated ankle/heel/toe triangles instead
of the existing monocular model-depth foot reference. This is conditional source
geometry, not recovered soles, anatomical roll, contact, load or physical accuracy.
The source is exposed development evidence, not unfamiliar/final evaluation.

## Fixed implementation protocol

- Inputs are bounded exact JSON bytes: saved predictions, both construction pose
  documents/cameras, an optional original root-only camera-world body, its supplied
  ground and an explicit world→body `world-alignment-v1` document.
- Reuse original two-ray limits and view/camera validation, `multiview_body._points`
  / `_confidence`, the existing camera/root/ground binding validators,
  `world_alignment`, and unchanged `foot_reference.FOOT_CONTRACT` shape bounds and
  weighted-median primitive. Never create artificial monocular pose records.
- Only the first 60% of the original source clock may supply foot geometry or
  fixed shape. Retain every source slot and original endpoint status, including
  unsampled, unavailable, metadata-only and later-excluded slots. No interpolation,
  body/point changes, floor fitting, clipping or fallback alignment.
- Missing or incompatible source/world/body/clock binding yields explicit
  unavailable foot slots. A malformed source envelope/resource violation rejects.
  Supplied calibration/ground labels remain claims; the body's root scale does
  not independently calibrate its monocular shape.
- The new distinct profile is not installed by existing `foot_body` consumers.
  Root coordinates any later consumer extension after independent review.

## Bounds and verification plan

One trusted local pure call: ≤900 sampled poses per view, ≤1,800 original source
slots, 32 MiB input, 10-second cooperative compute deadline, 8 MiB output. A later
saved-only study additionally uses a 30-second outer limit. No hard in-process
memory/blocking-I/O ceiling is claimed. Projected provider spend is $0.

Focused analytical tests will block native physics/model imports, network and
subprocesses; cover known triangles/mapping, unchanged shape bounds, exact
source/clock/body pins, insufficient/missing geometry, excluded-payload nonuse and
resource/deadline guards. No existing passing suite needs repetition.

Before any saved-source study, freeze source/test hashes and send root exact input
member hashes and one run plan. Candidate inputs are the named construction-only
members of `artifacts/multiview-jog-full-two-view-01`, the two saved Full pose
documents, a compatible root-only body and supplied ground. No third camera,
provider reference-body data, source media, model bytes or held-out comparisons
may be read. The exact approved one-shot study is recorded below.

## Preserved failures and next action

Interval's raw-pose fixed-camera explanation remains rejected; its 73.57° foot
plane is not a ground plane. Existing single-leg floor failures and kettlebell
stability failures remain unchanged. Source-surface anchors published at `9abbdff`
resolve conditional coordinates only, with contact/load/grasp unknown.

## Implemented API and verification

```python
from pathlib import Path
from weft.multiview_foot_reference import build_multiview_foot_reference

read = lambda name: Path(name).read_bytes()
result = build_multiview_foot_reference(
    read("two-view-triangulation.json"), read("first-evidence.json"),
    read("second-evidence.json"),
    camera_documents=[read("first-camera.json"), read("second-camera.json")],
    body_document=read("human.json"), ground_document=read("ground.json"),
    world_to_body_document=read("world-to-body.json"),
)
```

The supplied alignment must name `camera-world:<canonical camera SHA256>` in
metres and `body:<canonical human SHA256>` in human-model units, both z-up. Its
explicit scale is 1, rotation is identity, translation is
`[-origin_world_m[0], -origin_world_m[1], -ground.height_m]`, and clock is unchanged.
This declares the existing body placement; it does not measure body scale.
Missing/incompatible inputs produce complete unavailable results. Malformed
source envelopes and resource violations reject rather than publish partial data.

Verified once, 2026-10-31 13:47 PDT:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_multiview_foot_reference.py --junitxml=.local/multiview-foot-reference-tests-01.xml
```

**58 passed, 0 failures/skips, 0.809 s JUnit duration.** Analytical inputs only;
imports of physics/models, network and subprocess execution were blocked. Checks
include exact source/camera/body/clock binding, negative ground clearance without
repair, unchanged shape gates, missing/occluded/unsampled and 1,002 original-frame
coverage, poisoned later geometry and monocular depth, byte pins, early oversized
row rejection and final cooperative deadline enforcement. No saved-source API,
native simulation, inference, media decode or repeated triangulation ran.

Frozen hashes:

| File | SHA256 |
| --- | --- |
| `src/weft/multiview_foot_reference.py` | `d866ddc23a512fdee45bdc3762104dc59339ab074fc977322c89a1a9cee50e33` |
| `tests/test_multiview_foot_reference.py` | `9984e87ad1051962fd3ccc88fad5a8ff655b1ed3c5618b742110b44802549d0c` |
| `.local/multiview-foot-reference-tests-01.xml` | `530eaf54a85041d4932ed6e1275eda0e3a76337bf816b23fbfcbe785c1c51127` |

A later focused self-review reproduced four unexpected `AttributeError` failures
for null/list optional body coordinate records. The fix checks those record types
before invoking existing validators; valid geometry, thresholds and defaults are
unchanged. Initial source/test hashes were
`6e7909a29f402204d889ae98a344ab99df49b9584ef2bc5a8f6e36bada0ee8c0` /
`9a47242b699dfa356e61404b17ca679e1f26bc5464be53b3a10ae6eb8d1c59d6`.
Preserved red receipt `.local/multiview-foot-reference-optional-binding-red-01.xml`
(4 failed, 1 passed), SHA `c42f681bcc72ca33beaafd0249607123b441573d1a7a926cfcb4fdde180c4826`.
The corrected focused run passed all 6 affected cases (five malformed bindings
plus the valid-coordinate control), without rerunning the other 57 unchanged
cases. Green receipt `.local/multiview-foot-reference-optional-binding-green-01.xml`,
SHA `23cc0c5cd75fd33298c194a8d9825f41806415dc376ca14272f120ff153e8cc6`.

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_multiview_foot_reference.py -k 'malformed_optional_body or known_mapping' --junitxml=.local/multiview-foot-reference-optional-binding-green-01.xml
```

## Frozen saved-input protocol — executed once below

Use the seven construction members below from
`artifacts/two-view-root-jog-lab-01/run/`, each verified against that run's existing
manifest. Their exact canonical prediction/view/body pins match. The earlier
`multiview-jog-full-two-view-01/predictions.json` does **not** match this body's
prediction identity: the primary camera's provenance wording differs. Do not
rewrite that hash, silently equate the documents, or repeat triangulation. The
body package already retains its own exact `two-view-triangulation.json`.

| Member | Exact SHA256 |
| --- | --- |
| `two-view-triangulation.json` | `0ed173522834b43cbd0af4c683e3bf424acb9d7f6418c463a31260856d5e93c9` |
| `evidence.json` | `375277287e836df1fe4e8930d6a5203857df349dbdbf8752065d14d7ab2a326b` |
| `inputs/two-view-root/evidence.json` | `f32d63c236f16353ea8e35fe4b3edce0c1607938037563cbf09cc19e1a457e75` |
| `camera.json` | `a74764a1143ed8d7af042e45d988404aaa445d5f5e02bdcc1c7b4a3cd6a6ea4e` |
| `inputs/two-view-root/camera.json` | `ea6aebc58b3d4cb29b55d73186856751c545988a34dd39aa7ae68c2c7a60529c` |
| `human.json` | `849ece268cc6b091cf85460f3e22ecee172aa5323c4db88dac3e221e367d0867` |
| `ground.json` | `90768933ec0fb8e51ecbc1075745532496d61f68e331b82a54e265231ca8d86e` |

The existing manifest SHA is
`325fd37a2a0d856949c4d63513769b318f2af79558d02a65123f6543d3201eed`.
Full byte/canonical pins and seven successful binding comparisons are saved in
`.local/multiview-foot-input-pins-01.json`, SHA
`ce37aa4092d9fc4121c28a706fd1c466dc94487946ca939f2b4e08c589f66ee6`.
The authored explicit mapping is `.local/multiview-foot-world-body-map-01.json`,
SHA `b646b8c982ac864e63bb038ea5384a622b537161a29238de22f895e4d2287b4c`.
Its spatial/clock claims are **assumed**, not observed. No third-camera payload,
provider reference-body payload, comparison scores or source media was read.

Approved single output: `artifacts/multiview-foot-reference-jog-01`, new and sealed;
one call, no retries/tuning. Eight exact inputs total 7,903,965 bytes, 240 original
slots, 120 sampled poses per view, fps 30000/1001. Construction cutoff is
`0.6 * 239 / (30000/1001)` (first 144 original slots, 72 sampled rows); all 96
later slots remain excluded without evaluating their foot/image coordinates.
Retain 480 foot slots / 1,440 endpoint slots. Existing 32 MiB input / 8 MiB output
and 10-second cooperative limits remain; approved outer wall bound is 30 seconds,
one worker, no inference/native/network/paid calls, no consumer/body installation.
Freeze helper/runtime pins and preserve terminal receipt, failures and all input
copies. Report conditional-reference availability and every rejection; no fixed
shape/floor/contact outcome is a success gate. Root reviewed and approved this
exact one-shot execution; no new user permission gate was introduced.

The frozen implementation and exact run proposal were reviewed by root before
execution. No repeated source run or threshold tuning is authorized by this result.

## Preparation record — zero calls at this point

Root approved the exact saved-only study after reviewing the final module and
binding guard. Preparation command (no geometry/API calls):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 .local/multiview_foot_study_01.py prepare
```

New study directory exists with 210,428 prepared bytes. Plan SHA
`136950519b3e62cc389aad319d9d3fcc0dd424215077506404f240127296c7e0`;
helper SHA `13f3f13957d15a251a1b0d46225e53d11df5999fbfeed0a379d65c66bb4a973c`.
`plan.json` contains all 15 frozen pure-runtime hashes, exact input pins and bounds.
The 8 MiB cap covers the entire new study artifact, not just the API result;
original sealed inputs are referenced by exact pins without copying 7.9 MB again.
The original manifest and explicit assumed map are copied into this output.
A separate stdlib launcher enforces a 30-second process wait; the one worker uses
20-second CPU / 20 ms sampled 512 MiB peak-RSS guards (not an allocation ceiling),
forbids child processes/network/model/physics/media and rejects artifact reads
outside the seven approved members/new study. It disables triangulation and camera
fitting entry points. The API remains bounded to 10 seconds and 32 MiB input.

The following terminal record supersedes this preparation state. No consumer or
native lane ownership changed.

## Terminal study and independent saved-output audit

Completed 2026-10-31 13:54 PDT; PID 2839 exited 0, no live handle. One call only,
no retry. Actual worker command was `.venv-lab/bin/python
.local/multiview_foot_study_01.py run`, launched by a stdlib `Popen`/`wait(timeout=30)`
outer process with `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. The worker itself cannot spawn children.

- API call **0.282418 s**; worker through sealing **0.339969 s**; outer **0.385745 s**.
- Peak RSS **67,977,216 bytes**. Saved artifact **22 members + manifest,
  584,589 bytes**. No native/model/media/network/provider calls or triangulation.
- Seal `artifacts/multiview-foot-reference-jog-01/manifest.json` SHA
  `48fffcb9b529730faac556f993eec8ecb8e4c28610dd0989ca1cfd3076c37ae0`.
- Process receipt `.local/multiview-foot-source-process-01.json` SHA
  `242196ad0642f39ec47cf9435b65aae59c8a0f899d651a11f1177aa332afe70e`; stdout/stderr retained alongside it (stderr empty).

All **240 source frames / 480 foot slots / 1,440 endpoint slots** remain. The
144-frame construction prefix contains 72 sampled poses / 144 foot opportunities:
**94 conditional references** (45 left, 49 right) and **50 foot slots with
unavailable triangulated endpoints**. The prefix also retains **144 unsampled
foot slots**. The 96 later frames contribute
**192 excluded foot slots**, with no foot/image coordinates evaluated. No missing
point was filled, no ground-height objective was used, and no outcome was tuned.

Prefix-fixed triangle parameters (human-model units under the declared mapping):

| Side | Heel-to-toe length | Ankle along axis | Ankle radial height |
| --- | ---: | ---: | ---: |
| Left | 0.2348976963 | 0.0511182055 | 0.0763678213 |
| Right | 0.2299567029 | 0.0563215213 | 0.0750455650 |

The 282 retained source endpoint clearances include **6 negative values**;
minimum **−0.1762053430 m**, maximum **0.4668889136 m** under the supplied
reconstructed plane. These remain signed and unclipped. They are conditional
source-coordinate discrepancies, **not simulated contact tests or measured
penetration**. This result does not establish floor/body admission or improve an
existing scorer. Missing pose geometry, calibration/scale/synchronization accuracy,
sole orientation and physical contact/load/grasp remain unresolved.

Independent stdlib audit (no API import, original-input read, inference or new
reconstruction) verified every member hash, frozen input/runtime pins, 144/96
partition, original frame/decoder clocks, complete status counts, excluded-payload
absence and explicit world→body coordinate arithmetic (max error 0). Receipt
`.local/multiview-foot-saved-audit-01.json` SHA
`9fc5e945ca52f5d5b699bc8b5c5a529bbb316e2bd0280e46c62c9f3339466023`,
**0.004216 s**. It is not independent camera/biomechanics/physical validation.

## Release and next action

All three owned product/test/handoff paths are frozen and released to root for
review/publication. Original source studies, the four reproduced malformed-input
failures and all other lanes remain untouched. No Git mutation was performed;
root owns publication and cleanup. No live process or pending test exists.

Concrete improvement: existing calibrated 3D foot evidence now has an exact,
source-clock/body-bound reference representation without inventing monocular
coordinates or fitting a floor. It is a new **conditional** profile, deliberately
not installed by existing `foot_body`/task/lab consumers. Next integration must
explicitly support this profile, preserve unavailable slots and negative source
clearances, and validate physical feet separately. An unfamiliar single exercise
video still needs qualified calibration/ground/support evidence; this development
clip cannot close that generalization or contact gap.
