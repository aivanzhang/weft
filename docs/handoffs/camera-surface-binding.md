# Camera-body surface binding

## Frozen for review — October 31, 2026, 07:03:26 PM PDT

Exclusive new ownership: `src/weft/camera_surface_binding.py`,
`tests/test_camera_surface_binding.py`, and this note. All three paths were absent
before this lane. Root owns publication, CLI, collector/fitter/report, scorer,
lab and the released pose adapter. No shared file, Git, source study or native
job is part of this implementation. No live handles. The source annotation pilot
is terminal; do not rerun it.

The implemented API turns an **existing exact camera-body coupling** into the
existing interval body-binding schema, not recover a new coordinate system.
Inputs: prepared source envelopes, exact body/camera bytes, explicit existing
`world-alignment-v1` declaration from that raw camera's calibration world to the
same envelope world, caller-selected human entity and endpoint correspondences.
No world ID, person, endpoint, height, contact or calibration is guessed.

Only explicit identity calibration-to-surface spatial and original-clock
alignment is supported. Compose it with the existing coupling inverse: scale1,
identity rotation, `[origin.x, origin.y, supplied_ground.height_m]`. The original
body/camera/root/ground/source clocks remain byte-identical; camera metre scale
and anatomical shape remain priors, never independently measured anatomy.
The body period remains distinct from the original fractional source period.

Reuse `build_camera_coupling`, existing camera/world-alignment validators and
`prepare_surface_interval_objective`. Retain one prepared objective for actual
consumer use and complete cached source-slot inspection; do not evaluate geometry,
fit or install a candidate. Unknown/mismatched declarations remain unavailable;
malformed existing schemas fail. Explicit unknown correspondence rows are not
filled. Missing heights do not erase otherwise qualified image-source custody.

Bounds: body8 MiB, camera4 MiB, declaration16 KiB, binding64 KiB,1800 original
source frames/900 retained references/12 channels,10-second cooperative adapter
budget. The delegated validators retain their existing smaller bounds. No hard
process or memory isolation is claimed. Tests use analytical inputs, deny native,
media/model/network/process work, exercise the real interval/camera-fit contract
without evaluation, and measure bounded preparation/cached queries.

All three paths are frozen for root review. No source study or native call ran.
A follow-up must prioritize independent source evidence for ground/equipment,
not another packaging-only adapter; no new source read is authorized here.


## Public API and existing consumer use

```python
from weft.camera_surface_binding import prepare_camera_surface_binding

prepared = prepare_camera_surface_binding(
    prepared_envelopes,
    body_document=exact_original_body_bytes,
    camera_document=exact_same_camera_bytes_used_by_envelopes,
    calibration_world_declaration=explicit_world_alignment_bytes,
    human_entity_ref=explicit_observation_document_and_object_id,
    channel_correspondences=explicit_existing_correspondence_rows,
)
print(prepared.describe())
# Existing schema; can replace the same collector request body_binding field.
existing_binding_bytes = prepared.binding_document()
# Existing interval consumer, prepared once, without any evaluate()/fit call.
existing_objective = prepared.objective
original_source_slot = prepared.frame(0)
```

Each input keyword is required; explicitly passing null keeps the missing input
unavailable. No camera is silently taken from a different body or canonically
similar file. The supplied camera's raw bytes must match the prepared envelope's
retained camera and raw pin. Its canonical pin must match the camera body's
original pin; original source, full frame count, fps and each retained camera/body
timestamp must agree. The unchanged coupling validator checks original body,
root, shape, ground and source-clock constraints. Raw file hashes and canonical
nested hashes are named separately, never confused with authenticated history.

The explicit declaration uses the existing `world-alignment-v1` schema:
`from_frame.id = calibration-world:<exact camera raw SHA>`, metre prior and
`x_right_y_away_z_up`; `to_frame` must equal the envelope's supplied world.
Only exact scale1/identity rotation/zero translation and original-clock identity
(rate1, both declared origins0) are supported. Unknown declarations remain null;
known nonidentity frames, scale, rotation or clock return unavailable. Malformed
schema raises its existing validation error. This first conditional profile
accepts assumed/unknown claims without measurement refs; it cannot upgrade
supplied declarations to observed calibration. No target-world ID is generated.

Caller endpoint rows are the existing `channel_id`, `body_point`, `status`,
`method`, `uncertainty` schema. Every existing channel must be explicitly listed;
an unknown row uses null `body_point` and `status: unknown`. No landmark name,
heel/ankle substitution or human entity is inferred. Source evidence admission
remains in the existing objective's source/entity/evidence/frame/clock checks.

The generated body mapping's claims embed the original coupling and explicit
world-declaration hashes, keeping their conditional provenance in the actual
binding bytes. The result also retains exact input bytes, original clock/coupling
pins and source counts. It caches only its small report/binding and reuses the
existing prepared objective and envelope objects. `frame()` returns detached
source-envelope/body-join values for every original slot; it does not recompute
source qualification or geometry.

`conditional_binding_available` means the fixed coordinate binding is usable,
**not** that any interval is available or that a fit/body/physical task is admitted.
For example, unknown height can produce an available mapping with
`objective_status: unavailable`. The exact image-source admission stays available
independently where source custody qualifies. Missing camera/world mapping never
fills a zero-height surface. Contact/load/grasp, physical accuracy and anatomy
remain unknown. No CLI/common consumer route was changed by this lane.

## Verification and resource evidence

Thirty distinct analytical cases pass. The initial run passed28/29; its sole
failure expected the wrong unavailable reason for an unsampled slot. The actual
consumer correctly retained `source_not_sampled`; only the assertion changed.
The follow-up ran that corrected case, one strengthened exact-clock test, and one
new nonzero-ground case. All three passed. No unchanged full suite reran.

The actual existing interval objective and camera-fit contract accept the
constructed binding with geometry/evaluation methods denied. Fractional source
fps30000/1001 gives source period1001/30000 while body dt remains1/30. A separate
nonzero authored ground fixture retains height0.25 and translation[1,2,0.25],
without changing the supplied body, camera or floor. The exact-clock test first
repins its deliberately changed analytical camera ancestry, then proves that a
1e-9 timestamp discrepancy fails this exact join even though the camera's more
general tolerance admits it. Tests also cover missing/unknown declarations,
foreign entity, raw-versus-canonical custody, unsupported endpoint, no rotation/
scale/clock changes, measured-claim rejection, complete source slots, cheap byte/
frame limits, timeout and detached cached queries. No actual-source artifact was
read, and no objective evaluation, inference, native, media or network call ran.

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_surface_binding.py -o junit_family=legacy --junitxml=.local/camera-surface-binding-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_surface_binding.py -k 'exact_binding_reuses or supplied_nonzero_ground or original_camera_clock_change' -o junit_family=legacy --junitxml=.local/camera-surface-binding-tests-02.xml
```

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `src/weft/camera_surface_binding.py` | 14,668 | `dddc42328291d9122038e79809e28dfedb37fc5b2105dd798ed78f26656d1706` |
| `tests/test_camera_surface_binding.py` | 14,520 | `e6c8373c7662da79c3e11bb56de8612541844a963f6730633197ad9eacc891cd` |
| `.local/camera-surface-binding-tests-01.xml` | 10,119 | `f10b75cc162b8b51151e22a3a2b2c5e3e7063a3e57a77834f2b80e4837d1224e` |
| `.local/camera-surface-binding-tests-02.xml` | 873 | `b39f29e07068cb59b5252957bad11cd3c68a43756a35de2fb679ff1d7f3b7169` |

Receipt01 elapsed0.296s; receipt02 elapsed0.078s. The measured analytical workload
has10 source frames/3 body references/1 channel, body6,361B/camera1,688B:
preparation0.0022018s;1000 cached frame queries0.0060443s. Encoded retained
components: shared envelopes13,874B, objective join5,674B, image admission6,877B,
objective metadata5,923B and binding2,229B. The adapter report is separately
capped256KiB; exact original inputs and existing caches are retained once, not
copied per query. Existing objective caps remain8MiB ordinary cache +2MiB image
admission, and shared envelopes8MiB. This small analytical timing does not claim
maximum-size or sustained UI performance; no hard RSS/process ceiling is claimed.

All source/test pins are frozen. Root's preceding14-path milestone was published
at `9cbc2836c5fef361b378e932e4c0e8b57b42b26d`; these three new files are intentionally
excluded. Root reported the prior source/CLI saved audit passed at
`.local/surface-release-saved-audit-01.json`, SHA
`0139d2ee047c4a77bf2e0f76cd117c57399fe882c61079bc7033afbabd62155c`.
No Git mutation or scorer/shared-path edit occurred here; no live handles.

## Next evidence-based improvement, not started

Read-only code/handoff inspection identifies **already saved equipment
silhouettes**, a stronger spatial constraint than the previous bounding-box
co-motion screen. `scripts/studies/export_kettlebell_shared_evidence.py` records
`artifacts/kettlebell-shared-evidence-01/masks/mask-evidence.json` and per-row
`raster_path`/raw raster hash/dimensions/half-open pixel-cell mapping. The recorded
study retained771 slots,24 pre-initialization slots and747 verified PNG masks;
its original root seal is
`10b8067dbaada2046a5508da45c0a6d07c3a7dfdd8002b6f10e669ac59f5be24`.
This is a handoff/code fact, **not a newly verified artifact read** in this lane.

Existing exact pose/equipment/source-equivalence pins and the two previously
exposed co-motion windows are in `exercise-reconstruction.md`. The old result
has only two right-wrist/working-kettlebell candidates at original frames20/22;
a zero bounding-box gap cannot establish that a point is in the foreground
silhouette, much less a grasp. Human wrist estimates are not hand-contact points.

Next proposed step is bounded named-metadata discovery (no PNG/media decode yet)
to verify the mask manifest, exact mask-evidence/observation joins and the same
original/derivative clock proof. If compatible, a pure, uncertainty-aware
**image-silhouette exclusion** method could check an explicitly authored wrist
pixel rectangle against exact saved foreground pixel cells. It must retain every
missing/occluded/excluded slot, all original IDs and both clocks, and label its
result only conditional image consistency/exclusion. It cannot prove physical
contact/noncontact, load or grasp; mask/pose accuracy and occluded surfaces remain
unknown. Unlike another box-overlap wrapper, this would use actual retained shape
and expose box-only false support. Any raster read or saved study needs a new
exact pinned plan and root approval. No artifact was reread, mask decoded,
threshold changed, or new source experiment launched here.
