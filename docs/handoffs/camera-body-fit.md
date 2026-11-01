# Fixed-camera body angle proposals

## Implementation frozen — October 31, 2026, 5:51:34 PM PDT

Owner `/root/exercise_reconstruction`. Root approved only new
`src/weft/camera_body_fit.py`, `tests/test_camera_body_fit.py`, this handoff and
opt-in `camera_surface_interval_v1` changes inside
`src/weft/whole_body_reference_refinement.py`. Root additionally authorized the
opt-in image-admission cache in `src/weft/source_surface_interval_construction.py`;
default serialized outputs remain identical. Released radial files stay frozen.
Root owns CLI/pipeline/worker/report integration, native/source studies and Git.
No source/media reads, native physics, inference, network/provider calls or Git
mutations in this lane. No live handles.

## Boundary and reason

This is a construction **angle overlay, not a runnable/admitted updated body**.
The existing camera body pins its original angular reference twice:
`camera_reference.angular_reference_sha256` and the two-view root's
`unchanged_shape_sha256`. Mutating its angles while keeping those pins would
misrepresent provenance. The new mode retains the original body and supplies
proposed angles separately; it does not invent another admitted camera method.

The current fitter's orthographic pelvis-relative x/-z projection, torso pixel
scale and monocular root regeneration cannot simply be reused for camera bodies.
Initial scope is an original grounded `two-view-root-body-v1`, with exact
body/evidence/camera bytes and an explicitly prepared surface-interval objective.
No other camera profile, stacked correction or missing correspondence is inferred.

## Callable contract chosen before implementation

Extend `propose_whole_body_reference_refinement(...)` with explicit
`proposal_objective='camera_surface_interval_v1'`, `camera_document` exact bytes
and `camera_image_scale` (positive finite authored loss units per pixel), plus
the existing surface objective/weight/work settings. Camera arguments are
rejected outside this mode. The camera scale is numerical weighting, never a
measured metre conversion; the fixed 0.05 Huber threshold is in those declared
loss units. Default modes and outputs stay unchanged.

`camera_image_scale` is constrained to **[1e-6, 1]**. Image nonregression uses
unscaled pixels and a fixed 1e-8-pixel tolerance, independent of that authored
loss weight. Neither small weights nor missing intervals weaken this gate.

New report method: `blocked-camera-body-surface-interval-reference-v1`.
New result: `human_candidate=None`, separate `angle_proposal` under
`camera-body-angle-proposal-v1`, and `report`. The overlay contains the complete
original-time `reference_candidate`, exact input/coupling/root/ground/angular
pins, and `installation='unsupported_angle_overlay_not_a_human_spec'`.
No generic consumer may install it as a human model.

Reuse `camera_shared_contract.build_camera_coupling`, existing camera/body
validators and `camera_reference.project_world`. Native geometry, when later
separately authorized, evaluates trial q on the unchanged original model.
Restore absolute local endpoints and derivatives as relative + pelvis; then
`world = local + [origin_world_m.x, origin_world_m.y, ground.height_m]`, scale one.
Use supplied K/R/t/k without rounding correction, transpose-as-inverse,
recalibration or root/floor movement. Chain the perspective/radial derivative
through the full absolute joint Jacobian.

Camera rows apply only to their exact source knots. Original frame/source/body
clocks and decoder timestamps remain distinct and retained; there is no invented
grid-camera interpolation. Original local-y displacement, angular/rate bounds,
pointwise floor and image nonregression remain; the new profile labels local-y
honestly, not camera depth. It additionally restricts source-knot camera-depth
change to 0.08 supplied camera units and rejects invalid/behind-camera known
source endpoints (depth 0.2–100, normalized radius <=4). Missing/occluded/excluded
source/interval slots stay visible. Camera poses are never interpolated onto
the 30-Hz floor-check grid.

## Independent image admission and raw custody

Prepare the existing objective with `include_image_admission=True` **only** for
camera mode. `objective.image_admission(original_frame)` reads one detached,
cached `source-surface-body-image-admission-v1` row. It reuses the original
preparation's parsed registry, not another observation validator. A ready row
requires an included construction observation with detected/partial visibility,
the exact bound body/entity/evidence/frame and original source clock. Unknown
height, missing annotations or unknown world geometry do not discard otherwise
qualified image evidence. Excluded, unsampled or mismatched image payloads are
not used for target generation. The camera fitter separately requires the exact
explicit body-to-world mapping before geometry can run.

The additional cache is <=2 MiB, each row <=4 KiB, under the existing five-second
preparation deadline; it retains all <=1,800 source slots. Opt-in metadata reports
ordinary and image-admission row bytes and the shared envelope cache. The camera
report adds exact encoded metadata bytes and the aggregate encoded-cache count.
These counts are not Python object-memory/RSS measurements. The default creates
no admission cache and adds no serialized keys.

The helper receives original **raw bytes**, not reserialized mappings. Overlay
`input_bindings` separates:

- `body_raw_sha256`, `evidence_raw_sha256`, `camera_raw_sha256`;
- `body_canonical_sha256`, `camera_canonical_sha256`;
- `pose_evidence` (canonical hash/scope, plus original canonical hash only for a
  separately validated legacy `sealed_decoder_metadata` dimension overlay);
- camera-binding, coupling, root-reference, ground and original-angular-reference
  identities, each explicitly suffixed `_canonical_sha256`.

The preferred new loader route keeps raw evidence unchanged. Camera/pose
dimensions are qualified from the supplied camera and exact bound pose metadata;
the report calls this `supplied_camera_and_pose_metadata`, with
`decoder_verification='unavailable_no_decode_performed'`. No decoder provenance
is fabricated. All original decoder timestamps remain separately visible.

Complete camera/source slots use `max(declared_frames, decoded_frames)`. A
decoded-only tail has `camera_time_s=null` and
`camera_status='unavailable_original_decoded_slot'`, never an invented camera
pose. The objective must retain that same complete denominator before fitting.

```python
objective = prepare_surface_interval_objective(
    envelopes, body_bytes=original_body_bytes, binding_bytes=body_binding_bytes,
    include_image_admission=True,
)
result = propose_whole_body_reference_refinement(
    original_body_bytes, original_pose_bytes,
    proposal_objective="camera_surface_interval_v1",
    camera_document=original_camera_bytes, camera_image_scale=0.01,
    surface_interval_objective=objective, surface_interval_weight=2.0,
)
assert result["human_candidate"] is None
overlay = result["angle_proposal"]  # Not loadable as a human spec.
```

Those weights are explicit example assumptions, not recommended real-source
calibration. Preparing actual input envelopes/bindings remains a separate task.

## Approved work / verification plan

Inputs <=8 MiB each, <=900 references, <=1,800 complete source slots; existing
<=1,801 30-Hz grid samples, 30+8 second cooperative fit/finalization limits,
120,000 geometry calls, 20,000 interval evaluations and 4 MiB attempted-trial
journal. Preparation/transforms are bounded and cached once; final geometry is
fresh. No hard isolation claim for a cooperative Python deadline.

Pure tests only: exact input and source bindings; fractional source fps while
body.dt remains unchanged; nonzero radial/skew/nontrivial R/t derivatives;
absolute pelvis derivative; missing and invalid projection cases; angle-overlay
source immutability; unchanged default behavior; and the existing real bounded
QP with analytical geometry, including preserved failure gates. Native/model/
media/network/subprocess work is denied. No real-source or native acceptance.

## Verified result and preserved limits

**40 distinct checks pass:** 38 new camera/admission cases and two affected
legacy controls. They include the actual existing constrained QP with analytical
geometry, plus the unchanged `body_refinement._kinematics` with explicitly
stubbed forward/Jacobian functions. No native physics/model is imported or run.
Default legacy image/floor/surface admission and ordinary output shape remain.

The first solver case used three references, all ten source slots, one ankle
channel, authored scale 0.01, weight 2, and radial coefficient
0.00047528394168603385. It took **0.010319250 seconds inside the callable**
(excluding prior envelope/objective preparation), made 34 analytical geometry
calls, 14 camera projections and 14 interval evaluations, and accepted four
updates. All fourteen construction gates passed. Mean image error changed
0.163555304→0.055494700 pixels; summed interval loss changed
0.003592839090→0.001197613030 m². The maximum outside distance stayed
0.048941047 m because the original first reference remains fixed. Eight trial
records /4,480 bytes preserve the attempted directions. Full result: 74,989B.
Encoded prepared source caches total 31,787B (5,639 ordinary rows, 6,842 image
admission rows, 5,467 objective metadata, 13,839 shared envelope bytes).

Exact direct inputs/result are preserved outside pytest retention in
`.local/camera-body-fit-analytic-01/`, manifest SHA
`f0099ecb42a109d15902080f1b671cd1407300d6c3a3519b36c949971fd25aa3`, result SHA
`b6a28eae0ccbce130eaa0fec62056157bdc31277f15bf1289fed5fc88cf418f9`.
This is a **minimal authored analytical fixture**, not a native-validated human
or a qualified source run. Its origin is `test_camera_body_fit.fixture`, using
the existing whole-body test fixture and authored two-view binding structures;
some ancestor hashes are placeholders. It cannot prove source acquisition,
upstream custody, model admission or the full loader/native/CLI workflow.
The UI lane may inspect its saved producer/report bytes. Root separately
authorized a metadata-only authored source-adapter compatibility attempt;
that result is not claimed here and must not invent real ancestry.

One boundary test initially declared an unknown spatial claim while retaining a
numeric transform. The existing validator correctly rejected that malformed
**test fixture**. Its receipt is retained; setting only that test transform to
null yielded the targeted pass. No product guard was changed. The successful
first fit was not rerun after a later error-message-only change restoring the
legacy 30-Hz rejection wording; computational behavior was unchanged.

Published-code byte compatibility was checked separately under the same denials:
the frozen old objective module (`143853e9d12beccd0e90b32d44094e33bf1e278c2e1c4ba8c812b77911697100`)
versus the new **default** on known/unknown-height authored inputs. Both metadata
byte strings and all twenty source-row byte strings match exactly. Helper
`.local/check_camera_admission_legacy_01.py` produced receipt
`.local/camera-image-admission-legacy-01.json`, SHA
`cbb81c0e32bdb3471f8b0e243027398975d2efb06ea4974d44a724ccc35522a6`, in 0.043317s.
Only published source code and analytical fixtures were read; no real source
input or native study was repeated.

### Commands and receipts

All pytest commands use `env PYTHONPATH=src .venv-lab/bin/python -m pytest -q`
and `-o junit_family=legacy`:

1. `tests/test_camera_body_fit.py::test_existing_qp_improves_without_mutating_source_or_admitting_body`,
   with `WEFT_CAMERA_FIT_FIXTURE_OUT=.local/camera-body-fit-analytic-01`;
   `--junitxml=.local/camera-body-fit-first-01.xml`: one pass, 0.078s.
2. `-x tests/test_camera_body_fit.py --deselect=tests/test_camera_body_fit.py::test_existing_qp_improves_without_mutating_source_or_admitting_body`
   plus `tests/test_whole_body_surface_interval_refinement.py::test_default_and_explicit_original_objective_remain_identical`
   and `::test_opt_in_native_geometry_wrapper_captures_current_pelvis_without_a_second_forward`
   from that same file; `--junitxml=.local/camera-body-fit-rest-01.xml`: 35 passes,
   0.340s. This ran before the four extra boundary tests were added.
3. The four new cases `test_image_admission_does_not_require_world_alignment`,
   `test_decoded_only_slots_remain_without_invented_camera_poses`,
   `test_known_invalid_original_camera_geometry_abstains_and_closes`, and
   `test_input_and_opt_in_query_bounds_precede_expensive_work` in the new test
   file; `--junitxml=.local/camera-body-fit-boundaries-01.xml`: three passes and
   the malformed-fixture failure above, 0.108s.
4. Only the corrected `test_image_admission_does_not_require_world_alignment`;
   `--junitxml=.local/camera-body-fit-boundaries-02.xml`: one pass, 0.060s.
5. `.venv-lab/bin/python .local/check_camera_admission_legacy_01.py` for the
   published/default byte comparison. No unchanged passing check was repeated.

| Receipt | SHA-256 |
| --- | --- |
| `camera-body-fit-first-01.xml` | `43e9200d60d9f42ce29e81bf68a9ed0202bbb0428abbcd3b7ad485bea2f11e39` |
| `camera-body-fit-rest-01.xml` | `8e3ec449c4294e8b03edb91324ac844fe18907768280e1bd5932e9eeda8b6a29` |
| `camera-body-fit-boundaries-01.xml` | `10e21c5949e8eb334882eddb551b00f0ad7e343aa29c533136d04a26133a459d` |
| `camera-body-fit-boundaries-02.xml` | `6e53647de0510314a478568261e8d029921eb7ac6bdf43363b0433203ea65be3` |

### Frozen implementation

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/weft/camera_body_fit.py` | 17,854 | `e53d4a66581b1bdf7c16f4f1addb3977311eed48f65a65634cd9a400915a7232` |
| `src/weft/whole_body_reference_refinement.py` | 65,860 | `d6899edc246836d32e97a86f288388ee174b2c3375bd04c8d8b2a25d2159b4af` |
| `src/weft/source_surface_interval_construction.py` | 31,439 | `a30da16ac8682279404f3e166e38da60129e716b2d27551baadea33124d889f5` |
| `tests/test_camera_body_fit.py` | 29,490 | `d85eff246388ccf24ab32ea9e6006dfbd421233222864a739931cb702edb724f` |

New runtime modules beyond the existing fitter/surface closure:
`camera_body_fit`, `camera_shared_contract`, `camera_body`, `camera_reference`,
`multiview_root`, `multiview_body`, `multiview_pose`, `multiview_segments`.
Reuse their existing `camera_ground`, `camera_pose`, `spatial_pose`, `io` and
other transitive dependencies. No reconstruction/triangulation method is called
by their validation imports. Root owns the exact frozen worker closure.

Scoped Python whitespace checks passed. No Git mutation or unrelated avatar
edit occurred. The preceding checkpoint was published by root at
`61af0d6f37dff21310589e062308024b08db55de`; these five owned paths were excluded
pending review. No paid work, source/model/native job or live handle remains.

## Remaining reconstruction limits

Root's metadata-only jogging camera check places its exact positive coefficient
and whole-image radius in the new radial envelope domain. That does not supply
real height ranges, pixel uncertainty/correspondences, body/world bindings or
contact evidence. Original unavailable heel and negative ground gaps remain;
no fit, threshold, start or floor changes are implied. This overlay does not
recover contact/load/grasp, anatomy, biomechanics or arbitrary-video reliability.

Next: root reviews/integrates the five released paths with its independently
owned input loader/worker/CLI and the UI lane's offline report. A later frozen
native/source acceptance needs its own complete compatible input package and
plan. Do not load the angle overlay as a body, re-pin its original camera shape
as if source-estimated, rerun old heel studies or claim contact reconstruction
complete. Any future installation needs a distinct provenance-preserving body
derivation contract and unchanged independent admission checks.
