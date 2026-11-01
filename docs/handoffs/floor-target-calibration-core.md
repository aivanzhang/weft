# Floor-target calibration core

## Current ownership and status

November 1, 2026, America/Los_Angeles. Core/test implementation is frozen and independently reviewed clear; **43 distinct focused analytical cases pass**. Root reports that **all four actual public numerical cases passed** in `artifacts/floor-target-numerical-02`, using fixture02: three IPPE calls/six Rodrigues conversions total; supplied mode made neither call. This authored numerical milestone does not establish real-source or physical accuracy. Independent saved audits are with the UI/worker owners; this lane did not repeat output reads or execution. This lane owns only `src/weft/floor_target_calibration.py`, `tests/test_floor_target_calibration.py` and this note. Root owns files/CLI/public numerical acceptance; the reviewer owns the separate worker. The UI lane owns its bytes-only inspector. Published base `bf661cd`; no Git mutations. Preserve unrelated avatar work and all earlier failed calibration studies.

## Public API and exact serialized contract

`prepare_floor_target_calibration(observation_documents, *, source_sha256, request_document, measurement_documents, supplied_camera_document=None, supplied_ground_document=None, max_seconds=5) -> FloorTargetCalibration`. Both document maps use exact raw SHA256 keys and bytes. Immutable cached `describe()` returns a fresh parsed result; `files()` returns a fresh mapping of immutable bytes. No filesystem, detector, physics, body installation or report call.

Request exact keys (`schema_version=1`, `profile=floor-target-calibration-request-v1`):

```text
source_sha256, mode (supplied|planar_target), observation_sha256s,
measurement_sha256s, supplied_camera_sha256, supplied_ground_sha256,
world_frame {id,length_unit:m,axes:x_right_y_away_z_up},
anchor_frame (integer|null), check_frames [], evaluation_frames [],
fixed_interval (null|{first_frame,last_frame,camera_static,target_static,
                    method,uncertainty,evidence_refs}),
measurement_refs {camera,target,target_to_world,floor}
```

Each selected measurement reference is `{document_sha256,record_id}` or null. Supplied mode has null anchor/fixed interval/target/target-to-world and empty check/evaluation lists; it still requires full-camera and floor measurement support. Request pin lists match exact sets without duplicates; map insertion order and pin-list ordering do not choose geometry. Planar mode has one explicit original anchor, at most seven distinct construction checks and eight separately retained evaluation frames. No automatic frame or corner choice, fitting on later evaluation, or retiming.

Measurement documents have exactly `{schema_version:1,profile:floor-target-measurements-v1,source_sha256,records}`. Every record has `{id,kind,quality,method,provenance,uncertainty,evidence_refs,value}`. Quality is `measured|authored|unknown`; measured is the supplier's claim, never independently authenticated. Global required records must declare measured quality and reference construction-qualified observed evidence. Every uncertainty dictionary includes `interpretation=declared_absolute_bound|standard_deviation|unknown`; neither kind is propagated or treated as a certified interval. Unknown required support abstains. Record kinds/values:

- `camera_calibration`: value `{source_sha256,units:m,width,height,fps,metadata_frames,decoded_frames,K,distortion:{model:pinhole_k1,k1},world_frame_id,camera_sha256}`. Exact supplied raw camera pin binds the full sidecar, including extrinsics, in supplied mode; null in planar mode. Uncertainty `{intrinsics_px,radial_k1,translation_m,rotation_rad}` plus interpretation: numeric nonnegative declarations; translation/rotation may be null only before a planar solve.
- `planar_target`: value `{id,frame_id,units:m|mm,entity_ref:{document_sha256,object_id},corners:[{id,xy}]}`. Four to 64 unique corners; uncertainty `{point_position,scale_relative}` in declared target units/dimensionless units. Raw units stay retained; normalized points use only m→1 or mm→.001.
- `target_to_world`: value is the existing `world-alignment-v1` document, target metre frame→requested world, scale1 and identity clock. Uncertainty `{translation_m,rotation_rad}`. No inferred gravity/floor mapping.
- `floor_plane`: value `{world_frame_id,normal_world,height_m,origin_convention,target_to_world_ref}`; exact `[0,0,1]` is supported, tilted floors abstain. `target_to_world_ref` is the selected relation in planar mode and null for directly measured supplied camera/world floor. Explicit origin convention required even at height0. Uncertainty `{height_m,normal_rad}`.
- `target_corners`: value `{frame,nominal_time_s,source_time_s,observation_ref:{document_sha256,object_id,observation_id},partition,status,points}`. Status `observed|missing|occluded`; uncertainty `{pixel_error_note}`. Points `{corner_id,status:observed|missing|occluded,pixel:[u,v]|null,uncertainty_px:number|null,reason:string|null}`. Original nominal and observation-clock times stay distinct and exact. At most one record per original frame. Only explicit construction frames have point payload semantics inspected; excluded/evaluation payloads remain opaque retained input bytes.

The concrete fixture keeps the floor record in a separate measurement document: its raw relation reference points to the first document. Do not create an impossible self-referential raw document hash.

Result exact top-level keys (`schema_version=1`, `profile=floor-target-calibration-result-v1`):

```text
status (qualified_nominal|unavailable|rejected|ambiguous), mode, reason,
input_bindings, source, request, measurement_records, frames, alternatives,
selected_index, artifact_bindings, counts, work, limits, uncertainty, claims
```

`input_bindings` exact keys: `source_sha256,request_sha256,observation_document_sha256s,measurement_document_sha256s,supplied_camera_sha256,supplied_ground_sha256`; inventory lists are sorted. `source` has `sha256,width,height,fps,native_frame_times_s,decoded_frames,metadata_frames,retained_frames,time`; time maps each observation-document hash to its exact clock declaration. Missing nominal FPS returns unavailable, preserving native timestamps and null nominal times rather than inventing a camera timebase. `measurement_records` has `document_sha256,id,kind,quality,method,provenance,uncertainty,evidence_refs,target_units`; target_units is explicitly m/mm only for target records, otherwise null. Excluded point payloads never enter this metadata.

Each complete `frames` row has `source_frame,source_status,nominal_time_s,role,partition,observation_id,source_time_s,measurement_ref,status,reason,points`. Role is `anchor|construction_check|evaluation_not_used|not_selected`; point headers are `{corner_id,status,reason}`, with no excluded coordinates. No record means an explicit missing row, not a dropped source slot. Metadata-only tail rows are retained.

Each alternative has `index,R_ct,t_ct,R_cw,t_cw,duplicate_of,status,reasons,checks,projections`. Invalid/nonfinite solver returns retain their index and null numeric geometry with an explicit reason. Checks include depth/domain/residual validity and maximum/RMS pixel error. Each used-frame projection has `source_frame,measurement_ref,points,rms_px,maximum_px`; each point has `corner_id,pixel,uncertainty_px,predicted_pixel,residual_px,error_px,depth_m,status`. No withheld projection is computed. Numerical duplicates use max absolute R/t difference ≤1e-9 and remain listed. Only one distinct passing group is admitted, with first-index representative; this is nominal uniqueness under fixed checks, not uniqueness under measurement uncertainty.

`artifact_bindings` is null unless qualified; then exact keys are `camera_raw_sha256,camera_canonical_sha256,ground_raw_sha256,qualities:{camera,ground},generation:supplied_exact_bytes|planar_nominal`. Camera-body ground binds the canonical camera hash; no surface raw-hash contract is substituted. `files()` always includes `calibration.json`; only qualified results include `camera.json`/`ground.json`. Supplied artifacts stay byte-identical with unchanged quality. Reconstructed artifacts have quality `reconstructed`. No `surfaces.json` is emitted: uncertainty is not silently converted into authored ranges or claimed propagated by existing interval geometry.

`counts` exact keys: `source_slots,measurement_records,construction_frames_requested,evaluation_frames_not_used,prepared_construction_frames,used_corner_observations,alternatives,passing_alternatives,distinct_passing_alternatives,frame_status_counts`. The status histogram uses each frame's reason, or its status when no reason exists. `work` exact keys: `input_bytes,solver_calls,undistorted_points,point_reprojections,prepared_construction_frames`. `limits` is the frozen module LIMITS dictionary; no adaptive thresholds.

`uncertainty` explicitly says `propagation=not_propagated` and retains measurement claims. `claims` fixes physical validation false, contact/grasp/load unknown, geometry/body/floor installation false. Passing schema/measurement-reference checks does not authenticate surveying or grant body admission.

## Fixed bounds and math

Eight observation/eight measurement docs, total4MiB; request64KiB; 1800 original slots/60seconds; one target/64corners; one anchor+7checks; 8 unused evaluation frames. Five seconds cooperative, including preparation and encoding. One lazy OpenCV `solvePnPGeneric(...SOLVEPNP_IPPE)` call, no retry/fallback; at most two alternatives/1024 point reprojections. OpenCV cannot be preempted cooperatively; root owns hard worker limits.

Reuse `camera_pose.validate_camera` and `_undistort`, `camera_ground.validate_ground`, `world_alignment.load_world_alignment`, `source_surface_anchors._sources` and `io.canonical_hash`. Full-K/skew and exact k1 retained; richer distortion abstains, never truncated. Both target XY and undistorted image-ray layouts require rank2/normalized condition≤1e6 before solving; source-ray inverse uses existing20 iterations/radius4; depth0.2–100m; RMS≤1px and max≤2px on every required construction frame. Compose `R_cw=R_ct R_wt.T`, `t_cw=t_ct-R_cw t_wt`. Fixed-camera/fixed-target declaration is required before repeating nominal R/t across the source. It is not per-frame measurement.

Output4MiB/eight files; result1MiB. Missing measurements yield unavailable; evaluated inconsistent geometry yields rejected; multiple distinct passing alternatives yield ambiguous. Malformed custody/schema or resource exhaustion raises without a success object.

## Verification and retained evidence

- `PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_floor_target_calibration.py --junitxml=.local/floor-target-core-pure-01.xml`: 37 passed, two test-only nested `pytest.approx` matcher errors. The producer geometry checks had passed; no numerical tolerance was changed. Original receipt preserved.
- Same file with `-k 'existing_full_k or native_clock or unknown_pixel or known_nominal or excluded_and_later or supplied_exact_bytes or complete_1800'`, receipt `floor-target-core-pure-02.xml`: eight corrected/new/affected checks passed.
- New fractional-clock/oblique/radial/skew analytical payload case, receipt `floor-target-core-pure-03.xml`: one passed with stub IPPE. Saved exact inputs/result at `.local/floor-target-core-analytic-01`; eight files/34,871 bytes, fixture pin `766ab478fe12205d028c0fff9ca8511f5f297a89f7d48c8fa5fe45ea6672d0db`. Do not label this a real solver or source acceptance.
- Self-review found that target rank alone allowed collapsed image coordinates into IPPE. Root approved the same fixed rank/condition check on undistorted rays. New red receipt `.local/floor-target-image-rank-red-01.xml`, SHA `1f1474eb858d39661523b34bf440a3d1e8a43b37d3c5dd2cc93a4c3021a8fee1`, preserves the incorrect solver entry. New plus three affected checks passed in `.local/floor-target-image-rank-green-01.xml`, SHA `1f44c5e0b1f001c1dcff9c9bf83a74157abc44066cb9f82b782de9a005604260`.
- Deduplicating latest outcomes gives **43 passed**. No old unchanged suite was repeated. OpenCV, physics/model imports, networking and subprocess work were blocked in these tests. Existing NumPy projection/inverse/linear algebra was exercised.

Frozen product SHA `8ef5430ebc12fe80ccbe6d2e12ea8696ae57f360b3cfadd5bafca4bf9d6e6f04`; focused test SHA `67dddeb34474cd4ae72808c519f9182a489cbe3f2491ee71f9b0d239e4a49ac0`. Release receipt is `.local/floor-target-core-release-01.json`.

## Historical fixture01: preserved, not executed

`.local/floor-target-numerical-inputs-01/fixture.json` SHA `4237bf2c7afb5b3df5756ddd0e88d048f1752c8bdd98b2b693be999864106b73` inventories 18 JSON inputs/65,709 bytes across four cases. Preparation took .002407 seconds using saved authored bytes and scalar projection math only: zero core/IPPE/Rodrigues/media/model/physics calls. Helper `.local/prepare_floor_target_numerical_inputs_01.py` is pinned in the fixture.

1. Exact saved oblique target, k1=.08, skew4px, FPS30000/1001; one nominal solution is hoped for, not forced.
2. Identical physical target expressed in mm; pixels unchanged and uncertainty units converted explicitly.
3. Same oblique target at30m, a near-affine ambiguity case expected to retain two plausible alternatives; actual outcome must be preserved without tuning.
4. Exact saved camera/ground bytes plus simulated full-camera/floor receipt; no target and zero solver calls, original qualities preserved.

Root replaced fixture01 before any numerical execution to test a nonidentity target-to-world transform and a floor distinct from the tilted target. This was not a response to solver outcomes; fixture01 and its pins remain unchanged.

## Fixture02 and reviewed numerical preparation

`.local/floor-target-numerical-inputs-02/fixture.json` SHA `6a3f3b7887b8806e5fc8b9fa8148b34923abe0c2437fc0ba7a11b47763165887` inventories **18 JSON inputs/127,593 bytes**. Its inputs-only `prepare.py` SHA is `e7a1e75b4d24b9ad3e6b5f9954b8ad4ffd60431f6967e9538d36e76ae82f103a`. Preparation took .004928 seconds using scalar analytical projection only; zero core/IPPE/Rodrigues/media/model/physics calls.

The fixed 12-corner grid uses x={−.3,−.1,.1,.3}, y={−.2,0,.2} metres. Target→world is `Rz(.2) Rx(.6)` with translation `[.1,.05,.3]` m; floor height is independently −.2 m. Camera world→camera rotation is `diag(1,−1,−1)`, its world center is `[.15,−.10,1.5]` m, K=`[[900,13,640],[0,870,360],[0,0,1]]`, k1=.15, image1280×720, FPS30000/1001. Frozen truth includes target→camera and world→camera R/t separately. This tests composition direction rather than coincident coordinate systems.

All cases preserve 12 metadata slots/11 decoded slots. Planar cases use construction frames0/1 only, retain audit frame2 without evaluating its points, retain unsampled frames3–10 and metadata-only frame11. The audit frame's role remains `evaluation_not_used`, while its source-exclusion reason remains `excluded_source_partition`. No later frame substitutes for the original anchor.

1. Metre target: strict predeclared expectation `qualified_nominal`, one IPPE call.
2. Identical physical target in millimetres, with point-position uncertainty scaled consistently: same strict expectation and one call; pixels stay unchanged.
3. Same board/world/floor, camera center z=30m: strict expectation `ambiguous`, one call, both distinct alternatives retained and no camera/ground export.
4. Independently authored exact true camera/ground with simulated full-camera/floor support: strict expectation `qualified_nominal`, zero solver calls, exact original bytes and `authored` qualities preserved; no target route.

These are hypotheses fixed before execution, not promised solver outcomes. Any unexpected ambiguity, rejection or numerical mismatch must remain a failed acceptance; no tuning, threshold changes or retry is authorized by this note. All source identities are opaque authored test labels, not hashes of acquired media. Measurement receipts are simulated, not real surveying.

Original prepared public acceptance pins, retained as pre-execution review evidence:

- `artifacts/floor-target-numerical-01/plan.json`: `0f1b13ba3e91ed47c5cae25f568e998db3c9a3d0d517aa8c0e801bc33fd38c35`, status `frozen_not_executed`.
- `artifacts/floor-target-numerical-01/runner.py`: `7e434af090af0e3b2620ebeae49fc22258702f2cb4bc63d939ca2b6d70e412b5`.
- Exact copied `fixture.json`: `6a3f3b7887b8806e5fc8b9fa8148b34923abe0c2437fc0ba7a11b47763165887`.
- `.local/floor-target-calibration-runtime-01/manifest.json`: `84da0762dd8ab5dd9bf71b1c8b7930c50821fb4385f900bc945ea418057f24e3`; 291 files from committed `bf661cd6e8fccd58b22ac324a5374823dee7c9e2` plus explicit overrides. Its declared core/worker pins match the reviewed releases. This lane read the manifest, not every runtime member.
- Worker: `2a3aaaaa51210c18ae8642519cdda19d266c86efc6df69f27d64168a69028718`.

The plan invokes four exact public `floor-calibrate` commands, with two concurrent cases, three total solver calls, 30-second package limits and 35-second per-case outer limits. Aggregate execution bound78seconds; outer output132MiB/430files; observed child RSS acceptance cap512MiB. Launch/custody/resource enforcement is independently reviewed by the worker owner.

Read-only numerical-expectation review of this exact runner is **clear**. Checks match the fixture/core frame shapes, 24 used corner observations/48 retained point projections per planar case, full-K/radial forward arithmetic, world-transform direction, separate floor height, source-period camera rows, canonical-ground versus raw-camera hashes and supplied-byte preservation. Projection and truth tolerances are1e-6; metre/millimetre pose equivalence is1e-9. The runner's stronger strict expectations are preserved, not confused with general API guarantees. No tests, producer, solver or source work was executed for this review. At the recorded review time, `started.json`, `acceptance.json` and terminal outer `manifest.json` were absent.

## Actual public numerical milestone: root-reported passed

Root's terminal acceptance is `artifacts/floor-target-numerical-02`, outer manifest SHA `4af92dec9c72b9b83fa08ea39d382478c3f8c18cc35dad1d7d32679bc06baf56`. All four cases passed in **1.003870 seconds**, retaining **189 files/2,431,160 bytes**. Root reported plan02 SHA prefix `b2c82a54` and runner02 SHA prefix `e824cc79`; those are prefixes, not complete pins. Only root cleanup changed after the reviewed preparation; fixture inputs, runtime and numerical criteria were unchanged. Earlier preparation/evidence remains preserved.

- Cases1/2: qualified nominal calibration; maximum R/t truth error `1.6514567491299204e-15`, maximum independent projection error `1.1368683772161603e-13` px. Metre/millimetre nominal geometry was exactly equal.
- Case3: both distinct alternatives passed the fixed nominal checks; neither was selected and no camera/ground pair was emitted.
- Case4: exact supplied camera/ground bytes and their authored qualities were preserved; zero solver calls.
- Total actual work: three IPPE calls and six Rodrigues conversions. All12 original slots/11 decoded slots remained; evaluation point payloads were not used.

These metrics are root-verified results received through the owned handoff, not a fresh independent audit by this lane. The UI and worker owners are performing their bounded saved-only audits. This documentation update read only this note and performed no output scan, core/solver invocation or repeated test. Core/test pins above are unchanged and all three owned paths are frozen for publication.

Current live handles: none in this lane. Next action is root integration after the separate saved audits, not another numerical run. Real measured source acquisition, physical reconstruction accuracy, uncertainty propagation and contact validation remain unresolved.
