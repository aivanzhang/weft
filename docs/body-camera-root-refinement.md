# Camera/root proposal boundary

`weft.body_camera_root_refinement.propose_camera_root_refinement(input_document,
settings=None)` is an opt-in, native-free construction component. It fits one
clip-wide camera rotation and source-knot root translations to **absolute** image
points. It returns separate camera/root candidates and a report, never a modified
human, task, policy or admitted scene. There is no CLI/source adapter yet.

The ordinary angle fitter subtracts the pelvis before comparing image points.
This component tests a different assumption: whether a shared camera/root
misalignment explains errors while the body itself remains fixed. Image agreement
does not establish measured calibration, unique depth, anatomy or physical accuracy.

## Input contract

Pass exact finite JSON bytes, at most 8 MiB. Duplicate/unknown fields are rejected.
The document has `schema_version: 1`, profile
`fixed-body-camera-root-input-v1`, and these fields:

- `bindings`: lowercase SHA-256 strings named `source_sha256`, `body_sha256`,
  `evidence_sha256`, `body_angles_sha256`, `body_shape_sha256`.
- `scale_model_units_per_unit`: fixed positive scale, recorded but never applied
  again or optimized; supplied world coordinates already use model units.
- `camera`: `rotation` (proper world-to-camera 3×3 matrix), `center` (world
  coordinates), `K` (pinhole intrinsics, optional skew), `radial_k1`, `quality`
  (`authored`, `reconstructed`, `measured`) and bounded `provenance` text.
  Quality is a caller claim, not verified calibration.
- `floor`: fixed unit `normal` and `offset_model_units`, defining
  `normal · world_position = offset_model_units`.
- `samples`: 2–900 ordered source knots. Each has exact `evidence_id`
  (`source_sha256:frame:index`), `frame`, `time_s`, `decoder_timestamp_s`
  (number or null), original `root` and 1–12 `points`. Each point retains
  `name`, absolute `world` coordinates, absolute image `pixel` (or null) and
  `weight` (0 or 0.5–1). Every knot retains the same named endpoint slots.
  Times increase within 60 seconds; adjacent gaps cannot exceed 0.25 seconds.
  Native decoder time is preserved separately, not replaced by nominal time.
- `floor_checks`: null, or `{bindings, rows}`. Bindings exactly match the three
  body/angle/shape identities above plus `root_reference_sha256` (canonical hash
  of the ordered `{time_s, translation: root}` rows) and `floor_sha256` (canonical
  floor hash). Canonical hashes use the existing `weft.io.canonical_hash`.
  Rows contain `kind`, `time_s`, and a `gaps` map of the same 1–64 support IDs.
  The complete schedule is all source knots, then
  `floor((last-first)*30+1e-9)+1` grid rows at `min(last, first+i/30)`.

The report computes the exact input-document hash. External source/body/evidence
pins are **declared identities**, not authentication of bytes this function never
reads. Missing image endpoints keep null errors; wholly unavailable-image roots
stay unchanged. Grid rows never receive image targets. Source/grid overlap is
not additional observed poses.

## Fixed quantities and checks

Camera center, intrinsics, distortion, scale, first root, body angles/shape and
floor remain fixed. Camera translation is derived as `-rotation @ center`;
camera center and root cannot move together. No per-frame camera, scale/focal
fit, timing offset, confidence change, contact target or reward objective exists.

Analytic projection derivatives follow `camera_reference.project_world`. The
solver eliminates independent 3×3 root blocks, leaving a 3×3 camera system.
Data-only ranks are computed before regularization and after each proposed
update. Rank-deficient/ill-conditioned inputs return an unchanged unavailable
proposal. Full local rank is not global uniqueness or measured uncertainty.

Every accepted update reduces weighted image error without worsening any
positive-weight endpoint beyond the fixed 1e-8-pixel numerical tolerance.
Camera rotation is bounded by 0.15 radians, root displacement by 0.08 model
units, and root-correction speed by 0.3 model units/second. These are proposal
envelopes, not measured physical limits. Bounds may only be tightened.

If complete saved gaps are supplied, unchanged body geometry makes the effect
of root translation on each gap exactly linear, conditional on those supplied
values. A gap must remain at least `min(0, original_gap)-1e-8`; the separate
feasibility limit stays 0.025 model units. No floor loss selects updates.
Missing floor data produces null checks and unavailable geometry qualification,
never a pass. Supplied checks are not an independent geometry evaluation.

At most eight iterations × eight backtracks run within a ten-second cooperative
budget, with a 16-MiB output bound. No dense global Jacobian, native model,
physics step, media operation, inference, network call or file write occurs.
Timeouts during fitting return only the last fully accepted proposal, marked
partial. Parsing/initial diagnostics may instead raise a bounded-input or timeout
error. Native NumPy calls are not forcibly preempted.

## Integration with the existing body workflow

A future trusted adapter must reuse `whole_body_reference_inputs` source custody
and crop-target validation, then bind all retained body/reference identities and
missing source slots. Extract absolute endpoints from the existing native body
geometry, adding its pelvis offset to `body_refinement._kinematics` output;
do not use its pelvis-centered result as absolute world coordinates. Derive pixel
targets from the exact selected original-image landmarks, never interpolation.

The adapter must independently establish the fixed camera hypothesis and freeze
its construction/evaluation partition before fitting. Existing exposed source
studies are development evidence, not new holdouts. Preserve the full source
archive, clocks, unknown slots and baseline; do not select inputs using floor or
task outcomes. Recheck original source/body/runtime identities before sealing.

After a proposal, a separate fresh `whole_body_feasibility` geometry audit must
check the original source/30-Hz schedule with the unchanged floor, geometry and
limit. Native body admission needs an explicitly versioned root/camera binding:
ordinary `mobile_human` rejects nonzero inferred root depth. This component does
not bypass that guard. Root motion is not evidence of support/grasp, and the
19-axis capsule template still lacks recovered anatomy and physical validation.
