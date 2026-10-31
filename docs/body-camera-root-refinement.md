# Camera/root proposal boundary

`weft.body_camera_root_refinement.propose_camera_root_refinement(input_document,
settings=None)` is an opt-in, native-free construction component. It fits one
clip-wide camera rotation and source-knot root translations to **absolute** image
points. It returns separate camera/root candidates and a report, never a modified
human, task, policy or admitted scene. The saved-body adapter and existing
`body-reference-propose` command expose it as a separate diagnostic mode.

## Try it on an extracted body

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run artifacts/single-leg-person-workflow-01/build/body_source \
  --objective camera-root --camera-focal-px 640 --fit-seconds 5 \
  --out artifacts/my-camera-root-proposal
```

Use a new output directory. The command does not start a server or open a browser.
Inspect `proposal/index.html` yourself, or read `result.json` and the stage artifacts
from the terminal. An unchanged, unavailable or rejected diagnostic is a completed
inspection workflow, **not** an admitted body or a successful reconstruction.

Here **640 pixels is an explicitly authored focal-length assumption**, not a
calibration. This opt-in initialization fixes a centered principal point, zero
distortion and the original camera-to-model axis convention. It uses the original
image scale to choose camera distance, anchoring the first model pelvis to its
selected image estimate. These choices are frozen before residuals are evaluated;
there is no focal search or camera fallback. Source scale is still estimated.
The first pelvis must be available. Alternatively supply `--camera camera.json`
with the exact `fixed-body-camera-hypothesis-v1` contract below. Do not supply both.

The step artifacts preserve:

1. `inputs/`: exact selected original observations, body, clocks, source custody,
   and frozen camera configuration. Ordinary input does not decode or copy video;
   saved crop recovery retains its required original media/proof without decoding.
2. `geometry.json`: original absolute body endpoints and all sphere/capsule floor
   support gaps over every source knot and the full 30-Hz grid.
3. `camera-root-input.json`, `target-provenance.json`, `source-slots.json`: exact
   source-coordinate pixel targets, selected landmark provenance and unavailable
   slots. Interpolated grid poses receive no invented image targets.
4. `proposal.json`: bounded camera/root proposal, original/proposed errors,
   identifiability, rejected updates and all retained failures.
5. `geometry-check.json`: a fresh original-model diagnostic if roots changed;
   otherwise an explicit skipped check. No candidate is installed or stepped.
6. `worker.json`, `plan.json`, `supervision.json`, `manifest.json`: pinned code,
   settings, source identities, native-call counts, resource usage and exact outputs.

The offline page plots selected image points and model projections, **not actual
source photographs**. It renders one selected knot at a time and makes no network
requests. This report complements the existing video/lab workflow; it is not a new
simulator or automatic repair/admission route.

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

## Source and geometry integration

`body_camera_root_inputs` reuses `whole_body_reference_inputs` source custody and
crop-target validation, binding retained body/reference identities and missing
slots. `body_camera_root_geometry` extracts absolute endpoints from the original
native body geometry by adding its actual pelvis-site offset to the existing
pelvis-relative kinematics. Pixels come from the exact selected original-image
landmarks, never interpolation.

The camera document fixes `schema_version: 1`, profile
`fixed-body-camera-hypothesis-v1`, exact source/body/derived-evidence `bindings`,
an `image` object with width/height/fps/frame_count, `coordinate_frame:
original_body_model_units`, `scale_model_units_per_unit: 1`, `partition:
exposed_development`, `selection: explicit_before_proposal_no_search` and the
camera object described above with `quality: authored`. These metadata checks
authenticate selected bytes, not camera correctness or physical accuracy.

The parent rechecks original source/body/camera/runtime identities before sealing.
Previously exposed source studies remain development evidence, not new holdouts.
The original archive, clocks, unknown slots and baseline stay unchanged.

If proposed roots change, a fresh original-body geometry instance checks every
original source/30-Hz sample with the unchanged floor, geometry and limit. The
worker permits at most two model compilations/four data objects and tightens its
5,402-forward global cap to twice the input source-plus-grid row count. It forbids
physics steps, resets, rendering, media decoding, inference and network calls.
After one narrowly allowed read-only macOS runtime-startup probe, nested processes
are also forbidden. Fit time is at most ten seconds inside a 30-second owned child;
the parent retains failure receipts and kills its process group on timeout.
Input/output byte bounds and observed peak RSS are not a hard memory cap. These
guards constrain reviewed trusted code, not arbitrary generated code.

Native body admission still needs an explicitly versioned root/camera binding:
ordinary `mobile_human` rejects nonzero inferred root depth. This component does
not bypass that guard. Root motion is not evidence of support/grasp, and the
19-axis capsule template still lacks recovered anatomy and physical validation.

## Saved-source integration result

`artifacts/body-camera-root-cli-01` completed the public CLI and offline report on
the saved single-leg exercise body in 1.0655 seconds. The authored focal length
was 640 pixels, selected before fitting; no calibration or focal search occurred.
The original 118 source knots, 235 grid labels, 235 decoded-frame coverage and
236-frame container declaration remain distinct.

The result is **unchanged**, not an accuracy improvement. Image RMSE stays at
7.55056 pixels for 1,190 available endpoints; 226 endpoints remain unavailable.
Seven candidate directions exceeded the bounded correction envelope and one
worsened floor support, so no update was accepted. Original floor failures remain.
These perspective errors are not directly comparable with older pelvis-centered
image-fit metrics. The existing body, world, task and policies were not replaced.

The worker used one model, two data objects and 235 forward evaluations, with no
physics actions, resets, video decoding, new inference or API calls. It recorded
116,703,232 bytes peak RSS, not a hard memory cap. Unchanged roots correctly skip
the second geometry pass. Root manifest:
`1e07d5371215a3b9bb515ec7730ee81d5868d352e0219764f3dcc62c9f79deae`.
