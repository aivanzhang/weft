# Floor-target calibration contract

**Proposed, not implemented or executed.** November 1, 2026 (America/Los_Angeles). Root owns later implementation and acceptance. This review used existing code/handoffs only; no source, native, inference, network or paid job ran.

[Source readiness](exercise-floor-calibration-gap.md) found no qualifying recording with the required measurements. Preserve exposed FreeMoCap, Interval, jogging and carpet failures; do not rerun them unchanged.

## Minimal API and reuse

```python
prepare_floor_target_calibration(
    observation_documents: Mapping[str, bytes], *, source_sha256: str,
    request_document: bytes, measurement_documents: Mapping[str, bytes],
    supplied_camera_document: bytes | None = None,
    supplied_ground_document: bytes | None = None,
    max_seconds: float = 5,
) -> FloorTargetCalibration
```

New `floor_target_calibration.py` returns immutable cached `describe()`/`files()`. Malformed custody raises; missing, unsupported or ambiguous measurements return unavailable/rejected diagnostics without an admitted camera/ground pair. No filesystem, detection, body construction or installation.

Reuse `camera_pose.validate_camera`/its full-K radial inverse, `camera_ground.validate_ground`, `io.canonical_hash`, qualified construction references from `world_evidence`, and `world_alignment.load_world_alignment`. Use the existing `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)`/Rodrigues primitive demonstrated in `scripts/studies/diagnose_freemocap_board.py`; do not copy its source/job runner. No RANSAC, focal-length search, camera bundle adjustment or new optimizer.

## Input contract

`floor-target-calibration-request-v1`: exact source/raw-document pins; mode `supplied` or `planar_target`; world-frame ID; explicit original anchor frame and additional construction check frames; qualified measurement references; fixed-camera/fixed-target interval covering the complete source. Evaluation references remain separately frozen and cannot select a solution.

`floor-target-measurements-v1` holds source-bound records with IDs, method, provenance, declared quality and uncertainty:

- Capture-resolution K, exact distortion model/k1, original dimensions and nominal/native clocks.
- Unique measured target corner IDs/planar XY coordinates, explicit `m` or `mm`, scale/point uncertainty. Target z=0 defines the board, not the floor. Preserve raw units; derived metres use only declared conversion factors.
- Proper target-to-world rigid transform, explicit frame IDs, unit-normalized scale 1, numeric translation/angle uncertainty and independent measurement references. Board visibility does not establish gravity alignment.
- Independent horizontal floor height/uncertainty in that world. A tilted plane is unsupported, never flattened. Height 0 requires an explicit floor-origin convention and measured target-to-floor relation.
- Original corner frame/evidence ID, both clocks, partition, pixel/uncertainty or missing/occluded reason. Excluded coordinates never enter preparation.

Missing scale, floor relation, required uncertainty, source/clock join or fixed-interval declaration causes abstention before solving. The adapter retains supplied measurement claims; it cannot fabricate an observed receipt or authenticate physical surveying.

## Bounded calculation and outputs

Solve once from the explicit anchor's four or more non-collinear corners. Check finite bounds/rank/conditioning first. Undistort through the full-K inverse, then call IPPE with normalized pixels and identity K/no distortion. Reproject with the declared K/k1 convention, including skew.

For `X_world = R_wt X_target + t_wt` and `X_camera = R_ct X_target + t_ct`:

```text
R_cw = R_ct R_wt.T
t_cw = t_ct - R_cw t_wt
```

Retain every alternative, residual and rejection reason. Additional construction frames may reject alternatives under frozen checks, without another solve. Require one geometrically distinct admissible solution; preserve numerical duplicates even if clustered. Never select merely the lowest residual. Later evaluation grades the frozen result without selecting/refitting it.

Only the explicit fixed-camera/target declaration permits repeating R/t across the source. Preserve every source slot, missing calibration observation and metadata-only tail; repeated poses are not per-frame measurements. No later start or retiming.

`calibration.json` (`floor-target-calibration-result-v1`) records pins, complete slot/partition statuses, alternatives, selected index or null, checks, uncertainty and work counts. Qualified nominal artifacts are `camera.json`/`ground.json`, optionally `surfaces.json`. Fitted extrinsics and transformed ground are `quality=reconstructed`, not measured just because input corners were measured. The supplied route preserves compatible exact sidecar bytes and makes zero solver calls.

## Three existing-schema conflicts

1. **Hashes:** camera-body/ground uses `canonical_hash(camera)`; source-surface uses raw camera-byte SHA256. Record both. `ground.json` binds the canonical pin; `surfaces.json` binds the raw pin. Never interchange them.
2. **Clocks/distortion:** camera-v1 requires complete rows at frame/FPS with matching dimensions/count. Unmatched native clocks cannot be retimed. Only k1 in [-1,1] is supported; do not discard higher-order/tangential terms. Negative k1 requires the existing monotonic inverse branch; interval envelopes still abstain for negative k1.
3. **Uncertainty:** current surface intervals admit authored height ranges, not measured ranges, and camera/ground cannot carry covariance. Keep measurement uncertainty in the new receipt. Do not relabel it authored or claim existing interval certificates propagate K/R/t/target uncertainty; they remain conditional on nominal camera geometry.

No body, root, floor, task or scorer rewriting; no automatic ground application or contact/grasp/load inference.

## Proposed fixed bounds

- At most eight observation and eight measurement documents; combined input 4 MiB, request 64 KiB.
- Existing camera envelope: 1–1800 frames, at most 60 seconds, width 64–4096, height 64–2160, FPS 1–240.
- One target, 4–64 corners, one anchor plus seven construction check frames; separate later evaluation at most eight frames.
- One IPPE call, at most two alternatives, no retry; at most 1024 explicit preparation point reprojections.
- Existing inverse: at most 20 iterations/radius 4; proper-R tolerance 1e-5, translation components within 1000 m, camera depth 0.2–100 m.
- Freeze adapter consistency limits before implementation: normalized planar rank 2, condition at most 1e6, construction RMS at most 1 px/max 2 px. These proposed engineering limits do not establish physical accuracy.
- Output 4 MiB/eight files; result 1 MiB. Five-second cooperative checks cannot preempt OpenCV; a public runner needs separate hard wall/CPU limits. Unknown uncertainty never becomes zero.

## Ten analytical acceptance cases

1. Known oblique transform verifies composition and unchanged floor relation through existing validators.
2. Declared mm/m inputs preserve raw identities but yield equal normalized geometry.
3. Nonzero k1/skew preserves projection convention; richer distortion/invalid inverse branch abstains.
4. Two admissible planar poses remain ambiguous; both retained. Reflections rejected.
5. Collinear/near-singular/duplicate/insufficient corners reject before solving.
6. Invalid depth/excessive residual preserves failures without admitted outputs.
7. Source/resolution/frame/count/clock mismatch rejects without retiming; missing/tail slots retained.
8. Missing floor relation, unknown units/uncertainty or tilted ground abstains before solving.
9. Held-out/excluded payload sentinel remains unread and cannot change construction selection.
10. Supplied route makes zero solves and preserves raw/canonical pins; byte/work/deadline overflow fails without partial success.

An analytical release establishes adapter consistency only. Reconstruction accuracy still needs a separately frozen source with actual measurements.
