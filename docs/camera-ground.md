# Supplied ground: correct coordinates, unresolved body height

The camera-aware lab can now use an explicitly supplied ground height. This fixes the mismatch between its first-pose floor and the source's world coordinates. **It does not fix the estimated body:** the jogging example floats above the ground and contains a visibly inverted pose.

## Try the original supplied-ground example

The later [Full-model candidate](full-pose-model.md) corrects the inverted pose and has a new exact recording. It still floats; both candidates remain available for comparison.

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/camera-ground-jog-lab-01
```

Open the printed local URL. Inspect captured movement, try manual joint/release/push/support controls, and use **Record & start** / **Stop & save**. No learned policy is available for this candidate.

- [Source and inferred body](../artifacts/camera-ground-jog-interaction-01/index.html), including the largest ground gap and the unsampled final frame.
- [Verified action replay](../artifacts/camera-ground-jog-interaction-01/session/replay.html) and [video](../artifacts/camera-ground-jog-interaction-01/session/video.mp4).
- [Height comparison](../artifacts/camera-ground-jog-geometry-01/height-comparison.png).

The new recording contains 39 actions, one reset and 41 states. Every state, reward and completion flag replays exactly. This verifies the simulator/recording workflow, not the reconstruction's physical accuracy.

## What changed

The previous camera-aware example placed the floor under its first estimated body pose. In the supplied camera coordinates, that floor is at **z = 0.31403 m**. The source explicitly defines its reconstructed ground at **z = 0**.

The optional ground input uses that declared height directly. It changes only the world-to-simulation vertical offset. Raw pose estimates, body dimensions, angles, relative root travel, source times and forces remain unchanged. Converting all 1,680 saved native joint positions back to world coordinates before/after gives a maximum difference of **4.44×10⁻¹⁶** model units. Camera-root estimate files are byte-identical.

No pose or floor score chooses the new height. Reapplying first-pose floor fitting to a supplied-ground body is rejected. The original candidate and its failures remain saved.

## What the checks now reveal

The unchanged floor-penetration check passes all 112 available source poses and 239 control-grid poses. But every available source pose has a positive body/ground gap:

| Lowest native collision surface above supplied ground | Model metres |
| --- | ---: |
| Minimum | 0.0763 |
| Median | 0.3555 |
| Maximum | 0.6296 |

These are geometric gaps, **not inferred airborne phases or valid foot contacts**. The UI shows the gaps alongside penetration results. The body has no anatomically recovered feet, and contact timing is not known.

All 240 source frames remain accounted for: 112 estimates, 120 unsampled frames and eight unavailable sampled poses. Source frame 200, selected for its largest gap, shows the estimated body inverted while the person is jogging. The height plot and raw evidence retain this outlier; confidence alone does not make the 3D pose correct.

## Why changing the avatar alone will not solve this

A separate post-construction comparison uses the provider's reconstructed joints, never as fitting targets. It retains all 2,880 frame/joint slots and compares 1,344 available slots; 1,273 also meet the declared image-confidence/crop check.

Absolute 3D joint RMSE is **0.427 m for raw pose/root estimates** and **0.430 m for the fixed articulated body**. Their mutual RMSE is only **0.041 m**. These descriptive distances include body-position error; they are not the earlier hip-centered joint metric and do not establish measured anatomical accuracy.

For all 224 available ankle endpoints, the mean signed vertical error decomposes into:

| Component | Mean vertical contribution |
| --- | ---: |
| Estimated root height | +0.1309 m |
| Raw pose's ankle position relative to its root | +0.1007 m |
| Fixed body geometry versus raw pose | −0.0010 m |

The vectors add per endpoint; RMS magnitudes do not add. Root and raw-pose estimates account for most of this average height disagreement. The new ground placement changes none of them. Provider cameras, scale, ground and reference joints are themselves reconstructed, not independent laboratory measurements. [Source and license](multiview-jog.md#source-and-reference-quality).

The subsequent [two-view experiment](multiview-pose.md) uses two additional source videos to constrain 3D pose from image evidence and check a third-camera projection. Conditional agreement improves, but the inverted torso is rejected rather than repaired, and hidden joints remain missing. That experiment has not changed this lab or produced a learned policy; original-clip synchronization remains an explicit assumption.

## Build with your own supplied ground

```sh
.venv-lab/bin/python -m weft camera-body \
  --run artifacts/multiview-jog-image-01/run \
  --camera artifacts/camera-jog-root-01/camera.json \
  --ground artifacts/camera-ground-jog-plan-01/ground.json \
  --out artifacts/camera-ground-new
```

Use a new output directory. Omitting `--ground` preserves the original first-pose floor behavior. The [example ground file](../artifacts/camera-ground-jog-plan-01/ground.json) requires schema version 1, source SHA-256, the camera JSON's canonical hash, `right_handed_z_up`, metre units, horizontal `height_m`, provenance and `measured` / `reconstructed` / `authored` quality. Quality is a supplied claim, not independently verified. The first root must remain 0.01–5 model units above the plane. Unknown fields, mismatched calibration, nonfinite heights and later offset tampering are rejected.

The CLI stores exact ground bytes, provenance, a signed native-surface clearance report and the unchanged floor audit. Geometry audit and manual controls run locally; no new inference, training or API call occurs.

## Verification

**116 scoped Python checks and 28 deterministic UI checks pass.** Tests cover world-origin invariance, unchanged native points, floating bodies that pass penetration checks, retained intersections, invalid/calibration-mismatched input, offset tampering, preservation of immutable source runs and ground changes during packaging. The final post-test edits affect coordinate/provenance wording and placement-method metadata only; the actual build and interaction use that final implementation.

The actual build took 0.387 s internally. Across 120 in-process action/state/JSON calls, median latency was 0.800 ms and p95 0.816 ms, with a maximum response of 35,132 bytes. This excludes HTTP/browser painting and sustained idle behavior. Selected saved images were visually inspected; no browser interaction or server was started.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| [Plan, inputs and verification](../artifacts/camera-ground-jog-plan-01/report.json) | 7 | `4ae126e1d6c26c780c8728694699eb62339153e7a6a4f8873ad5c3a5d81bb4d2` |
| [Ground-aware lab](../artifacts/camera-ground-jog-lab-01/run/manifest.json) | 242 | `c53a576093cbd834b94a055d4f928e51db435816fdbdb1bb0fa30c6bdef876f9` |
| [Action replay and performance](../artifacts/camera-ground-jog-interaction-01/report.json) | 21 | `77598894959bc6a9b97c1536947b5ed39d2573a287dd2de6afba9b55e0b9e30b` |
| [World-coordinate geometry comparison](../artifacts/camera-ground-jog-geometry-01/report.json) | 5 | `997a68ae4cd8805a97b17a4816814399ca0f3113a712fb23d2e5370dce4d1338` |

At **October 28, 2026, 10:54 PM PDT**, Meta spending remains six settled calls, $0.166152 of $5, with no unsettled reservation. The broad reconstruction goal remains incomplete.
