# Appearance tracking beyond colored masks

`weft.generic_vision` adds explicit ROI (rectangle) initialization and grayscale normalized template matching. It does not require a saturated color and does not identify objects automatically. The original template stays immutable: a failed match cannot silently become a new identity.

New runs also retain source-region color evidence when available. A grayscale shape match with incompatible color is rejected as `appearance_changed`, with no accepted position. Achromatic sources remain usable, explicitly without color evidence. This is an appearance check, not a visibility detector or proof of identity; same-color objects, changing illumination and occlusion remain limitations. [Measured collision follow-up](appearance-guards.md).

## Run and inspect

```sh
.venv-lab/bin/python -m weft.generic_vision \
  --video data/balls/bouncing-ball.webm \
  --config data/general/annotations/electrostatic.initialization.json \
  --annotations data/general/annotations/electrostatic.manual.json \
  --out artifacts/my-new-appearance-run
```

Output directories must be new. Inspect `tracks.gif`, `tracks-contact-sheet.png`, `appearance-tracks.json`, and `manual-evaluation.json`. Each frame/object has a status, correlation score, second-peak separation, source-based evidence ID, and either an accepted rectangle or no coordinates. Scores are similarity measures, **not calibrated probabilities**. `initialized` is an authored region, not a measured detection; it is excluded from downstream measurements.

The source hash pins the initialization. ROI coordinates use original source pixels; output coordinates use the declared bounded analysis dimensions (maximum dimension 960). Position means the center of the authored rectangle translated by an appearance match, not a segmentation-derived center of mass. The box size never establishes a measured physical shape.

The search radius is bounded, ambiguity rejects competing peaks, and overlap rejects both conflicting object assignments. A short absence may recover within the local search; after the configured gap the track stays `lost` until a new authored run. This prevents automatic long-gap identity guessing. Blur, rotation, scale change, specular appearance change, repeated textures, poor initial boxes, and fast motion remain material limitations.

Longer files can use an explicit [source-pinned interval](source-intervals.md). The full original is retained, source frames stay absolute, and earlier frames before object initialization remain in coverage checks. This does not expand the tracker into automatic object discovery or make difficult collisions physically reconstructed.

An opt-in `"association_mode": "joint"` assigns competing template matches across objects and retains rejected hypotheses separately from measurements. The default stays `independent`. [Method, frozen comparison and unresolved failures](joint-tracking.md).

## Simulation adapter

The local sandbox and benchmark runner support `--profile appearance`. The region
configuration is required; no automatic semantic object discovery is implied:

```sh
.venv-lab/bin/python -m weft lab \
  --video data/general/rolling-ball-grayscale.avi --profile appearance \
  --config data/general/annotations/rolling-grayscale.scene.json \
  --out artifacts/my-grayscale-lab
```

This example retains the entire original video but reconstructs only the explicitly
selected first visible pass (frames 39–60). It is a grayscale derivative of existing
real evidence, **not a new independent physical trial**. Its second pass is excluded.

`inspect_generic(video, out, config, progress=None)` returns the existing planar `evidence, scene` pair for `scene_vision.fit_scene` and `SceneEnv`. `out` already exists. Configuration:

```json
{
  "tracking": {
    "schema_version": 1,
    "source_sha256": "<64-character source hash>",
    "start_frame": 0,
    "end_frame": 119,
    "objects": [{
      "id": "object", "label": "Selected object", "box_xywh": [50, 80, 30, 30],
      "role": "active", "shape": "circle", "provenance": "User-selected source-frame rectangle"
    }]
  },
  "world": {
    "controlled_id": "object", "target": [0.75, 0.5],
    "gravity": [0, 0], "damping": 1.0, "restitution": 0.5
  },
  "source_label": "User video; world properties are authored hypotheses"
}
```

The adapter requires explicit roles, shapes, world values and target; these are **assumptions**, not recovered physics. It rejects detected camera/background motion and joint tracking coverage below 80% of the authored interval. It preserves full decode audit, all visibility states and [full-interval coverage](full-interval-reconstruction.md). New runs no longer select only the first episode or cap replay at six seconds: one continuous simulation retains the complete declared suffix through gaps, with unavailable measurements unscored. Before common model initialization, no simulated state is invented. Construction and development split by source time, not successful-detection count. Old runs retain their original subset semantics. No manual annotation enters construction or fitting. Bounded damping and [early initial-velocity proposals](initial-motion.md), checked by the unchanged per-object reconstruction scorer, determine replay eligibility; tracking alone cannot mark a world ready.

A negative background-motion screen does not prove a fixed camera. Gravity, camera calibration, mass, depth, spring/electrostatic/contact forces, hidden shapes, task intent, and physical transfer are not recovered. Circle/box collision geometry comes from the initial rectangle; angular state is unknown. Interactive controls are authored accelerations and braking, not inferred actuators.

## Recorded development evidence

Baseline: `artifacts/appearance-tracking-01/`, two real clips; no paid calls. Small manual labels were saved before tracking. Full decode was audited, including the electrostatic source's WebM metadata discrepancy (661 advertised, 656 encoded/decoded).

| Source | Accepted tracks | Manual-label results | Outcome |
| --- | --- | --- | --- |
| Electrostatic scene | 199/342 frame-object rows, including 2 authored initializations in denominator | Charging sphere: 7/7 visible labels, 0.7–2.9 px center error. Hand-held sphere: 1/4 visible labels, 2.1 px accepted error; 3 absent labels have no position. | Broad background movement; not fixed-camera physics. Later hand-held reentry remains lost. Tiny suspended pellet untracked. |
| Elastic pendulum | 1/504 rows, including 1 initialization in denominator | 0/11 visible labels accepted; no error metric fabricated | Blur/appearance change loses the weight; broad background motion also detected. Unsupported, not a reconstruction success. |

Measured decode/tracking time in the concurrent baseline: approximately 0.37 seconds and 0.73 seconds respectively; preview serialization is outside that timer. This is one local measurement, not a sustained responsiveness guarantee.

Ten deterministic tests cover grayscale translation and brightness shift, short occlusion recovery, permanent long-gap loss, repeated appearance ambiguity, texture rejection, full decode/source pinning, camera-motion mismatch, explicit missing-label/false-visibility scoring, and a grayscale analytic clip through scene construction, unchanged reconstruction scoring, and interactive stepping. Synthetic success is software verification, not independent physical accuracy.

Template matching reference: [OpenCV tutorial](https://docs.opencv.org/4.13.0/d4/dc6/tutorial_py_template_matching.html). Source/license details: [development data notes](../data/general/README.md).

## Metallic-pendulum follow-up: candidate not promoted

A separate `generic_flow_prototype.py` tested immutable rotation/scale/blur variants plus affine Lucas–Kanade optical flow (image-feature motion) as a **search proposal only**. Every accepted position still required the original 0.7 appearance score and 0.06 separation from competing peaks. Missing positions, permanent long-gap loss, overlap rejection, and the existing camera-motion screen remain enforced. The ellipse support is an authored compact-object assumption, not recovered segmentation. No model/API/download was used.

The exact parameters and assumptions were saved before evaluation in `artifacts/appearance-flow-pendulum-01/predeclared-design.json`; the existing 11 manual labels stayed unchanged and never entered construction. Results, sources, code, and checksums are retained alongside `comparison.json`.

| Check | Immutable translation baseline | Rotation/blur + flow prototype |
| --- | --- | --- |
| Accepted frames, excluding initialization | 1 | 14 |
| Frozen visible evaluation labels accepted | 0/11 | 0/11 |
| Accepted-label position error | Undefined: no accepted labels | Undefined: no accepted labels |
| Missing evaluation labels | 11 | 11 |
| Real absent-label test coverage | None | None |
| Measured decode/tracking time | ~0.73 s | ~4.04 s |
| Fixed-camera physics eligibility | Rejected | Rejected |

The baseline's frame-2 similarity drops to 0.60, below its 0.7 gate. The candidate sustains frames 1–13 and briefly recovers frame 20. At the direction change, blur, reflections, apparent size, and competing matches reduce ambiguity separation below 0.06; flow features also fail the forward/backward consistency check. Frame 25 is rejected as ambiguous and frame 30 onward remains lost. More accepted **unlabelled** early frames do not establish improved measured accuracy.

The frozen real labels all show the object, so zero false-visible labels does **not** establish real occlusion accuracy. Six analytic tests cover rotated/blurred grayscale tracking with immutable references, duplicate appearance rejection, complete occlusion and long loss, no flow-only published detections, invalid initialization, and preserved camera-motion ineligibility. These software tests are not physical validation.

Decision: keep this as an isolated experiment, not the default tracker or a successful pendulum reconstruction. Prototype API: `track_prototype(video, new_output_directory, tracking_config)` returns explicit per-frame motion evidence with flow diagnostics and `physics_ready=False`. It is not connected to the lab or simulation adapter. The 4.04-second timing includes a permanently lost tracker for most frames and excludes preview writing; it does not demonstrate sustained tracking speed.
