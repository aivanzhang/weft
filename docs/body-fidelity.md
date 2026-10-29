# Body-orientation fidelity check

This development check measures one visible defect: whether the simulated body's
hip-to-shoulder direction matches approximate torso axes in source video frames.
It does not establish valid anatomy, biomechanics, force, balance, or muscle activity.

## Evidence boundary

[Nine frozen labels](../data/body-orientation/annotations.json) cover three squat,
three single-leg squat, and three horizontal kettlebell-pushup frames. The assistant
visually inspected original decoded frames and selected approximate centers before
inspecting candidate pose predictions. Labels are independent of candidate outputs,
but **not independent human-expert annotations**. Clothing, rear/side views, and
occlusion make centers uncertain. All axes use a preset 15° tolerance. Individual
endpoint uncertainty is recorded separately; it is not a measured confidence interval.

The frames are purposively selected familiar development examples. Do not tune a
candidate against these results and then call them final-holdout accuracy. Preserve
the annotation file and its checksum; label corrections require a new benchmark
version and an explicit reason unrelated to candidate performance.

## Grader

```python
from weft.body_fidelity import load_annotations, grade_torso_projection

labels = load_annotations("data/body-orientation/annotations.json", project_root=".")
predictions = [{
    "id": frame["id"],
    "source_sha256": frame["source_sha256"],
    "axis_xy": [0, -1],  # Analytic legacy upright-body baseline.
} for frame in labels["frames"]]
report = grade_torso_projection(labels, predictions)
```

Prediction axes must already be in original-image x-right/y-down coordinates.
For a body rotation whose columns are local right/forward/up and simulation mapping
is camera `(x, y, z)` → simulation `(x, z, -y)`, its local-up source projection is
`[R[0, 2], -R[2, 2]]`. This assumes an orthographic directional projection; it is not
a recovered camera calibration. Never optimize camera yaw or an image transform
inside the grader to match the labels.

The directed angular error is between 0° and 180°; an upside-down torso fails.
Translation and uniform scaling do not affect angle. Normalized-image coordinates
must be converted to original pixels first: different image width and height would
otherwise distort the direction. Optional `hip_xy` and `shoulder_xy` predictions
replace `axis_xy` and also report mean endpoint distance divided by labeled torso
length. That secondary metric requires a real, independently specified position
and scale mapping; axis-only predictions do not pretend to provide one.

Missing frames remain explicit failures, not dropped successes. Source checksum
mismatches, duplicate/unknown prediction IDs, nonfinite values, and zero axes are
rejected. The result records the canonical annotation hash, frame-level errors,
per-clip summaries, and total coverage. Source verification optionally checks all
original video and extracted image checksums.

## Baseline evidence

`artifacts/body-orientation-baseline-01/` preserves the original labels, grader,
predictions, hashes, and results. Its baseline is the analytic image-plane axis
`[0, -1]`, not a rendered or dynamically calibrated body.

| Clip | Mean error | Frames within 15° |
| --- | ---: | ---: |
| squat | 8.45° | 3 / 3 |
| single-leg squat | 36.18° | 0 / 3 |
| kettlebell pushup | 68.85° | 0 / 3 |
| All | 37.83° | 3 / 9 |

This confirms a near-horizontal source torso cannot be represented by an always-
upright root. A fixed mean orientation can improve this defect while still missing
real torso movement across frames. Passing this narrow axis check is not passing
full-body reconstruction.

Verification: `.venv-lab/bin/python -m pytest tests/test_body_fidelity.py` — 13 passed.
All three local video hashes and nine extracted frame hashes matched at baseline
creation. No model/API calls were used to author or grade these labels.

## Anchored orientation candidate

`body_frame.py` estimates torso up from shoulder/hip midpoints and lateral direction
from the shoulders. It maps model-camera coordinates to simulation axes, then fits
one proper mean rotation. This uses pose evidence only, not the frozen scoring labels.
`body-frame.json` retains per-frame rotations, rejected evidence and fit residuals.
The inferred rotation changes the actual MuJoCo body frame, including gravity loading;
it is not a renderer-only rotation. Old saved models without this field remain upright.
MuJoCo's [body-frame conventions](https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst)
define the local-to-parent geometry used here.

The direct study `artifacts/body-frame-fit-01/` reused the baseline pose evidence:

| Clip | Upright mean error | Fitted mean error |
| --- | ---: | ---: |
| squat | 8.45° | 4.80° |
| single-leg squat | 36.18° | 11.94° |
| kettlebell pushup | 68.85° | 8.50° |

Eight of nine frozen axes fall within 15° (previously three); the remaining single-leg
frame is 18.98° off. All 144 joint/action trials stayed finite. Push-up reference-servo
RMSE worsened from about 0.168 to 0.218 radians under changed gravity loading; the
visual improvement is not an across-the-board dynamics improvement.

The root still has **one fixed mean orientation and position**. Camera up is assumed
to align with gravity; camera roll, perspective, scale, floor/kettlebell support,
translation, out-of-plane joints and muscle force are not calibrated. A body can now
tilt while its limbs or contacts remain wrong. Fresh full-pipeline results and
recording checks are recorded in [follow-up verification](iteration-followup.md).
