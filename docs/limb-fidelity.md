# Limb-direction development check

This source-only check measures projected upper-arm, forearm, thigh, and shin
directions. It exposes missing lateral motion without fitting a camera or swapping
left/right labels to improve the candidate's score. It does not measure anatomical
accuracy, 3D depth, force, muscle activation, or dynamic balance.

## Frozen evidence

[Annotations](../data/limb-orientation/annotations.json) contain twelve source frames:

- Jumping jacks/burpees: 20, 30, 40, 60, 169, 180, 200, 330.
- Squat: 10, 50.
- Kettlebell pushup: 30, 120.

All eight directed limb-segment slots remain present in every frame: 96 observations,
84 scorable and 12 source-occluded. The assistant estimated visible joint centers
from original decoded frames before inspecting candidate limb outputs. These are
**not independent human-expert ground truth**. The clips are familiar development
examples, not representative samples or a final holdout.

Each endpoint records visual uncertainty and visibility. Missing source endpoints
remain null. Segment tolerances were fixed before scoring using
`degrees(atan2(sum(endpoint_uncertainty_pixels), segment_length_pixels))`, clamped
to 15–35°. These limits are heuristic allowances, not calibrated confidence intervals.

Anatomical left/right is inferred from the original view: left is image-right in
frontal jumping jacks, image-left in the rear-view squat, and near-camera in the
pushup. This interpretation is not certified anatomy. Never choose a different side
assignment after inspecting candidate scores.

Frozen annotation SHA-256:
`f051db850105a77e0350cdfd7128f84d340ad1a18dafcfeda9c6f4ab722290bd`.
Corrections require a new dataset version and a source-based explanation; do not
silently alter labels or tolerances to improve a candidate result.

## API

```python
from weft.limb_fidelity import load_annotations, grade_limb_projection

labels = load_annotations("data/limb-orientation/annotations.json", project_root=".")
predictions = [{
    "id": "jumping-jacks-burpees:20",
    "source_sha256": labels["frames"][0]["source_sha256"],
    "segments": {
        "left_upper_arm": {"axis_xy": [1.0, 0.0]},
        # Supply the other candidate axes and frames; missing predictions fail.
    },
}]
report = grade_limb_projection(labels, predictions)
```

Segments are `left_upper_arm`, `left_forearm`, `left_thigh`, `left_shin` and the
four `right_` equivalents. Directions are proximal to distal: shoulder→elbow,
elbow→wrist, hip→knee, knee→ankle. Each segment accepts `axis_xy` or
`start_xy`/`end_xy`, never both. A null or absent segment is a missing prediction.

Predictions must use original-image x-right/y-down directions. Convert normalized
image coordinates using source width/height before computing axes; treating the two
normalized coordinates as equal-length units distorts directions. A simulation using
camera `(x,y,z)` → world `(x,z,-y)` can project a world-space segment delta as
`[delta_x, -delta_z]`. Do not optimize projection, scale anisotropically, or fit a
camera against these labels inside the grader. Uniform scale and translation do not
change the angular score. The comparison is directed: reversing endpoints gives 180°.

## Coverage and failures

The report includes overall, per-clip, and per-segment summaries, plus all 96 original
observation slots. Source-unavailable observations are not successes. Missing candidate
predictions and zero-length projected axes are explicit failures against the 84
source-scorable slots; neither disappears from the pass-fraction denominator. Mean
angular error includes only finite, nonzero scored predictions and must be reported
alongside coverage. Malformed/nonfinite inputs, duplicate/unknown frame IDs, and
source checksum mismatches are rejected.

Verification: `.venv-lab/bin/python -m pytest tests/test_limb_fidelity.py` — 12 passed.
All three source video checksums and twelve original-frame checksums matched.
No model/API calls or candidate predictions were used to author this benchmark.

## First candidate result

The final spatial-body pipeline scores 60/84 source-visible segments within tolerance versus 26/84 for the legacy model. Mean error on 80 scored segments improves 40.69°→14.09°; four missing predictions remain failures. The worst 174.77° error is an upstream anatomical identity swap, preserved rather than corrected in the labels. [Full evidence and limitations](spatial-human.md).
