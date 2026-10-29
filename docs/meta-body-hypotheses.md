# Independent checks of Meta's body/contact hypotheses

The live model proposed three explanations. Local construction-only checks supported one limited simulator claim, corrected two factual details, and left the physical explanation unresolved. **No proposal was accepted or applied.** No new API calls, physics steps, training, model changes, reference lifting, floor adjustment, or success-threshold changes occurred.

Inputs: [retained proposal](../artifacts/meta-body-contact-live-02/hypotheses.json), original squat pose/root evidence, and the previously [sealed full-reference feasibility audit](../artifacts/whole-body-feasibility-01/report.json). Outputs: [sealed checks](../artifacts/meta-body-hypothesis-checks-01/report.json), [comparison plot](../artifacts/meta-body-hypothesis-checks-01/projection-check.png), and [check script](../scripts/studies/check_meta_body_hypotheses.py). The artifact includes input hashes, unchanged model/proposal copies, five exact source-frame PNGs, and a relative-path manifest verified across 12 files.

## Results

| Proposal | Outcome | Independent finding |
|---|---|---|
| `h_floor_anchor` | Supported within the model | Recomputed first-pose clearance and later lower-leg overlap match the sealed geometry audit. This does not identify the real physical cause. |
| `h_pelvis_projection` | Physical interpretation unidentified | The requested vertical `root_y` is wrong: vertical is `root_z`; `root_y` is authored zero depth. Root z matches the image-pelvis equation exactly, so their agreement is not independent validation of camera motion or real vertical displacement. |
| `h_support_load_unknown` | Physical support/load unidentified | Feet/soles are incompletely visible and barbell mass/forces/EMG are absent. However, the blanket “low-confidence ankles” wording is contradicted by the retained raw confidence values. |

## Floor construction check

The script constructs the original body, assigns the original reference angles/root positions, and calls forward geometry only. It computes each capsule/sphere's lowest surface and compares the result with the sealed audit. `mj_step` is replaced with a function that raises, preventing dynamics or policy actions.

| Source frame | Lowest collision surface z | Lowest shape |
|---|---:|---|
| 0 | +0.005000 | Left lower-leg capsule |
| 72 | −0.044565 | Right lower-leg capsule |
| 174 | −0.033117 | Left lower-leg capsule |

Values are model units relative to the unchanged floor at z=0. Negative means intersection; both later samples exceed the unchanged 0.025 penetration limit. Joint names containing `knee_geom` identify shin capsules extending from knee to ankle, not isolated knee points. Their original source IDs remain in every measurement.

The first-floor alignment and later overlaps are reproducible model facts. Scale error, camera geometry, pose error, actual foot support, and floor offset remain alternatives; this check does not distinguish them causally.

## Projection and joint-angle check

The corrected equation is:

`root_z = −(image_pelvis_y − first_image_pelvis_y) × 0.002117377354`

| Frame | Image pelvis y, pixels | Root z, model units | Left/right knee angles, radians |
|---|---:|---:|---:|
| 0 | 387.046 | 0.000000 | 0.292 / 0.504 |
| 72 | 576.071 | −0.400237 | 2.014 / 1.694 |
| 110 | 358.797 | +0.059815 | 0.128 / 0.148 |
| 212 | 363.894 | +0.049022 | 0.117 / 0.107 |

Recomputed root z has zero numerical discrepancy at these four retained samples; root y is zero throughout. The plot also retains both hip angles and floor overlap. These variables share their inference pipeline, so correlation cannot establish camera motion, calibrated depth, gravity alignment, or real contact. Lines in the plot connect only the four requested samples, not a newly measured trajectory.

## Visibility and missing measurement check

Visual inspection used only exact frames 0, 72, 110, 174, and 212. Each shows a barbell; complete sole-to-floor contacts are outside the lower image boundary or obscured. The video cannot establish heel lift, load distribution, or force magnitude from these views.

Across these frames, the minimum of each ankle's retained visibility/presence values ranges from **0.7176 to 0.9602**, above the existing 0.5 threshold. Thus low raw ankle confidence is not demonstrated here. Conversely, high model confidence does not make cropped feet into measured contact points.

The supplied evidence contains images, estimated landmarks/angles, inferred root references, and authored simulator parameters—not measured muscle electrical activity (EMG), force-plate readings, contact forces, or calibrated barbell mass. Simulated motor commands are not replacements for those measurements.

## What this establishes

This is a completed model-guided test loop: proposal → coordinate/evidence checks → local geometry calculations → supported, contradicted, and unresolved findings. It is **not a completed repair**, a physical validation, or a final held-out evaluation. The existing failed trajectories and trial scores remain unchanged; `accepted:false` and `applied:false` remain explicit in the result.
