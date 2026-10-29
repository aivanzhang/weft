# Temporal identity development evidence

Verified **October 28, 2026, 1:47 PM PDT**. Explicit temporal correction is implemented and tested; anatomical identity remains an inference.

## Try the corrected environment

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/temporal-identity-e2e-05/jumping-jacks-burpees

# A new local clip; use a new output directory.
.venv-lab/bin/python -m weft lab --video /path/to/exercise.mp4 \
  --profile exercise --body-model spatial --pose-identity temporal \
  --out artifacts/my-temporal-exercise
```

The saved jumping-jack case uses explicit `--pose-mode image`; new clips default to `video`. Neither inference mode guarantees person count. `--pose-identity raw` remains the default. In the lab, inspect **Checks & limits** and **Files** for identity assumptions, `identity-audit.json`, corrected `evidence.json`, and unchanged raw `pose-landmarks.json`.

## Implemented correction and limits

`pose_identity.py` checks isolated bilateral swaps of at most three sampled poses over 0.20 seconds, with two original supporting poses on each side. It requires high-confidence torso/limb points, stable supporting torso geometry and a strong reduction in continuity error. It does not sort anatomy by screen side. Multiple poses, long gaps, uncertain torso geometry and conflicting correction candidates remain unresolved.

Corrections permute both image and world landmarks, preserve the full original sample under `identity_raw`, and recompute angles/confidences before body construction. Derived identities have explicit provenance IDs. The audit records the exact mapping, thresholds, evidence IDs, decisions and source/module hashes. No person-count or visibility gate is relaxed. The lab labels corrections as neighboring-pose inference, not verified anatomy.

Thresholds and 21 analytic tests were frozen before real candidate scoring. Controls cover injected flips, genuine smooth turns, crossed arms, occlusion, long gaps and multiple poses. Initial/global left/right identity, long errors and fast real motion indistinguishable from a detector flip remain unresolved. Normalized-image continuity is a heuristic, not calibrated geometry.

## Final results

`artifacts/temporal-identity-e2e-05/` reproduces inference, reconstruction, learning, actions and recording for five real sources in **51.156 seconds**, with at most two workers. Four cases pass unchanged gates; lunge/twist remains a desired-case execution failure for competing pose hypotheses. Each successful case has 12 recorded transitions and 13 fully decoded video frames, with unchanged run manifests. Study seal: 434 files, `a4527f97653f6ac2cd10d1e16eacc27c7f686173241ee746a3ff1d7c1f10af3e`.

Only jumping-jack source frames **330 and 332** are reassigned. The other four clips are unchanged. Servo RMSE remains within the 0.35-rad gate: single-leg 0.120, squat 0.081, pushup 0.150, jumping jacks 0.233. Jumping-jack servo error is slightly higher than the prior raw-reference 0.231; following an inferred reference is distinct from matching source anatomy. One-target RL results are unchanged and do not establish whole-exercise learning.

| Frozen source check | Raw spatial model | Temporal candidate |
| --- | ---: | ---: |
| Shoulder/hip directions within tolerance | 30/39 | 36/39 |
| Mean shoulder/hip error | 34.56° | 7.64° |
| Limb directions within tolerance | 60/84 | 67/84 |
| Mean limb error, 80 scored segments | 14.09° | 7.30° |
| Maximum limb error | 174.77° | 32.35° |

All 39 observable shoulder/hip pairs are scored; 15 source-unobservable pairs remain separate and unchanged. There are no new pair failures. Six newly passing pair observations reuse **two** corrected sampled poses: source frame 331 maps to sampled 330 within one source-frame interval. These are not six independent corrections. Four visible limb predictions remain missing; 17/84 visible segments still do not pass. No annotations, tolerances or camera alignment changed.

Final pair study: `artifacts/temporal-identity-score-final-02/`, three-file seal `d4b285d9212136c3be867a4f28b7e859f28ce8e64a78064d384b953e0c11e622`. Final limb study: `artifacts/limb-temporal-evaluation-03/`, 14-file seal `f4439e118dca45c1b7f6b964da48b237f6d62266022d0bc65f1ea05766281eca`. Direct saved-evidence exploration remains under `artifacts/temporal-identity-direct-01/`; final claims above use the full rerun.

**347 Python tests passed**, no failures/errors/skips (`.local/temporal-full-tests.xml`); 15 deterministic UI checks passed (`.local/temporal-ui-tests.log`). Actual browser/device interaction remains unverified. No paid API calls were made. The $5 Meta ledger remains configured with $0 committed and no project key at this observation.

## Source-check design

This benchmark checks whether left/right shoulder and hip assignments agree with
source-visible pair directions. It distinguishes an exposed jumping-jack diagnosis
from source-only negative challenges involving real torso turns, moving overhead
arms, profile overlap, and partial occlusion.

## Frozen source evidence

[The manifest](../data/pose-identity-labels/annotations.json) contains 27 visually
inspected original frames from four clips. It was frozen before inspecting new
identity-candidate outputs. No raw detector outputs were inspected for the new
negative windows. Labels are assistant visual estimates, **not independent expert
anatomical ground truth**.

| Window | Source frame indices | Source challenge |
| --- | --- | --- |
| Jumping-jack diagnosis | 328, 330, 331, 332, 334, 336 | Front-facing subject keeps continuous side order; frame-330 diagnosis already exposed |
| Overhead moving arms | 40, 41, 42 | Hands approach above the head without a torso turn |
| Profile jumping jack | 750, 751, 752 | Shoulder and hip pairs have overlapping/hidden far sides |
| Lunge before turn | 200, 201, 202 | Three-quarter frontal view, right arm loaded overhead |
| Lunge real turn | 300, 301, 302 | Genuine torso rotation changes projected shoulder direction and separation |
| Lunge return | 450, 451, 452 | Returns toward frontal view while the right arm remains loaded |
| Single-leg squat | 100, 101, 102 | Profile view with obscured far-side joints |
| Pushup | 120, 121, 122 | Partially visible shoulders, hidden far hip |

The loaded lunge arm provides source-visible continuity through the turn; a changing
projected pair angle is not itself evidence of an identity error. Negative challenge
does **not** certify that the unseen raw detector is correct. These frames can still
contain detector mistakes; judge predictions against source labels, not a blanket
"never correct" label.

Some source frames already have prior torso/limb labels, and all clips have development
history. New windows were withheld from this candidate inspection only. They are not
a final holdout or evidence of generalized identity accuracy. Detailed exposure notes
are stored per window.

## Scoring contract

Every frame includes:

- Original source/video and decoded-image hashes, frame index, timestamp, width/height.
- Approximate `left_shoulder`, `right_shoulder`, `left_hip`, and `right_hip` pixel centers
  with uncertainty or explicit null for unobservable joints.
- Source-facing direction and shoulder/hip pair visibility.
- `axis_xy`: anatomical **left→right**, calculated as right minus left, in original
  image pixels with x right and y down, including letterbox bars.
- Fixed angular tolerance and uncertainty-aware horizontal pair-order evidence.

There are **39 observable pair directions and 15 unobservable pair observations**.
Retain all 54 observations in the result. Unobservable source pairs cannot establish
that a correction or unchanged output is accurate; report their action count and
coverage separately. Missing predictions on observable pairs must remain failures or
explicit unresolved observations, not disappear from the denominator.

Tolerances were fixed source-only before candidate evaluation:
`degrees(atan2(sum(endpoint uncertainty), projected pair length))`, clamped to 20–45°.
This is a heuristic allowance, not an expert confidence interval. Horizontal order
is unresolved when horizontal separation does not exceed summed pixel uncertainty.

Compare raw and corrected **directed** pair axes against the same frozen labels.
An actual label reversal changes the direction by 180°. Do not optimize the camera,
permute sides, alter tolerances, or relabel ambiguous points to improve a result.
Convert normalized coordinates using the original image width and height before
computing directions; equal normalized units distort axes in nonsquare images.

Report per-window and overall observable coverage, raw/corrected angular errors,
tolerance passes, correction counts, and changes on source-unobservable pairs.
Preserving a genuine turn and abstaining under ambiguous visibility matter alongside
correcting the already known diagnostic. This small source check does not validate
anatomy, depth, biomechanics, muscle activation, or long-term temporal identity.

## Integrity verification

Frozen annotation SHA-256:
`89f70a41d934989343f7836bb224208484164f9781e0a4c82b8353124be4b120`.

All four original video hashes and 27 extracted-frame hashes matched. Checked unique
frame IDs, original-image point bounds, all 54 visibility records, left→right axis
arithmetic, and tolerance limits. Selected marked overlays were visually checked
against their original images. No paid or model calls were used to create these source
labels. Candidate implementation and evaluation are separately recorded above.
