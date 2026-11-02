# Explicit distal-image uncertainty — verified optional candidate

November 2, 2026, PST. Previous goal turn is **progress**: `012b54d` was
published and independently remote-verified, with 176 Python /72 mocked UI
checks and actual exact recording. Its optional source candidate still fails
one source and three interpolated floor checks. The full combined goal stays
active; this work does not replace coupled contact/control, learning, broader
video coverage or independent physical validation.

## Hypothesis and frozen method

The v1 distal fit uses normalized image residuals plus a fixed angle penalty.
Saved curvature evidence shows that this penalty can dominate the image signal.
Expose a separate opt-in v2 with explicit **assumed**, not calibrated, image and
angle error scales. Preserve v1 numerical behavior, stored contract and defaults.

For residual `r = pixels / max(image dimensions)`, use
`sum(confidence * r²) + (image_std / angle_std)² * correction²`, plus the
same ratio for the adjacent-correction prior when available. This is a scaled
least-squares objective; pose confidence is a heuristic weight, not a measured
probability. Report image-derivative strength relative to the original-angle
prior, including weak directions. Do not claim depth uncertainty is calibrated.

Keep the existing .30-rad correction, .10-rad update, 12-iteration/four-backtrack
and 20-second-per-body bounds. Preserve all views, source-pixel nonregression,
fixed upper axes/lengths/root/ground/forces, complete fits and zero independent
3D measurement confidence. No floor score, reference labels or contact target
may enter optimization or parameter selection.

If controlled qualification passes, freeze exactly one new ASPset development
candidate using assumed normalized image standard deviation **.0025**, original
angle standard deviation **.30 rad**, and adjacent-correction standard deviation
**.45 rad**. These are illustrative assumptions, not estimates from that clip.
No parameter sweep or candidate selection on floor results. Preserve every
failure and compare all image/floor slots to v1.

## Ownership, bounds and acceptance

Root owns the distal fitter/adapter, body constructor/CLI/lab glue and their
tests/docs. Preserve the independent avatar session's files and its `dbdd7e2`
checkpoint. Existing dependencies only; provider cost $0. No source acquisition,
video decode, inference, API/key/ledger access, server/browser or deployment.

- [x] Implement explicit settings, source custody, versioned validation and
  understandable lab labels, preserving all legacy methods.
- [x] Verify resized-image invariance, near/far and weak/zero derivative cases,
  temporal behavior, invalid settings, two-to-four-view integration and replay.
- [x] Freeze and run at most one new source construction after qualification.
  Separate native/interaction plans must bound time, memory, calls and output;
  pin source/runtime/settings before execution. Reuse existing source copies.
- [x] Audit all saved fits/slots independently, retain mixed results, verify
  actual recording and document exact evidence. Publication is verified
  separately in `.local/aspset-distal-uncertainty-publication-01.json`.

No root job is live at the start. Prior native/interaction supervisors are
terminal/reaped. Publication/state receipt:
`.local/aspset-distal-pixel-publication-01.json`. Do not repeat those runs.

## Qualification and native launch

**213 Python /73 mocked UI checks pass**, with no test corrections required.
Receipts: `.local/distal-uncertainty-tests-01.xml` and
`.local/distal-uncertainty-ui-tests-01.log`. Checks include exact legacy manual
recording, explicit v2 CLI recording, image resizing, near/far information,
zero initial twist derivative, temporal/exact targets, two-to-four-view native
upper-bone invariance, tampered assumptions and changed-input rejection.

`examples/partial-pixel-uncertainty.json` declares the preselected assumptions.
CLI `--partial-pixel-uncertainty PATH` requires all existing pixel-fit flags.
The new body method is `camera-body-partial-pixels-v2`; v1 contracts/numerical
branches remain unchanged. Source bytes and canonical settings hashes are
preserved separately. The lab labels error scales and weak directions as
assumptions, not measured confidence; no new geometry request or parsing.

Frozen controls: `.local/aspset-distal-uncertainty-lab-01`; output:
`artifacts/aspset-distal-uncertainty-body-lab-01`. `plan.json`/`launch.json`
pin committed runtime plus six owned overlays, original source/camera/ground
packages, exact settings and passed checks. Bound: 30 s external /25 s CPU,
768 MiB sampled RSS, six compiles/64 data allocations/1,000 static forwards,
zero dynamics/source decode/inference/API, 64 MiB per file and post-exit
400 MiB/3,000 files. Read its process receipt or live handle before resuming;
do not duplicate an existing run. Next, independently audit its saved result.

## Native completion and recording plan

Native construction completed/reaped: worker 1.707408 s /151,519,232 B peak RSS,
supervisor 1.765524 s, exit 0; four compiles, 53 allocations, 253 forwards and
zero dynamics. Run has 381 members, seal
`672234299b8645616a8fb2b4787591774aa4bea9e9623d3af7a6915bc5bdf449`.
All original inputs remain unchanged. The independent saved audit passes;
final metrics and seals will be consolidated after recording below.

Recording controls: `.local/aspset-distal-uncertainty-interaction-01`. The plan
pins the native runtime and candidate, 39 actions/one explicit reset, exact
replay and 60 in-process action/state/JSON calls. Bounds: 45 s external /35 s
CPU, 768 MiB sampled RSS, six compiles/16 allocations, 3,000 physics calls,
1,000 forwards, 64 video writes/256 verification reads, 64 MiB session output.
Only exported session video may be decoded; no original-source decode,
inference, network/API, browser/server or deployment. An explicit after-close
receipt must verify all run members and unchanged lifecycle metadata.

## Results and limits

The shared saved-only verifier now accepts `--uncertainty` for this pinned v2
candidate; its original mode retains the same pinned v1 inputs and criteria.
No floor evaluator, threshold, original observation or task gate changed.
`artifacts/aspset-distal-uncertainty-audit-01` checks all **1,139 manifest members**
across the transport baseline/v1/v2, 45 unchanged input files, 24 pixel fits,
259 trial losses, independent finite-difference image strength and 163 unchanged
complete-limb fits. Four-member seal:
`ccfc1361e4c0ccda6ff691abe0c2549c6b6a9691605c20ea69e50b0f493cbdff`.
Time 0.375934 s; no new candidate solve, native work, source decode or inference.

| Check | v1 fixed penalty | v2 explicit assumptions |
| --- | ---: | ---: |
| Failed source floor poses | 1/47 | 1/47 |
| Failed interpolated floor poses | 3/111 | 2/111 |
| Worst penetration, model units | 0.072313 | 0.049340 |
| Left-camera mean error, px (497 points) | 10.6286 | 10.4368 |
| Middle-camera mean error, px (561 points) | 6.5939 | 6.6740 |
| Right-camera mean error, px (562 points) | 6.2201 | 6.3521 |
| Aggregate weighted normalized pixel SSE | 0.00466618 | 0.00428583 |

All 6,768 image slots remain: 1,620 compared, 72 low-confidence/outside and
5,076 unsampled. **39 comparisons regress** (1 left, 18 middle, 20 right).
Six fits reach the unchanged .30-rad correction bound; 22/24 fits have at least
one image direction weaker than the assumed original-pose prior. This is a
local derivative diagnostic, not measured statistical confidence. The candidate
remains **infeasible/manual-only**. No default, task, policy or physical-accuracy
promotion. Do not describe this as a generalization result.

The revised verifier's legacy regression is
`artifacts/aspset-distal-pixel-audit-regression-01`, seal
`f838d691c9bd4195a144245dbdecb4115821791a0f5de32359106f7de7a60053`.
All v1 numerical report fields (excluding elapsed time), fit rows and image slots
equal the original audit exactly. This was a verifier-change check, not another
source construction or changed evaluator.

## Interaction and after-close integrity

Actual recording: 39 actions, one reset, 41 states/video frames, all states,
rewards and flags replayed exactly. Session
`artifacts/aspset-distal-uncertainty-body-lab-01/sessions/session-c5c1fedf9c4f`,
six-member seal
`2ebb335f74b36c06391c62060ff9076e3436abfe4e7502691c7bef32ee4b9ce8`.
Seven files total 1,912,456 B including the manifest. Worker 1.347172 s /
162,103,296 B peak RSS; 1.378744 s including close. Supervisor exit 0/reaped
in 1.471918 s. Two compiles/allocations, 145 forwards, 1,104 physics calls,
41 video writes and 42 verification reads. Median/p95/max over 60 in-process
action-state-JSON calls: 0.855/0.898/0.909 ms; not HTTP/browser/sustained-use QA.
`after-close.json` verifies all 381 run members and exact lifecycle hashes;
status remains `ready` / result `manual_candidate`.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-distal-uncertainty-body-lab-01
```

Use `controls`, `action`, `step`, `record start`, `record stop`, `files`; use
`lab --resume` instead to start the visual interface yourself. [Guide and
artifacts](../multi-view-body.md#explicit-image-and-pose-uncertainty).

## Geometry diagnosis and next safe action

`.local/aspset-distal-uncertainty-reachability-01.json` retains all 70 source rays
and both positive intersections with each fixed-knee, fixed-length endpoint
sphere. It uses the existing monotonic radial inverse, then the quadratic
`|camera_center + distance * ray - knee|² = length²`. Every retained root
reprojects within 1e-6 px and preserves length within 1e-10. No pose fitting,
floor-based branch selection, new observations or physics occurs.

**44/70 exact rays do not intersect the fixed-length sphere**; only 2/70 have an
exact branch within the current correction bounds. This is exact-pixel
inconsistency under the fixed geometry, not noise-qualified physical
impossibility. Camera/pose/length uncertainty remains material.

At frame 84, two exact ankle depths exist. Their twist/bend corrections are
`[1.150627, -0.227985]` and `[0.086425, 0.826546]` rad; both lie outside the
local .30-rad correction contract. Their endpoint centers are respectively
0.129197/0.096990 model units above the supplied plane; neither is selected
using that height. The executed v2 fit stops at `[0.184808, .30]`, leaves
27.39797 px prepared-camera error and retains one weak image direction.

Next: qualify explicit ray/sphere alternatives and consistent articulated
geometry under noisy observations, preserving depth ambiguity and temporal
evidence. Do not widen this method's bounds or sweep penalty values to fit the
floor result. Any next method needs its own bounded contract and controlled
qualification before a new source candidate. The complete combined goal still
requires coupled contact/control, meaningful independent learning, broader
unfamiliar-video coverage and physical evaluation.

All root jobs are terminal/reaped. No paid calls/key/ledger access; no deployment.
Frozen runtime is `dbdd7e2` plus six owned overlays, separate from dirty avatar
work. Preserve that session's files; publication receipt records refreshed Git
state, exact remote revision and scoped changes. Do not rerun unchanged checks.
