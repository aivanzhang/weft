# Distal image constraints — verified optional candidate

November 2, 2026, PST. Previous goal turn is **progress**: `0d90496` published
partial multi-view transport, 123 Python/71 mocked UI checks, actual recording,
and a saved mixed result. Root-owned paths/index/stashes were clean; avatar work
remains owned by its separate session. The full combined goal remains active.

## Hypothesis frozen before implementation

An unavailable 3D wrist/ankle can still have useful image evidence. The partial
candidate at frame 84 has a supported upper leg and one middle-camera ankle
observation, but retains an inconsistent distal bend. Add an opt-in bounded
pixel proposal after partial transport, using all qualified original distal
images without inventing a 3D observation or changing complete-limb fitting.

Keep the proximal world origin/direction, torso, root, both lengths, floor and
forces fixed. Two variables: axial rotation around the fixed upper bone and
distal bend. Each correction is limited to .30 rad, each update to .10 rad;
bend stays in [0, 2.6]. Use at most 12 iterations/four backtracks, original-angle
regularization 1e-3 and adjacent-correction regularization 5e-4 across gaps ≤.25 s.
Residuals are original pixel distances divided by each view's maximum dimension,
weighted by its minimum visibility/presence. Preserve every source view/status.
Reject nonfinite/out-of-envelope projections and source-objective regression.
No floor, reference annotation, task score or contact label enters the fit.

Reuse the existing radial pinhole projection/Jacobian and explicit ≤1e-5
rotation-roundoff preparation. Preserve original calibrations and derived
rotations. All resulting depth/twist/bend remain inferred, even with a rank-two
local derivative. This is not an anatomy/contact or generalization claim.

## Work and bounds

- [x] Implement the two-angle numerical fit and qualify analytic derivatives,
  native upper-bone invariance, bounds, confidence exclusions and rejection.
- [x] Integrate versioned source custody/validation, explicit CLI options, lab
  uncertainty labels and exact recording/replay. Preserve legacy methods.
- [x] Freeze one real-source candidate with unchanged left/mid/right evidence,
  manual ground, lengths and source slots; inspect all camera and floor results,
  retaining failures. Run no unchanged candidate again.
- [x] Verify actual manual interaction/export/replay and update TODO/handoff.

Publication is a separate verified receipt, not inferred from local completion:
`.local/aspset-distal-pixel-publication-01.json` pins the exact remote revision.

Existing dependencies; provider cost $0. No acquisition, decoding, inference,
API/key/ledger access, server/browser or deployment. Pure fit is capped at
20 seconds per body. Before native execution, pin runtime/input identities and
30-second external/25-second CPU/768-MiB sampled-RSS bounds, six compiles,
64 allocations/1,000 static forwards, zero dynamics, 400 MiB/3,000 output files.
Recording will have its own bounded plan and owned output.

## Implementation and native launch

`distal_pixel_fit.py` is a pure two-variable numerical fit; the source-bound
adapter is `multiview_distal_pixels.py`. Explicit CLI flag `--fit-partial-pixels`
requires `--fit-limbs --partial-segments` and direction-enabled construction.
New method `camera-body-partial-pixels-v1` binds original camera hashes, all
per-view statuses, normalized source objectives, retained priors and changes.
Every affected joint remains confidence-zero for independent 3D measurement.
Complete fits and upper-bone native positions remain unchanged across 2–4 views.

Scoped union: **176 passing Python cases /72 mocked UI checks**. Receipts:
`.local/distal-pixel-core-tests-01.xml`, `distal-pixel-integration-tests-01.xml`,
`distal-pixel-fixture-tests-01.xml`, `distal-pixel-ui-tests-01.log`.
Initial failures were test-fixture issues, preserved in the first two receipts:
changing primary evidence after binding correctly rejected custody; after
rebuilding, exact synthetic geometry correctly produced zero corrections.
The fixture now binds its full input first and supplies an explicit biased
monocular depth prior against unchanged known image targets. No production
guard, evaluator threshold or fitted correction bound was relaxed.

Native controls/output: `.local/aspset-distal-pixel-lab-01` and
`artifacts/aspset-distal-pixel-body-lab-01`. Frozen committed `0d90496` plus
seven root-owned runtime overlays; no dirty avatar files. `plan.json` and
`launch.json` pin all 311 runtime members, exact original input packages and
the worker. Same native bounds above, adding only the new pixel flag to the
prior frozen construction. The supervisor records terminal/reaped state; check
that receipt or live handle before resuming, never duplicate this job.
The saved audit below checks the candidate without changing its source inputs,
corrections or floor limit in response to the result.

## Native completion and recording plan

New run has 380 members, seal
`da5aca806abd17cf9e8c09cc895e0134151cc4530a900e72ba0f25656cbe0909`.
Worker 1.653624 s /151,388,160 B RSS; supervisor 1.743881 s, exit 0/reaped.
Four compiles, 53 data allocations and 253 static forwards; zero dynamics,
source decode, inference or API. Output 383 files /43,810,060 B including metadata.

Independent recording uses `.local/aspset-distal-pixel-interaction-01` with the
same frozen runtime and candidate. Bound to 39 actions/one explicit reset,
exact replay and 60 in-process action/state/JSON calls; 45 s external /35 s CPU,
768 MiB sampled RSS, six compiles/16 allocations, 3,000 physics calls/1,000
forwards, 64 video writes/256 verification reads, 64 MiB session output. No
source decode, inference, network/API, browser or server. Preserve immutable
run and lifecycle metadata. Read its process receipt before any resumption.

## Saved audit and actual interaction — November 2, 2026, 1:06 AM PST

The saved-only audit in `artifacts/aspset-distal-pixel-audit-01` passed in
0.274759 s. Its four-member seal is
`07e0d038fb3e134dbeef889287476609db439bc0c3eb5d1eee1e9eeaad22e054`.
`scripts/studies/verify_aspset_distal_pixels.py` independently checks 758 sealed
baseline/candidate members, 45 identical input files, all 24 pixel fits and 214
retained trial losses. All 163 complete-limb fits and supported upper-bone
positions remain unchanged. The original 188 source slots, 47 body poses and
6,768 image slots survive: 1,620 compared, 72 low-confidence/outside, 5,076
unsampled. No new solve, native physics, source decode or inference in this audit.

| Check | Partial directions | Added distal pixels |
| --- | ---: | ---: |
| Failed source floor poses | 1/47 | 1/47 |
| Failed interpolated floor poses | 3/111 | 3/111 |
| Worst penetration, model units | 0.077483 | 0.072313 |
| Left-camera mean error, px (497 points) | 10.6642 | 10.6286 |
| Middle-camera mean error, px (561 points) | 6.6110 | 6.5939 |
| Right-camera mean error, px (562 points) | 6.1971 | 6.2201 |

The aggregate weighted normalized pixel SSE falls 0.00480541→0.00466618, but
**37 individual comparisons regress** (3 left, 14 middle, 20 right). All remain
in `source-images.json`; source agreement is not independent physical accuracy.
The unchanged 0.025 floor limit still fails at source frame 84 and grid times
1.633333, 1.666667 and 1.7 s. The candidate remains **infeasible/manual-only**;
no default, learned-policy, contact or physical-validation promotion.

The recording supervisor exited 0 and reaped its child in 1.121941 s. Worker
time 1.035472 s /161,939,456 B peak RSS, two compiles/data allocations, 145 static
forwards and 1,104 physics calls. It records 39 actions, one explicit reset and
41 states/video frames; all states, rewards and flags replay exactly. Session:
`artifacts/aspset-distal-pixel-body-lab-01/sessions/session-9f4a8ded64d7`, six-member
seal `6ef806a2c4215fb478f366d20362d8d9c0a4ed8bffc4cd8938a9a2809a632c7d`.
Seven session files total 1,911,115 B including the manifest. Median/p95/max
action-state-JSON latency over 60 in-process calls: 0.846/0.900/0.910 ms;
this is not HTTP, browser painting or sustained-use verification. Metadata
was unchanged before close. Publication must also check the 380-member run
after close and preserve lifecycle `ready` / result `manual_candidate`.

Try it without starting a server:

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-distal-pixel-body-lab-01
```

Enter `controls`, `action`, `step`, `record start`, `record stop`, and `files`.
Use `lab --resume` instead to start the visual interface yourself. See the
[guide](../multi-view-body.md#optional-distal-image-constraints) for artifacts.

## Resume guard and next work

Both supervisors are terminal/reaped; do not duplicate either run. Native and
interaction plans/workers/receipts are in `.local/aspset-distal-pixel-lab-01`
and `.local/aspset-distal-pixel-interaction-01`. Runtime remains committed
`0d90496` plus seven hashed root overlays. The independent avatar checkpoint
`5783a36` arrived on local main during this work; its dirty follow-up files are
not part of this checkpoint. Preserve them. Index/stashes were empty at
resumption; refresh before publication. Publication identity and after-close
verification belong in `.local/aspset-distal-pixel-publication-01.json`.
No paid calls or key/ledger access; the shared $5 authorization is unchanged.

The saved-only `.local/aspset-distal-pixel-curvature-01.json` diagnoses all 24
fits without another solve. At frame 84, normalized image-derivative curvature
eigenvalues are 5.64134e-6 and 3.55971e-4; the original-angle prior is 1e-3,
2.809 times the largest image curvature. The accepted corrections are
0.101946/0.095389 rad; prepared-camera ankle error falls 47.0702→39.1188 px.
This suggests the prior may overpower the image constraint, not that depth is
identified or a weaker prior is valid. Next, qualify dimensionally explicit
image/angle uncertainty and weak-direction behavior on controlled examples
before freezing any new source candidate. Keep correction bounds, unknown
depth, all rejected attempts and the independent floor scorer unchanged.
Do not tune on floor results or repeat this unchanged candidate.

Full coupled contact/control, useful independent learning, unfamiliar-video
coverage and physical validation remain open. The overall goal is active.
