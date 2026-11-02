# Partial multi-view limbs — verified checkpoint

November 2, 2026, PST. Root owns reconstruction, lab integration and this note;
preserve the independent avatar edits. Published predecessor `aa1b0b4` is verified
on remote main; `.local/aspset-fixed-limb-publication-01.json` retains the receipt.
The full combined goal remains active, not completed by this development clip.

## Hypothesis and boundaries

The fixed-length candidate retains four floor failures between frames 80–88.
At frame 84, multiple cameras support the left hip and knee but not the ankle;
discarding the supported upper segment leaves an inconsistent whole-leg prior.
Qualify the existing minimum-rotation upper-segment transport for two to four
cameras and its composition with fixed-length fitting. Use each joint's actual
contributing cameras, not every camera indiscriminately. Preserve the missing
endpoint, original distal bend and uncertain axial twist; do not invent contact.

Complete limbs must remain exactly equal to the corresponding fitted/unfitted
constructor. Partial transport runs afterward and cannot affect its temporal
fit history. Root, floor, dimensions, forces, source slots and evaluation limits
remain unchanged. Old methods retain their versioned behavior. New versions
must validate contributor sets and uncertain components, with native controls,
CLI/record/replay checks and clear compact lab labels.

## Work and verification

- [x] Implement versioned multi-camera partial transport and fitted composition.
- [x] Verify native proximal directions, retained unobserved bends, unchanged
  complete limbs, rejected provenance claims, old methods and manual recordings.
- [x] Freeze one new candidate using the exact existing left/mid/right evidence,
  manual ground and fixed-length settings, adding only `--partial-segments`.
  Bound native execution to 30 seconds, 25 CPU seconds, 768 MiB sampled RSS,
  six compiles, 64 data allocations and 1,000 static forwards; zero dynamics,
  original-video decoding, inference, network or paid calls. Preserve failures.
- [x] Audit saved geometry/source slots and unchanged floor checks independently;
  verify actual interaction/export/replay, document results and publish under
  standing approval. Publication receipt records the final remote identity;
  no default or learned-policy promotion.

123 Python checks and 71 mocked UI checks pass. Receipts:
`.local/multiray-partial-tests-01.xml` and `multiray-partial-ui-tests-01.log`.
Native controls include two, three and four cameras; the two unseen distal
points never acquire invented contributors or fitting metadata. Multi-view
partial geometry is `multi-view-body-v2`; fit plus partial is
`camera-body-length-fit-v2`. Existing methods remain admitted unchanged.

Native job prepared at `.local/aspset-partial-limb-lab-01`; output
`artifacts/aspset-partial-limb-body-lab-01`. Frozen runtime is `aa1b0b4` plus
five root-owned overlays, excluding dirty avatar files. Worker hash
`59b6a256d7d168d0dca523e48b58ad4cc5a69cb7f18ac18a4198b2804dfe95f6`,
plan hash `88df9ee55f122a519532a1ff726fcfbc48c38934c346f969406edcc19bd2020d`.
Launch/process receipts identify completion and ownership; do not duplicate it.
Existing dependencies only; local provider cost $0.
No API key/ledger access, server, browser, deployment or new source acquisition.
The frozen candidate and its saved audit are complete; results follow. Do not
rerun unchanged checks or change limits to hide the observed regression.

## Native result

Candidate run: 378 members, seal
`0254ca5f0bf4e97b7ec3d220170bfbd13f61c2076a14a55b93a15b2e1fc1b945`.
Worker 1.601199 s /152,551,424 B RSS; supervisor 1.674647 s, terminal/reaped.
Four compiles/53 data allocations/253 static forwards, zero dynamics or new
media/inference/API work. Native output 381 files /43,462,109 B including metadata.

Saved-only audit `artifacts/aspset-partial-limb-audit-01` verifies 24 partial
limbs, all 163 unchanged complete-limb fits, and all 6,768 image slots in
0.189649 s. Audit seal:
`4487592d948ebb340b9fdf284b93fba4870332a8185c457233e41e55b7ce801d`.
Source floor failures **0→1**, grid failures **4→3**, worst penetration
**.055751→.077483**. Mean image error left **9.2281→10.6642 px**, middle
**8.2954→6.6110**, right **9.2751→6.1971**. All 41 worsened left-view points
remain recorded; no other view drops evidence or improves its denominator.
This is a mixed construction result, not a physical success or default promotion.

The completed job `.local/aspset-partial-limb-interaction-01` uses that exact run
and frozen runtime: 39 actions, one explicit reset, exact replay and 60 in-process
latency calls; 45 s external /35 s CPU /768 MiB RSS, six compiles/16 allocations,
3,000 physics calls/1,000 forwards, 64 video writes/256 export verification reads,
64 MiB output. No source-video decoding, inference, API, browser or server.
Run and lifecycle metadata remain unchanged; its process receipt is terminal
and reaped. No duplicate execution is needed.

## Recording and review evidence

Actual 39 actions, one reset and all 41 states/rewards/completion flags replay
exactly. Export has 41 frames. Recording:
`artifacts/aspset-partial-limb-body-lab-01/sessions/session-4f6d03008f86`, seal
`fd366136d410fb14547769c86ad520340263ee02f47d27abff7ac7a2c1d05191`.
Worker 1.031344 s /161,611,776 B RSS; supervisor 1.132402 s. Two compiles/two
data allocations, 1,104 physics calls, 145 forwards, 41 video writes and 42 reads.
Session output 1,909,768 B. Sixty in-process calls: median .824646 ms, p95
.860656 ms, maximum .880875 ms; response at most 35,562 B. These are not HTTP,
browser painting, sustained-idle checks or measured speed improvements.

The 123-case suite takes 14.386 s. Current UI receipt is
`.local/multiray-partial-ui-tests-02.log` (71 passes); after native freezing,
only the label was clarified to apply the bounded fit to **complete** limbs.
No source/runtime geometry or numerical settings changed afterward. The full
audit verifies 756 sealed run members, 45 identical source-input files and all
6,768 image slots: 1,620 compared, 72 low-confidence/outside, 5,076 unsampled.
All 41 image regressions are left-leg points: 18 knees and 23 ankles.

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-partial-limb-body-lab-01
```

The [guide](../multi-view-body.md#optional-partial-limb-directions) links replay,
video and complete comparisons. `lab --resume` starts the visual server only
when the user runs it. Reproduce construction with the prior exact camera/ground/
root-view/additional-view inputs and `--body-view both --fit-limbs --partial-segments`,
always into a new output directory. Saved-only independent audit:

```sh
.venv-lab/bin/python scripts/studies/verify_aspset_partial_limb.py \
  --out artifacts/aspset-partial-limb-audit-new
```

Publication uses standing approval and scoped staging, not the foreign avatar
files. Final local/remote hashes, test/evidence hashes, owned cleanliness and
stash/index status live in `.local/aspset-partial-limb-publication-01.json`.

## Next safe action

Saved diagnosis `.local/aspset-partial-limb-diagnosis-01.json` distinguishes joint
positions from collision-shape names. At frame 84, knee center is .335090 model
units above ground; ankle center is −.039483. The `left_knee_geom` is the **shin
capsule**, radius .038, reaching the ankle. Its .077483 penetration is not a
knee-center intersection. Do not change the scorer or rename frozen evidence.

The fixed thigh is .389886 m; observed hip–knee distance .322469 m, and modeled
hip–observed knee distance .318694 m. This implies a .071192 m minimum positional
mismatch at the fixed origin/length even with unrestricted direction. Those
source estimates are not measured anatomy. The new orientation is correct for
the supplied direction, not proof the body fits the source.

The left ankle still has a qualifying **middle-camera image observation**:
47.070 px current error versus 80.111 px before partial transport. Left/right
ankle image observations at this frame are low confidence; no supported 3D
ankle exists. Next qualify a bounded proposal using available distal pixel
constraints, with explicit unknown depth/twist and fixed root/shape/ground.
Keep every view, confidence failure, rejected alternative and original score.
Do not rerun this candidate unchanged or tune control against an infeasible
reference. The full generalized coupled-world/RL goal remains active.
