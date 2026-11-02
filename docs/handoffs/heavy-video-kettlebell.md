# Explicit single-person pose tracking

## Status — November 1, 2026, 7:34 PM PST

Implemented and verified an opt-in single-person detector capacity. The cached
Full model now provides 127/129 usable kettlebell poses and constructs a separate
mobile body, but independent development directions regress and the complete
floor check fails. **No model, body, floor, controller, task or lab is promoted.**
The full generalized-video goal remains active. All owned workers have exited;
no live job, paid call, new download, dependency install, server or deployment.

## Product change and verification

`exercise-body --video ... --max-pose-hypotheses 1` explicitly assumes one person;
it can miss competing people. The default remains two hypotheses. The option
passes through `exercise_vision.py`, `exercise_body_source.py` and `cli.py` into
model options, raw poses, source config and saved-body metadata. Saved imports
cannot override it; absent/conflicting/claimed-verified assumptions reject.
Confidence, sampling, source clocks, root gaps, floor and task limits stay fixed.
See [the user command](../equipment-proposals.md#prepare-body-evidence-without-a-lab).

VIDEO mode permits, but does not prove, region reuse. The
[upstream MediaPipe graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_graph.cc)
reruns detection below configured capacity and enables smoothing only at capacity
one. This current upstream source suggests the cause of the identical capacity-two
results below; installed-binary internal ROI reuse was not measured.

The producer retains the original complete fixed-resolution decoder. Register its
reviewed SHA `774dccf1cade873ba4ff2932aa3d77ce8fdb3abc28c250f7530e54f56796112c`
alongside the previous producer in `whole_body_reference_inputs.py`. Unknown
producers still reject. Existing crop/densify/whole-body refinement tools retain
their stricter two-hypothesis contracts; no hidden option injection into them or
the ordinary lab/build raw-video defaults.

**234 distinct focused cases pass**: capacity/CLI, raw and saved source custody,
model/recovery, producer and camera-input compatibility. Summary
`.local/pose-capacity-test-summary-01.json`, SHA
`09ff378cc3828e2ccbe4e01e313839433c07c7a4e96e1482aa417af3b8c6bf15`.
Initial 98 cases had 97 passes and one fixture error: the expected quality
exception occurred while preparing the rejected fixture. Only that corrected
case reran; it passed. Additional 52 and 84 cases pass. Preserve all four XML
receipts named in the summary. Root reviewed the changes; agents remain at usage
limits and no peer review is claimed. These are scoped checks, not entire-repo
or physical validation.

## Exact source and bounds

Source `data/exercise/kettlebell-pushup.webm`, SHA
`4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`;
Taco Fleur/Cavemantraining, CC BY-SA4.0, attribution copied from the existing
source metadata. All runs decode 257 actual frames; container metadata 259 remains
distinct. Each takes 129 samples at even frames 0–256, retaining absent/low-confidence
slots, nominal and decoder clocks. No crop, seek, fallback or identity repair.
These are exposed development inputs, not a fresh holdout.

Cached float16/1 models:

- Heavy: `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b`.
- Full: `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.

Inference uses the frozen e5 runtime plus pinned capacity-module overrides. Each
new source pass has one worker, 45 CPU/cooperative seconds, 60 external seconds
plus at most 5 seconds cleanup, 128 files/120 MiB, 48 preview frames and 256 KiB/log.
1 GiB RSS is an observed completed-child acceptance ceiling, not a hard memory
cap; MediaPipe internal threads are not claimed capped. $0 provider/download/
recurring cost; local electricity unmeasured. No key or shared Meta ledger change.

## Actual complete-source results

| Run under `artifacts/` | Usable poses | Root prerequisite | Process seconds / peak RSS |
| --- | --- | --- | --- |
| `heavy-video-kettlebell-01` (capacity 2) | 73/129 | Reject gaps and pose coverage | 5.454354 /452,624,384 B |
| `heavy-video-single-person-kettlebell-01` | 91/129 | Reject five gaps | 4.291980 /452,968,448 B |
| `full-video-single-person-kettlebell-01` | 127/129 | Ready for construction | 3.157895 /363,118,592 B |

Capacity-two Heavy VIDEO is **exactly equal to all 129 saved Heavy IMAGE pose
rows**, excluding only inference timing. Capacity one changes 128 rows. Heavy's
five unsupported intervals are 56→68,96→110,140→158,188→202 and 232→250,
lasting 0.4–0.6 s. Full has missing poses at frames 0 and 4, starts at frame2 and ends
at 256. Its complete body-control grid is255 rows, not253.

Source manifests, in table order:

- `72475e4b4a840704e1a1d50e1421e4aaf29c83f466905e3945263d5493d9f16d`.
- `17ebdc199473851358806cd58031042e443845d20b6b8084fe9d45196018aa08`.
- `3cb55e99d3bace25ba1540444b73ee39458b04657326cdb1b1a43cf3d048573b`.

Frozen plans are `.local/heavy-video-kettlebell-plan-01.json`,
`.local/heavy-video-single-person-kettlebell-plan-01.json` and
`.local/full-video-single-person-kettlebell-plan-01.json`; respective hashes:
`440f38d37ef70662bd5ce3355bb906bcb529cda4a32fc2356caadf379c2e7e77`,
`98ae3f3484cc7ccbd5ad7dbdb228cf000990302fc64c83d9c82c88f26e7def27`,
`017484198af4266eb1ddfe822e4112a683a046fa898a54c0fefe4cbabee51f23`.
Each pins helpers, inputs, runtime and resolved settings. Matching `.local/`
process directories hold terminal receipts/logs. Do not repeat completed plans.

## Independent saved checks and accuracy limits

All 36/37/38 source members, 257 clock rows, 129 samples and 128 adjacent pairs per
variant were checked in separate saved-only reviews. Original pair and limb
labels and scorer definitions remain frozen; no new media/model/native work.

| Configuration | Pair directions passing (3 observable) | Limb directions passing (12 observable) | Missing limb predictions |
| --- | --- | --- | --- |
| Saved Full IMAGE/capacity 2 | 3 | 10 | 0 |
| Heavy IMAGE and VIDEO/capacity 2 | 3 | 8 | 4 |
| Heavy VIDEO/capacity 1 | 3 | 8 | 4 |
| Full VIDEO/capacity 1 | 0 | 9 | 2 |

Retain 3 unobservable pair slots and 4 unobservable limb slots. These are
assistant-labelled 2D development checks, not independently measured anatomy.
Coverage/smoothness gains do not erase direction regressions.

Review folders and seals:

- `artifacts/heavy-video-kettlebell-review-01`:
  `08db2a32b43a09bd2c19e7e7af633b3832908ae3677ce191bef0bc0cfcd15862`.
- `artifacts/heavy-video-single-person-kettlebell-review-01`:
  `5cedef602b776120e6a152a91c29c31ee904c7d873f41ff8a80b2b7b3e146156`.
- `artifacts/full-video-single-person-kettlebell-review-01`:
  `de6887c793025d4538e3a9ce99985502663cab6d1b2e562a3f4985a5f13f57a2`.

## Body construction and complete floor result

The actual public `exercise-body --evidence-run` path creates
`artifacts/full-single-person-kettlebell-body-02`, seal
`abd4ceb7c8ca699454b0a5d162f893dee5f31cd2990b751d627324751879fbc5`.
All 127 references,257 source slots and 129 sampled clocks remain; source video,
raw poses, model/config and identity audit are exact copies. One model/data
allocation and first-pose floor forward; zero dynamics, inference or training.
1.013979 seconds /228,573,184 B, 39 files /16,688,095 B. Plan
`.local/full-single-person-body-plan-02.json`, SHA
`3bd65f2cd66e6756bd74d6911d9d498223fad16daaaffecbc6be519357fe4914`.

The existing `audit_reference_feasibility` then checks all 127 source poses and 255
control-grid poses under its original 0.025-model-unit floor limit. Result:
**one failed source pose and one grid pose**, both near source frame 132/time 4.4 s;
right-knee penetration reaches 0.07187404503200923 model units. Units are not
calibrated metres. This is an inferred-geometry inconsistency, not a measured
physical contact. No equipment is installed, no lab or policy promoted.

Saved `artifacts/full-single-person-kettlebell-floor-02`, seal
`47690146a0f018dc30390756873e1d547594c24abc79b38172978a42dc2568eb`;
0.355738 seconds /134,103,040 B, 8 files /1,065,629 B. Actual native work: 1 model,
2 data allocations,382 forwards,0 resets/steps. Plan
`.local/full-single-person-floor-plan-02.json`, SHA
`05c876b4fbdae080fe4031c999af4fda182cb9b68ef6e1517370d0ee38dda1ab`.
The separate `.local/full-single-person-kettlebell-floor-02/saved-audit-01.json`
checks all 382 rows, brackets, contact arithmetic, original threshold and 7 exact
source-copy joins without native work; SHA
`12fb15cc651ac745407f3851ee2f9e84426152ceaf03e9f04014e5032061a1e7`.

Preserve two preparation/verification failures:

- Body attempt01: minimal frozen package lacked its existing palette resource;
  failed before output/native construction. Attempt02 adds only that pinned
  resource, with unchanged product/source/gates/bounds.
- Floor attempt01: a wrong expected grid count 253 asserted after the checker
  returned, before saving its numerical report. That result was **not retained**
  and is not claimed as evidence. Attempt02 verifies input-derived time bounds
  before native work and saves the report/counts before result assertions.
  Original process/stderr remain; no physical code or input was changed.

## Publication, ownership and next action

Root owns the capacity feature, producer registration, two test files, user guide,
this note, TODO and combined handoff. Standing merge/push approval applies;
publication receipt is `.local/pose-capacity-publication-01.json`. Preserve all
independent avatar changes. Main/origin last verified at `8afc084` before this
checkpoint; index and stash were empty. Recheck the receipt for the final hash.

Next substantive work: inspect source-bound geometry at frame132 and the retained
pair-direction regressions before considering this body for equipment alignment.
Do not interpolate away the knee failure, move the floor, drop references or
reuse old-body equipment/camera transforms without a new explicit reconciliation.
Do not repeat unchanged Heavy runs or controller sweeps. Independent geometry,
contact and full-task learning remain unfinished; the existing playable
`scripts/play_sandbox.py --example auto-equipment --console` is unchanged.

The [official SAM 3D Body installation guide](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)
still requires approved checkpoint access. The read-only model/source check made
no request, agreement, account action or acquisition; it is not the progress claim.
