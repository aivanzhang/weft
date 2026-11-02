# Source-ground and working-equipment reconstruction

## November 2, 2026 (PST)

Continues [native endpoints and conditional geometry](native-interval-and-association.md).
The prior goal turn made verified progress and published `7b01ffd`. This turn
targets the remaining body/floor and working-equipment failures, not UI polish
or another run of the unchanged rejected models. The full goal remains active.

## Native reconstruction inputs now compose correctly

`native_reference_inputs.py` binds original dimensions, raw detector evidence,
the reversible native-time view and original decoder identity in a separate
sidecar. It does not insert dimension metadata into either preserved pose view.
The existing body validator checks complete source/crop/capture custody first;
detached worker inputs retain exact body/evidence/clock/sidecar identities.

The squat body contains a later body-constructor snapshot, not the original pose
decoder. Original decoder `37934f7d…` was independently audited and attached from
its sealed original pose run; neither the body nor its manifest was repinned.
Current decoder `fee07a21…` and the intervening complete whole-image producer
were separately reviewed. The old nominal reference route rejects hidden regional
or single-pose assumptions, and still directs native inputs to the explicit native
adapter rather than silently altering its old contract.

29 focused adapter tests and 116 earlier compatibility/crop/callback checks pass
(overlapping invocations, not a unique total). Saved squat and selective-crop
packages qualify in **1.41 s /184 MB**, all input bytes unchanged, with no decode,
inference or physics. Receipt: `.local/native-reference-admission-01/result.json`.

## Coupled source-image root/leg fit: improved, still rejected

`root_leg_source_fit.py` tests a new joint camera/root/lower-leg hypothesis instead
of repeating the previous fixed-root or floor-first methods. It uses the new native
composition, existing kinematic geometry/Jacobians and the bounded QP solver.
It changes no source observation, body shape, upper-body joint, native timestamp,
task, force cap or independent floor threshold.

The explicit camera gauge uses source width as focal length, fixed center/scale,
zero distortion and fixed first root; it is not recovered calibration. Corrections
are limited to .15 rad for leg/camera angles, .08 model units for root/depth,
.3 model units/s root correction and 1.5 rad/s angle correction. Two sweeps use
seven-variable root/one-leg blocks and three-variable camera blocks, preserving
the existing eight-variable QP cap. Inputs retain 3–128 complete observed-root
references; longer clips are rejected, never truncated. Every accepted update
must preserve each confident endpoint and every original source/30 Hz floor gap.
A fresh model independently audits the complete result.

One frozen source fit in `.local/root-leg-native-01` processes all 34 squat knots:

| Measure | Original | Candidate |
| --- | ---: | ---: |
| Source poses failing floor limit | 13/34 | 10/34 |
| 30 Hz samples failing floor limit | 30/80 | 21/80 |
| Worst source penetration, model units | .073842 | .055167 |
| Sum of source floor-limit excess | .302454 | .141163 |
| Confidence-weighted image error, pixels | 9.93570 | 9.18295 |

All nonregression checks pass, but the unchanged .025 floor gate still fails.
**Rejected/unpromoted: no body installation, training or task-success claim.**
The fit takes 3.00 s /79 MB, 3,722 kinematic geometry calls and 84 accepted local
updates, zero physics steps/inference/API calls. Eleven focused checks include
finite-difference Jacobians and independent floor rejection. Result SHA:
`2645dec57e6e9785b0a37e1e3e3a33d763e7b5f787aef3049a18a4bfec7ba9e6`.

Preserve the first launch's missing-NumPy failure: resolving the virtualenv
interpreter symlink selected the base interpreter. `attempt-02/` uses the actual
virtualenv executable; the first attempt made no fit call. Original plan, logs,
bindings and both process receipts remain in the study.

Saved-only diagnosis finds 497 rejected local proposals: 407 endpoint-image,
69 correction-bound, 15 floor and six depth rejections. Root/camera changes are
negligible, despite 84 accepted leg updates. Separating global camera and local
root/leg blocks does not explore every simultaneous compensation. This does not
prove global infeasibility or justify relaxing the image/floor gates. Next use a
bounded data-only feasible-direction/rank diagnostic before another fit; use it
to define a genuinely coupled low-dimensional proposal only if supported.
`bound-diagnosis/REPORT.md`, `comparison.png` and `comparison.svg` retain the
readable findings/plot; no second native evaluation was performed. Diagnosis seal:
`48687206f33db7b736c8c617d44339c9df4c25ef58ecd24bc8e3696f48fad21b`.

## Exact saved-trial recordings and contact ordering

`shared_exercise_trial_replay.py` reconstructs a completed saved trial from its
exact actions, outcomes and per-boundary state hashes. Original physics/evaluator
code hashes, runtime versions, seeded reset and final metrics must match. It never
chooses a new action or searches a controller; an authored label also requires
recomputing and matching every original authored action. Divergence stops at the
first mismatched boundary and seals retained partial evidence as failed.

The unchanged scorer observes substeps before the existing diagnostic observer.
Output contains ordinary MP4/GIF/interactive HTML recordings, complete substep
telemetry, original trial, runtime snapshot and exact source joins. Source-package
custody remains the caller's responsibility, explicitly checked in the real study.
Fifteen tests cover real replay/export, original runtime/hash requirements, trace
completeness, illegal actions, false controller labels and immediate-stop failure.
Independent review found no acceptance blocker.

### Reuse baseline, replay candidate once

The older `artifacts/auto-equipment-console-01/lab/sessions/session-0f8f6d170f69`
authored episode (reset index 10) has all 254 exact baseline actions/outcomes.
Every current baseline state hash matches a derived view changing **only** the
explicit evaluator stamp; both original artifacts remain unchanged. This is a
documented metadata comparison, not rewriting old evidence or claiming equal
runtime hashes. `.local/working-equipment-diagnosis-01` verifies it in .225 s /
72 MB, zero physics. Seal:
`3a89ca5a750510dab737bb95681b4ff66223287769098ebfa6fc46f49d72da08`.

Only the candidate was re-executed, once, under its exact original code/runtime:
**254 verified actions /4,064 substeps**, 7.12 s /520 MB peak RSS under 60 s /
512 MiB limits. All original trial/scene/task/source files remain unchanged.
No new control trial, inference, training or API call was made. Study:
`.local/saved-equipment-trial-recording-01`; replay seal:
`ae5ee4e19a615931b379b74a9c76da690affb1b39b95bf1e2635335493aea612`.

The first spin-limit crossing occurs at control step 57, substep 1, between
**1.866667 and 1.868750 s**: .023225→.927068 rad/s against the .75 limit.
The coincident right-forearm contact has normal force 52.249 in authored
mass/scale units. Source bracket is frames 58–62 under the original nominal pose
clock, not verified native capture time. This is simulated contact/impulse timing,
not an isolated causal experiment, measured force or proof of physical grasp.

Open the new saved recording at
`.local/saved-equipment-trial-recording-01/replay/recording/replay.html`.
The live world remains available using the preceding handoff's ordinary lab.

## Working-equipment image candidate

`equipment_cell_occlusion_geometry.py` replaces whole-row occlusion freezing with
cell-level uncertainty for this opt-in method. It retains every original target
row-hull cell and every baseline-reachable competing-foreground cell. Empty target
rows and missing-competitor views stay wholly uncertain. It only narrows outer
left/right band faces; holes remain solid, and depth, mass, centered box inertia,
pose, floor and raw identities remain unchanged. Final serialized bounds must
again cover every protected ray. Eight analytic counterexamples pass.

One fixed sam-2 proposal uses the original seven frames and archived association
proof, without repeating source video decode or model inference. Cell-level
exterior mismatch falls **1,498→1,480 (1.20%)**; proxy volume shrinks **.397%**.
Protected positive/uncertain cells lost: **zero**. These new cell-level counts
cannot be directly compared with the prior whole-row exterior count of 549.
The maximum width reduction is .002916 world units; this does not establish a
solution to the forearm impact or recover hidden depth/contact.

`.local/working-equipment-cell-fit-01` completes in 1.10 s /252 MB, one metadata
model compile, 14 cached binary masks and seven saved PNG decodes, zero physics
steps or API calls. All 332 sealed members verify; seal:
`be368a47df299aef0fb73f3840ca8fe811b3adacc5ed8d7b38e82b74c7d8930d`.
### Normal admission and actual original-task comparison

`.local/working-equipment-cell-admission-01` admits the fixed candidate and runs
one original-task authored trial, seed 1729, no jitter, **254 controls /4,064
substeps**. The exact saved baseline is reused, not simulated again. Both still
fail only working-object stability:

| Peak working-object measure | Baseline | Candidate |
| --- | ---: | ---: |
| Displacement, model units | .491753 | .489879 |
| Orientation change, rad | 1.11120 | 1.08905 |
| Speed, model units/s | 1.36593 | 1.28883 |
| Spin, rad/s | 6.53164 | 6.74394 |

Spin worsens; **no policy, default or task-success promotion**. Body, original
task gates, source identities and clocks stay unchanged; the task only rebinds
its scene hash. Source fit remains conditional/unverified in ordinary geometry
metadata. This is not routed through the association-only study reader.

The complete cell and association proof archives remain inside the candidate.
Its 466-file /109 MB input stays below the existing 512-file /128 MiB caps.
The ordinary lab package contains 843 files /140 MB under its separate output
contract; input limits were not increased. Actual build/trial/load: 4.37 s /
532 MB peak RSS within 60 s /512 MiB, total study output 342 MB within 512 MiB.
No additional refit, model inference, API call or training. Study seal:
`914bd35944cb5e30f831a0a8915a2df34f00db3cd25dbe197e183ee185062779`;
lab seal `32e1d2b6ef4a0a747e3ccb5058f9ff5050f1880108e6d8bb0f9dec496cb55cc2`.

```sh
.venv-lab/bin/python -m weft lab --resume .local/working-equipment-cell-admission-01/lab
```

Normal loading/summary at reset step zero is verified. Manual/authored controls
and recording use the existing lab; no learned policy is attached.

## Ownership, remaining work and next action

All three agent lanes, independent code review and root integration are complete
for this checkpoint; all checkpoint-owned processes are terminal/reaped. Do not
rerun the frozen rejected fit/model experiments or duplicate saved diagnostics.

Next qualify source-supported ground/root/contact hypotheses and useful full-task learning
without replacing the original target or weakening gates. Broader subjects,
camera/identity uncertainty and independent physical accuracy remain open.
The normal portal still builds a body-only exercise world; explicit inventory
selection and automatic equipment routing remain a product gap, not a completed
general-video path. In particular, `portal_pipeline.execute_job` chooses
`scene_scope="human_only"` without collecting a separate inventory declaration;
do not present that software choice as the user proving equipment absence.
Address inventory/routing before claiming ordinary uploads reconstruct mixed
exercise scenes. No deployment, provider calls or ledger changes are included.
Preserve separate avatar edits and all original studies. Last published root
checkpoint is `7b01ffd`; this turn's publication receipt will be
`.local/source-ground-contact-publication-01.json` after independent remote
verification. Recheck worktree/remote on continuation; only the 12 listed
checkpoint paths belong to this work, not the parallel avatar changes.
