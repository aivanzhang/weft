# ASPset different-motion reconstruction

## Verified milestone — November 2, 2026, PST

The different-motion clip now reaches one shared 3D simulator, manual/programmed
control, recording, export and exact replay. The complete goal remains active.
Root owns both tracks; preserve the separate avatar session's changes. The
previous learned example at `9002c79` remains unchanged; no new training occurs
here. Standing user merge/push authorization applies to tested scoped changes.

## Source and frozen assumptions

CC0 ASPset `04ac-0006` was selected in archive order before viewing/inference.
Three original 4K videos each have **185 frames at 50 Hz**, 555 complete slots.
Native and decoder clocks agree at every slot, with zero offset. Common-take
association is supplied metadata, not an independent synchronization measurement.
This is a different motion with the same subject/cameras, not a new subject,
fresh final holdout or independently validated physical world. Reference
coordinates remain excluded from construction and have not been acquired for
this clip.

The pose method is unchanged: Full model
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`,
VIDEO/raw, stride four, two hypotheses, and the **unchanged manually selected
regions from clip 0003**: left `[1450,760,2010,1420]`, mid
`[1580,680,2140,1360]`, right `[1720,800,2280,1440]`. No new per-clip box tuning
or automatic-person-selection claim. Each view retains 47 sampled /138 unsampled
slots. Left has 16 tracked/31 partial poses; mid 46 partial/one low-confidence;
right 38 tracked/nine partial. Quality flags do not establish joint accuracy.

All three publisher camera files are byte-identical to clip 0003. New camera
documents bind the new source hashes and exact 185-frame clock, with explicit
millimetres-to-metres conversion. The prior conditional plane basis and height
`-1.3111770047585218` are carried as an **authored static-setup assumption**.
Camera-file equality does not prove an unchanged physical floor or measured
gravity. `ground-left.json` says `quality: authored`; the prior annotation,
estimate and source manifest are retained. No placement fitting to floor scores.

Body construction uses frozen fixed-length + partial-segment + joint-image
settings: image fraction .0025, angle prior .3 rad, correction speed 4.5 rad/s,
12 iterations/four backtracks, .3-rad correction and .1-rad update caps. Root,
torso, shape, ground, forces and task/floor gates remain outside that fit.

## Implementation and source results

`camera-body --control-joint right_knee` now accepts a sealed construction-only
body directly. It validates the selection before reconstruction and records it
as separate manual metadata, preserving the original body/task/policy bytes.
It does not fabricate a policy, reward or inferred task intent. Without the
flag, existing policy/task selector behavior remains unchanged.

**39 checks pass** in `tests/test_camera_body.py` and
`tests/test_multiview_extra_lab.py`; receipt
`.local/camera-body-manual-selection-tests-01.xml`. Cases cover original-source
integrity, explicit override, native lab selection, missing/invalid joint
rejection before construction and existing recording compatibility.

The public `exercise-body --evidence-run` and `camera-body` paths build the new
candidate in **2.923880 s /292,225,024 B peak RSS**, with five model compilations,
54 data allocations, 254 forwards and zero physics steps/decode/inference.
All **47 source +111 grid floor checks pass** the unchanged .025 tolerance.
Worst penetration is .0204390972 model units; three source poses intersect
slightly and **44 source poses have a positive floor gap**, median .0601963.
Passing the tolerance does not mean verified ground support.

The fit uses 853 original image constraints and changes 185 limb poses; all 185
are weakly constrained in at least one direction. **All four trajectories hit
their 12-iteration cap**. Weighted image error decreases 1218.5117→598.2802;
this is construction agreement, not measured depth/anatomy/contact accuracy.
Saved-native joint projection errors average 5.33093/6.17542/6.22002 pixels in
left/mid/right views, across 479/388/550 available joint comparisons. All 6,660
frame/view/joint slots remain: 1,417 compared, 275 low-confidence/outside and
4,968 unsampled. No missing slots are relabeled observations.

## Shared-world interaction

The existing candidate/lab adapters preserve the exact reconstructed body and
all three cameras. Controls are 19 body motors plus three **explicit virtual
root forces**. Equipment/contact requirements are not applicable, not passed.
The unchanged source interval is 184 steps /3.68 seconds.

One programmed-reference episode passes the existing simulator gates: angular
RMSE .21442435 rad, root RMSE .02819549 model units, maximum simulated floor
penetration zero and mean reward .69048039. This is assisted authored tracking,
not learned balance or physical validation. The recording has six manual
actions, one reset and 184 programmed actions: **190 actions /192 events**, all
exactly replayed, with a completed export receipt.

Construction, interaction, export and replay take **11.488739 s /435,240,960 B**,
42 compilations, five data allocations, 645 forwards and 4,640 physics calls.
Ten in-process controls including JSON take 1.500 ms median /1.557 ms p95;
these are not browser or sustained-idle measurements. The 394-member lab run
and ready lifecycle remain unchanged after close. Actual console resume,
reference step and quit exit zero in 1.252 s, without a new recording.

## Inspect it

```sh
PYTHONPATH="$PWD/.local/aspset-next-body-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/aspset-next-shared-01/lab
```

Use `controls`, `record start`, `mode reference`, `step 6`, `reset 29`,
`record stop`, `record evidence`, `quit`. Manual and coast also work; learned
mode is unavailable for this untrained lab. Replace `lab-console` with `lab`
to start the graphical interface yourself. No server/browser was launched here.
The source-only manual lab is `artifacts/aspset-next-joint-body-lab-01`.

## Evidence identities

All paths below are relative to the repository; generated artifacts are local,
not committed. Each manifest preserves exact member hashes.

| Package | Members | Manifest SHA-256 |
| --- | ---: | --- |
| `data/aspset/04ac-0006-01` | 13 | `adad325e9fa9c187c2448512e518b69dbe7982671483ae0c60971d2c0369a680` |
| `.local/aspset-next-media-inspection-01` | 24 | `bbe3ce977d0b30967fa50a06704f19d1adb3170ffa5ee2cc484ecf7a652b0357` |
| `artifacts/aspset-next-person-region-01` | 46 | `7fbd4a63e15e8fe0b921f0de33df3005d0171f32b22b1d2c4e6d636ef117cd86` |
| `artifacts/aspset-next-pose-source-01` | 12 | `e1ddfa60086d6d4438c028b7dc20b86dfdc0c52bfc7fc1d5bb949477f722a464` |
| `artifacts/aspset-next-camera-input-01` | 22 | `7cfcfb7fe97874ec1022eb1dfb0d70de72d017aa7efa50991bea40198a6e6ab7` |
| `artifacts/aspset-next-original-body-01` | 38 | `057bafb56a925eda036fe96416203ff3a512af9c903ae3a9ba49b17498d7ba85` |
| `artifacts/aspset-next-joint-body-lab-01/run` | 383 | `c4000ffcdd2aca83cbb1d1a7e2d6c677a6744b7f3cbbb88e78c4b9d46e8e1ef4` |
| `artifacts/aspset-next-shared-01/candidate` | 67 | `cd746d2807f6e03d48d99ccd96e0a4e3b5ce597d74ffba7b9bdc8942e7cbcb7b` |
| `artifacts/aspset-next-shared-01/lab/run` | 394 | `d998da1621c5d70a935d5492f2b07df326fc8e25d69194d25e56f06b2f94691c` |
| `lab/sessions/session-bba962504d70` under that shared output | 6 | `4623f2aecc2f1566bfc9be5b8d0b8093e16389810eb2fe99da65b515e1d6dcff` |
| That recording's `evidence/` | 1 | `7a8366358b492bba2f1bf10ce8d5ff633ca94425f5a1ae09a3161a82652a83fa` |
| `artifacts/aspset-next-saved-audit-02` | 7 | `a6a6249f01ab7417ca9f3d88709f26398aebbe8189eb5e97a35529598b13bcac` |

Saved-only independent arithmetic verifies nine packages/**969 members**, all
48 retained trial losses, source bindings, native endpoint projection, original
pre-image-fit reference hashes and all image/floor slots. It takes 1.122 s
/138,641,408 B without native physics, inference or another optimization.
The first audit package accidentally included its own unfinished manifest hash;
`aspset-next-saved-audit-01` remains untouched. The `-02` package retains that
failed manifest and the byte-identical verified result files, with a separate
seal correction receipt. Calculations were not rerun or results changed.

## Resume guards and next action

- Pose worker: `.local/aspset_next_person_region_01.py`; at most two workers,
  each 60 wall/55 CPU seconds/768 MiB RSS, 185 decodes/47 inferences; aggregate
  192 MiB/128 files. All three terminal/reaped; partial observations retained.
- Body controls: `.local/aspset-next-body-01/{plan,launch,input-bindings,native-result,process}.json`.
  Frozen 315-file runtime from `9002c79` plus only the CLI/manual-selector edits.
  Bound 60 wall/55 CPU seconds/768 MiB, eight compiles/72 data/1,000 forwards,
  zero steps; output 425 files/68,193,523 B. Terminal/reaped, exit zero.
- Shared controls: `.local/aspset-next-shared-01/{plan,launch,report,process,console-result}.json`.
  Same frozen runtime; 60 wall/50 CPU seconds/768 MiB, 48 compiles/8,000 steps,
  256 MiB/1,500 files. Output 475 files/111,823,779 B. Terminal/reaped, exit zero.
- No API/key/ledger access, paid call, deployment, browser or server. The shared
  $5 budget is unchanged. Do not repeat completed jobs to regain their results.
- Git base is `9002c79`; scoped publication receipt will be
  `.local/aspset-next-publication-01.json`. Stashes were empty; avatar edits stay
  with their owner. Refresh Git/ownership before the next mutation.
- The subsequent [published-joint assessment](aspset-reference-assessment.md)
  now preserves the matching 0006 reference and compares this unchanged body and
  recording without fitted alignment. On 479 matched points, body RMSE is
  69.433 mm versus 156.583 mm for assisted simulated motion. Its tracking gap
  remains open; assessment coordinates must stay outside fitting/selection.
- Then qualify unfamiliar subjects/cameras and independently supported ground/
  equipment contacts. This same-camera, assisted case does not complete them.
