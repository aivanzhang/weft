# New portrait exercise input

## Result — November 2, 2026, 10:24:54 AM PST

Acquired one complete, previously uncached published exercise video for exposed
development: **Ej Exercise - ESP ICT**, by **EjSupra08**. The uploader identifies
it as their own work under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
on the [primary file page](https://commons.wikimedia.org/wiki/File:Ej_Exercise_-_ESP_ICT.webm).
The complete original release passes its published SHA-1 and decodes to EOF.
No re-encoding, cropping, inference or physics ran.

This is a new portrait-format source, not calibrated physical truth. No person
identity, scene geometry or exercise quality was inferred from metadata; this
lane did not visually inspect frames. The next reconstruction pass must retain
ambiguous people, occlusion, missing observations and source timing.

Owner: `/root/unfamiliar_exercise_input`; root owns reconstruction and publication.
Only this handoff, `.local/unfamiliar-exercise-input-20261102-01/`, and
`artifacts/unfamiliar-exercise-input-ej-20261102-01/` were created. No live handle
remains. No shared code, TODO, combined handoff or avatar file changed. No commit
or push. Last observed Git: `main`, HEAD
`58e6ac96801bab15764e40586222661d8b3d0456`; unrelated peer edits remain untouched.

## Frozen input and verification

| Property | Verified value |
| --- | --- |
| Original | `artifacts/unfamiliar-exercise-input-ej-20261102-01/original.webm` |
| Bytes | 2,334,882 |
| SHA-256 | `bc7cacd1effaaa41ff2c537e6417ef79e5bf6cfa75de1ab5202444d57d753430` |
| Publisher SHA-1, matched | `43e2cd38a7ebf75d10ed9ab76a0420b1391ecf8c` |
| Video | VP9, 368 × 656; 1,107 sequential decoded frames |
| Native clock | 1/1000 time base; frame PTS 0–37,093 ms |
| Video-stream duration | 37.126 s; final frame has 33 ms duration |
| Native intervals | 708 × 33 ms, 392 × 34 ms, 1 × 66 ms, 5 × 67 ms |
| Decode | EOF, exit 0, 0.422952 s; one decoder thread, 30 s wall cap, 1,501-frame safety cap not reached |
| Partition | Exposed development; no published test/final holdout used |

All six longer timestamp intervals remain exact. Do not replace native PTS with
frame index divided by nominal frame rate, or invent upstream camera-frame IDs.
The container reports a -7 ms start while the retained video PTS begins at 0 ms.
Audio remains in the original, undecoded; its capture-clock relation is unknown.

The existing FFmpeg 7.1 binary decoded to a null output using `-copyts`,
`-vf showinfo`, `-fps_mode passthrough` and `-threads 1`. No existing `ffprobe`
binary was found in the three standard locations checked; nothing was installed.
The full command and every native frame clock are in
`.local/unfamiliar-exercise-input-20261102-01/native-clock.json`. Its SHA-256 is
`6f741dd0b89aaa1fd8551fb9caeaff58dab09e72e2fbce965a8e900c72575f46`.
Raw `native-clock-ffmpeg.log` preserves decoder metadata and diagnostics;
showinfo checksums are not cryptographic raw-frame identities.

The source directory also contains `source.json`, `inspection.json` and
`manifest.json`. Manifest SHA-256:
`3c0d0013d5c6cae82887454369d6280ab2557073fed81e0756fde79cb8106784`.
The published WebM is complete; an upstream camera-original or parent recording
was not supplied, so their completeness and clock relation remain unknown.

## Acquisition bounds and provenance

The bounded search/acquisition interval was November 2, 2026,
**10:20:26–10:24:54 AM PST (268 seconds)**. Two search queries, two primary-page
opens and one successful direct metadata GET completed. One earlier in-sandbox
DNS failure occurred before HTTP, for six attempted metadata operations against
the eight-operation cap. Narrow reviewed network escalation then fetched the
same public metadata and the single original; no failed external route was
retried. Media download took 0.395361 seconds, below 60 seconds/64 MiB.

Full search returns, the primary HTML, HTTP headers, failure receipt and exact
curl commands are retained in the job directory. Individual web-tool server
retrieval times were not supplied. The selected page revision is
[1225724282](https://commons.wikimedia.org/w/index.php?title=File:Ej_Exercise_-_ESP_ICT.webm&oldid=1225724282).
The direct source URL is recorded in `source.json`; no transcoded derivative was
substituted. Receipt SHA-256:
`92be7703d3569c7991a171dec915f734eb87b1819537cdbd913777049c34f687`.

Previously blocked ASPset archive, BioCV, SportsPose, UMPM, TUM Kitchen and other
readiness routes were not retried. Existing jumping-jacks video was recognized
as exposed and not reacquired. One other metadata-only CC0 candidate was not
downloaded because this lane was limited to one input. External and recurring
provider cost: **$0**. No keys, ledger, account, forms, agreements, contacts,
browser/UI, model install, server or deployment was used.

## Next safe action

- [x] Run the ordinary source-only reconstruction workflow against the
  unchanged original in a new output directory. Pin its SHA-256, retain all
  1,107 source slots/native clocks, and preserve failures rather than selecting
  an easier crop or constant-rate derivative.
- [x] Report actual visual/source coverage separately from calibration and
  physical accuracy. Intrinsics, extrinsics, measured floor/equipment geometry,
  force/contact labels, independent pose truth and upstream capture timing are
  absent; this source cannot complete those evidence gaps.

Acquisition and native-clock qualification were completed first. The later
authorized construction attempt is recorded below; it did not produce a body.
No new source search or unchanged download is needed.

## Ordinary body follow-up — November 2, 2026, 10:35:10 AM PST

Root authorized one construction run owned by this lane while other agents work
on motion/contact code. New control directory:
`.local/unfamiliar-exercise-body-01/`; new output:
`artifacts/unfamiliar-exercise-body-01/`. Do not start a duplicate.

The frozen runtime is committed HEAD
`58e6ac96801bab15764e40586222661d8b3d0456`, 317 files/6,871,637 bytes, no working-tree
overlays. Plan SHA-256:
`5e14c80a5575c85c3e580ee59eefc95734dbca741863587098784da8f37e7fdc`.
`launch.json` pins the plan, worker and supervisor before execution. Existing
`.venv-lab` and Full model SHA
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`
are used without installation/download.

The ordinary public command uses `exercise-body --pose-mode image
--pose-identity raw --max-seconds 120`, the unchanged original, default two pose
hypotheses and no crop/selection. Expected stride 2 gives 554 inference calls.
Bounds: external 150 s, CPU 120 s, sampled peak RSS 1 GiB, all newly produced
runtime/output/log/cache bytes 128 MiB, at most 600 inference calls, one decode,
zero physics steps, one optional first-pose static floor-geometry evaluation.
Invocation: `.venv-lab/bin/python -I -B
.local/unfamiliar-exercise-body-01/supervisor.py`. Preserve a rejected result;
do not force admission, change thresholds or rerun inference unchanged.

Root inspected unedited source frames 0/553/1106: a floor exercise, one visible
person at those moments, and a shifting/tilting camera. This is not fixed-camera
or metric-floor evidence. Instrumentation records all 1,107 OpenCV timestamps
against the retained native clock, expecting a +7 ms container-relative offset
without rewriting either clock or replacing the six longer native intervals.

The single attempt completed and was reaped: **rejected by unchanged source
quality gates**, CLI exit 1. `failure.json` records `UnsupportedVideo`; the CLI
turns that rejection into `SystemExit(1)` in the worker receipt. There is no
`human.json` or `body-source.json`. Do not retry, lower 65% or discard ambiguity.

| Retained result | Value |
| --- | --- |
| Decoded source slots | 1,107; independent WebM block count also 1,107 |
| Declared decoder frame count | 1,113; six-frame discrepancy retained, not truncated decoding |
| Inference | 554 Full/IMAGE/raw calls, two-pose capacity, no crop |
| Usable samples | 329 tracked +24 partial =353/554, **63.7184%**, below 65% |
| Other samples | 190 low-confidence, 9 no-pose, 2 competing hypotheses |
| Ambiguous frames | 1004 and 1082; decoder timestamps 33.697 and 36.299 s |
| Work bounds | 11.306051 s supervised; 10.967777 CPU s; 371,245,056 B peak RSS |
| Native construction/dynamics | Zero model compilations, allocations, forward calls or physics steps |
| Cost | Zero API calls, training or provider spend |

The quality loss is concentrated around decoder time 15–35 seconds. Two model
hypotheses do not establish two physical people. All sampled raw rows match the
saved evidence exactly. Standard annotated previews are separate derivatives;
the original video remains byte-identical.

Failed-package manifest SHA-256:
`e0d322539a50fb4c6edb283814b4df352f9755fb3f4bb5187a1d48df2a52c294`.
It covers 27 members; 28 files including the manifest total **24,159,210 bytes**.
The saved-only acceptance audit verified all member hashes, every sample ID,
source equality, clocks, unchanged mode/capacity and rejection: **12/12 checks**.
Audit `.local/unfamiliar-exercise-body-01/acceptance-audit.json`, SHA-256
`a693475c6a2c5c143dec3d7cdc8c496178b18f0c1714f279bbd8e88d108af527`.

All 1,107 decoder timestamps equal native video PTS **+7 ms**, tolerance 1 ns,
with no mismatches. Full clock SHA-256:
`3ff2f4169a3d99d407dc9c4e1dda154132ecc06c51105d81ff8aed1114d07783`
(`decoder-clock.json` in the body job directory). The ordinary evidence retains
its separate nominal `frame/fps` field; its maximum difference from native PTS
is **0.200635539 s**, ending at nominal 36.892616925 s versus native 37.093 s.
Do not call this nominal timeline native-time physical motion.

The audit hook denied three `subprocess.Popen` attempts; the callers were not
captured. Their exceptions were handled upstream and inference completed. No
subprocess or network request executed inside the worker. No re-run is needed
to identify those callers. Parent hard limits, source failures and logs remain.

## Native-clock integration review — proposal only

Root requested a concrete downstream correction before shared edits. None were
made. `.local/unfamiliar-exercise-body-01/native-clock-binding-review.json`
records reviewed file hashes, exact reuse boundaries, proposed ownership and
verification. **This is a coordinated contract change, not a one-field fix.**

Existing reuse points and constraints:

- `world_alignment.py` converts a constant clock rate/offset. It cannot recover
  this clip's irregular intervals.
- `source_equivalence.py` already verifies full frame/PTS maps and `ClockOrigin`,
  but its API requires an original/derivative YUV-equivalence bundle. This source
  has no derivative or cryptographic raw-frame equivalence proof. Its current
  origin contract also excludes negative native-minus-decoder offsets; this
  clip needs **-7 ticks**. Do not fabricate a derivative bundle or pixel hashes.
- `observations.py` schema 3 already supports all-frame timestamps; the exercise
  adapter emits nominal schema 1. Schema 3 currently requires equal metadata
  and decoded counts, so preserving **1113 versus 1107** needs an explicit,
  verified encoded-block-count exception—not relabeling metadata.
- `exercise_body_source.py`, `root_motion.py` and
  `whole_body_reference_inputs.py` require original nominal frame/fps identities.
  Root interpolation and gap gates currently use that nominal time. Editing only
  final reference times would preserve incorrect interpolation and break seals.
- Human/shared-world runtime reference arrays already accept strictly increasing
  nonuniform knots and calculate slopes from their differences. Reuse them;
  no resampler or replacement derivative solver is needed.

**Proposed ownership:** one future clock-binding lane, coordinated by root; no
active shared-file lock is claimed here. New `src/weft/source_clock.py` and
`tests/test_source_clock.py` would hold immutable clock validation. Coordinated
opt-in integration belongs in `exercise_body_source.py`, `root_motion.py`,
`spatial_pose.py`, `human.py`, `observations.py`, `shared_world_human.py` and
`shared_world_lab.py`, plus receipt/loader compatibility in
`whole_body_reference_inputs.py` and `shared_world.py`. Do not split mutually
dependent contract edits across uncoordinated owners.

The new binding must pin source/evidence/clock hashes, every decoded frame ID,
native and decoder ticks, signed constant origin and original metadata counts.
Raw nominal times, decoder fields, statuses, IDs and source bytes stay unchanged.
A separately versioned derived body maps references by evidence/frame ID to
the observed clock **before** root interpolation, time-gap checks, velocities,
duration and simulation. Preserve every source slot; insert/drop no observations.
Existing legacy outputs/equivalence tests remain unchanged. Camera/multiview
native-time admission stays unsupported until its own clock contracts agree.

Acceptance needs saved-clock tests for this exact 1,107-row table/-7-tick origin;
irregularly sampled constant-velocity controls; native-time gap-boundary tests;
rejection of missing/reordered/nonmonotonic clocks, source mismatch, drifting
origins and unproved count discrepancies; and unchanged raw bytes and legacy
body/seed equivalence. The actual clip must **still fail** its original 65% and
competing-hypothesis gates. No inference, source decode or physics is needed for
that first integration verification.

### Current fail-closed boundary and next action

The sealed failed source has no admissible body. Existing
`exercise_body_source._validated_pose` rejects failed pose quality, while
`shared_world_human._not_failed` rejects its failure marker before person-only
construction. No actual native-time body/simulation exists for this clip. This
is the concrete dependency, not a claim that every irregular source is rejected
by a generic clock gate.

- [ ] Root: assign the coordinated native-clock integration above if it is the
  next selected generalization step. Preserve the failed run and all gates.
- [ ] Separately address source visibility/identity and moving-camera geometry
  before attempting admission; improved clocks alone cannot qualify this clip.

No live process, lock, source edit, commit, push or deployment remains in this
lane. Root may integrate the handoff; media/artifacts remain local and ignored.
