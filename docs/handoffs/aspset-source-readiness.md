# Calibrated sports-video source readiness

## Active — November 1, 2026, PST

Previous turn made progress: `4a7628f` publishes the new body/equipment build,
complete recorded interaction/export/exact replay and retained stability failure.
The source-mode repair does not resolve equipment geometry. Do not repeat the
old fixed-controller or shape sweeps, invent a grip, or relax task checks.

Root is checking a new source of calibrated video before further source fitting.
No other root-owned job is live. Preserve the separate avatar session's files;
known peer usage limits remain. No new branch, server, browser or deployment.

The author identifies [ASPset-510](https://github.com/anibali/aspset-510) as CC0
and supplies video, camera and 3D-joint archives through Internet Archive.
One matching development unit is now acquired, but its calibration, clocks and
reference semantics remain unqualified. Check capture/reference method, camera
convention, units and clocks before using the unit for reconstruction.
Keep the publisher's test partition untouched. A published 3D reconstruction
would be a separate reference, not automatically measured biomechanics/contact.

`sam-3d-body` is not an immediately usable replacement: the [official installation
guide](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md)
requires checkpoint access approval and authentication. No request, agreement,
account, checkpoint or dependency installation was attempted. Panoptic and
SportsPose retain research/access restrictions; no such data was downloaded.

## Frozen metadata-only screen

`.local/aspset_metadata_01.py` owns five fixed public GETs: author README,
license, downloader text, published checksums and archive item metadata. Never
execute the downloaded script or invoke its full-dataset downloader.
Output: `.local/aspset-source-readiness-01`, new and exclusive.

Bounds: two workers, five requests, 256 KiB per response, 15 seconds per
request/12-second socket timeout, 60-second external maximum, no retry. Estimated
retained output at most 2 MiB; external/recurring provider cost $0. No video,
pose inference, native simulation, keys, Meta calls or ledger access. Preserve
failures and exact response hashes. Next: execute this read-only screen once;
acquire media only after its bounded matching unit and rights are established.

The five metadata requests completed in 0.865403 seconds. Both the publisher and
archive declare CC0. Exact metadata SHA:
`bc44aef130b19fc55915b384b4082302fa5ca5c8d81302d1f7ab8db4c31a7bcf`.
The train/validation video archive is 23,719,067,106 B; do not download it whole.
Its camera archive is 2,423 B and split archive 1,578 B. Joint references are
18,292,550 B. Author checksum file agrees with every archive MD5 in the metadata.
No payload has been decoded or used for fitting/evaluation.

Next bounded acquisition: `.local/acquire_aspset_unit_01.py`, new output
`.local/aspset-source-acquisition-01`. Four requests, two workers, 60-second
external bound: two complete tiny camera/split archives, a 1 MiB leading joint
archive range and a 32 MiB leading video archive range. A range must return the
exact 206/Content-Range before any body is read; ignored ranges fail immediately.
Retain compressed bytes and all source/header/checksum provenance. Partial parent
archives cannot claim complete-archive checksum verification. Estimated retained
download below 34 MiB; external/recurring cost $0. No extraction, source decoding,
inference, code execution, test-partition payload or key/API use. Next inspect the
retained archive headers for a complete matching development clip and calibration.

## Acquisition completed; qualification next

Attempt01 retained its successful split archive and three redirect-guard failures;
this was a local exact-host guard, not an approval rejection. Attempt02 used the
archive metadata's declared `ia601605.us.archive.org/4/items/aspset510` mirror,
retained the guard and downloaded only three bounded requests. Exact 206 ranges
were verified before reading either large archive prefix. Total: 33 MiB plus
2,423 B, 1.209622 seconds, no retry of the already acquired split.

`data/aspset/04ac-0003-01` contains two complete MKV members (left/mid), matching
cameras, a complete 52,672 B C3D reference, rights and provenance. The publisher
marks this clip `train,all`; Weft marks it exposed development. Selection was the
first complete matching unit in the leading archive ranges, before viewing or
inference. The incomplete right view was not extracted. No test payload was read.
Selective bounded extraction took 0.044819 seconds. Source manifest SHA:
`f7a57a345529fff170977b6f9fc1db04bc66af7a5241213013f90663875c46e1`.
Full parent video/reference checksums are explicitly unverified; retained complete
member identities and exact tar offsets do not imply complete parent integrity.

The pinned MIT `c3d==0.6.0` pure-Python wheel is downloaded, not installed, at
`.local/aspset-readers-01/c3d-0.6.0-py3-none-any.whl`, SHA
`86928e4bd46aac107b1288981e4fee0daaca7ca24524bef6e890a99b7d1421a4`.
Publisher camera code and constants are saved alongside it. World-to-camera
projection uses `K @ extrinsic`; left extrinsics are identity, so the original
world is camera-aligned, not demonstrated gravity-aligned. No silent z-up or
metre conversion is allowed. Keep the reference entirely outside model inputs.

Next: one no-network, credential-free C3D parse (15 CPU/30 wall seconds, at most
1,000 frames/64 points, no analog channels); then two bounded media/clock
inspections using existing FFmpeg/OpenCV, at most two workers and 60 seconds per
view. No installation, model inference or physics in this qualification. Estimated
additional output below 10 MiB, external/recurring cost $0. Record complete source
frame identities, actual EOF, native timestamps and coordinate uncertainty.

## Source inspection result and frozen pose screen

Both views decode completely: 188 frames, 3840×2160, 50 fps, native 1/1000 timebase,
0–3740 ms in 20 ms steps, zero native/decoder offset. Original YUV hashes are
retained per frame. Each native decode and OpenCV clock verification ran once;
left 7.216534 s, mid 7.482659 s, at most two workers/two decoder threads. Peak
OpenCV RSS was 286,343,168 B; highest FFmpeg child RSS 470,564,864 B. Matching
packaged indices do not independently establish physical synchronization.

The C3D contains 188 frames numbered 1–188, 17 named joints, 50 Hz, declared mm,
no analog channels. All 3,196 residuals and camera flags are zero: these metadata
do not establish observed optical markers. The first parse saved its complete NPZ
but failed when serializing a NumPy scalar; preserve inspection01. Inspection02
uses builtin scalar conversion, completed in 0.016836 s and retains all values.
No dependency was installed. Capture method and gravity/floor remain unqualified.

Next bounded source-only pose screen: `artifacts/aspset-pose-baseline-01`, new
exclusive output. Existing pinned Full model, VIDEO/raw, default capacity two,
stride four, 47 inference calls and 188 decoded frames per view. No crop, person
choice, reference-pose input, physics or task. Two workers maximum, 60 seconds and
768 MiB sampled peak-RSS ceiling each; 128 MiB output and 64 files total. Model
and code pins, exact source clocks and complete failed/ambiguous observations
must survive. Exposed source thumbnails show bystanders and a small central actor;
do not silently claim this is a single-person scene. Estimated external/recurring
cost $0; no Meta/key/ledger activity. Goal remains active.

The full-image screen completed all 94 calls/376 source decodes in 3.877565 s
supervised wall time. Left retained six usable poses and 41 misses; mid missed
all 47. Both correctly fail existing source-quality gates. Baseline01 preserves
a harness-only import failure: the optional TensorFlow documentation import
requires `ModuleNotFoundError`, not its `ImportError` parent. Baseline02 retains
the same import prohibition with the correct exception; no install or threshold
change. Baseline02 seal:
`d3b05f43939aca71bcf3dec38182a6ec5766c0ca6c2fa608b7d39dc8836b70ac`.

The changed hypothesis is resolution/attention, not calibration or controller
tuning: the central person occupies roughly 90 pixels in the default 960-wide
inference image. A new opt-in `exercise-body --person-region LEFT TOP RIGHT
BOTTOM` uses one explicit fixed source region and maps every detection back to
the original video. It preserves region-relative raw observations, competing
hypotheses, source identities/clocks and confidence for joints outside the crop.
Default inference stays unchanged. Saved-body import verifies the mapping and
region provenance; it cannot silently omit or override the region.

Core/source/CLI checks passed (exact XML: `.local/person-region-core-tests-01.xml`).
Next real-source study: `artifacts/aspset-person-region-01`, same Full/VIDEO/raw
two-hypothesis model, cadence and bounds as baseline02. Freeze left rectangle
`[1450,760,2010,1420]` and mid `[1580,680,2140,1360]` from original thumbnails before
region inference. These are explicit reviewer-selected attention boxes, not an
automatic detector or annotation truth. Both use original source pixels, not
reference joints. Preserve both failures and successes; no native physics or
reference-pose input. Compare coverage, mapped coordinates, runtime and memory.

Both region views now pass unchanged pose-quality gates: 47/47 usable sampled
poses each, preserving all 188 source slots and 47 model observations per view.
Left: 15 fully tracked /32 partial; mid:45 /2. Highest worker RSS402,522,112 B;
supervised wall3.799792 s. This is detection coverage, not accurate joints or
physical stability. Seal:
`b1a5f2e252d4f886a3664e538835df0844a8f3052f9b08daa99a2d97df10528b`.
Review tightened the general crop boundary: predictions outside the input region
receive zero effective visibility, while exact original model scores remain in
`region_raw`. This prevents downstream camera consumers from mistaking extrapolated
points for observed pixels. All3,102 actual raw region landmarks lie inside the
fixed boxes, so the saved study is unchanged and requires no inference rerun.

Next saved-only comparison uses the existing `multiview_pose.triangulate_pair`
and `camera_reference.compare_frame`, no new detector or physics. Candidate
construction reads only saved image poses and publisher cameras. A fixed proper
rotation maps camera-right/forward/up to nominal x/y/z, with declared mm→m scale;
nominal camera-up is **not a measured gravity or floor orientation**. Freeze the
two-view output before reading reference coordinates. Then compare the twelve
common named joints to the published C3D in exact index order (C3D frame1 maps
video0), without fitting alignment, timing, scale or side permutations. Report
unavailable slots and reference uncertainty, not a post-hoc pass threshold.
Bounds:188frames,33joints,2views,30seconds,128MiB output,$0 provider cost.

Saved-only comparison02 completed in0.505748 s (0.708087 s supervised). The first
comparison supervisor rejected OpenCV's import, not a decode; comparison02 permits
the existing module dependency but prohibits video decode/encode entry points.
No inference or physics was repeated. Frozen two-view candidate seal:
`68337c06676bb9ff2263c3df6ec3dfeed625a1f15c82f9058bdecc43cba6ec67`.
Across12commonjoints×188frames,492 absolute two-view comparisons have84.6372mm
RMSE against the published reference;1,761 slots are unavailable and3 have
inconsistent rays. Left/mid image RMSE20.1836/8.0951pixels at3840×2160. Monocular
root-relative RMSE163.2688/109.1116mm uses different coverage and centering, so do
not report a direct percentage improvement against the absolute stereo metric.
Offline report: `artifacts/aspset-reference-comparison-02/index.html`. Ground,
physical contact and independent reference accuracy remain unqualified.

Next compatibility step: repackage each successful region observation tree as
sealed original pose evidence, then call public `exercise-body --evidence-run`
without decoding or inference. `artifacts/aspset-person-region-body-01` is a new
construction-only result, not a calibrated body or promoted default. Allow at
most two model compilations/data allocations/static forward calls total (one
initial collision-surface fit per view), zero physics steps,30seconds,768MiB
sampled RSS and128MiB per view. Keep both original 47-pose lists and source
regions exact. No reference pose, camera fit, API, training or controller input.

## Accepted checkpoint — November 1, 2026, PST

Public saved-evidence CLI qualification02 completed both bodies in 1.234845 seconds
(1.340665 seconds supervised), peak RSS 411,041,792 B. Exactly two model
compilations, two data allocations and two static forward calls; zero dynamics,
inference, video decodes, API calls or training. Each body retains 47 references,
its exact original source/pose bytes and explicit region. Qualification01 stopped
at import because the frozen runtime omitted `assets/palette.json`; qualification02
pins that existing resource. No model/body work was repeated. Both bodies remain
ordinary model-derived `exercise-v3` inputs, not calibrated bodies or ready labs.

| Saved artifact | Manifest SHA-256 |
| --- | --- |
| `artifacts/aspset-person-region-body-02/left` | `c981f2d02e80dcbfcb71a0100b3e2e8903a61ae644953ddff5274e0ba42e5983` |
| `artifacts/aspset-person-region-body-02/mid` | `2a98b5ad478e6035c6222825bdb7f1ea22941841f1b83bb82649a6657e7082ca` |
| `artifacts/aspset-reference-comparison-02` | `840312e0b6da9e0de3ee55e351da4cf8eddfc93129e007473c4ffdba772a0a2b` |

Independent saved-only acceptance audit:
`.local/aspset-person-region-acceptance-audit-01.json`, SHA
`f019ea5e401ce32474859ef5aac9d74ba8bb96d831a81f9667b1f015ddb97247`.
It checks 221 members across ten packages (nested coverage counted separately),
all 3,102 original-coordinate mappings, source/body byte equality, all 9,588 C3D
coordinates directly against the raw float32 payload, every report metric's
saved-error aggregation and all 11 local report links. It verifies 183 distinct
passing cases, not 207 by double-counting the 24 reruns. Audit: 0.157692 seconds,
no native/model/media work. Its first clock check required exact binary float
equality; the corrected audit uses the already-frozen 1 ns tolerance. No clocks
or scoring limits changed. Tests: `.local/person-region-{core,compat,final}-tests-01.xml`.

Changed production files: `exercise_vision.py`, `exercise_body_source.py`, `cli.py`.
Tests cover bounds, coordinate/depth mapping, preserved raw scores, competing
people, original clocks, saved import and resealed tampering. Existing capacity,
crop and mode recovery checks pass. User commands are in `docs/equipment-proposals.md`
and `docs/local-lab.md`. No lockfile/environment installation changed. Downloaded
media stays local through the scoped `.gitignore` entry.

The reusable saved-only study command is:

```sh
.venv-lab/bin/python scripts/studies/check_aspset_pose.py \
  --source data/aspset/04ac-0003-01 \
  --poses artifacts/aspset-person-region-01 \
  --reference-inspection .local/aspset-reference-inspection-02 \
  --media-inspection .local/aspset-media-inspection-01 \
  --out artifacts/aspset-reference-comparison-new
```

Do not rerun it now: the checked offline report already exists. Final study review
adds early thumbnail size checks and an explicit 188-entry clock check; existing
qualified inputs satisfy both, with no changed estimates or rescoring needed.
Two supplied views/calibration are additional inputs, not single-video recovery.
All new data is exposed development; no final holdout was consumed.

No live root-owned worker remains. The avatar session independently published
`a460faf` during this work; its two remaining viewer edits stay outside this
checkpoint. Origin fetched successfully; no stashes exist. Publish only the
explicit owned files under standing approval and independently verify remote
HEAD and owned-path cleanliness. Publication receipt:
`.local/aspset-person-region-publication-01.json`.

Next: carry the frozen calibrated image evidence into source-qualified body/ground
construction, keeping nominal camera-up and inferred/unknown ground separate.
Use published poses only for assessment after candidate freezing. Equipment
contact, full-interval learned task success and independent physical evidence
remain unresolved. Preserve the playable default and previous failures; do not
repeat unchanged controller or shape sweeps. The complete goal remains active.
