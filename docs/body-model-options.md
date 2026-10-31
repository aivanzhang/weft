# Body-model options and access

## Propose a correction from a saved body

```sh
.venv-lab/bin/python -m weft body-reference-propose \
  --run artifacts/my-exercise-body --out artifacts/my-body-proposal-01
```

This reuses `exercise-body` output in one local workflow: verify saved metadata,
freeze exact inputs and fitting code, run a bounded child, then generate the
existing offline comparison. Open `inspection/index.html`; inspect
`proposal/plan.json`, `proposal/worker.json`, `proposal/report.json` and
`result.json` for exact settings, work counts and failure reasons. Each output
directory must be new. It never starts a server or opens a browser.

The default is `--objective image-first`. `floor-first` is an explicit experimental
alternative, not an automatic retry. `--manifest-sha256` binds a previously known
body-source seal; without it, the selected local seal is frozen and rechecked,
not independently authenticated. For ordinary bodies, source-media bytes are
not read: the loader verifies selected metadata, retained pose slots, raw-pose
identities, clocks and an explicitly reviewed saved decoder version. It adds
only dimension provenance to a separate evidence copy. Unknown decoder versions,
existing dimension claims, supplied cameras and stacked corrections require
separate adapters and fail closed.

Bodies from the normal **selective-crop recovery** pipeline use the same command.
This route reads and hashes the original video bytes, then replays the existing
source-only derivation once: all crop plans, source-coordinate mappings,
accepted/rejected decisions and selected estimates must match their saved proof.
It does not decode frames, rerun a detector, fit a body or consult simulation
outcomes during selection. Original whole-frame estimates, all candidates,
decisions, original video and ten exact verifier-source snapshots are copied.
Later rechecks verify hashes without repeating selection. Missing proof or a
changed verifier version stops the proposal; no fallback discards the history.

The report identifies **Selected crop estimate, mapped to source image** and
shows the preserved observation, proposal and decision hashes. It compares
**Input body → Proposed body** against the same selected image targets. These
are fitting errors, not before/after detector accuracy. Fixed root/shape/floor
means fixed relative to the supplied body after any pose recovery. The original
pre-crop estimates remain provenance; they do not silently replace accepted
targets. None of this establishes calibrated imagery or physical accuracy.

The fit retains its fixed 0.025 model-unit floor gate, first pose, shape, root,
clocks and motor settings. No candidate is installed, including one whose saved
gates all pass. Rejected and partial proposals remain inspectable. A failed
worker returns nonzero with preserved receipts; an interrupted worker's missing
counts stay unknown. Successful delivery of a rejected proposal returns zero
and explicitly reports `proposal_status`, not simulation readiness.

Bounds: 30 seconds fitting plus 8 seconds final checks; a 45-second hard child
deadline; 60-second workflow checks plus bounded owned cleanup/sealing. Options
can only tighten time bounds. One worker, two model compilations, six data
allocations, 120,010 native forward calls, zero dynamics/reset/training/inference
or paid calls. Inputs are bounded, logs are capped at 256 KiB per stream, proposal
JSON at 16 MiB combined. Ordinary input snapshots remain 32 MiB and workflow
packages 64 MiB/96 files. Recovery preserves the body-source allowance of one
64-MiB original video, separately from 32 MiB of metadata (8 MiB per metadata
file): at most 96 MiB of inputs plus the bounded fitting-code snapshot, and a
160-MiB/96-file workflow package. The larger file allowance applies only to that
exact video copy, not arbitrary files or metadata. CPU and per-file child limits
are enforced; peak memory is observed, not hard-capped. The
allowlisted environment excludes caller credentials and Python startup hooks.
These are restrictions on reviewed trusted code, not an OS sandbox for generated code.
Saved recovery checks are synchronous CLI preparation and load the existing
numerical/image libraries without invoking decoding or inference; UI integrations
must call this workflow from their background worker, not a request/UI thread.

**Verified composition:** `artifacts/body-reference-cli-01` runs the ordinary CLI
once on five authored poses, then verifies exact inputs, code pins and all output
seals. The process takes 0.455 seconds; the child uses two models, four data objects
and 215 geometry forwards, with 74,317,824 B observed peak RSS. It performs no
dynamics, media decoding, inference, training or API calls. The five-pose/nine-grid
checks pass, but the candidate stays unadmitted and uninstalled. MuJoCo's macOS
startup attempts one system lookup; the worker blocks it and records the denied
attempt. This fixture verifies product composition, not reconstruction accuracy.
Seal: `a5919d1ece9bf17d9c5d522eade9f56b2f095d0d012149416b0c3141e5de23eb`.

**Verified recovery input:** the saved kettlebell body passes this adapter in
0.510 seconds, preserving its 129 sampled slots, 121 usable references, eight
unavailable poses and all five accepted/seven rejected crops. A separate
0.212-second fitter-input check verifies all 1,305 confident endpoint coordinates
exactly and preserves 147 unavailable endpoints. These checks perform no fit,
geometry, decoding, inference or physics; they do not qualify a reconstructed
body. [Receipts and limits](handoffs/combined.md#crop-recovery-integration--october-31-2026-538-am-pdt).

**Real recovered-body workflow:** one first public-command run now also completes
the fit and report on that saved kettlebell input in **2.663 seconds**, using a
five-second fitting budget. Mean image-target error falls **13.25875→11.57002 px**
over 1,305 confident endpoints; 121 source poses and 255 intermediate checks have
zero floor-limit violations before and after. All eleven unchanged fit gates
pass. The worker performs two model compilations, four data allocations and
12,580 geometry forwards, with 98,729,984 B observed peak RSS; no dynamics,
decoding, inference, training or API call occurs. One macOS startup subprocess
attempt is denied, not executed.

Open the [saved comparison](../artifacts/body-reference-crop-cli-01/workflow/inspection/index.html).
Five crop-derived targets, eight missing-reference slots and 147 unavailable
endpoints remain explicit. The original body, root, first pose, clocks and floor
are unchanged. Only the proposed reference joint angles differ: **this does not
change physical geometry or validate equipment contact**. The proposal remains
unadmitted and uninstalled. Replacing the reference in an existing task would also
change its scoring target; any controller trial must keep the original scene,
task and scorer fixed. Saved audit:
`.local/body-reference-crop-cli-saved-audit-01.json`; acceptance seal
`a154fd6e97b50f11cc22dc2ea6651d9269c47084bb7607db7d4434bc27ff1918`.

## Saved-source whole-body fitting

The opt-in `whole_body_reference_refinement.propose_whole_body_reference_refinement`
API fits the existing 19 joint axes in blocks of at most 8 variables. It preserves
the first pose, body shape, root, floor, source clocks and missing estimates.
Every accepted change must keep confident source-endpoint image errors and all
affected source/grid floor gaps no worse, with bounded angle, correction-rate
and depth changes. A separate final geometry instance checks all samples.
This is an unpromoted construction proposal, not a new default or recovered anatomy.

One Heavy VIDEO/raw saved-source trial completed in **1.478 seconds**, with
10,262 geometry forwards, two models, no dynamics/reset/video decode/inference,
88,522,752 bytes observed peak RSS and no API cost. Mean image-endpoint error falls
**5.13984→3.86933 pixels** across 1,197 confident endpoint pairs. Floor failures fall
**22→14 checks**:117 source + 235 control-grid checks retain 5 source and 9 grid failures.
The worst penetration remains **0.0718557 model units**, above the unchanged 0.025
limit. The candidate is **rejected and not installed**.

Frame 8's right-shin primitive remains the worst case. Its relevant corrections
are zero, not at angle/rate/joint bounds. Saved results do not identify which
optimizer constraint prevented a useful direction; do not infer that no repair
is possible or that more fitting would necessarily help.

Original sealed evidence lacks dimension provenance. The trial authenticates
its exact bytes through the source/body manifests, then adds only a
`sealed_decoder_metadata` declaration in a new derived input. This reuses saved
decoder metadata; it does not decode the video again or verify physical scale.
All 118 retained pose slots, 117 body knots and declared 235 decoded frames remain
distinct from the container's 236 frame count.

[Saved report](../artifacts/whole-body-reference-candidate-01/report.json), seal
`42f43355225785365baffbfa77cf559a7e94aca951ed75d19176df0d4c2f235a`;
[saved-only audit](../.local/whole-body-candidate-saved-audit-01.json).
The audit verifies 13 output files and 46 ordinary pins and recomputes saved floor
arrays and gates; it does not independently reconstruct native geometry or
endpoint positions. Thirty pure checks and one 39-forward fixed-pose Jacobian
fixture verify the new implementation. Equipment/contact/balance and independent
physical accuracy remain outside this trial.

### Floor-priority follow-up

An explicit `proposal_objective="floor_first_v1"` changes only the proposal
objective. It preserves the image-first default, every hard constraint, the
0.025 floor limit, work bounds and final fresh geometry audit. Its local QP
targets zero gap, but actual acceptance prioritizes reducing squared penetration
**beyond the unchanged limit**; image/prior loss only breaks ties.

The single follow-up is also **rejected and uninstalled**:

| Saved result | Failed floor checks | Mean image error | Excess-floor loss |
| --- | ---: | ---: | ---: |
| Original body | 22 | 5.13984 px | 0.002490 |
| Image-first proposal | 14 | 3.86933 px | 0.001571 |
| Floor-first proposal | 19 | 3.98639 px | 0.002226 |

Loss is half the sum of squared excess penetration over 235 unique sampled times
and 23 support endpoints, in squared model units. Source/grid checks may share
a time; they are not extra independent poses. Worst penetration stays 0.0718557
in every result. Floor-first underperforms image-first on all three metrics;
neither qualifies. No default, body, task or policy is replaced.

Sixteen new pure cases and independent math review pass. The one source fit
takes **1.488 seconds**, two models/four data objects/10,259 forwards, zero
dynamics or new inference, and 88,621,056 bytes observed peak RSS. Saved counters
record eight floor-descent and 371 image-tie-break acceptances, with 27 QPs
returning `dependent_blocker`. Individual QP inputs are not saved; a bounded
independent duplicate-constraint example found no solver defect. Do not treat
those statuses as proven infeasibility or justification for an unchanged retry.

[Follow-up report](../artifacts/whole-body-floor-first-01/report.json), seal
`af5836b30d64624522eb09ab80f4ce8656097c7ab2b4a86242a9c6a5ca5b817b`;
[independent saved audit](../.local/whole-body-floor-first-saved-audit-01.json).
The audit checks final values and counter arithmetic, not every optimizer
decision or physical validity.

### Inspect the saved proposals

Open the [image-first comparison](../artifacts/whole-body-reference-views-01/image-first/index.html)
or [floor-first comparison](../artifacts/whole-body-reference-views-01/floor-first/index.html).
Choose retained source slots or the control grid, then select a sample or use
Previous/Next. The chart compares floor penetration; the tables show signed
support gaps and available source-endpoint errors. Exact saved IDs remain in
details. Frame 44's missing body estimate stays selectable and unknown; grid
samples never acquire source-image targets. Both pages retain **rejected—not promoted**.

Create a new report from either sealed proposal with the ordinary CLI:

```sh
.venv-lab/bin/python -m weft body-reference-report \
  --run artifacts/whole-body-reference-candidate-01 \
  --out artifacts/my-body-reference-report-01
```

No server, browser, inference, media processing or physics starts. An optional
`--manifest-sha256` binds an expected seal; without it, the selected local seal
is frozen and rechecked, not externally authenticated. Reports copy the four
supporting JSON files without altering them and require a new output directory.

Both real CLI reports completed together in **0.533 seconds**, with
102,072,320 bytes observed peak RSS and no paid calls. Independent audit verifies
23 terminal files, 189 code pins, eight exact source copies and 12 local links;
both retain 118 pose slots/117 references/235 grid samples, 1,197 known and 207 unknown
endpoint pairs. Actual saved-page scripted DOM checks preserve unknowns, sample
navigation and chart identity across 100 detail updates, with no requests/timers.
These are not live-browser responsiveness measurements.

[Report acceptance](../artifacts/whole-body-reference-views-01/worker.json), seal
`05906bb23c2fbd140359cc0c23e90a7836659eee4fc4e9a5007cd274e9ef0c91`;
[saved audit](../.local/whole-body-reference-views-saved-audit-01.json).

## Previously reviewed model options

Model options reviewed **October 28, 2026, 8:29 PM PDT**; public-data access findings updated **October 28, 2026, 9:44 PM PDT**. No model weights were downloaded, dependencies installed, agreements accepted or paid requests made. The separately linked jogging study acquired one small public video and reconstructed reference files.

## Meta SAM 3D Body

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) estimates a full-body mesh, including feet and hands, from an image. Its output representation is Meta's Momentum Human Rig (MHR). This is a candidate alternative to the current MediaPipe pose estimates, not a validated fix for Weft's floor conflicts.

The [official installation instructions](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md) require requesting access to the Hugging Face weights and authenticating after acceptance. The current Meta API key does not establish that access. The [hosted API catalog](https://dev.meta.ai/docs/models) lists SAM 3.1 boxes/masks; it does not list a SAM 3D Body endpoint. Do not invent a body endpoint or send weights requests through the segmentation adapter.

The [official demo](https://github.com/facebookresearch/sam-3d-body/blob/main/demo.py) selects CUDA when available and otherwise CPU. That code path is not a tested Mac performance guarantee. The installation example uses Python 3.11. Our local lab is Darwin/arm64 with Python 3.14.7; PyTorch, MHR and SAM 3D Body are absent. A separate compatible environment and bounded single-frame measurement would precede video inference.

## MHR alone

[MHR](https://github.com/facebookresearch/MHR) is a body representation with shape and pose parameters, not an image-to-pose estimator. Its README documents CPU/GPU inference and a TorchScript option; the repository [license file](https://github.com/facebookresearch/MHR/blob/main/LICENSE) is Apache 2.0. Rendering an authored MHR mesh would not demonstrate recovered anatomy or physical accuracy. Runtime compatibility and asset-download size remain unmeasured here.

## Independent calibrated evidence

The [2023 OpenCap paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462) describes synchronized videos, camera calibration and laboratory measurements. Its [reproduction script](https://github.com/opencap-org/opencap-core/blob/main/ReproducePaperResults/labValidationVideosToKinematics.py) points to the `LabValidation_withVideos` dataset on [SimTK](https://simtk.org/frs/?group_id=2385). Such data could support stronger independent checks than our current silhouette proxies; no dataset has been acquired or evaluated in this review.

SimTK returned HTTP 403 through the web reader, but a direct bounded public HTTP read succeeded. The index lists the video archive as **19,051 MB**, dated July 29, 2023, with an Apache 2.0 license notice. The public download landing page returns JavaScript directing visitors to login. That script was read as text, not executed; no login, direct-file bypass or dataset download was attempted. The listing is public, but anonymous archive access is not established. A complete archive download would exceed this review's bounds.

Keep OpenCap core and OpenCap Monocular distinct. The [2026 monocular paper](https://arxiv.org/html/2603.24733v1) describes its code as Apache 2.0, but the [current repository README](https://github.com/utahmobl/opencap-monocular) says PolyForm Noncommercial and lists additional SMPL restrictions. It requires Ubuntu, Python 3.9 and an NVIDIA driver. Preserve this conflict; do not infer permission from the paper or accept model terms on the user's behalf.

## Additional public-data findings

| Source | Observed access boundary | Action taken |
| --- | --- | --- |
| [Synchronized fitness dataset](https://zenodo.org/records/19720803) | Metadata describes video acquisition, but the published markerless ZIP preview lists 294 CSV entries and no video extensions | Saved public metadata/inventory only; no 486-MB archive download. Availability of the required RGB videos is not established. |
| [BioCV](https://researchdata.bath.ac.uk/1258/) | Video/mocap/calibration data requires an access agreement; large participant archives | No agreement, contact or download. The paper's license does not grant dataset access. |
| [MoVi access guide](https://www.biomotionlab.ca/Data/Tutorials/DataverseTutorialBMLmoviV2.pdf) | Institutional account, data request and research agreement | Read-only review; no account or request. |
| [UCOPhyRehab++](https://www.nature.com/articles/s41597-026-07362-5) | Published motion data and derived images; RGB videos available on request | No request or archive download; derived MP4 files are not proof of original RGB availability. |
| [Panoptic toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) | Research/noncommercial data; its 3D labels are multiview reconstructions | CMU server request timed out; no data acquired. The official toolbox documents a mirror, not yet tested. |

The [Zenodo metadata/inventory snapshot](../artifacts/calibrated-source-discovery-02/report.json) preserves the initial DNS failure and successful bounded read: four-file seal `a4bc5a9a61322fce153ea419c028895a8c628cf2d65bcf522d4675743be35c9f`. The other entries are primary-source web findings, not raw-byte snapshots in that bundle. These access limitations do not block unrelated local development.

An accessible [Autelica jog-turn sample](https://autelica.com/sample/) supplied one unchanged 11.2-MB video plus small camera/3D reference files under its sample-specific CC BY 4.0 license. [The completed study](multiview-jog.md) tests both inference modes, retains the default rejection and verifies an interactive recording. Its camera/joint data is reconstructed, with height-based scale and stance-foot ground alignment—not independently measured calibration, anatomy or physical contact. Do not substitute its model agreement for laboratory ground truth.

## October 30 follow-up: local temporal-model options

Heavy is already cached and supported by the existing CPU pose path. It was
evaluated on the exact single-leg source in **IMAGE/raw** mode: 114 of 118 samples
were usable, but competing hypotheses and a root-motion gap prevented body
construction. Do not repeat that experiment. Before the October 31 pass below,
the retained Heavy runs contained no **VIDEO/raw** result. Temporal region reuse
is a distinct configuration, not a new temporal 3D model. The source, cached model,
rights-review document and prior results are pinned in
`.local/mediapipe-heavy-triage-01.json` (SHA-256
`538d811ff3acaaeb15c977be6be03b994063172bae8dfec82807ccd92ef19d24`).

**October 31 result:** the distinct VIDEO/raw run completed in 4.995 seconds,
retaining all 235 original frames and 118 sampled slots. It produced 117 usable
partial poses and one low-confidence sample (frame 44), with no competing
hypotheses; unchanged pose/root prerequisites pass. The large frame18→20 inferred
joint jump falls from Lite VIDEO's 1.1014 to 0.4279 radians RMS. Missing-pair
comparisons remain unavailable; these numbers do not measure anatomical error.
`artifacts/heavy-video-single-leg-02` is independently audited in
`.local/heavy-video-source-verification-02.json` (SHA-256
`acdacc9446a03a67814183223fdc0937d0d7e658646bc634f60e0664d5c8e3fd`).
The subsequent existing-path body/world build preserves all slots but still
fails the floor limit at eight source poses and fourteen intermediate poses.
It is an untrained simulator candidate, not qualified physical reconstruction;
the Lite default is unchanged. Attempt01's zero-frame startup failure is preserved;
attempt02 only disables an unnecessary system-font scan during model import,
without allowing subprocess or network access.

[MotionBERT-Lite](https://github.com/Walter0807/MotionBERT) remains an unqualified
alternative. The nominated artifact is
`FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`, not the generic pretrained
Lite model advertised as 61 MB. The review found a published CPython 3.14/macOS
ARM64 Torch wheel, so the current Python version alone is not a demonstrated
blocker. That does not establish compatibility of the older application or a
measured CPU inference cost.

One bounded anonymous metadata chain reached the author-linked OneDrive viewer,
but its retained HTML prefix did not identify the exact checkpoint, size,
checksum or weight-specific terms. Do not describe this as a verified download
or a login requirement. `.local/motionbert-eligibility-01.json` and
`.local/motionbert-direct-metadata-01.json` through `-04.json` preserve the source
review, initial DNS failure and later public responses. No weights, dependencies,
credentials, agreement acceptance or inference were involved. Keep the existing
environments unchanged; establish the exact artifact and a bounded isolated
runtime plan before acquisition or execution.

## Resume without repeating work

1. Reuse the acquired jogging source and frozen comparison for a clearly labeled camera-aware additional-input candidate; exclude reference joints from fitting. Keep its missing poses, full interval and floor failures visible. Stronger independent measurement remains open. Do not repeat gated OpenCap/BioCV/MoVi acquisition paths or download the Zenodo CSV archive expecting RGB video.
2. If SAM 3D Body becomes accessible under the user's instructions, pin weights/code and measure one frame in an isolated local environment before committing to a video run. Preserve input timestamps, missing detections, source identities and original candidates; apply the same independent checks.
3. Keep costs explicit: this review spent $0 in provider fees. Future local runtime and any hosted model price are unknown, not covered by an assumed rate. The existing $5 total is for the guarded Meta API adapters, not authorization for a new paid compute service.

Six raw public documents, URLs, acquisition times and individual SHA-256 hashes are saved in [the source bundle](../artifacts/body-model-source-review-02/sources.json): 42,154 bytes, eight-file seal `dda2b7342a4833f7112819794b429ac12e46154412d87c0cf6f51c1125f3ee98`. These are exact snapshots of moving branch URLs, not commit-pinned releases. API catalog and paper findings came from the web reader and are not raw-byte snapshots in that bundle. The earlier DNS failure and a seal-preparation error are preserved; neither caused source replacement or another model request.

The [SimTK index snapshot](../artifacts/calibrated-source-discovery-01/report.json) is 49,742 bytes, two-file seal `4afd0dba3926c25f2b7d7fb795b0c707892d7dfcf41189641f6419788ed2c8c5`. The [login redirect snapshot](../artifacts/calibrated-source-access-01/report.json) is 203 bytes, two-file seal `5e0c921a31bd9fce7502f21ba78ea0033a8b3ec6973098857ca86ec0a5fc9d8e`. Neither snapshot contains credentials or participant video data.
