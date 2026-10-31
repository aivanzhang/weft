# Body-model options and access

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
partial poses and one low-confidence sample (frame44), with no competing
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
