# Additional-view landmark evidence

Published multiview checkpoint: `9dae370`. Owner:
`/root/exercise_reconstruction`. Both strict adapter and explicit numerical prior
passed review and focused analytical checks. Study 01 remains an immutable camera
admission failure. The single approved study 02 completed, but the original-start
right heel still fails the unchanged all-view consistency gate. Numerical
compatibility improved; heel/contact recovery did not. No inference, decoder,
physics or paid call ran. Root owns integration, coordination and Git.

## Source-surface envelope milestone — frozen October 31, 2026, 4:20 PM PDT

Root approved only new `src/weft/source_surface_envelopes.py`,
`tests/test_source_surface_envelopes.py` and this handoff for the next slice.
The published multiview/rotation files remain unchanged. No source study or
native/model/media/network operation is authorized or running.

Existing wrist/box co-motion, mask/contact projection and silhouette-band
construction already address projected proximity/shape; do not duplicate them
or invent cavities/contact from mask absence. The next reusable improvement is
an explicit static-support/ground **height interval** plus authored pixel
rectangle, producing a conditional 3D anchor enclosure without choosing a
nominal height or pretending a manual constraint was recovered automatically.

Reuse the source-surface source, camera, annotation and world/evidence validators.
For undistorted pinhole calibration, constant-sign ray/plane denominators imply
coordinate extrema at four pixel corners × two height endpoints. Exact rational
arithmetic on supplied numeric coefficients and outward-rounded floats bound
the modeled geometry; calibration accuracy itself is not certified. Raw camera
R is inverted algebraically, never silently normalized. Missing inputs, radial
distortion without a bound proof, horizon crossing or support-domain mismatch
abstain. All original source/channel slots, clocks and partitions remain.

Bounds: existing observation-loader caps (eight documents / 64 MiB), 4 MiB
additional JSON, 1,800 original source slots, 900 annotated frames / 12 channels,
8 MiB cached output and ten-second cooperative preparation. No hard OS memory
claim. Pure tests cover analytic extrema, widening, point-limit agreement,
nonidentity transforms, missing/rejected inputs and bounded cached queries.
The plain JSON consumer contract exposes outer bounds and exact input pins,
not a midpoint measurement. Existing construction still needs an explicit
world/body join and interval-aware objective; no automatic install is planned.

Root independently reviewed the exact fractional inverse, corner-extrema proof,
outward rounding, source admission, clocks/partitions and consumer limitations;
no blocker was found. Current pins:

- `src/weft/source_surface_envelopes.py`, 17,784 B, SHA
  `60c85dd3df9e0e638052d60df2635f342430554e71fa340678426af9499e8383`.
- `tests/test_source_surface_envelopes.py`, 15,167 B, SHA
  `f7656ec6a5904400cf53d90fbbb8243b86ca4b22ed6cefc064f0b331d8043987`.

### Public API and concrete interval input

Reuse the exact observation/camera/annotation bytes accepted by the existing
`surface-anchors` workflow. The annotation schema is unchanged; its channel's
`surface_id` must name the supplied interval surface. Minimal example:

```python
from hashlib import sha256
import json
from weft.source_surface_envelopes import prepare_source_surface_envelopes

# Example user-authored range, not a measurement or inference from a video.
intervals = {
    "schema_version": 1, "profile": "source-horizontal-surface-intervals-v1",
    "source_sha256": original_source_sha256,
    "calibration_sha256": sha256(camera_bytes).hexdigest(),
    "world_frame": {"id": "supplied-room", "length_unit": "m",
                    "axes": "x_right_y_away_z_up"},
    "calibration_evidence_refs": [],
    "surfaces": [{"id": "bench_top", "kind": "static_support",
                  "height_interval_m": [0.4, 0.5],
                  "bounds_xy_m": [-0.5, -0.5, 0.5, 0.5],
                  "quality": "authored", "provenance": "Example user-supplied range",
                  "evidence_refs": []}],
}
prepared = prepare_source_surface_envelopes(
    observation_documents, source_sha256=original_source_sha256,
    camera_document=camera_bytes,
    surface_interval_document=json.dumps(intervals).encode(),
    annotation_document=annotation_bytes,  # Channel surface_id: bench_top.
)
metadata = prepared.describe()
original_slot = prepared.frame(0)
complete_rows = prepared.iter_frame_bytes()
```

`SourceSurfaceEnvelopes` retains exact inputs once; queries return defensive
cached JSON, without reloading evidence or recomputing geometry. Its exported
`AxisAlignedEnvelope` typed/plain JSON contract has `minimum`, `maximum`,
`unit: "m"`, `kind: "outer_axis_aligned_enclosure"` and an arithmetic certificate.
Available rows use `status: "conditional_surface_envelope"` and preserve qualified
entity/observation/annotation/surface IDs, partitions and exact source times.
Unavailable rows give a reason and `world_envelope_m: null`. No chosen height,
point or midpoint is returned. Unknown height uses `null` / `quality: "unknown"`;
measured intervals are unsupported, not guessed. Measured camera claims still
require the existing exact evidence receipt.

The arithmetic certifies the enclosure of a **conditional mathematical model**
using the supplied finite coefficients. It does not certify calibration/metre
accuracy, normalize rotations or establish physical support membership. The
axis-aligned box is an outer set: arbitrary combinations of its coordinates
need not be attainable by the original pixel/height ranges.

### Exact construction boundary

Current `source_surface_anchors` emits points; existing CLI/construction
consumers do **not** consume intervals. A future join must retain this new
profile and exact source/entity/frame/clock pins, map the body endpoint into
the declared world through qualified `world-alignment-v1`, and use an
interval-aware objective such as distance outside the lower/upper bounds.
It must not copy the midpoint into a point-target field or treat zero interval
error as contact/feasibility proof. No body/floor, CLI/UI, native consumer, policy
or scorer was installed or changed here.

### Verification and limits

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_source_surface_envelopes.py -o junit_family=legacy --junitxml=.local/source-surface-envelope-tests-01.xml
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_source_surface_envelopes.py::test_complete_frame_cap_precedes_camera_geometry -o junit_family=legacy --junitxml=.local/source-surface-envelope-tests-02.xml
```

Receipt 01 SHA `1629ae6aedd685169c785bb57dd67b9a0367741ba3669cadc15b74a84dfc2517`
retains 30 passes / one test-only error-text assertion failure (0.355 s pytest).
The upstream validator correctly rejected `metadata_frames` before the new
module's redundant cap; the assertion expected only the new message. No
production code changed. Only that assertion was corrected/rerun in receipt 02,
SHA `d90c53eed23484e2d716a2f25157350cf967533e53201948d8652147ae13010b`:
one pass / 0.039 s pytest. All **31 distinct cases** now pass; no unchanged suite
rerun. Cases cover exact/outward arithmetic, oblique transforms, interval
widening, point-limit agreement, raw near-rotation nonmutation, missing/excluded
slots, clock mismatches, unsupported claims, caps, deadlines and cached queries.
Native/model/media/network and process work were blocked.

Representative analytical workload: 600 source frames / 1,200 envelopes,
834,871 B input, 1,294,368 B cache, 0.190679 s preparation and 0.000171 s for
twenty cached queries. This is not sustained-service or physical-accuracy
evidence. The ten-second bound is cooperative, not OS isolation. No real-source
study was needed; all existing contact/heel failures remain unchanged.

Live handles: none. These two new files and this handoff are released for root
publication. Previous modules/shared files are untouched; no Git mutation
occurred. Next is the explicit interval-aware consumer join above, if selected,
not another unchanged source fit.

## Published scope and ownership

Own `src/weft/multiview_landmark_evidence.py`,
`tests/test_multiview_landmark_evidence.py`, new `src/weft/camera_rotation_prior.py`,
`tests/test_camera_rotation_prior.py` and this handoff. Do not change the
published two-view producer, strict calibrated-foot consumer, partial-foot API,
body/floor, task/scorer, lab, UI, recording or avatar files. Previous camera-play
and partial-foot studies are complete and must not be repeated.

The reusable gap is a bounded, source-qualified **additional-view adapter** for
the existing all-observation pixel triangulator. It must not duplicate that
solver or add view selection, fitted timing, a new heel-roll prior or floor
selection. Supplied camera calibration and cross-view timing remain conditional
priors. Any result is a world-point estimate, not reconstructed anatomy, a sole,
contact/load/grasp, native foot admission or independent physical validation.

## Source-backed hypothesis

The original jogging frame-zero right heel is unavailable in the exact saved
two-view reference (`inconsistent_image_rays`). Its ankle/toe are available.
Partial evidence retains that rejection and all six original negative ground
gaps; none may be altered or relabeled.

An additional original-frame image observation exists locally:

- `artifacts/multiview-jog-drone3-image-01/evidence.json`, 1,799,298 B, SHA
  `0bfe50ccc899c87293b0adf906bb690bae8d6943e62e0479241fe79fb709f673`;
  ancestor seal `52683b836b2a938adda15182dd8ef17e8b209f5ed1aec840d3512510af7cecdb`.
- `artifacts/multiview-jog-full-two-view-01/camera-3.json`, 127,613 B, SHA
  `ce2e79b3e182ca7733ee1e89244fc2bb4886d68a88315861af270338426becb0`;
  ancestor seal `f8d302915b5f17c0fcab0797493fdcacb71b803af1466baf3bc433e52a1f9cff`.
- Source `cc7d176f46bcfd172e2f310764d3266eebb9d944fe1ca075532decb8f091df9c`;
  240 original frames, 120 stride-two IMAGE samples, fps 30000/1001.
  Saved decoder clock SHA
  `c9eda8c20c7e0099ab37939923735d8541a00598558119971d20a1e3a7219eac`.

The third view uses saved Lite model SHA
`59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`;
the first two approved body views use Full. Preserve mixed model identities, not
a claim that all three ran the same model. At frame zero, the third-view right
heel has visibility 0.9959903359413147 and presence 0.9990230798721313 inside the
image. All six original foot endpoints meet the unchanged per-joint confidence
rule despite the whole-pose `low_confidence` label. Discovery inspected metadata
only; no geometry was fitted. The third camera has already been used in
development comparisons and is **not** an untouched holdout.

The camera parameters were reconstructed using multiple views. Equal original
frame numbers and decoder times do not independently prove synchronization.
Retain the original assumed common-world/aligned-clock provenance. Additional
views require explicit validated `world-alignment-v1` identity mappings into
the first camera's world; no implicit shared origin, units or clock.

## Implementation protocol and bounds

Use exact JSON bytes for two to four named evidence/camera pairs and explicit
additional-view alignments. Reuse `multiview_pose._view/_observation`,
`world_alignment.load_world_alignment`, bounded input helpers and the unchanged
`multiray_pose.triangulate_pixels`. Missing/unknown/incompatible alignment yields
unavailable geometry, never a silently dropped view.

Only the first 60% of the original clock may supply requested landmark
coordinates. Preserve all source/landmark/view slots and original
missing/unsampled/later-excluded statuses. Every available qualified observation
participates; a conflicting view cannot be discarded. Return fixed solver
diagnostics and sensitivity separately from any physical-confidence claim.
Raw inputs/pins survive once; per-row queries must not copy complete evidence.

Implemented caps: 32 MiB exact inputs, 8 MiB retained result, 1,800 source slots,
900 sampled rows/view, at most 6,000 fitted landmark slots, ten-second cooperative
preparation. Oversized work must reject before a fit, not truncate observations.
Analytical tests block native/model/media/network/process work and exercise
known three-view geometry, conflicting third rays, missing alignment,
mixed-model custody, excluded-payload nonuse, full missing denominators, cached
query behavior, unchanged limits and early resource/deadline failure.

Runtime reuse pins:

- `multiray_pose.py`: `c5fcffd7c609df25ffb19780d78423988dc0f2e4df5d5737604ea246f80ef769`.
- `multiview_pose.py`: `1b1cd67c59cedf23a93a7ab3420ce91f15d5d7e1b616d8d5ee3119cbf733bfc0`.
- `world_alignment.py`: `05ffb0cbb279a05c8dcd80e68576825ddfa5f0f677ffbf24bc0466423ea11d69`.

## Public API and analytical verification

```python
from weft.multiview_landmark_evidence import prepare_multiview_landmark_evidence

prepared = prepare_multiview_landmark_evidence(
    [{"name": "primary", "evidence_bytes": primary_pose_bytes,
      "camera_bytes": primary_camera_bytes, "clock_provenance": clock_note},
     {"name": "additional", "evidence_bytes": additional_pose_bytes,
      "camera_bytes": additional_camera_bytes, "clock_provenance": clock_note}],
    landmark_indices=(27, 28, 29, 30, 31, 32),
    alignments=[explicit_identity_alignment_bytes],
)
summary = prepared.describe()
original_start = prepared.frame(0)
all_rows = prepared.iter_frame_bytes()
```

The alignment references `camera-world:<canonical camera SHA256>` frames with
`m` units and `x_right_y_depth_z_up` axes. Its spatial/clock claims retain their
original `assumed`, `inferred`, `observed` or `unknown` status; validation checks
the supplied claim structure and qualified evidence references, not independent
calibration/synchrony truth. Missing/unknown/nonidentity mappings abstain. Only
construction-prefix evidence IDs can support a qualified alignment claim.

Initial strict module SHA (frozen inside study 01 runtime)
`f5ebee3c3dd2dccb995c43e3fd1a567f11eba939673a7a4eb7d9c264d4e5b8dc`
(16,279 B); test SHA `7d0839006637506e1e365fb973531096292aacc79c61aee5a2a5d5d62c7a50ab`
(18,348 B). Command, executed once:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_multiview_landmark_evidence.py --junitxml=.local/multiview-landmark-evidence-tests-01.xml
```

Receipt SHA `1d887f7549bfaf5818517a7319aa4a31bdc6a7da37457cefcb17d3acc4fed457`:
29 passed, zero failures/errors/skips, 0.753 s pytest / 0.849 s command wall.
One `record_property`/xunit2 compatibility warning occurred; the four measured
properties are present in the saved XML. No unchanged suite was repeated.
The larger analytical case retained 1,002 original slots, 6,012 landmark slots,
18,036 view-landmark slots and 366 all-view fits: 1,318,299 B input,
5,593,540 B cached result, 0.458765 s preparation and 0.000386 s for twenty
cached row queries. This is bounded analytical responsiveness, not real-source
accuracy or a sustained service benchmark. Tests blocked native/model/media,
network and child-process work.

## Predeclared saved-study hypothesis

After implementation freezes, send root exact six source members, explicit
alignment documents, ancestor seals, helper/runtime pins and one run plan.
Candidate scope: six foot endpoints across the first 72 sampled source rows,
at most 432 unchanged pixel fits using all qualified views. Retain all 240 source
slots / 1,440 landmark slots / 4,320 view-landmark slots and every failure.
One stat-only size check found the six evidence/camera files total 6,036,196 B.
Bounds: 32 MiB input, 8 MiB result cache, **32 MiB aggregate sealed artifact**
(one exact copy of each input, ancestor seals/provenance and diagnostics),
ten-second cooperative preparation / thirty-second outer deadline, one worker,
$0. Root approved that aggregate cap. No inference, decoding, new acquisition
or native use. Do not confuse the result-cache limit with all artifact bytes.

Falsifier: the original-start heel remains unavailable or any qualified supplied
view violates unchanged consistency limits. Do not change model, thresholds,
clock, view membership, starting frame or floor afterward to obtain acceptance.
The prior FreeMoCap all-view coverage/accuracy regressions also remain evidence;
an extra observation does not imply a universal quality improvement.

## Frozen saved-study preparation

Independent review found no blocker and reused the 29 existing analytical
passes. Preparation ran once in 0.089 s; zero adapter/solver/source-media calls.
The preparation helper only checks exact source bytes/selected ancestor members,
parses camera metadata for explicit assumed bindings, syntax-checks code and
archives the declared committed pure closure. It does not fit geometry.

- Plan `.local/multiview-landmark-jog-preparation-01/plan.json`, SHA
  `868c072c72200ec4a174be310d48f2fdfdb56f0d944a54429dc1658f58a59a70`.
- Worker/runner `.local/multiview_landmark_jog_01.py`, SHA
  `45edf5173858b76b0cf55662bfe33e032ad633941358fb2deea7d26ad20e3716`.
- Runtime `.local/multiview-landmark-jog-preparation-01/runtime.zip`, SHA
  `1b9861f2cc1deb17c8712bb13ccc3f152fe44811325495ec0f8315554fca9c1b`,
  35,895 B / 11 members: nine pure Python modules, reviewed test and committed
  lockfile. Base `6c73bf77295fca7fdf0dec479b46c28d4afea981`; only new adapter/test
  overrides, no dirty avatar/learning files. No study tests execute.
- Preparation helper `.local/prepare_multiview_landmark_jog_01.py`, SHA
  `e11fdbfb37be82d7ae13952fcc44de0aa1254cf4241bf87c538d9cdcd37b98f6`.
- Assumed alignment SHA values:
  secondary `8ccd290847d5caabdb0f75396baace56c437a4f00d545c66f15c0edbbc1ac139`;
  additional `6dcb8a292c0757b96eab61f81247000e2b3700e5d926455f42ff0f9b6611177b`.

Exact six input pins (paths relative to `artifacts/`):

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `two-view-root-jog-lab-01/run/evidence.json` | 1,903,436 | `375277287e836df1fe4e8930d6a5203857df349dbdbf8752065d14d7ab2a326b` |
| `two-view-root-jog-lab-01/run/camera.json` | 127,993 | `a74764a1143ed8d7af042e45d988404aaa445d5f5e02bdcc1c7b4a3cd6a6ea4e` |
| `two-view-root-jog-lab-01/run/inputs/two-view-root/evidence.json` | 1,949,660 | `f32d63c236f16353ea8e35fe4b3edce0c1607938037563cbf09cc19e1a457e75` |
| `two-view-root-jog-lab-01/run/inputs/two-view-root/camera.json` | 128,196 | `ea6aebc58b3d4cb29b55d73186856751c545988a34dd39aa7ae68c2c7a60529c` |
| `multiview-jog-drone3-image-01/evidence.json` | 1,799,298 | `0bfe50ccc899c87293b0adf906bb690bae8d6943e62e0479241fe79fb709f673` |
| `multiview-jog-full-two-view-01/camera-3.json` | 127,613 | `ce2e79b3e182ca7733ee1e89244fc2bb4886d68a88315861af270338426becb0` |

Plan also pins all three exact ancestor manifests; verification covers these
selected source members, not every unrelated ancestor media/runtime member.
Both Full and Lite model identities remain exact. All spatial and clock claims
are explicitly **assumed**. Only first 144 original frames / 72 sampled rows
per view may supply coordinates; all 96 later source frames remain excluded.
No landmark/view membership depends on the forthcoming result.

Anticipated output is 18 files, approximately 9 MiB, under the approved 32 MiB /
32-file cap; exact copies appear once. Worker CPU 20 s / wall 28 s; parent 30 s
through verified custody sealing. Peak RSS is measured, not a hard memory limit.
The custody manifest intentionally has no completed-success label; only the
separate process receipt can report on-time completion after final seal I/O.
An unavailable heel is a completed diagnostic outcome, not a reason to retry.

Prepared command, subsequently approved and **executed exactly once**:

```sh
.venv-lab/bin/python .local/multiview_landmark_jog_01.py 868c072c72200ec4a174be310d48f2fdfdb56f0d944a54429dc1658f58a59a70
```

## Study 01 terminal failure and diagnosis

`artifacts/multiview-landmark-evidence-jog-01` is immutable: 16 files /
6,143,910 B, 0.339962 s launch through verified custody seal. One public adapter
call / one solver entry; `constrained_stereo._camera` raised
`Camera geometry must be finite, bounded and a proper world-to-camera transform`.
No estimated point, JSONL row export, summary or completed result exists.
The six exact input documents and all three ancestor manifests survive once;
their 240-source-frame / 1,440 requested-landmark / 4,320 view-landmark target
denominators remain known, but **no derived output coverage was produced**.
Frame-zero endpoint statuses therefore are unavailable from this run, not new
ray rejections. The prior strict heel rejection and six negative gaps remain
unchanged. Failure-path RSS was not recorded and is unavailable.

- Process `.local/multiview-landmark-jog-process-01.json`, SHA
  `40ab5b87cd20fde8e5dd1e07011ed8f4ba790f2b58ffe05277e182ba83329dad`.
- Custody seal `manifest.json`, SHA
  `928523bdbe2fe1084bb540ae6ecd163d41032eca884f867b04800370a33ee684`.
- `failure.json`, SHA
  `2a00da847b6c32ca85f03d53457673afafe0487c6daa4453488f66d0ce9092f2`.
- Root's independent saved audit
  `.local/multiview-landmark-jog-saved-audit-01.json`, SHA
  `17d747d7b7fef3781632fe7f90220d78abb144f45afbead99be1c81d50883bc7`:
  all 16 files / 6,143,910 B hashed once in 0.004963 s; selected six ancestor
  members, assumed alignments, absent result and first solver entry verified.
  No re-fitting or source/native/media/model work occurred in that audit.

Static code and copied-camera arithmetic found no field-order, units, shape,
radial or image-size adaptation error. The existing source camera validator
permits rotation orthogonality residual up to `1e-5`; the existing pixel solver
requires `1e-8`. At original frame zero, the exact supplied rotations give:

| Camera | max abs(R Rᵀ − I) | determinant |
| --- | ---: | ---: |
| Primary | 4.4907297436225235e-8 | 1.0000000005816836 |
| Secondary | 4.576341561257635e-8 | 1.0000000094056836 |
| Additional | 5.3044961312309624e-8 | 1.0000000121147392 |

All exceed the solver's unchanged orthogonality limit. These small residuals
are consistent with finite serialization precision, but no higher-precision
source rotation was established, so this does not authorize silently correcting
them. The calculation used only first-frame scalar arithmetic on already copied
cameras; it did not invoke either validator, a fit, an image decoder or physics.

No retry, normalization, threshold change, view removal, source shift or floor
movement occurred. A further fit requires explicit source-qualified corrected
calibration or a separately versioned, reviewed numerical-correction hypothesis
that retains all original camera bytes and alternatives. Merely catching this
exception and exposing unavailable slots would improve diagnostics, not
reconstruction. The current strict solver and original results must remain.

## Explicit numerical-rotation prior — pure verification only

Root authorized a distinct **assumed numerical rounding** hypothesis after
study 01. Default `rotation_prior=None` keeps strict raw-camera behavior; no
upstream/downstream validator or solver changed. Opt in only with:

```python
prepared = prepare_multiview_landmark_evidence(
    views, landmark_indices=(27, 28, 29, 30, 31, 32), alignments=alignments,
    rotation_prior="bounded-polar-rounding-prior-v1",
)
```

`camera_rotation_prior.prepare_camera_rotation_prior(R)` preserves the exact
bounded 3×3 source numbers and returns the closest proper SO(3) polar factor.
It rejects reflections, singular matrices and anything outside the existing
upstream `1e-5` orthogonality/determinant envelope **before** decomposition.
It requires the unchanged downstream `1e-8` limits afterward. The fixed
Frobenius correction bound is `2e-5`, derived from the upstream envelope rather
than the observed study residuals: with ε=`1e-5`,
`||R Rᵀ−I||F ≤ 3ε` implies
`||R−Q||F ≤ 3ε/(sqrt(1−3ε)+1) < 1.501e-5`; `2e-5` provides numerical slack.
This bounds a dimensionless matrix change, not physical calibration error.

Each prepared record retains raw/corrected R and canonical hashes, exact
Frobenius/max-entry corrections, both orthogonality residuals/determinants and
an immutable identity. The adapter creates/caches it once per actually used
construction view/frame; landmarks reference that record. Unbound, insufficient,
unsampled and excluded slots get no correction. Raw camera bytes remain exact;
intrinsics, extrinsic translation, radial distortion, source times, ground and
body stay untouched. Geometry remains conditional; rounding and world/clock
alignment remain **assumed**, never newly measured.

Current frozen pins:

- Prior module `920bd77e967f82e557fb88817491237970aab6a4e7a03628211f6850f9092a97` (5,048 B).
- Prior tests `2979428467ac6b6d3033663a637f9e51343fe7fccef9b40337afad5f9e9fa455` (5,066 B).
- Adapter `3ea727bdd232d9f7862341a9949d047af367dad3d2bc5376baa2ff2b6e2e916c` (18,401 B).
- Adapter tests `e215ea2e1d2035b74e66a0440ebf0b147ffecc39134291e7bcf6128a6df36c60` (24,556 B).

Executed once, only new and affected cases:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_rotation_prior.py tests/test_multiview_landmark_evidence.py::test_rotation_prior_default_stays_strict_without_silent_correction tests/test_multiview_landmark_evidence.py::test_rotation_prior_cached_per_used_frame_preserves_raw_cameras_and_clocks tests/test_multiview_landmark_evidence.py::test_rotation_prior_does_not_touch_unbound_or_insufficient_ray_slots tests/test_multiview_landmark_evidence.py::test_rotation_prior_requires_exact_explicit_profile tests/test_multiview_landmark_evidence.py::test_exact_geometry_uses_every_supplied_view_and_all_slots tests/test_multiview_landmark_evidence.py::test_cached_queries_do_not_reparse_evidence_fit_or_hash -o junit_family=legacy --junitxml=.local/camera-rotation-prior-tests-01.xml
```

Receipt SHA `7ee0761db7b687b9ea0a0fbea3e976155736a9fd5cbf7ef1e5dbcae4bcf1d104`:
27 passed, zero failures/errors/skips, 0.161 s pytest / 0.258 s command wall.
This includes 23 new cases and four affected strict/cached controls, not a full
rerun. The cached analytical case performed 12 frame/view corrections for 24
all-ray fits in 0.015374 s; raw camera hashes, fractional source clock, excluded
payload nonuse and every denominator remained intact. Native/model/media,
network and subprocess work were blocked. No real source fit was repeated.

Any study 02 must use a new frozen plan/runtime/output and explicitly activate
this prior after review. Its all-view result can only be a conditional estimate:
the older two-view producer used a different objective, so no causal claim that
the third view alone improves reconstruction is permitted. Do not add a fitted
baseline merely to support a claim we will not make. Study 01 and every original
rejection remain immutable.

## Frozen study 02 preparation

Root's independent review cleared the prior/helper adapter delta and reused the
27 checks. At October 31, 2026, 3:44:56 PM PDT, a new preparation copied the
previous verified minimal runtime closure and changed only the four reviewed
source/test overrides. It performed zero original-source reads and zero fits.
No dirty tree archive, installation or repeated tests were used.

- Plan `.local/multiview-landmark-jog-preparation-02/plan.json`, SHA
  `c2f8782f4f52a4cb1d0c696f6fb5333eb39fb4bd5359f80cd1c20191d6ba77b7`.
- Worker `.local/multiview_landmark_jog_02.py`, SHA
  `aec0b7791d7bf60ba65069e44792d458192638e6396810b416491fc50bb1d16f`.
- Runtime `.local/multiview-landmark-jog-preparation-02/runtime.zip`, SHA
  `a8eefda3cef8f6d656e00e57ddea2f476aa0f5f03d264da62249e4b9592ed8dd`,
  41,011 B / 13 members. Base remains `6c73bf77295fca7fdf0dec479b46c28d4afea981`.
- Preparation `.local/prepare_multiview_landmark_jog_02.py`, SHA
  `7a69fcbd4c6c362e92b1c4e3a86c86d3d7dca18727cc4af97608e3153c1da8a2`.

All six original input pins/paths, both exact assumed alignments, landmarks,
first-60% partition, all-qualified-ray membership, solver limits and time/byte
bounds are identical to study 01. The **only scientific delta** is explicit
`bounded-polar-rounding-prior-v1`. At most 216 used view/frame corrections feed
at most 432 fits; complete raw/corrected records remain in the original-frame
JSONL. New counters/statistics expose that cache behavior. Failure telemetry
now records RSS, without backfilling study 01's missing measurement.

Study 02 also retains the three exact small prior failure/process/seal receipts,
not another copy of study 01's source documents. Anticipated output is 21 files
/ about 9 MiB, within the unchanged 32-file / 32 MiB aggregate cap. Worker
network/media/model/native exclusions and post-seal completion rules remain
unchanged. The historical two-view result and study 01 retain separate identities.

Root reviewed this exact plan and authorized the command, **executed once**:

```sh
.venv-lab/bin/python .local/multiview_landmark_jog_02.py c2f8782f4f52a4cb1d0c696f6fb5333eb39fb4bd5359f80cd1c20191d6ba77b7
```

## Study 02 result — completed diagnostic, heel hypothesis rejected

Terminal October 31, 2026, 3:46:06 PM PDT:
`artifacts/multiview-landmark-evidence-jog-02`, 21 files / 8,563,906 B.
One adapter call, 432 all-ray fits and 216 cached numerical-prior preparations;
0.511336 s preparation / 0.656857 s launch through verified custody seal.
One cached original-frame-zero query took 0.0000803 s. Peak worker RSS was
65,404,928 B. Cache 2,386,700 B; exact six input bytes 6,036,196 plus 2,714
alignment bytes. All eleven custody/partition/assumption checks passed.
Native, model, media, network and paid calls were zero.

All 240 source frames / 1,440 landmark slots / 4,320 view-landmark slots survive:

| Landmark status | Count |
| --- | ---: |
| Conditional estimate | 279 |
| Inconsistent image rays | 153 |
| Insufficient qualified views (unsampled source frames) | 432 |
| Later source excluded | 576 |

The 4,320 view slots contain 1,228 available, 68 low-confidence, 1,296 unsampled
and 1,728 later-excluded observations. Construction remains original frames
0–143 (72 sampled rows/view); original frames 144–239 remain excluded. No
missing or rejected slot was discarded, filled or used to select another start.

At original frame zero, landmarks 27/28 (ankles), 29 (left heel) and 31/32 (toe
indices) have conditional all-view estimates. Landmark 30 (right heel) remains
`inconsistent_image_rays` with **all three qualified observations retained**.
Its per-view reprojection errors are 20.967651306663655,
17.2651596471732 and 10.371173985952751 px; the primary error exceeds the
unchanged 19.2 px gate. No view was removed, threshold changed or later pose
selected. This falsifies the predeclared original-start conditional-heel
hypothesis under these fixed observations/assumptions; it does not isolate which
image estimate, calibration or clock assumption is wrong.

All 216 correction records include raw/corrected matrices and pins. Maximum
Frobenius correction was 5.140209654675284e-8 (bound 2e-5); maximum changed
entry was 3.6447474704992544e-8. Maximum raw orthogonality residual
7.999051820917202e-8 became 2.4424906541753444e-15, below the unchanged solver
limit. The correction remains an **assumed numerical prior**, not measured
calibration. No contact/load/grasp, floor/scale recovery, body installation,
independent accuracy or causal third-view-only gain is claimed. The old strict
foot rejection and six negative ground gaps remain unchanged and were not
recomputed by this study.

Frozen output pins:

- Process `.local/multiview-landmark-jog-process-02.json`:
  `be53e08e30e00495a64f243e913897c1a4d484c7941de683ba29db6a995217e2`.
- Custody `manifest.json`:
  `41d529548fcfffcb065cc69a6ad091d064997bc7253a0fb4381614694d2786ca`.
- `result.json`:
  `a18d4c4e743b4f5a070b14ccdc9211d4e07e42ed7f1ce9249a7726549354c2ef`.
- `summary.json`:
  `89617c62f0d9d38b3a39338fdf2d5ab47ae60b0034e8eb1586bf47f578f0cf13`.

Root's independent saved-only study 02 audit passed:
`.local/multiview-landmark-jog-saved-audit-02.json`, SHA
`b569ff9b0db1720697c37e7f6e90ee8e313a7d1b15434c8dea09bbcbeacb7cac`.
One pass read all 21 files / 8,563,906 B in 0.026174 s and checked 240 rows,
1,440 landmarks, 4,320 view slots, 432 saved fit diagnostics and 216 rotation
priors. It verified raw camera matrices, original input pins, retention of every
qualified ray, later-coordinate nonuse, correction norms, proper rotations and
the symmetric-positive polar condition without SVD, fitting or native work.
The original-start right heel remains inconsistent; this audit verifies saved
custody and numerical consistency, not reconstruction accuracy or contact.

Live handles: none. No Git mutations. All five owned product/test/handoff paths
are released for root integration. The independent saved-only audit is complete;
do not rerun fits, repeat checks or tune this exposed source. A new reconstruction
attempt needs distinct source-backed image/calibration/synchronization evidence,
not another rounding correction or relaxed admission rule.

## Published checkpoint — October 31, 2026, 3:59 PM PDT

Product/test bytes are published at `34d406ef0a1dca2e77f925182a8a479d2396b48f` with exact private
remote verification. Receipt `.local/camera-learning-publication-01.json`, SHA
`7529b3fb4827c0abdf3de402607adfe34f6cf1f5c28a8ed64fe829751f59b9c6`. This follow-up changes notes only. Owned
paths/index were clean and stash empty; all 1,249 unrelated avatar status entries
were preserved. See [combined next steps](combined.md); no study is live or should
be repeated merely to resume work.
