# Silhouette constraint for equipment co-motion

## Current status — October 31, 2026, 9:50 PM PDT

Owner `/root/exercise_reconstruction`: NEW
`src/weft/equipment_silhouette_constraint.py`, NEW
`tests/test_equipment_silhouette_constraint.py`, this NEW handoff, and only the
necessary opt-in call in `src/weft/exercise_support_evidence.py`. The new paths
were absent. Root approved pure implementation and small analytical checks;
no saved-source evaluation, raster/pose/native/model/API work was performed.
Release02 fixes the independent review's raw-to-canonical custody gap; 34 distinct
focused cases pass. It is frozen for delta review. No live process. Root owns
saved loading/workflow/CLI and publication;
`support_ui` owns the separate report, and `support_learning` owns independent
review. Do not edit their paths or rerun completed component/source studies.

Previous registry/silhouette/report milestone is published and remote-verified
at `791127de2a71939ab7d251a8d703b41d7b93652f` (root receipt
`.local/equipment-evidence-publication-01.json`, SHA
`b15241d87e972779a886a87896f359a355f1dad129a6e21fd0d4a212a2d34533`).
Its sixteen released paths stay frozen. The selected-frame correction, 32
analytical passes, real02 composition and preserved01 failure are in
[the reconstruction handoff](exercise-reconstruction.md).

## Concrete integration

Add only an explicit optional keyword to the existing estimator:

```python
estimate_exercise_support(
    pose_evidence,
    camera_prior=explicit_fixed_camera_prior,
    equipment_observations=exact_equipment_bytes,
    equipment_ids=explicit_ids,
    equipment_source_equivalence=verified_proof_if_needed,
    equipment_silhouette=prepared_silhouette_evidence,  # Default None is unchanged.
)
```

The existing box/co-motion computation remains the baseline. The new consumer
can only restrict its equipment candidates, not generate a candidate from an
image intersection. It must qualify every point in the candidate's original
temporal window: any image exclusion removes it; unavailable or conflicting
evidence abstains. Other baseline outcomes stay unchanged. No body/floor/root
shift, physical contact/noncontact/grasp/load/support/ground admission or scorer
change is allowed. Image masks and wrist landmarks are model estimates, and the
pixel uncertainty remains explicitly authored rather than confidence-calibrated.

Exact required joins: existing canonical pose hash; supplied pose raw identity
from the prepared annotations; equipment raw bytes/IDs; verified source alias
and both clocks; original frame/evidence/observation IDs; the same construction
prefix; named wrist and exact pixel already used by co-motion. Excluded/later
silhouette frame payloads must not be queried. Different point definitions or
incompatible bindings do not authorize reinterpretation.

Retain all original candidate fields, windows, coverage and slot records.
Record each original pair status, the original whole-baseline canonical hash
and original equipment status counts; add a bounded per-candidate image-support
decision. The default None branch must remain byte-identical and avoid importing
the new consumer. Prefix-count caches avoid rescanning overlapping windows.

## Implemented bounds and checks

- Keep the estimator's existing 10-second cooperative deadline and 8MiB complete
  output cap. New preparation uses at most5 seconds within that same deadline,
  at most 2MiB serialized point/prefix cache and ≤540 construction poses ×8
  equipment ×2 wrists. This is not a hard process-memory or time ceiling.
  Existing prepared silhouette cache ≤8MiB is checked once; no raster loading,
  coordinate inference or repeated complete evidence copies per window.
- Pure analytical cases: supported window, one mask exclusion, missing/occluded/
  excluded/conflicting point, identity/clock/canonical-pin mismatch, unchanged
  baseline noncandidate, original window/count retention, exact default golden,
  native/model/network denial, deadline/cache bounds and a representative cached
  overlapping-window workload. Reuse existing passing component checks.
- No real saved-source evaluation before a concrete interface/runtime/input
  plan is frozen and reviewed. The latest02 result is exposed development
  evidence, not an unfamiliar/final holdout or physical validation.

## Public callable and saved reload boundary

`constrain_equipment_support(baseline, prepared_silhouette, *, deadline)` is
installed through the estimator's optional keyword above. The consumer intersects
its five-second cooperative allowance with the existing absolute deadline,
including final baseline/result hashing and size checks. The default branch does
not import the consumer and retains the exact published output bytes.

The prepared object must be an actual `EquipmentSilhouetteEvidence` with exact
`PoseSurfaceAnnotations`. It reads the silhouette metadata and immutable row
cache, the annotations' metadata and annotation document, and pins their exact
pose/canonical-observation/equipment/mask bytes. It does not call the annotations'
`frame()` or live precursor qualifier; restored historical precursor provenance
is not a new qualification. A derivative source requires a real bounded
`load_source_equivalence` result, not an inspection dictionary or copied source
hash. Root is implementing that strict saved-object loader separately.

Only the exact first `floor(.6*N)` original pose frames are decoded from the
silhouette cache. All other cached bytes participate only in a length-prefixed
opaque digest; no excluded coordinate payload is parsed. The digest prefixes each
row by its unsigned eight-byte big-endian byte length and is explicitly not a
source-file hash. Every selected wrist checks original nominal/decoder clocks,
verified derivative mapping, canonical entity/observation/evidence IDs, exact
pixel/confidence, point origin and equipment box. A differently defined
`identity_raw` point cannot qualify from pixel equality alone.

Every original window remains exact. Missing/conflicting image evidence takes
precedence over exclusion, and both counts remain visible. Only original
`projected_co_motion_candidate` pairs can be kept. Other original outcomes cannot
be promoted. Prefix counters answer each overlapping window without rescanning
its points. All original pose rows and complete original-source coverage remain.

## Serialized result contract

The baseline profile stays `exercise-support-evidence-v1`. Its rows, coverage,
window definitions, ground fields and headers remain unchanged except active
equipment-pair status counts. The added `equipment_silhouette_constraint` member
has profile `equipment-silhouette-window-constraint-v1`, status
`applied_restrict_only`, original canonical baseline hash/counts, exact input
bindings, construction frames, per-frame point decisions, counts/work/bounds and
explicit unknown contact/noncontact/grasp/load. It never installs geometry.

Each pair adds `co_motion_baseline_status` and `silhouette_constraint` containing
`status`, `reason`, half-open `window_point_range`, and the complete
`window_image_counts` (`intersection`, `exclusion`, `unavailable`). Decision
statuses are `kept`, `excluded`, `unavailable`, `not_candidate`, `excluded_later`.
The active status is unchanged when kept, `image_silhouette_exclusion` when
excluded, or `unavailable` when a required point is unavailable. Point rows are
ordered original construction frames, with equipment order × left/right wrist.

The exact baseline can be recovered by removing the added top-level field,
restoring original equipment counts, and removing each pair's two added fields
while restoring its saved original status. This reconstructs its canonical hash;
it does not rerun the selector or claim physical validation.

## Initial verification and preserved release01 pins

No source/raster decoder, inference, native physics, network or API calls. The
tests block these imports/process routes and use authored binary arrays. Five
derivative cases use the real source-equivalence loader on a clearly analytical
sealed fixture, retaining its different original/derivative decoder times.

Commands:

```sh
WEFT_SILHOUETTE_CONSTRAINT_FIXTURE_DIR=.local/equipment-silhouette-constraint-analytic-01 \
PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q \
  tests/test_equipment_silhouette_constraint.py \
  --junitxml=.local/equipment-silhouette-constraint-tests-01.xml

PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q \
  tests/test_equipment_silhouette_constraint.py::test_default_none_is_lazy_and_identical \
  tests/test_equipment_silhouette_constraint.py::test_actual_loaded_derivative_proof_preserves_both_clocks \
  --junitxml=.local/equipment-silhouette-constraint-joins-01.xml
```

- Initial 28 cases pass in 1.808s; receipt SHA
  `4de639cf150627836b11cb456c391d9f6f4ff08f5efd54cf25a08c6ecc39a692`.
- Five new derivative cases plus the amended published-byte golden pass in
  0.205s; receipt SHA
  `b82891d286723174d1c712d27d6bc6879ab5ba72ed9a0c776c38be48159507b7`.
  Total 33 distinct cases; no unchanged suite rerun.
- Published module `791127d` was executed once on the same retained analytical
  pose/equipment bytes; its 38,621-byte default output exactly equals the current
  default, SHA `adabae52b761e440a7b3e9db84a6ea12e9ac4b7d170dfb489fb92bec64652417`.
  Receipt `.local/equipment-silhouette-default-golden-01.json`, SHA
  `a14f606461364210e5f8704d9208771dfdb20bc9a670cf7d1c24083ff293a53d`.
- Representative analytical workload: 90 construction frames, 180 point slots,
  180 kept original windows, 546 prefix-counter entries; one point pass and no
  new raster/annotation queries. The full test including fixture preparation
  took 0.555s. The consumer's measured `<2s` assertion passed; its transient
  detailed timing file was removed by pytest retention, so no exact isolated
  timing or RSS is claimed. This is not an actual-source responsiveness result.
- Scoped diff whitespace check passed. No Git mutation or stash operation.

Preserved release01 product/test hashes (superseded below):

| Path | SHA256 |
| --- | --- |
| `src/weft/equipment_silhouette_constraint.py` | `eba1d20a758e5c1c10c88887b5f6f79c8111119504d117efec401be19eea9d70` |
| `src/weft/exercise_support_evidence.py` | `59c14428cefa257d07cd718cf874d0f845d50a2abd89211552d7cdb56b7140fc` |
| `tests/test_equipment_silhouette_constraint.py` | `21e2bac22de05e9ce3be6b3d65de6d49b696700dbf6d68e5d813a8ad2ed280f9` |

Actual small analytical result is preserved at
`.local/equipment-silhouette-constraint-analytic-01/support-evidence.json`,
52,979B, SHA `ce3c8a7055cb8b23df8aff5e719493146cb9a146f30c60047872f03afb46ece9`.
Its eleven-file package retains the original baseline and exact authored
input/cache bytes. Counts: 31 source slots, 15 retained poses, nine construction
frames, 30 retained pairs, 18 original candidates all kept; physical claims remain
unknown. Report owner received this saved payload for schema composition without
rerunning the producer.

## Next action and remaining gaps

Independent reviewer checks only the release02 delta, reusing the passes. Root
then joins its strict saved-object loader/workflow with the actual analytical
payload and report. A later bounded consumer evaluation may reuse completed real
composition02 from frozen runtime; no such evaluation has run in this lane.
No reraster, original-source scan, threshold change or candidate tuning is needed.

This restrict-only feature improves the existing selector, not automatic ground,
equipment pose, physical contact/grasp/load, unfamiliar-video accuracy or body
admission. The real source remains exposed development evidence. Original source
failures and all excluded windows remain intact; source-compatible image evidence
can still abstain or reject every candidate.

## Release02: raw-to-canonical custody correction

Independent review identified that release01 compared the baseline's canonical
pose hash with a declaration while separately checking raw bytes against their
raw pin. Coherently replacing raw bytes and raw-pin declarations could leave a
stale canonical declaration undetected. Root's strict saved loader already
checked this, but the generic estimator API must independently join the bytes.

The consumer now parses the bounded ≤8MiB exact pose document once and derives
its canonical hash under the existing 32MiB canonical-input/deadline guard. It
compares that derived hash to the already-matched declared/baseline canonical
identity before selected silhouette queries. The full pose is processed solely
for integrity, not for excluded coordinate evaluation or candidate selection.
The excluded silhouette row cache remains opaque. No output schema, default
behavior, source threshold, clock, scorer or physical claim changed.

The focused adversarial case changes only a later excluded raw landmark, updates
both raw-pin declarations plus the embedded annotation provenance/hash, and keeps
the stale canonical declaration. It formerly reached the selected frame cache;
after the correction it rejects before any such query. The smallest actual
selector success also still passes. No whole suite or source study was rerun.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q \
  tests/test_equipment_silhouette_constraint.py::test_coherently_repinned_raw_pose_must_derive_the_declared_canonical_identity \
  --junitxml=.local/equipment-silhouette-constraint-custody-red-01.xml

PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q \
  tests/test_equipment_silhouette_constraint.py::test_coherently_repinned_raw_pose_must_derive_the_declared_canonical_identity \
  tests/test_equipment_silhouette_constraint.py::test_existing_selector_opt_in_keeps_supported_windows_and_exact_baseline \
  --junitxml=.local/equipment-silhouette-constraint-custody-green-01.xml
```

- Red receipt: one expected regression failure, SHA
  `b74e04a098d33e69f495a25966fc86daf0fd395a9bc55f27f9dc94b1d357a350`.
- Green receipt: two affected passes in 0.162s, SHA
  `12a6ad66aeb5860cfa5d0ac7657555ca619b5e7561993495cf19268e02cb83b3`.
- Total now 34 distinct passing cases. Release01 receipt
  `.local/equipment-silhouette-constraint-release-01.json`, SHA
  `2169ebf2d337975e8ba3668fac67280239111861a60a0d7fa37176fb639dc0e4`,
  and its exact analytical package remain untouched.
- New module SHA
  `5e6b73eddeab1b22f3d67a29e647f19ac3a60ee9280eadbf26847c347bc75463`;
  new tests SHA
  `83c53c6d0f8f29f6386d66e270025d29fd43ea810b7ac5835b33fc0fff75c628`.
  Estimator SHA remains
  `59c14428cefa257d07cd718cf874d0f845d50a2abd89211552d7cdb56b7140fc`.

No live handle, original-source read, raster decode, inference, native work,
network/API call or Git mutation. Root owns integration and any later saved-only
consumer run after independent delta review.
