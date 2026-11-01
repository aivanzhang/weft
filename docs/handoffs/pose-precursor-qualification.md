# Explicit pose precursor qualification

## Frozen for integration — October 31, 2026, 8:10 PM PDT

Exclusive active paths: new`src/weft/pose_precursor_custody.py`,
new`tests/test_pose_precursor_custody.py`,
new`tests/test_pose_precursor_annotations.py`, this note, and the narrowly opt-in
integration in`src/weft/pose_surface_annotations.py`. All new paths were absent.
The completed custody note and silhouette paths stay frozen for root publication.
Root owns CLI/worker/report/native jobs/Git; no overlap. No live job here.
Implementation and52 distinct focused analytical cases now pass. These five
paths are frozen. Root reviewed the typed correction and approved one saved
source-prefix custody pilot, now complete. Canonical annotation/raster execution
was not performed; the final section records the exact result and next join.

Initial authorization covered pure implementation/tests. The later single
context-read authorization is retained separately from its frozen prepared plan.
The closed three-file metadata inspection is recorded in
`docs/handoffs/pose-precursor-custody.md`. Its raw precursor has129 original rows
and exact seal/audit pins but lacks full source headers. The prior candidate
observation pins a different pose document; no compatibility is assumed.

## Implemented contract

`prepare_pose_precursor_custody(context_pose_document, raw_pose_document,
identity_audit_document, manifest_document)` accepts four exact byte documents.
It verifies the existing hash-only run manifest's fixed member names and pins,
source/model/frame/audit metadata, and retains all original temporal/identity
labels. It does not fabricate a new raw declaration or rewrite any source.

The optional`precursor_custody` annotation argument opens one bounded in-memory
qualification context. Only the existing admitted branch, after canonical
source/entity/frame/clock/payload checks, requests a raw-frame comparison.
Excluded/missing slots never compare landmark payloads. Each queried row must
match the saved precursor's exact image coordinates/scores and source/frame/
clock/model bindings. Zero repairs alone cannot qualify it. Derived/crop/unknown
representations and mismatches abstain. Original frame0 remains unavailable
when its model produced no pose; no later start is substituted.

Qualified output remains a saved model estimate with unverified anatomical/
person identity, not measured contact, calibration or physical accuracy. The
ordinary annotation provenance embeds compact exact custody pins and explicit
mode/uncertainty labels; complete audit bytes and per-frame unavailable reasons
remain retained. Existing default raw guards and serialized outputs stay
unchanged when this opt-in input is absent.

Bounds:4 documents/16MiB aggregate,≤900 pose rows/1800 original slots,5s combined
cooperative preparation/qualification processing,≤2MiB prepared cache. Exact
input bytes are retained once; frame inspection uses small cached JSON and
never deep-copies all evidence. The synchronous bound is not hard isolation.
Pure analytical tests block native/model/media/network/subprocess work. No
additional original source, raster or provider call is included in this step.

## Concrete public use

```python
from weft.pose_precursor_custody import prepare_pose_precursor_custody
from weft.pose_surface_annotations import prepare_pose_surface_annotations

custody = prepare_pose_precursor_custody(
    context_pose_document=exact_evidence_bytes,
    raw_pose_document=exact_pose_landmarks_bytes,
    identity_audit_document=exact_identity_audit_bytes,
    manifest_document=exact_run_manifest_bytes,
)
annotations = prepare_pose_surface_annotations(
    exact_context_observation_documents,
    pose_document=exact_evidence_bytes,
    source_sha256=source_sha,
    channels=explicit_named_channels,
    landmark_bindings=explicit_landmarks_and_pixel_uncertainty,
    minimum_confidence=authored_threshold,
    uncertainty_method=authored_method,
    uncertainty_note=authored_limits,
    precursor_custody=custody,
)
```

The existing canonical observation documents must pin this exact context's
canonical pose hash. A different older candidate observation is not relabeled.
If new observations are needed, existing`observations.adapt_exercise` can make
them in a **new**, explicitly identified output from the unchanged context.
This helper does not create files, decode media, repair poses or run that export
automatically. It has no independent acquisition or simulator path.

`custody.describe()` exposes source/model/input identities and original audit
counts. `custody.frame(i)` is metadata-only: raw/context source clocks, original
bounded status/identity labels, model-hypothesis count/type, derivation-marker
names and audit unavailable reason. It never returns image arrays or arbitrary
identity payloads. Exact four input documents remain in`input_documents`.

Inside annotation preparation one private qualification context parses each
pose input once, after bounded header preparation. The existing admitted branch
alone requests image comparisons; repeated channels reuse a small frame cache.
All33 image entries must retain the exact five original fields x/y/z/visibility/
presence and match via compact canonical payload hashes. Derived markers,
unknown statuses, noninteger/ambiguous hypotheses, missing decoder times,
changed clocks, missing rows or recorded repaired IDs abstain before comparison.
Unchanged arrays still do not authenticate model left/right identity.

The final annotation object retains the custody inputs; its existing frame rows
contain the matching/unavailable/not-queried diagnostics. The ordinary annotation
document's≤2000-character provenance includes context/raw/audit/manifest pins,
temporal/raw mode and audit-unavailable count, plus authored pixel uncertainty
and unverified identity/contact status. Complete original audit/decisions remain
in exact retained input bytes rather than a truncated or discarded sidecar.
Its larger inspection metadata records every source denominator and comparison
count. Prepared custody cache bytes are included in the annotation's aggregate
cache check. Successful qualification releases its temporary parsed payloads.

The hash-only supplied manifest establishes matching members, not authenticated
ancestry, calibration, decoder timing or physical truth. Header geometry/fps is
still supplied context; exact raw/context clock equality is not independently
verified synchronization. Identity-continuity audit abstentions remain visible
even when an unchanged image estimate is usable for conditional image analysis.
No ground/equipment/contact claim or body admission is enabled here.

## Verification and failures retained

All commands used the existing environment and blocked native/model/media,
network and subprocess work during tests:

```text
.venv-lab/bin/python -m pytest tests/test_pose_precursor_custody.py tests/test_pose_precursor_annotations.py -q -o junit_family=xunit1 --tb=short --junitxml=.local/pose-precursor-tests-01.xml
.venv-lab/bin/python -m pytest tests/test_pose_precursor_custody.py tests/test_pose_precursor_annotations.py -q -k 'zero_repairs_never or complete_metadata or original_labels or wrong_prepared_context or no_opt_in or actual_annotation_path or unqualified_rows' -o junit_family=xunit1 --tb=short --junitxml=.local/pose-precursor-tests-02.xml
.venv-lab/bin/python -m pytest tests/test_pose_precursor_custody.py::test_representative_preparation_qualification_and_cached_inspection -q -o junit_family=xunit1 --tb=short --junitxml=.local/pose-precursor-tests-03.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/pose-precursor-tests-01.xml` | 35 pass/1 fixture expectation failure,.275s | `ab016ae6a1d5e3098c3ad0631a0a9cd0d10ecb3fea6091d7fa732605e85d4bda` |
| `.local/pose-precursor-tests-02.xml` | 20 new/affected checks pass,.163s | `cb547abfedc22ef4f4993d4edc71a03cbc77b443767c139c2d1708c6297e02c7` |
| `.local/pose-precursor-tests-03.xml` | Updated metadata-workload timing check passes,.065s | `50d5fbfe89a23a6e0bcab6c9e5ea8b9c0bf4fd0b8dcbc62426c78b3c0aa1c1f5` |

The first mismatch fixture changed fps without changing its recorded clock, so
the existing canonical validator correctly rejected it **before** the intended
exact-context check. The fixture now varies only a scope label; the precise
context-byte rejection is exercised. Original failure receipt retained; no
production validation was weakened. After adding bounded original status/marker
metadata, only affected checks and its representative timing were rerun.

There are37 distinct latest passing cases: source/model/member/mode/repair-count
contradictions, exact coordinates and score mismatches despite zero repairs,
derived/crop/unknown statuses, missing clocks/rows, repaired-row exclusion,
withheld/payload-mismatched canonical sources, metadata sentinel non-disclosure,
byte/cache/deadline exhaustion, original unavailable start, cached query
ownership and actual annotation composition.

Default compatibility was first compared directly against the published module
from commit`9cbc2836c5fef361b378e932e4c0e8b57b42b26d`, module SHA
`a4d29c84f46f0be63566c32947ad4909a5ff2d54758b1ec75a9823cc408076e6`.
Read-only snapshot and provenance:
`.local/pose-precursor-default-baseline-01.py` and
`.local/pose-precursor-default-baseline-01.json`. Three unmarked/raw/temporal
analytical examples matched annotation bytes, metadata bytes and every cached
frame exactly. Receipt01 records all nine expected hashes. The final test uses
those fixed hashes and is self-contained; it does not depend on ignored local
files, Git access or importing a second implementation in future runs. No old
pose test suite was rerun.

Representative current-code workload:61 original source slots/30 image-array
comparisons prepared and qualified in0.020298s;1,000 cached metadata queries
took0.002331s. All61 slots remained:30 matched/30 unsampled/1 container-only.
This is one local analytical workload, not maximum-bound stress or real-source
accuracy evidence. Native/media/model/provider calls and paid cost:0/$0.

## First freeze, preserved

| File | SHA256 |
| --- | --- |
| `src/weft/pose_precursor_custody.py`,15,415 B | `e5a4d3e2fbf660e177ec5e37216c5508f5c416109367fa9a4c23c74ae2f9bd83` |
| `src/weft/pose_surface_annotations.py`,20,447 B | `49da6ee96c30c943c7db07b4450c3ed35298276654428330b7cf663a720167db` |
| `tests/test_pose_precursor_custody.py`,10,961 B | `0e5c3c2c94dc9d3b5b4dfab57f1a2eef4f3df6d1e5b926795a0bc9ed524d9b65` |
| `tests/test_pose_precursor_annotations.py`,7,819 B | `b49cd0d4b05c099cd79e847a122d9cb32451b6f91b7bffe21a990fe0bfe977f9` |

All five final hashes including this handoff and exact verification receipts
are in`.local/pose-precursor-qualification-release-01.json`. Scoped whitespace
checks passed. No Git changes, other agent paths or frozen silhouette/custody
notes were edited. Root owns publication and runtime integration.

Next action is independent review. Actual kettlebell applicability is **not
tested**: its fourth full context member has not been reread, and the previous
canonical pose mismatch remains. If root later approves a named one-file
context read and frozen metadata-only qualification plan, use the exact existing
source/precursor/audit/manifest pins and preserve any incompatibility without a
retry or label change. No source study, raster read or automatic inference is
authorized by completion of these analytical checks.

## Typed audit-method correction

Root identified that raw-mode `audit.method` could contain a dict/list and leak
that structured payload into public metadata. Two focused cases reproduced the
leak before the fix. It now accepts only null/missing or a nonempty string of at
most160 characters; temporal mode still requires its exact known method and
implementation hash. The supplied label remains unauthenticated. Original raw
audit bytes, mode, identity uncertainty and all unavailable reasons remain.

Review of adjacent output fields found that source IDs/clocks/counts, original
status/identity labels and audit unavailable reasons already have typed bounds;
model/declaration payloads expose hashes, not dictionaries. Four new sentinel
cases prove those guards reject coordinate-shaped metadata. Arbitrary retained
input payloads never become projected public metadata merely because they are
present in an audit or excluded frame.

```text
.venv-lab/bin/python -m pytest tests/test_pose_precursor_custody.py -q -k raw_audit_method_cannot_export_coordinate_payload -o junit_family=xunit1 --tb=short --junitxml=.local/pose-precursor-audit-method-red-01.xml
.venv-lab/bin/python -m pytest tests/test_pose_precursor_custody.py -q -k 'audit_method or other_public_metadata_sentinels' -o junit_family=xunit1 --tb=short --junitxml=.local/pose-precursor-audit-method-green-01.xml
```

| Receipt | Outcome | SHA256 |
| --- | --- | --- |
| `.local/pose-precursor-audit-method-red-01.xml` | 2 expected failures, .056s | `38d305ccd72c317dc9048172319c1697bfe06e37c718985f3c5ad956278af71c` |
| `.local/pose-precursor-audit-method-green-01.xml` | 15 new focused passes, .078s | `56f5b6a7d1198a205aedfcfe204c48b10a5a004ed5fee3e7f49a0080f62e9d20` |

Total latest distinct passing cases:52. The ordinary annotation module and its
integration test were not edited; the earlier exact default-output comparison
remains applicable. No unchanged passing suite or workload was rerun. Original
failure receipts remain intact. Native/media/model/source/raster/API calls:0.

| Current file | SHA256 |
| --- | --- |
| `src/weft/pose_precursor_custody.py`,15,601 B | `1041d19d19f0a30425218ddaa3a6bbe75e9c9f776832ed8d4f9eade1d30f573b` |
| `src/weft/pose_surface_annotations.py`,20,447 B | `49da6ee96c30c943c7db07b4450c3ed35298276654428330b7cf663a720167db` |
| `tests/test_pose_precursor_custody.py`,14,063 B | `416ead4155ea82ba0956e18e684b3206b55a1cc109d5e379bd33f1b247bdc6a8` |
| `tests/test_pose_precursor_annotations.py`,7,819 B | `b49cd0d4b05c099cd79e847a122d9cb32451b6f91b7bffe21a990fe0bfe977f9` |

All five current path pins, including this note, are frozen in
`.local/pose-precursor-qualification-release-02.json`. The previous release01 is
preserved. Completed `docs/handoffs/pose-precursor-custody.md` remains untouched
at `aab2b0df098cb80f651f1afc0bf81b2010fa651237fd25a7af5e7d2fa392653c`.

## Prepared saved-metadata pilot — pre-execution freeze preserved

Concrete next question: do the exposed kettlebell context's requested original
image arrays and scores exactly match its sealed raw precursor despite the
retained temporal audit? Matching establishes only saved model-array custody.
It cannot establish correct anatomical/person identity or contact.

The plan opens exactly one new original source member and reuses the three
existing exact copies from the closed metadata discovery:

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| `artifacts/mobile-exercise-e2e-10/kettlebell-pushup/run/evidence.json` | 2,162,428 | `df21a959787fc098711f4762151cc72c3d53bb87e4199fd2e901fc465a1b4ec8` |
| `.local/pose-precursor-custody-01/inputs/pose-landmarks.json` | 2,024,499 | `e60c7257dead659a5d4add202e2c45c0196556140978b59903b489f24bc7e264` |
| `.local/pose-precursor-custody-01/inputs/identity-audit.json` | 16,851 | `2c74ba5a3c842a9bb8921f89f4b9b50763e813b652f310f4f1eebd1575ea7d5b` |
| `.local/pose-precursor-custody-01/inputs/manifest.json` | 10,122 | `e6e88ded526ec582b138dfd90285a0cdde65ddfd33596d255cc23f95afe5529b` |

Original source identity is
`4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`.
Expected supplied metadata:640×480,30fps,259 container slots/257 decoded slots,
129 sampled rows at original frames0,2,…256. These are predeclared identity
checks, not newly verified decoder measurements. Raw precursor and audit
originals will not be reread; all four selected bytes are copied once into the
new output. Source member hashes are checked against the retained original
hash-only manifest; no authenticated ancestry claim follows.

Selection is fixed before the context read: first77 of129 sampled rows
(`floor(.6×129)`), original frames0,2,…152. One custody preparation and one
qualification context make exactly77 requests. All259 original slots remain:
77 construction/52 excluded later sampled/128 unsampled decoded/2 container-only.
Later/missing slots retain metadata and unavailable reasons but never compare
or normalize landmark payloads. Original no-pose frame0 is retained; no later
start is selected. Any header/member incompatibility fails; each unsuitable
construction row abstains without retries or changed guards.

This deliberately does **not** call `observations.adapt_exercise`: that exporter
reads later landmark payloads before assigning its partitions. The older
candidate observation also pins a different canonical pose (`0a0128e9…` versus
context `78dd09f9…`). Neither observation relabeling nor a new canonical export
is part of this pilot. The product's actual canonical admission/integration
path remains covered only by analytical tests. Successful prefix comparison
would make a later source-qualified observation join actionable, not complete
it. No confidence threshold, pixel-error assumption, annotation, silhouette,
surface, ground, heel, contact or geometry is invented in this custody study.

Frozen preparation files under `.local/pose-precursor-kettlebell-01/`:

| File | SHA256 |
| --- | --- |
| `plan.json` | `731a5c919de869e7bbfdc3ca529b95a77d91c7b303180fe402da102c9c5fea44` |
| `study.py` | `bb193bd1b7f93b32bc8b0e96ec109bf75c7535c5ab3de71e80a4a9a661471d12` |
| `runtime.zip` | `c1eebb3a34373d5eb9fda6a0324a4155204ecce168feec2fc60a4b56cc1f8a9c` |
| `prepare.py` | `0190e11c2ab5129523ed112353235c90d4be4c0a8b1c4803d711c6c1253016cb` |

Runtime is17 explicitly named modules/307,543 uncompressed bytes/88,119 zipped
bytes from committed `9cbc2836c5fef361b378e932e4c0e8b57b42b26d`, plus only the
two current owned product overrides above. No dirty shared/other-agent files
were archived. AST-only inspection verified25 top-level project import edges
are included; it did not import the adapter or read source data. The annotation
override is pinned for interface identity; the pilot invokes only custody.

Bounds:exact4,213,900 input bytes; one original≤4MiB read/5s cooperative read
bound; helper preparation+comparison5s cooperative; one worker20CPU/28s worker/
30s outer wall;16MiB/24 output files including capped logs and reserved receipt.
Observed terminal child peak RSS must stay≤512MiB, but no hard memory ceiling
is claimed. Trusted exclusive writes are counted; parent stats at most once
per second and once after termination. Guards forbid native/model/media,
network/nested subprocess and extra original artifact reads. They are not a
hostile-code filesystem sandbox. Cost:$0. A member seal proves byte integrity;
the separate process receipt decides bounded completion. Exhaustion may leave
an honest unsealed failure. No overwritten output, automatic retry or repair.

Preparation command (already completed; code/lock inputs only):

```text
PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python .local/pose-precursor-kettlebell-01/prepare.py
```

Future command, **not executed; root review/approval pending**:

```text
.venv-lab/bin/python .local/pose-precursor-kettlebell-01/study.py 731a5c919de869e7bbfdc3ca529b95a77d91c7b303180fe402da102c9c5fea44
```

Target `artifacts/pose-precursor-kettlebell-01` is absent; no process/logs/native
handle exist. Next safe action is root review of the typed fix and this frozen
plan, not execution or additional discovery. Root owns common integration and
Git. This lane performed no Git mutations or edits to other agents' files.

## Approved one-shot result and saved-only audit

The preceding status describes the immutable preparation checkpoint. Root then
reviewed the typed fix and exact plan/runner, authorized **one** invocation, and
prohibited retries or additional source reads. Authorization is a separate
`.local/pose-precursor-kettlebell-01/authorization.json`, SHA
`9a0e04aac522d80b7abeaa25cf9de04f3711ccba32ab336122221a195aa46c4b`;
the prepared plan was not rewritten. The exact command above ran once and
finished October31,2026,8:08:57PM PDT, exit0. No live handle remains.

| Denominator/outcome | Count |
| --- | ---: |
| All original source slots retained | 259 |
| Fixed construction-prefix requests | 77 |
| Exact original image-array/score matches | 75 |
| Prefix abstentions | 2 |
| Later sampled poses, not queried | 52 |
| Unsampled decoded slots | 128 |
| Container-only tail slots | 2 |

Original frame0 remains unavailable: its no-pose/no-model-hypothesis status
fails the identity-label guard (`unsupported_precursor_identity_status`). One
other prefix row has `unavailable_precursor_pose_status`. No coordinate or
confidence-score mismatch was found among the75 arrays actually compared;
that does not validate model identity or accuracy. The temporal mode, zero
recorded repairs and **all90 identity-audit unavailable records** remain.
No audit uncertainty was promoted to a verified person or hand correspondence.

The worker read exactly one original source context file,2,162,428 B, and three
previous copies,2,051,472 B. It made one preparation, one qualification context
and77 requests. Preparation took0.042390s; qualification0.074293s; parent wall
0.291786s. Terminal child peak RSS was59,654,144 B, with no hard-memory-limit
claim. No bound violation or forbidden attempt occurred. Native/media/raster/
model/network/API calls and cost:0/$0.

Output:12 sealed artifact files/4,525,193 B. Including two empty logs and the
external process receipt there are15 execution-output files. The seal records
only member integrity; the separate successful process receipt records bounded
completion. Important exact pins:

| Artifact | SHA256 |
| --- | --- |
| `artifacts/pose-precursor-kettlebell-01/manifest.json` | `36bf66c2ac6d070c865593d93028367f6bea2a06857146776b5b1bb8b1d786fd` |
| `artifacts/pose-precursor-kettlebell-01/result.json` | `40f46fb9970be90c81fced33a2d227918900880e04ad1dfe258b7683701fd0fd` |
| `.local/pose-precursor-kettlebell-01/process.json` | `a75a535cbe3201f977bf8fc2279481d847714eaa988b363cf7b768d18d6f85c4` |
| `.local/pose-precursor-kettlebell-01/saved-audit.json` | `d1216068447bf723fef9f900415fdd2189e50d2e73c2ed9fe9a507caf28893cb` |

One separate stdlib saved-output audit ran via:

```text
PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python .local/pose-precursor-kettlebell-01/audit.py
```

It passed in0.032426s, reading14 saved files/4,527,129 B exactly once, verifying
all12 artifact files including the seal,17 runtime members, exact selected
input bytes, source/frame/clock labels, all259 partition rows,75 recorded
payload comparisons, preserved abstentions and successful bounded-process
receipt. It did not reread the original source, reexecute the adapter, inspect
excluded coordinates, decode media or run geometry. This is an independent
saved-byte/diagnostic consistency check, not independent image truth or contact.

## Smallest next integration — proposal only

The remaining join is a **partition-safe canonical exercise export**, not new
pose inference. Existing `observations.adapt_exercise` computes measurements
and deep-copies every sampled payload before calling the canonical validator.
Simply exporting then changing partitions would inspect excluded coordinates.
The old candidate's canonical pose pin still differs; it must not be relabeled.

Proposed narrow change: add an explicit fixed construction-frame selection to
that existing adapter, with its current default byte-identical. In the opt-in
path, record/derive measurements only for selected original pose IDs. For later
sampled slots retain bounded typed IDs, both clocks, original status and an
explicit exclusion reason, with no coordinate payload copy or measurement.
Keep all decoded observation slots and container/decoded source denominators;
retain the exact original source document separately and its original full
canonical fingerprint as custody, not a repaired/new pose. Do not interpret the
fingerprint operation as permission to extract excluded coordinates. A
coordinate-sentinel test must prove excluded payloads cannot reach exported
measurements or source-point queries; no new favorable frame selection.

The resulting **new** canonical document can feed the current
`prepare_pose_surface_annotations(..., precursor_custody=...)` with explicitly
named left/right wrists and separately fixed authored pixel uncertainty and
confidence threshold. Qualified array custody does not guarantee either wrist
passes that point-level threshold. The existing silhouette adapter can then
reuse the original equipment document, mask identities and verified
source-equivalence clock mapping. All masks, occlusions, holes and excluded
slots still require their existing checks. No contact/noncontact/grasp/load or
ground geometry follows from that image consistency route.

No exporter edit, annotation run, mask read or additional source study was
started. Next action is root ownership/scope review of this narrow integration;
any actual raster study needs its own frozen bounds and evidence plan. Product
files remain at release02 hashes; only this handoff gained terminal evidence.
Current five-path and study receipt pins are in
`.local/pose-precursor-qualification-release-03.json`. Root owns publication.
