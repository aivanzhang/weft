# Exercise surface collection workflow

## Ownership and current milestone

This lane owns only `src/weft/exercise_surface_pipeline.py`,
`tests/test_exercise_surface_pipeline.py`, and this note. Root owns CLI routing,
common documentation, Git and any later native acceptance. The presentation lane
owns `exercise_surface_report.py` and its tests. The released collector,
`exercise_surface_inputs.py`, remains unchanged.

The filesystem workflow is implemented and released. Its analytical orchestration
checks and real collector/reporter/runtime integration check pass.
No live processes, fitting, native models, media decoding, reconstruction,
network calls or paid work were started. This is a conditional-input collection
tool, not a reconstruction, fitting or simulator-admission result.

## Public API and stages

```python
prepare_exercise_surface_collection(
    observations, out, *, source_sha256=None, camera=None, body=None,
    selection=None, request=None, resume=None, expected_manifest_sha256=None,
    max_seconds=30, progress=None,
)
```

- Fresh invocation takes one to eight exact saved observation paths, optional
  camera/body, and optional edited selection. Source identity can be inferred
  only when every observation document declares the same original source.
- Without channels, the successful result is `needs_selection`. Discovery
  validates existing metadata but never prepares interval geometry or chooses
  a person, channel, point, surface or weight.
- Resume requires exactly one edited selection or request, a new output, and no
  raw-source options. An optional exact parent manifest pin is supported. A
  request cannot bypass a missing usable prior selection.
- Selecting existing channel rows invokes the released collector and qualifies
  its unfilled draft, yielding `incomplete_request` and complete worksheets.
- Edited requests use the same collector's qualification. Status remains
  `incomplete_request`, `qualified_but_unavailable`, or
  `conditional_inputs_available`. Unknown values are not zero residuals or
  evidence of contact.

Copy editable files **outside** the sealed package before editing. Editing a
sealed parent in place invalidates its manifest; the workflow will not repair
or silently reseal it. Root exposes these stages through `surface-inputs`.

Successful workflow results have `status: completed`, `sealed: true`, an offline
`report_path`, and a separate `collection_status`. Incomplete input is therefore
not an execution failure. Errors return `status: failed`, no success report/config
link, and a sealed failure package where publication remains possible. A valid
completed seal is terminal and cannot be relabelled failed by a late notification.

## Exact saved contracts

`selection.json` uses `exercise-surface-selection-v1`:

```json
{
  "schema_version": 1,
  "profile": "exercise-surface-selection-v1",
  "input_bindings": {
    "source_sha256": "exact original source identity",
    "observation_document_sha256s": ["sorted exact raw-document hashes"],
    "camera_raw_sha256": null,
    "body_raw_sha256": null
  },
  "channels": [],
  "human_entity_ref": null
}
```

Channels use the existing annotation schema without replacement:
`{id, entity_ref: {document_sha256, object_id}, point_role, surface_id}`.
The initial selection contains no chosen rows. During request qualification,
channel selection and original input pins stay fixed; the collector still permits
an explicitly supplied/edited `body_binding.human_entity_ref`. That request value
is not falsely described as immutable or inferred from the original selection.

`inventory.json` uses `exercise-surface-inventory-v1`. It contains exact bindings,
root alias-to-raw-hash `input_files`, original source metadata, per-document
`source_clocks`, and construction-filtered entity rows:

`{entity_ref, properties, source_slots, construction_annotation_slots,
partition_counts, visibility_counts}`.

Properties remain existing filtered claims; raw global labels and withheld
payloads are not recovered. All original source/entity slots remain represented
in coverage counts. Source/media identity is a retained declaration, not a new
video hash, decoder check, calibration measurement or ancestry authentication.

`workflow-state.json` records the operation, collection status, selection/request/
parent/collector-manifest hashes, original bindings, and plain-text `next_action`.
Its scope always retains unknown contact/load/grasp, unverified physical accuracy,
and false body/floor installation and fit/simulator admission.

Original bytes occur once under the collector's fixed root filenames:
`surface-observation-NN.json` ordered by raw hash, optional `surface-camera.json`
and `original-human.json`. Selected stages preserve exact collector outputs,
including `collection-request.json`, `collection-report.json`, complete
`collection-frames.jsonl`, and `collection-manifest.json`. Qualified surface
inputs still use unchanged `source-surface-fit-config-v1` relative filenames.
The package is **not** a qualified body-source run.

`plan.json` pins original files, edits, parent seal, exact current runtime files
and Python/NumPy versions. Resume validates every member of the previous complete
seal, cross-checks nested collector seals and original-input inventories, then
copies only original source bytes and the retained selection. Prior runtime code
is never executed. The exact parent manifest is retained separately.

## Bounds, publication and reporting

- At most 96 files / 64 MiB in the resulting package; selected original inputs
  at most 32 MiB. Observation reads enforce the collector's tighter aggregate
  16 MiB limit **before** reading another file; camera shares that surface limit.
  Original body is at most 8 MiB.
- Exact runtime archive at most 8 MiB. It includes the actual pure collector,
  reporting and filesystem dependencies, plus the once-read `assets/brand.css`
  (at most 16 KiB). No native engine is invoked by this path.
- Reporter receives only saved selected root bytes, at most 32 files / 32 MiB,
  never archived runtime, the plan or parent manifest. It returns at most 32
  safe flat HTML/CSS/JSON names / 16 MiB, each at most 2 MiB, beneath `inspection/`.
- Workflow budget is 1–30 seconds cooperative, with up to five seconds for
  failure preservation. Existing collector calls have their own cooperative
  bounds; this is not hard native-call preemption, process isolation or RSS
  enforcement. Deadline checks precede final seal publication.
- Paths reject traversal/symlinks. Writes are exclusive, do not replace files,
  and verify copied bytes. Original sources, edited documents, parent members
  and current runtime identities are rechecked before publication.
- Failed pre-publication terminal attempts are preserved with `attempted-`
  names when safe. An already valid completed seal remains authoritative.
  Unsealable failure writes `failure.json` when possible; such a directory
  cannot be resumed as a successful package.

The report callback is
`build_exercise_surface_report(files, *, max_seconds=5) -> dict[str, bytes]`.
It consumes saved bytes only. Bounded progress callbacks fire at inputs,
selection, qualification and report milestones, not per frame. The CLI owns
stderr formatting; result JSON remains separate on stdout.

## Verification and next safe action

Preserved receipts:

- `.local/exercise-surface-workflow-tests-01.xml`: 24 orchestration cases passed.
  Native/media/network/process operations were denied; actual collector joins
  ran, while report/runtime adapters were analytical stubs.
- `.local/exercise-surface-workflow-tests-02.xml`: four affected/new boundary
  cases passed after review: aggregate read reservation, failure before final
  seal, valid completion remaining terminal, and filtered reporter/resume path.
  Unchanged passing cases were reused, not rerun. The first receipt's former
  post-publication-failure expectation was explicitly replaced by the approved
  terminal-completion rule.
- `.local/exercise-surface-workflow-tests-03.xml`: one complete authored
  discovery → selection → qualification sequence passed with the actual reporter,
  actual collector and current 23-file runtime archive, including exact brand CSS.
  No runtime or report stubs were used for this integration check. Original
  package seals remained intact, all four source rows survived, and the unchanged
  fit-config contract was emitted. Shell wall time was 0.216 seconds; this small
  authored fixture is not a maximum-size responsiveness benchmark.

There are 27 distinct current cases: 23 applicable cases reused from run01,
three new/replacement boundary cases from run02, and the actual integration case
from run03. Run02 also reran one affected routing case. The replaced old terminal
expectation remains preserved in its historical receipt, not counted as current
behavior. Final source/test/receipt hashes are recorded in
`.local/exercise-surface-workflow-checks-01.json`.

Covered behavior includes no automatic selection, complete source rows,
conditional/unavailable status separation, exact unchanged config semantics,
deleted-member/resealed-parent rejection, source mutation detection, symlink and
report-path rejection, and preserved failed edits/terminal attempts.

The actual public CLI acceptance below is now terminal. Root owns the independent
saved-output audit and publication. Do not rerun passing checks or launch a fitter
merely to inspect collection output.
No real-source readiness or model accuracy claim follows from these authored
analytical checks; the missing real interval/calibration/correspondence inputs
documented elsewhere remain missing.

## Actual public CLI acceptance

`artifacts/exercise-surface-cli-acceptance-03` passed the three actual
`python -B -m weft surface-inputs` stages. Exact commands and predeclared authored
edits are retained in `plan.json`, not inferred from the outcomes.

| Stage | Saved status | Child wall time |
| --- | --- | ---: |
| `discovery/` | `needs_selection` | 0.383939 s |
| `selection/` | `incomplete_request` | 0.267594 s |
| `qualified/` | `conditional_inputs_available` | 0.270258 s |

The run used exact existing authored metadata from
`.local/camera-surface-native-inputs-02`, not a real source study. Its one channel
and person reference came from the existing authored annotation/binding. Pixel
ranges, surface intervals, body/world mapping, camera, motion and floor were not
invented or changed. Weight was the existing authored value 2. The request set
`envelope_bindings` to null explicitly so the collector could bind its own exact
serialized document hashes.

All 10 original source/channel slots survived: three conditional envelopes and
three conditional body terms, at original body/image-ready frames `[0, 2, 4]`,
plus seven retained unsampled slots. The parent package manifests remained
unchanged through both resumes. The final `surface-config.json` reached actual
`prepare_surface_fit_inputs` and `prepare_surface_fit_objective` successfully in
0.063903 seconds. This was preparation only: zero objective evaluations, fitting,
models, dynamics, media decoding, inference or network calls. Each successful
CLI guard recorded no denied attempts. Native-library loading is distinct from
model/image operations and was permitted.

Outer wall time was 1.007830 seconds. The sealed acceptance contains 155 files /
1,616,663 bytes; its active frozen runtime contains 43 files / 576,489 bytes,
including the manifest. Observed peak child RSS was 87,490,560 bytes, not a hard
limit or simultaneous aggregate. This is one small authored fixture, not a
maximum-size responsiveness or physical-accuracy benchmark. Spend was $0.

The active runtime is `.local/exercise-surface-cli-runtime-04`, from committed
`25eac48` plus exactly the reviewed `cli.py`, pipeline and reporter overrides.
Committed brand CSS and the CLI's import-time palette are included. No dirty
whole-tree or avatar code was executed. A strict project-module finder admitted
only the pinned runtime; operation guards denied model, image, fitter, network
and nested-process entrypoints. These are reviewed-path guards, not an OS sandbox.

Essential pins:

| Artifact | SHA256 |
| --- | --- |
| Acceptance manifest | `95a105f09762b9fc136fa551e120c272b51c64e9ba00c76c15165a3da2591c1b` |
| Plan | `beb29aed3369739ab5b23565e807461d552fde7af8acebe8116db11d0046684e` |
| Runtime manifest | `de2ce7038df42e7d1a20ce5265a1f95d451cd1ac5680afdb4779f90f8a9dc655` |
| Result | `02bb4ab8468ed46597ad294477f5e7e9370b3eb22467b135701ee2eb0bbdd174` |
| Process receipt | `d1e2aa19d0c5ccfbe6468eb53bf44909bde280f9a2e2110b563fe2f177d23c6a` |
| Fitter-contract preparation | `7ad506e7b3c7a64b8492c620fbe2b45c982cb9411890f660f92b9c7bed7a7415` |
| Discovery seal | `49da50596a49f60aa579bd03d6741b4af4c0bf0b3328b2e7ce0348c628ef7dea` |
| Selection seal | `cd1ef1cf9412a3c7e388d766c063cdddbfd1a45919baaecd03370addceafa03a` |
| Qualified seal | `3b21bdba2684bd4cbc9c27c94dad61fff9640adf74eab16e660b501da9d7a47f` |

Open the saved report at
`artifacts/exercise-surface-cli-acceptance-03/qualified/inspection/index.html`
to inspect this result. It does not fit or install anything. Root is auditing
saved outputs independently; no independent-audit pass is claimed here yet.

### Preserved harness failures

The first two public CLI attempts each stopped during eager imports, before input
collection. They remain sealed failures:

- `artifacts/exercise-surface-cli-acceptance-01`, seal
  `cc52d0389c355cde851d3a7da25fee20413ce2f6ea6638f4c93cd9bd6719ee9e`:
  an overly broad harness path filter hid installed dependencies along with live
  source paths. The corrected guard excludes only exact live repo/source paths.
- `artifacts/exercise-surface-cli-acceptance-02`, seal
  `38cac93471cfc35a7ad4ed571173ded87ca9775da4cfefcfd8ef04c8cb1eb324`:
  the harness prohibited Pillow's import although library imports were authorized.
  The final guard permits imports but denies image operations. It also waits for
  module initialization to finish before installing operation guards.

The final run followed a complete read-only eager-import review: NumPy, OpenCV,
Gymnasium and Pillow. Exact guard deltas and previous failure identities are
retained in acceptance03. This was changed-harness verification, not an unchanged
retry or a product correction. There were five actual CLI calls total: one in
each preserved failure and three in the passing sequence. No further retry or
source/native study is active.
