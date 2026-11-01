# Optional calibrated two-view body fitting

## Current status — October 31, 2026, 11:10 PM PDT

The optional two-view fitter is frozen. Root executed one reviewed saved-source
fit: the proposal completed and all fourteen saved gates passed. The outer
workflow failed only while generating the offline viewer, which exceeded its
unchanged 2MiB HTML limit. **Do not repeat fitting.** Root and the UI owner are
repairing only the bounded saved-report presentation in a new output directory.
No native, source, projection or fitting job is live in this lane.

The independent saved-fit audit below passed. Both supplied-camera pixel errors
decreased; this is conditional source consistency, not verified physical
accuracy. All 480 surface channels remain unavailable and contact remains
unknown. The original body, ground, root, clocks and first pose stay fixed.
Ground/equipment recovery and unfamiliar monocular generality are unresolved.

Current ownership: this lane's implementation, audit and note are released.
Root owns CLI/workflow/runtime/publication; `support_ui` owns the overflow-only
report patch. Independent repair review is clear. No further science run is
authorized here. Next: root freezes the final report-only plan, performs one
bounded recovery and publishes the completed code/evidence. Preserve every
failed prior workflow and checker result; no Git mutation by this lane.

## Initial implementation freeze — October 31, 2026, 10:27 PM PDT

Owner `/root/exercise_reconstruction`: new `camera_multiview_fit.py`, new
`tests/test_camera_multiview_fit.py`, this note, and only the necessary opt-in
change in `whole_body_reference_refinement.py`. These new paths were absent.
Inspection found no need to edit `camera_body_fit.py`: reuse its existing
projection/Jacobian class. Root owns source preparation, worker/CLI/report,
native acceptance and Git; other agents own independent review and UI. The four
owned paths are ready for review. Thirty distinct focused cases passed, including
the existing QP with analytical kinematics and an exact single-view golden.
At that freeze there was no saved-source/native/media/model/API/network run. Analytical
geometry stubs are not native physics or evidence of real reconstruction accuracy.

The preceding equipment checkpoint is published at
`41cfee28ceef55146fbbf6e29ef8248c475bd46d`; root receipt
`.local/equipment-support-publication-01.json`, SHA
`c461b151ee7ddc39059ec99334e935cfc49cfed6e3c62e173ba3b7eaebeaedf8`.
Both exposed kettlebell co-motion candidates were rejected by existing mask
evidence; no physical contact was recovered. Prior source/heel/floor failures
remain. This next feature is optional calibrated two-view construction, not a
new two-video requirement or proof of unfamiliar monocular reconstruction.

## Implemented public interface

`prepare_camera_body_fit` currently prepares one camera and source-target set.
Its `CameraBodyFit.project()` restores absolute pelvis values and derivatives,
then applies unchanged K/R/t/radial projection and depth/domain checks. The
whole-body fitter already runs one bounded QP with pointwise pixel, depth,
original-floor, joint/rate, interval and original-first-pose constraints.

Use explicit `proposal_objective='camera_multiview_surface_interval_v1'`
and optional `camera_secondary_view` to that existing fitter. Primary camera and
surface arguments remain unchanged. The secondary dictionary contains:

```python
{
    'camera_document': exact_camera_bytes,
    'evidence_document': exact_pose_bytes,
    'observation_documents': {raw_sha256: exact_canonical_observation_bytes},
    'world_alignment_document': explicit_world_alignment_bytes_or_none,
    'correspondence': {
        'primary_entity_ref': {'document_sha256': '...', 'object_id': '...'},
        'secondary_entity_ref': {'document_sha256': '...', 'object_id': '...'},
        'status': 'assumed',
        'method': 'Explicit caller-supplied same-person/model-endpoint hypothesis',
        'uncertainty': 'Not independently verified identity or anatomy',
    },
    'image_scale': 0.01,  # Explicit authored loss units per pixel; [1e-6, 1].
    'weight': 1.0,        # Explicit authored relative loss weight; (0, 1000].
}
```

`prepare_camera_multiview_fit(primary, secondary_view, *, deadline)` reuses
the original prepared primary object. It joins the second camera/evidence
to `body.camera_reference.root_refinement.views[1]` and the exact original root
`source_evidence_ids`, not infer a match from equal video hashes or frame numbers.
Reuse `_view`/`validate_camera`, canonical `_sources`/`_image_admission`, and the
existing `world-alignment-v1` validator. Only an explicitly declared identity
same-world spatial/clock map is supported initially; unknown/missing maps produce
unavailable inspection, and the fitter rejects before geometry. Assumptions stay
assumed. Original body period and fractional source period remain distinct.

Secondary coordinates are read only after exact body/source/entity/evidence/
frame/clock construction admission. Retain every original source slot and
partition, including missing, excluded and unsampled rows. Confidence is a model
score, not calibrated error or identity proof. Unknown surface height does not
erase otherwise qualified image evidence. No new first-60% crop or retiming.

## Existing optimizer integration and versioned output

A prepared composite reuses the same twelve absolute body endpoints through
both fixed cameras, stacks their 24 image rows and preserves named view metadata.
The existing QP uses those rows with its existing bounded block/backtrack loop.
No extra geometry model, optimizer or floor/root/shape fitting. Source confidence
stays `point.weight`; `authored_view_weight` and `objective_weight` are separate.
Each camera has its own authored loss scale; raw-pixel gates never divide by the
other camera's scale. Secondary eligibility is independent of primary image or
surface availability. Each view must have a nonzero qualified target count.

The original fourteen top-level gate keys are unchanged. The global
`source_endpoint_image_nonregression`, `camera_projection_domain`, and
`camera_endpoint_depth_change` are exactly the conjunction of the corresponding
two per-view gates. Failure/status accounting remains the original top-level
gate set. No view may worsen behind a pooled mean decrease or a small view weight.

Projection accounting charges both actual view projections and reserves the
complete final source audit under the existing 120,000-call cap. A supplied cap
must cover both views' baseline and final audits before geometry is constructed.
Source geometry
still contains twelve points and nineteen joint columns. Fresh final geometry
remains independent of optimizer caches; rejected QP/backtrack alternatives stay
in the existing bounded journal. No midpoint or contact target is introduced.

Method: `blocked-multiview-camera-body-surface-interval-reference-v1`.
Distinct overlay: `camera-body-multiview-angle-proposal-v1`, with
`human_candidate=None`. Primary `input_bindings` stays unchanged. The new
`additional_view` is identical to
`report.camera_fit.multiview.additional_view`: `{binding,
binding_canonical_sha256}`. Its complete binding is versioned
`camera-multiview-evidence-extension-v1` and includes exact raw secondary inputs,
canonical camera/evidence/correspondence pins, original root-view provenance,
both authored scales/weights, alignment assumptions and qualification counts.
The hash covers the complete binding without hashing itself.

`report.references[].points` has primary twelve then secondary twelve entries,
uniquely identified by `view_id` plus `joint`. Secondary target origins and full
original slot/partition/nominal/decoder clocks are in
`report.camera_fit.multiview.secondary_source_slots`. Declared, decoded and pose
counts are separate; metadata-only tails retain their unavailable camera status.
`report.multiview_images` is the dynamic `camera-multiview-image-audit-v1`: each
view saves known/missing/invalid counts, raw-pixel means, depth-change maximum,
the three gates and its exact `[0,12]` or `[12,24]` point range.

No human installation or physical-contact admission is emitted. Existing
single-view serialization/default behavior remains unchanged. Worker/report/
controller admission is a separate root-owned boundary; old consumers must not
silently treat the new profile as supported.

## Bounds and runtime closure

- Two supplied views total; at most 900 original references and 1,800 original
  source slots per view. Additional raw inputs total ≤16MiB, each pose ≤8MiB,
  at most eight canonical observation documents. Five-second cooperative extra
  preparation intersects the original fitter deadline; no hard process ceiling.
  Alignment is at most 16KiB, extension metadata 2MiB and combined prepared
  metadata 4MiB. Final serialization and iterative preparation check that same
  shorter deadline. Resource exhaustion cannot drop original slots.
- Existing 30-second fitting +8-second finalization, QP, geometry/projection and
  complete-output/journal limits remain unchanged. No silent dropping of slots.
- Pure tests block native/media/model/network/process work. They check exact raw and
  canonical joins, fractional clocks, missing/unknown/conflicting mapping,
  construction/excluded sentinels, both-view projection/Jacobians, raw-pixel
  gates, default golden, original-first-pose/body/floor preservation, bounds and
  cached-query ownership.
- The actual existing QP with analytical kinematics improves a declared
  two-camera ambiguity case without worsening either view. Contradictory and
  degenerate inputs retain rejection/unavailability. This is not a real-source
  reconstruction or performance benchmark.
- At initial freeze no saved-source run was authorized. Root later reviewed and
  executed the separately pinned preparation and one fit documented below.
  No repeat fit or expanded source study follows from that authorization.

The only new helper module in the fitter closure is `camera_multiview_fit`.
It imports standard library, existing NumPy, `camera_body_fit`, `io`,
`multiview_pose`, `source_surface_anchors`,
`source_surface_interval_construction`, and `world_alignment`. These dependencies
are already in the existing camera/surface closure; no model/native import is
introduced by the new helper. Root owns `camera_multiview_inputs` and its frozen
input aliases/worker route.

## Verified results and preserved failures

All commands use `PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python -m pytest -q`
with focused `tests/test_camera_multiview_fit.py::<node>` selections and
`--junitxml=<receipt>`. Reuse these receipts; do not repeat unchanged passing
nodes. Latest outcomes cover **30 distinct cases**, not a sum of repeated runs.

| Receipt under `.local/` | Outcome | Meaning |
|---|---|---|
| `camera-multiview-preparation-01.xml` | 18 failed | Test fixture used an invalid `world_landmarks` string before the new helper; preserved, not a source result. |
| `camera-multiview-preparation-02.xml` | 18 passed | New fixture sets that field to null before generating exact pins; source joins, clocks, exclusions, projection/Jacobian and preparation bounds. |
| `camera-multiview-qp-01.xml` | 4 passed, 1 failed | Actual QP checks; the ambiguity test incorrectly demanded a pooled 90% improvement despite the mandatory fixed first pose. |
| `camera-multiview-qp-02.xml` | 1 passed | Corrected only that assertion: mutable references improve by over 99%; the first reference stays fixed. |
| `camera-multiview-final-01.xml` | 4 passed | Final complete slot schema, exact clock mismatch, metadata tail, default golden and current saved QP payload. |
| `camera-multiview-budget-red-01.xml` | 1 expected failure | Too-small two-view projection budget previously reached geometry. |
| `camera-multiview-budget-green-01.xml` | 4 passed, 0.151s | Pre-geometry projection reservation, aggregate bytes before parsing, original deadline, complete final audit at the minimum cap. |

The last command selected
`test_projection_bound_reserves_both_view_baseline_and_final_before_geometry`,
`test_aggregate_extra_input_bound_precedes_json_parsing`,
`test_final_preparation_serialization_uses_the_original_shorter_deadline`, and
`test_minimum_projection_budget_retains_complete_final_audit`.
Its receipt SHA is
`3ada839f2fe8dce1731598400e15a90d36d7452504f84e2c5e9fde126a8d8fa0`.

Exact old single-view output is pinned in
`.local/camera-multiview-default-golden-01.json`: canonical result SHA
`bebe97ae1de1af55d62b08cb5244d06f55f3aab243fd81ab94a9ea782ce2c839`.
The passing current default check also rejects importing the extra-view helper.
The original fitter snapshot at committed `41cfee2` is retained as
`.local/whole-body-before-multiview-01.py`, SHA
`d6899edc246836d32e97a86f288388ee174b2c3375bd04c8d8b2a25d2159b4af`.

### Saved authored producer payload for downstream consumers

`.local/camera-multiview-analytic-02` preserves exact final-schema inputs and
actual QP output. Three references, ten original slots/view, 36 known endpoints
per view, scales `.01`/`.025`, authored weights `1`/`3`. Three accepted updates,
44 analytical geometry calls and 38 view projections; all fourteen gates pass.
Both view means decrease from about `.16339869281` to `.05451604147` pixels.
The separate ray-ambiguity case has zero primary-only accepted updates but
positive two-view updates without either view regressing. Neither case recovers
real contact, calibration or biomechanics.

- Manifest: `0527213eff0296380ada56b73b1e8d6c313e28d3a52344ee3cd19b4754921cc7`.
- `result.json`: `b659939717fc7a43f7edefacc19dafcfbe8443532e138abd9f173427a99fa89e`.
- `report.json`: `711233c4fa074a2b40d14bb224540a909182a24d84708d36d4749b2487231ccb`.
- `angle-proposal.json`: `f8129287b3132c4cdbb0c10271f931cfbae1bfbc2c2f5071bd0e25beeaccbdd1`.

The earlier `analytic-01` remains unchanged; it predates explicit secondary
declared/decoded/pose counts and metadata-tail status. Use `analytic-02` for
schema composition, without rerunning fitting. The authored metadata is not
authenticated source ancestry.

## Frozen product pins

- `camera_multiview_fit.py`:
  `33dfd19a656f8893680c93b139860f45068881d9854fa59d3b09cbe873a6a749`.
- `whole_body_reference_refinement.py`:
  `1749db879d283a6e3767fdff4e1e9b8deda1c47e5a25b0a3818ec6362fafcbab`.
- `test_camera_multiview_fit.py`:
  `3f080db48962bb3fc0d466e81da9a8b98e78f7fb38e10dc83e4424b7e76b8028`.

Root config/pipeline/worker/CLI and consumer/controller integration subsequently
passed independent read-only review. Existing analytical producer output was
reused by the pure controller and report owners without rerunning the QP.
Actual jogging clocks remain explicitly assumed, not independently synchronized.
Do not duplicate `.local/camera-multiview-source-discovery-01.json` discovery.

## Saved source preparation and one real fit

The metadata-only preparation retained 240 source slots per view and the exact
112 original body reference frames. Secondary poses: 120 supplied, 112 selected,
8 excluded and 120 original unsampled slots. Qualified image targets are
1,241/704 known and 103/640 missing for the primary/secondary views. Same-world,
clock and person correspondences remain supplied assumptions. This is the
already exposed jogging development source, not an unfamiliar clip or fresh
holdout. No causal benefit is attributed solely to the second view.

- Preparation output: `artifacts/camera-multiview-real-01/secondary-inputs-01`;
  manifest `9e6f6c9b2e62c0e0ebb2612e40e01a05d4bf4777676136d7d696e707ab038867`.
- Saved preparation audit: `.local/camera-multiview-source-saved-audit-01.json`,
  SHA `2e91824d7f2f07acb58fb99932bd61013c27ec3c1597e1f1aea7cfc963f2494f`;
  nine selected files, 5,668,976 bytes, 0.030636s; no source or geometry rerun.
- The losslessly compacted `secondary-inputs-02` preserves those JSON values;
  manifest `fac738246d603a213e885e5898a40ac0499689ed7435f61ee7e60922e66003ca`.
  Raw pins changed only with serialization; original prepared bytes remain.
- Actual fit plan: `.local/camera-multiview-fit-plan-02.json`,
  SHA `50b8305687964245df576be36e28ee8a55026ca9a6a246cff34a063776b3394c`.
  Plan01 was never executed: review found the missing plan-hash child argument.
  The corrected supervisor pins the command and verified plan hash.
- Frozen runtime: `.local/camera-multiview-real-runtime-01`, manifest
  `5d22340e520e958c31d41392121103aa5f8f9a8ff116226219a2a7f6957a6de0`;
  committed `41cfee2` plus fourteen reviewed overrides, no dirty avatar bytes.
- Bounds/settings: 25 exact metadata inputs / 13,901,664 bytes; both image
  scales `.002`, secondary weight `1`; 30s fit plus 8s finalization,
  75s outer deadline. Original body/root/floor/task and 112 references unchanged.
  No media, inference, API or trial. Report input cap 24MiB, output cap 16MiB,
  HTML cap 2MiB; the caps were not relaxed after failure.

The guarded worker completed once in 4.1672s. Root reports two native models,
four data allocations, 18,266 forwards and zero physics steps. These native
counters were not independently reexecuted by the saved-only audit.

Completed proposal: `artifacts/camera-multiview-real-01/fit-01/proposal`.

| Saved member | SHA-256 |
|---|---|
| `manifest.json` | `27c1922017155f2160b805af3fdccf4d3620933944dfab56911d921e9b797adc` |
| `report.json` (6,924,187 bytes) | `b9ea18c28116b65d65902816c53bf88142e5ba09609b42c6b5ebf0b2b9666971` |
| `angle-proposal.json` (117,144 bytes) | `bdd428382939883640a3733ddb620ddca0ad73c4fd02b93d060238a1d4aebd5d` |

The outer workflow failure remains sealed at
`7454d904787761c163a52328cb7df17f62d951efced7c4bf4572f8ed43f7806a`.
Its HTML was 3,032,873 bytes, not evidence of a failed fit. A new viewer must read
the completed proposal, not rerun its optimizer or relabel the failed workflow.

## Independent saved arithmetic audit — October 31, 2026, 11:05 PM PDT

Receipt `.local/camera-multiview-real-fit-saved-audit-02.json`, SHA
`991993f9800efd5f7d90ef3ef7d1b513125d0d96d6c3e4be935b0e76f6be5b98`,
passed in 0.095850s. Four selected files / 7,049,736 bytes; cumulative selected
reads including the preserved first checker attempt total 14,097,886 bytes,
below the 24MiB allowance. It checks exact saved custody, per-point scalar
arithmetic, view/global summaries and gates, original clocks, retained surface
unknowns, floor ledger and journal accounting. No original source, projection,
fitter, selector, media or native call was made; this is not an accuracy audit.

| Saved result | Primary (`drone1`) | Secondary (`drone2`) |
|---|---:|---:|
| Known / missing endpoints | 1,241 / 103 | 704 / 640 |
| Mean error before → after (pixels) | 12.792154124 → 10.868088367 | 27.036613683 → 24.971869424 |
| Invalid known endpoints after | 0 | 0 |
| Maximum camera-depth change (supplied units) | 0.07987277236 | 0.06660360709 |
| Per-view pixel/domain/depth gates | All true | All true |

All fourteen original gates pass; global camera gates equal the conjunction of
the two view gates. There are 375 accepted updates, 104 changed references,
666 QP calls, 6,860 view projections and 18,266 saved geometry calls. The journal
retains 3,206 records / 3,401,477 bytes: 375 accepted, 666 optimal directions and
2,165 gate rejections. Maximum correction/rate are approximately `.15` rad /
`1.5` rad/s. All 351 original floor checks (112 source +239 grid) retain zero
penetration. No floor improvement or contact recovery is claimed. All 480
surface-channel slots remain unavailable; the overlay is an unadmitted
construction proposal and physical validation is unverified.

Checker01 is preserved at
`.local/camera-multiview-real-fit-saved-audit-01.json`, SHA
`f991a718af0aaba440e32981ebfdbf15a06e8ef6e6e8cf59a9a5b0638dd18a2a`.
It failed only because it replaced the producer's floating comparison
`after <= before + 1e-8` with subtraction. Concrete boundary: `drone1`, original
frame 120, `right_wrist`, evidence
`b1b362fa7516879c8ae37c48c40972b0090680e6bdcb6933fcf9ccb8c2e2b0cd:frame:120`:
before `25.341928092928363`, after `25.341928102928364` pixels. `before + 1e-8`
equals the saved after value; subtraction rounds to `1.000000082740371e-8`.
A second boundary is frame 186/right_shoulder, before `8.12505179236417`, after
`8.12505180236417`. Audit02 independently verifies the saved norms and uses the
unchanged producer comparison. Neither tolerance nor scientific artifact changed.

## Read-only recovery review boundary

The CLI error message now distinguishes a completed fit from an inspection
failure and supplies a shell-quoted `body-reference-report` command with the
completed proposal's exact manifest pin and a distinct output directory. Review
found no automatic fit/trial retry; its focused space/`$literal` path test checks
the exact parsed command. Existing generic failure status stays unchanged.

The reviewed report-only helper/supervisor perform full saved output inventory,
logical gzip/alias hash and local-link checks, prohibit native/media/network/
child work, and require caller-observed timely zero exit plus a pinned receipt.
Their final plan must pin the released report repair before execution. The UI
owner is retaining the complete plain JSON and using a lossless packed display
copy only when the original inline HTML exceeds its limit. No fitted value,
source partition, gate, selected evidence or output cap changes. Node DOM checks
are not a browser responsiveness claim. The initial external-script fallback
was changed before execution to inline packed data, avoiding an unverified
`file://` CORS/SRI dependency; final release pins belong in the recovery receipt.

The final inline-only patch passed read-only review at these exact pins:

- `whole_body_reference_report.py`:
  `7e60de58a186b59e04ae8ba1703beeba250d1e454b231bf9af1ea43480aef1e1`.
- `camera_multiview_report.py`:
  `cc72b973d95a694849eb557207444c043a786bb9992195e2c47b650742432729`.
- `tests/test_camera_multiview_report.py`:
  `f37d8d50d8e844f9b6d67d9681a4910f09f98b57c82b73cf8b2c0420ce2f0422`.

The UI owner reports seven new recovery cases plus six affected checks,
including four unchanged-mode byte goldens. Review reused those passes; it did
not rerun tests or generate another report. A single cached-model size receipt,
`.local/camera-multiview-report-recovery-size-check-03.json`, SHA
`2da8096ada0f13068211a87a9b5bf8cc90bc1886d52b23694148272adb753a25`,
measures 1,891,052-byte HTML, 2,997,652-byte complete plain JSON and
16,614,274 aggregate bytes including a terminal-manifest size estimate, below
the unchanged caps. Its same-width gzip-hash placeholder is sizing only, not
custody or public recovery. The final root run must verify actual stored bytes,
logical decompressed originals, inline payload hash/length and terminal output.
The packed copy preserves all point fields, absent/null distinctions and original
slots; only the selected twelve points are unpacked on a view change. Required
string dictionaries and malformed JSON/layout/index cases fail visibly. The
inline fallback keeps the previous hash-only script policy and introduces no
external load, server, fetch, fitting or physical claim.
