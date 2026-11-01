# Guided shared-exercise configuration

## Current status — November 1, 2026, PDT

Core and focused tests are frozen; independent read-only correction review is clear. **37 distinct analytical cases pass**, with all failed checker/implementation attempts preserved. No source discovery, video decode, model inference, native construction, physics, API call or live job occurred in this lane. Root owns the forthcoming one-shot public CLI discovery/resume acceptance; no world build is part of this preparation.

Owned paths: `src/weft/shared_exercise_config.py`, `tests/test_shared_exercise_config.py`, this note. Root owns CLI/common guides/publication. `support_ui` owns the bytes-only offline report. `support_learning` owns the pure proposal-config validator extraction and independent review. Prior ordinary video/equipment evidence milestone is published at `7d604e4`; do not repeat those source studies.

Frozen implementation SHA256: `856e8532fa9be0164ebab82aa0fea0c08afe43e5e49683ee21d522b7ac95dc91`; test SHA256: `e9fda5a69afb9e2204f106a9565b91bfe125f509eaa5e0566bc6e90f5df6a58b`. Final release receipt is `.local/shared-exercise-config-release-02.json`; release01 is retained as the pre-review documentation checkpoint. Earlier pre-review pins `8c2846df…` / `3ef26b1c…` are superseded, not claimed approved.

## Public API and exact output

```python
prepare_shared_exercise_config(
    human_run, evidence_run, out, *, source_mappings=(), resume=None,
    request=None, expected_manifest_sha256=None, max_seconds=30., progress=None
)
```

Discovery requires explicit saved human/equipment paths and no resume/request. A lab path resolves to its saved `run/`; outer lifecycle metadata remains separate. Resume requires both positional source arguments `None`, no replacement mappings, a prior sealed setup, and a separately edited request. Its optional expected hash pins the prior setup seal. Never edit files inside a sealed setup.

Result profile: `shared-exercise-config-package-v1`. `needs_input`, `config_ready` and `unavailable` have a complete terminal seal/report. Malformed custody or a failed operation returns `failed`, unsealed, with a failure receipt; original evidence and any partial writes remain. The result includes `sealed`, `output_path`, exact `manifest_sha256` when sealed, `metadata_path`, `request_path`, nullable `config_path`, `report_path`, `build_inputs`, `build_options`, `full_reference:true`, complete counts, input accounting, runtime pins and zero scientific-work counters. A ready config is **not** an admitted body, reconstructed world, contact claim or task success.

Successful packages retain exact complete parents under `sources/human/`, `sources/equipment/`, and `sources/mappings/<proof seal>/`; original manifests are unchanged. `inputs/human-lifecycle.json`, when supplied, retains the separate outer status. `inputs/human-observations.json` retains the exact observation bytes used for guided qualification. Parent inputs are never trimmed to fit a limit. Ready `build_inputs` points to these usable complete copied parents plus `config.json`; the existing builder revalidates them normally.

Root files: `snapshot.json`, `request.json`, `metadata.json`, optional `config.json`, `result.json`, `implementation-source.zip`, terminal `manifest.json`. The saved result is explicitly pending-terminal-seal; only the final manifest establishes package completion. Report files live under `inspection/` and link to exact root downloads. The archive includes exact loaded Weft implementation bytes and `assets/brand.css`; report CSS must equal the archived bytes. This is a local source snapshot, not an installed environment or independently authenticated source.

## Editable request and current builder boundary

Request profile `shared-exercise-config-request-v1` has `schema_version:1`, exact `snapshot_sha256`, `human_entity_ref:null`, `camera:null`, `geometry_prior:null`, `equipment:[]`, `target_object_id:null`, and separate `build_options:{body_scale:1.0,root_assistance:false}`. Optional `frame_translation_xy` is absent. No entity, equipment frame, target, camera distance or translation is chosen automatically.

A supplied human reference is exactly `{document_sha256, object_id}`. The current builder binds `person-0`: another explicitly selected ID stays inspectable but cannot produce a ready config; no renaming or identity authentication is inferred. Every selected human body reference must match its retained source ID, frame and time. All other source entities remain in the snapshot.

Camera mode must explicitly be `fixed_pelvis_perspective_prior`, with finite non-bool distance .1–50 **original body-model units**. Geometry must explicitly choose `floor_box_depth_equals_width`; it is an authored unchanged-floor box/depth prior, not measured support. Select 1–12 distinct equipment IDs and original frame IDs 0–1799, with target among those objects. The unchanged pure `validate_proposal_config` validates the final original `source-equipment-proposal-config-v1`. Existing optional XY translation remains explicit and bounded; this helper does not execute it. Separate build options are finite non-bool body scale [.5,2] and exact bool root assistance. Root's next command always includes `--full-reference`; no build/train action is automatic.

## Source semantics, clocks and partitions

Exactly two canonical observation sources are exposed: human and original equipment evidence. Each preserves its own full decoded-frame × entity grid, source clock, partition label and missing reason. `original_frame` is null without a verified mapping; those rows remain inspectable and ineligible. Original metadata-only tail slots remain separate, with no invented observations. Missing/withheld measurements never become zeros or coordinates in the display snapshot. Contact, grasp and load stay unknown.

Legacy human derivation now matches the existing builder exactly: `adapt_exercise(pose, scene=original_scene)`, then `provenance.input_artifacts` with exact original `evidence.json`/`scene.json` raw pins. Newly derived JSON is compact but lossless; original input bytes are untouched. This pure adapter imports `spatial_pose`/`body_frame` constants for existing authored topology, never the native builder. If an extra saved `observations.json` differs in semantics or partitions, its bytes/slots remain visible and the route is **unavailable**, rather than silently widening construction. Known body-source inputs retain their supplied canonical observations; full body-source/body geometry validation remains deferred to the existing builder.

Complete SourceEquivalence bundles use the real existing validator, no fabricated mapping dictionaries. Original and derivative frame IDs remain aligned only through that proof; separate decoder clock offsets remain explicit. Canonical binding uses the existing registry/source-clock validators. Hashes establish byte identity, not physical truth. Source eligibility is not a positive-width world, valid torso/root reconstruction, camera calibration or initial/full-horizon collision admission.

## Bounds, immutability and costs

- 64 MiB aggregate input read bytes / 256 unique input files; full proof manifest duplicate reads are counted.
- At most 2 observation sources, 12 entities/source, 1,800 original/decoded slots/source and 900 body references; at most 43,200 observation slots, snapshot ≤8 MiB.
- Output ≤96 MiB / 320 files, with final-seal/failure reserves; report ≤16 MiB /32 files.
- Shared cooperative 1–30-second deadline; report receives at most5 seconds with final write/hash reserve. This is **not** a hard worker/RSS limit.
- Every successful setup reserves a future resume envelope: exact complete copied source bytes + repeated proof manifest + lifecycle + exact snapshot +256 KiB prior-seal reserve +64 KiB edited request, ≤64 MiB/256files. Resume never rereads prior report/generator assets.
- All paths reject traversal/symlinks; outputs are exclusive/new. Existing file identities and optional failure/lifecycle marker presence are checked before publication. A newly added failure marker cannot escape a prior absent-file check.
- Final publication is monotonic. A valid completed seal is never rewritten as failed after caller I/O; any late caller failure is an exclusive sibling receipt. Invalid/partial attempted seals remain unsealed, with failure evidence. No cleanup deletes evidence.
- Local-only work; $0 provider calls. No source re-acquisition or duplicated scientific study.

## Verification and preserved failures

Only this new focused test file ran. Receipts are under `.local/`:

1. `shared-exercise-config-tests-01.xml`: 8 passes /20 failures. Most failures exposed an omitted registry label (the registry intentionally strips unqualified labels); corrected to null, not a raw-label leak.
2. `shared-exercise-config-discovery-check-02.xml`: authored fixture lacked required camera `screen`; no product source relaxation. Corrected fixture; `...discovery-check-03.xml` passes one case.
3. `shared-exercise-config-affected-checks-04.xml`:19 affected failed nodes now pass; prior unchanged passes reused.
4. `shared-exercise-config-boundary-checks-05.xml`:4 passes /2 failures. Alternate-person fixture observation order fixed; large derived pretty JSON exceeded16 MiB, corrected with lossless compact serialization (no rows removed, no bound raised).
5. `...boundary-checks-06.xml`:3 targeted passes. Maximum authored900-reference/1,800-frame case:1.090392seconds,6,400,736input bytes,2,801,808snapshot bytes,9,530,224reserved resume bytes. This measures one local metadata case, not native/video responsiveness.
6. `...unavailability-checks-07.xml`:3 specific floor/camera-body/failed-parent reason checks pass.
7. Independent review found exact builder-derivation drift and absent-marker races. `...review-red-08.xml` preserves4 reproduced failures; `...review-green-09.xml` has8 targeted passes after correction, including ready resume and real pure derivative-proof loading. Latest outcomes deduplicate to37 passed cases.

Tests block subprocess/network/model/native imports and use authored bytes. Core tests stub only the report; actual source registry, SourceEquivalence and pure config validator run. They cover complete slots, defaults, source/clock/partition conflicts, known body-source metadata, invalid choices/options, wrong snapshot pins, copied-source mutation, missing mapping, full proof custody, budgets, deadlines and terminal behavior. No full old suite was rerun.

UI independently consumed one preserved actual authored core payload at `.local/shared-exercise-config-analytic-01`; receipt `.local/shared-exercise-config-report-composition-01/receipt.json` SHA `9d8f9d2634e2c39dd7caeb7b469f20cd8aec852c8e53d4c1c1a985422052f6e2`:5 selected metadata reads/17,159bytes/.00706seconds. It is a saved schema-composition check of the earlier payload, not validation of the later derivation fix, authenticated ancestry or a new source run. Root's upcoming public acceptance uses the final corrected runtime.

## Root's prepared next acceptance

Inputs-only fixture `.local/shared-exercise-config-cli-inputs-01` is complete and materialized; **zero producer calls** occurred during assembly. Its `fixture.json` SHA is `5950e86584b01c2d78db45c4c07e5509358d0fccff22bc6f107278ec7f327c58`; human manifest `901fe2da47da4a619d60330eb0484e5065d6f020d47da083b0b10d01a80338ad`; equipment manifest `edc55c09e4c5fecc2adc66b6ef25ddd89d9ac3676f50f5bfed9469694e7f5dd5`. The authored media bytes have their real hash but are not a video; minimal body metadata is not a physics-valid human. `edited-request-settings.json` fixes person-0/weight/frame0/camera-distance3/box prior/body-scale1/no root assistance; root applies exact emitted snapshot/document pins after discovery. No intentionally missing declared file exists.

Root alone prepares/runs one frozen public discovery→edited-request resume acceptance after review. Separate real-input readiness metadata is root-owned at `.local/shared-exercise-config-real-input-discovery-01.json` SHA `93832c32e17b3873287f031b40191313d6ced8131621dfc98913c2e23d68c7ba`:176 complete source/proof files/52,881,882bytes before setup metadata, within declared bounds. No source processing was performed here and no later real run is authorized by this note.

Independent correction review cleared the exact final source/test pins without reruns. A separate report-only nullable-translation compatibility correction remains with root/UI, not this core. Next action: root captures the final runtime and performs the public CLI acceptance. Preserve all original human/body/root/floor geometry, protected gates, failed proposals and unrelated avatar work. No Git mutation occurred in this lane.
