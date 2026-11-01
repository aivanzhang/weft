# Guided shared-exercise configuration

## Current status — November 1, 2026, PDT

Implementation, focused checks, both public CLI workflows and both saved-output
audits pass. Scoped publication outcome is recorded separately below. Pure configuration checks cover 41 cases, CLI 20,
producer 37 and offline report 18. These are scoped analytical checks, not video,
native-world or physical validation. Independent review closed builder/source
derivation, late failure-marker and nullable-translation inconsistencies.
One general Weft pipeline remains the goal. This adds preparation and explanation
to the existing `shared-exercise-build`, not another simulator, constructor or
contact model. Root owns integration and any later frozen acceptance.

The preceding ordinary pose/equipment workflow is published on private `main`
at `7d604e4fada2fd8939752efffb2557056b6525ef`, exact remote verified. Receipt
`.local/exercise-equipment-publication-01.json`, SHA
`e9ba56cbbc26572e58d650419e43531e7602d0ab9e8982a40cbd6f1a3bd66f2b`.
Its twelve paths/index were clean and stash empty; two separate avatar edits
were preserved. All previous source/CLI/audit processes are terminal. Do not
repeat them. [Completed public evidence workflow](exercise-equipment-workflow.md).

## Missing user step

The existing builder already performs body seed → equipment proposal → strict
coupled-world construction → one interactive lab. Users still manually assemble
source hashes, equipment IDs, original source frames and camera assumptions.
The new preparation should expose missing inputs and frame eligibility, create
an editable source-pinned request, validate explicit choices, and print the
existing build command. A configuration-ready result is **not world admission**.

Do not automatically choose an equipment object, frame, task target or camera
distance. Body scale and root assistance must stay explicit authored choices.
The current floor-box/depth-equals-width prior is not recovered object shape,
held equipment, handles, grasp or measured contact. Hand-image evidence remains
diagnostic and cannot supply those physical claims. Keep failed pose-only
parents distinct from usable mobile bodies; a new form cannot waive admission.

## Parallel file ownership

- Reconstruction: new `shared_exercise_config.py`, its tests and
  `shared-exercise-config.md`; exact saved-input snapshots, complete slots,
  original source/clock/proof joins and discovery/request-resume validation.
- Contract extraction/reviewer: new `equipment_proposal_config.py`, its tests
  and handoff; narrow import/re-export and dependency-list changes in
  `equipment_proposal.py` / `exercise_scene_inputs.py`. Preserve public validator
  behavior and exception types while avoiding native-engine initialization for
  configuration validation. Do not change numerical proposal code or old runs.
- Report: new `shared_exercise_config_report.py`, focused tests and handoff;
  bounded offline editing/downloads and complete frame navigation. No source
  reads, browser work or avatar edits.
- Root: `cli.py`, new CLI tests, this handoff/common docs, report integration,
  frozen acceptance and scoped publication. The API and source-copy contract
  below are agreed before dependent implementation.

## Agreed API and user path

```python
prepare_shared_exercise_config(
    human_run, evidence_run, out, *, source_mappings=(), resume=None,
    request=None, expected_manifest_sha256=None, max_seconds=30., progress=None,
)
```

Discovery requires the two source paths; resume requires both positional inputs
to be `None`, no source mappings, and a prior sealed package plus edited request.
Only the new package's complete copied sources may be used on resume. Keep each
original source manifest verbatim; do not present a selected partial snapshot as
an original complete run. Exact copied build inputs live under `sources/human`,
`sources/equipment` and `sources/mappings/<seal>`.

Result profile: `shared-exercise-config-package-v1`. Fields include `status`,
`sealed`, `output_path`, `manifest_sha256`, `report_path`, `metadata_path`,
`request_path`, nullable `config_path` / `build_inputs`, `build_options`, and
`full_reference=true`. `needs_input` and `config_ready` are successful setup
states, not world admission; `unavailable` / `failed` retain the report and return
a nonzero CLI status. A ready result supplies exact copied human/equipment/proof
paths plus the existing ordinary `config.json`, not original mutable paths.

The request is versioned `shared-exercise-config-request-v1` with an exact
snapshot pin, explicit human entity confirmation, camera, geometry prior,
equipment selections and target. Object/frame/target/camera choices start empty.
Separate `build_options` show the existing authored defaults: body scale `1.0`
(unchanged size) and root assistance `false`. Validate those choices without
adding foreign fields to `source-equipment-proposal-config-v1`. Optional XY
translation remains explicit; no automatic floor/body/frame adjustment.

The offline report takes only exact metadata/request bytes, copies the existing
brand CSS unchanged, and performs no source reads. Reserve five seconds plus
remaining write/hash time before its single terminal publication. Do not embed
a fabricated self-referential final seal into a page; the CLI prints the actual
pin after publication. Complete copied inputs count within 64 MiB / 256 files;
output within 96 MiB / 320 files. Reserve/account for control metadata so every
successfully emitted setup can be resumed without rereading all report assets
or dropping source files.

```sh
.venv-lab/bin/python -m weft shared-exercise-config \
  --human-run BODY --evidence-run EQUIPMENT --source-mapping MAPPING \
  --out NEW_SETUP
.venv-lab/bin/python -m weft shared-exercise-config \
  --resume NEW_SETUP --manifest-sha256 PRINTED_SEAL \
  --request PATH_TO_DOWNLOADED_REQUEST.json --out NEW_CONFIGURED_SETUP
```

The page downloads an edited request **outside** the sealed package. The CLI
prints the existing builder command only after configuration readiness, with
explicit scale/root options and `--full-reference`; it never executes a build.

## Existing contracts to reuse

- `equipment_proposal.py::validate_proposal_config` and the unchanged
  `source-equipment-proposal-config-v1` output schema; keep draft metadata outside
  that schema.
- `exercise_scene_inputs.py` for source/frame/proof admission semantics.
- `exercise_scene_pipeline.py` for the existing one-world orchestration.
- `exercise_body_source.py` and `shared_exercise_lab.py` for actual body/lab
  admission, control and recording; do not weaken them to make a draft pass.
- Existing focused proposal/input/pipeline fixtures. Preserve positive saved
  `artifacts/shared-exercise-assembly-01` and rejected
  `artifacts/direct-exercise-build-01` (knee/equipment collision); neither is a
  fresh holdout or permission to rerun an unchanged trial.

The resolved command continues to use the existing route:

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --human-run BODY --evidence-run EQUIPMENT --config CONFIG \
  --source-mapping MAPPING --full-reference --out NEW_BUILD
```

## Execution boundary and next action

Both frozen public discovery → edited request → resume checks and their saved
audits are terminal and passed. Finish scoped publication; do not
repeat the source stages. Native jobs require a separate frozen root
plan; agent assignments do not start them. No model, decoder, physics, API,
deployment or server is running in this slice. Provider and recurring cost: $0;
the existing $5 Meta ledger is untouched. Keep complete input slots and honest
unsupported output, bounded source/hash/snapshot I/O, terminal seals and new
output directories. Preserve every actual failure without repeating successful
source work. The complete generalized video-to-RL goal remains active.

## Scoped checks and metadata-only preparation

- Pure validator extraction: **41 passing cases**, independent root review clear.
  The saved original proposer/input files match committed `7d604e4` exactly; all
  unrelated function ASTs are unchanged. Receipt
  `.local/equipment-proposal-config-checks-01.json`, SHA
  `202041f4158da346115207c8ac0adc660f284f08e0daec95b086d62f805efbf2`.
  [Exact API, aliases, dependencies and pins](equipment-proposal-config.md).
- Root CLI: **20 passing cases**, independent review clear. No producer or build
  ran. Discovery/resume exclusivity, quoted pinned follow-ups, explicit complete
  build options and failed/unsealed handling are checked. Receipt
  `.local/shared-exercise-config-root-cli-checks-01.json`, SHA
  `84fa2b511360cb86db4be0f29214357580b48d1000c1c8bfa2ce62cc170a1636`.
  CLI SHA `612165e74d6a83d3ca469c31e1053081946deeb18d3ff16ccb69f55f8a35445e`;
  test SHA `02336b5fa77299a13ddf4813bec087d3dd80ece21fffaa7ca51af6a98751e2bd`.
- Metadata-only future-input note
  `.local/shared-exercise-config-real-input-discovery-01.json`, SHA
  `93832c32e17b3873287f031b40191313d6ced8131621dfc98913c2e23d68c7ba`:
  `artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run` has 145
  files / 39,580,233 bytes; exact seal
  `a8f390f8cf4e04071ec5e8ff3140a2006985abc83371824bf5ff02ba84220019`.
  With the previously verified ordinary equipment/proof packages, the projected
  complete source snapshot is 176 files / 52,881,882 bytes, within declared caps.
  One new 15,690-byte manifest read / 144 stats; equipment/proof metadata reused.
  No body arrays, media, proof bodies or pipeline were read/run. This is not new
  body/world admission or a completed setup; revalidate under the eventual plan.

## Frozen authored public CLI acceptance — passed

- Runtime `.local/shared-exercise-config-runtime-01`, exact committed `7d604e4`
  plus six reviewed setup-specific overrides; 287 files / 6,155,386 bytes.
  Manifest SHA `f9d6f6fa03691f697557415c76bbf64d8ff6a99b5b218334609faf139d5b592d`.
  Dirty avatar assets are not included.
- Plan `.local/shared-exercise-config-cli-acceptance-01/plan.json`, SHA
  `d3a6ad0e8d456979186a11fe1ecdd8d2bee3c66c9497b604b5175d664e20ba9c`;
  helper `.local/run_shared_exercise_config_cli_acceptance_01.py`, SHA
  `567944aef3b30839c75747d51c4a1c68299afa1a0bf15f5310e80032af9d12ac`.
- Complete authored fixture `.local/shared-exercise-config-cli-inputs-01`,
  fixture SHA `5950e86584b01c2d78db45c4c07e5509358d0fccff22bc6f107278ec7f327c58`.
  Eight original slots, six decoded slots and three body references; opaque
  source bytes are not a decoded video or a validated body.
- One child, 60-second wall / 45-second CPU envelope, observed RSS acceptance
  ≤512 MiB (not a hard memory limit), 64 MiB / 384 aggregate output files.
  Frozen imports, minimal environment, forbidden provider/network/decoder/native
  work. A ready configuration only prints the existing build command.
- Product pins: producer `856e8532fa9be0164ebab82aa0fea0c08afe43e5e49683ee21d522b7ac95dc91`,
  report `81e84996b225e5700fbacb55782f7c7043061dcc5937310020c9d09ad4f7c5e9`.
  All earlier red attempts remain preserved; no passing source study is repeated.

Actual discovery → explicit edited request → pinned resume passes in **0.461681
seconds**, 53 files / 1,047,745 bytes, observed peak child RSS 89,194,496 bytes.
Outer seal `a8ecb386be35961a93cee1c7ccd3e2b29610537d89edf432ea8b2cf5464af268`.
Process receipt `.local/shared-exercise-config-cli-acceptance-01/process.json`,
SHA `d1b736a2552cbf2ea513c451e4cfb87a642bbca3d066c7921c13712d3131a869`.
Independent execution review `458a67abd74184c0ed60448dd42dfa3643d14c10fbb07b0a09bd9165691c23a6`
precedes the run. Saved-only output audit passes all 53 files, both complete
parent copies, both 29-member generator archives, 20 local links and exact
request/config downloads. Receipt `.local/shared-exercise-config-cli-saved-audit-01.json`,
SHA `cf160d4e9a42954e7af9f8a7b7960f2e74e1e005be27e46d8010b85e0e125628`:
59 reads / 1,141,091 bytes / 0.172548 seconds; 4,000 repeated selections cause
zero writes in the saved scripted harness (not browser QA). No producer repeat.
The three explicit references and all eight original slots
remain; no scientific or native work occurs. Both CLI stages are terminal.

## Separate real saved-input acceptance — passed

Reuse the unchanged runtime above and complete original body/equipment/proof
packages listed in the metadata note: 176 files / 52,881,882 bytes. Explicit
choices are preserved from `shared-exercise-assembly-01/proposal/input-config.json`
and `.local/mobile-source-assembly-plan-01.json`: person-0, sam-2/sam-3 at frame 8,
camera distance 1.587591529575929 original model units, floor-box prior,
XY translation [0.2, 0], unit body scale and virtual root assistance enabled.
No new fitting, contact inference or changed source selection is authorized by
this setup check. Previously exposed development evidence is not a holdout.

- Plan `.local/shared-exercise-config-real-01/plan.json`, SHA
  `f1ec7fb27e42effe8d8a6071cea4e205de7311249c0851656f3b800f05b9b244`.
- Helper `.local/run_shared_exercise_config_real_01.py`, SHA
  `4873cbf9c379cc04a762ed1ecc7d17681039f1624c80c5e9219183eb30ffac33`.
- Input metadata `.local/shared-exercise-config-real-inputs-01/fixture.json`, SHA
  `20e194140ab5616a5dd9b937d1a725efac6ca02d379ee468ec2dab90bee594c8`.
- One child, 60 seconds wall / 45 CPU, observed RSS acceptance ≤512 MiB,
  aggregate output ≤192 MiB / 640 files, per-file ≤16 MiB. Both setup calls
  retain their own complete immutable source copies. No decoder, inference,
  native builder/physics, API, new service or paid work. Root owns execution
  after the exact plan/helper delta review, receipt
  `.local/shared-exercise-config-real-execution-review-01.json`, SHA
  `614b3a0feeff59af78131683d6d5b3614443668b34b9b0a09bd9165691c23a6`.

Actual public discovery and resume complete in **2.088460 seconds**, 389 files /
115,764,340 bytes, observed peak child RSS 239,337,472 bytes. Discovery takes
0.749929 seconds and explicit-request resume 0.895819 seconds. Root process
`57054` is terminal, exit 0; no live source/native process remains.

- Output `artifacts/shared-exercise-config-real-01`; outer seal
  `7c088698f8cb31acbf36ddb14bfde40204f48707f6c2a8356a915ab5c2573677`.
- Discovery seal `16a8f6efd036a74b650f94663233c71dfc82d9830785134f35b1e800118848c9`;
  ready configuration seal `285fcd17017c9c9523a65e9310eb730ebbe53f8889533df336ba0337b0ffd6de`.
- Process `.local/shared-exercise-config-real-01/process.json`, SHA
  `170dad574edfa3ae00b76703bb9789567044064825e8b181a3244aaf1d253e46`.
- Both preserve 259 original / 257 decoded source slots, 121 body references,
  257 human + 771 equipment entity slots and two explicit metadata-only tails.
  Source counts are not inferred from sampled poses. Discovery leaves choices
  empty and root assistance off; resume preserves the explicitly supplied
  original settings, including root assistance on.
- Discovery reads 176 files / 52,884,515 bytes (including the repeated proof
  manifest); resume 179 files / 53,672,525 bytes. Reserved future resume envelope:
  179 files / 53,975,123 bytes. No source trimming or budget increase.
- [Ready setup page](../../artifacts/shared-exercise-config-real-01/configured/inspection/index.html).
  It prints, but does not execute, the original builder with complete newly
  copied parents/proof, exact config, explicit scale, full reference and root
  assistance. The old native assembly is not rerun; physical accuracy and
  unfamiliar-source support remain unverified.

### Independent saved-output audit

The single streamed pass verifies all 389 files, outer/stage seals, each stage's
145 human + 4 equipment + 27 mapping members, both 29-member generator ZIPs,
and every copied source/document identity. The exact copied original evidence
confirms 259 declared / 257 decoded frames, independent of sampled-pose counts.
Saved Node checks cover 518 original-frame views, 2,056 observation slots,
16 metadata-tail views and 44 local links; 8,000 unchanged updates produce zero
writes in 39.505 ms. This is scripted saved-page behavior, not live-browser QA.

Final receipt `.local/shared-exercise-config-real-saved-audit-03.json`, SHA
`ee7340661471995509f31cd4b4d6244e38159f20d54937d2e30324abbde25cf2`:
398 aggregate reads / 116,051,377 bytes / about 0.349 seconds,
77,594,624-byte peak auditor RSS. Opaque/unneeded files are streamed/discarded.
Audit01 stopped before any output reads because a JSON/JavaScript nanosecond
stat comparison rounded its cached fixture identity. Audit02 completes the
single output pass; amendment03 replaces only that inadequate fixture-stat check
with one exact-hash, full-precision 2,002-byte fixture reread. Both earlier
receipts remain; no output scan, producer, renderer, source model or native
operation is repeated. No product defect or source-data failure was hidden.

## Publication and next work

`.local/shared-exercise-config-publication-01.json` is the authoritative actual
commit/private-remote receipt; absence means publication is not finished.
The implementation/test pins remain identical to the checked frozen runtime.
Protect `shared_exercise_task.py` SHA
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
Only reviewed setup files and related handoffs are staged. Preserve unrelated
`avatar_replay.js` / `avatar_replay_smoke.mjs`, all source/failed/successful runs
and the empty stash list. Fresh fetch matched local/remote `7d604e4` before
publication; no history rewriting, cleanup, deployment or external message.

The overall goal stays active. [The bounded calibrated-source search](exercise-floor-calibration-gap.md)
found no acquisition-ready complete clip with independent floor measurements.
Existing clips remain exposed development evidence. Next implement the smallest
measured planar-target adapter into existing camera/ground contracts, with
explicit source identity, distortion, target-to-floor measurements, uncertainty
and ambiguity handling. Do not substitute authored geometry, image overlap or
a better source fit for measured floor/contact accuracy. This next adapter is
proposed, not implemented or a reason to rerun old failed source studies.
The [proposed API, schema conflicts, limits and ten analytical cases](floor-target-calibration-contract.md)
are ready for the next implementation slice. In particular, preserve raw versus
canonical camera hashes, unsupported clock/distortion abstention and the
existing interval certificates' inability to propagate calibration uncertainty.
