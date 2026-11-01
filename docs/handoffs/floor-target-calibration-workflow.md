# Source-bound floor calibration workflow

## Current status — November 1, 2026 (America/Los_Angeles)

Implementation, pure authored checks, product reviews and all four public
numerical cases pass. Three actual IPPE solves ran on synthetic analytical
points; supplied mode made zero solver calls. No new source acquisition,
decoder, inference, body constructor, physics or paid API call ran in this
slice. Independent saved-output audits pass; scoped publication is next. The complete
Weft goal remains active. The previous
goal turn was **progress**: guided shared-world setup and both public acceptance
paths were published as `bf661cd6e8fccd58b22ac324a5374823dee7c9e2`, independently
verified on private `main`. Receipt `.local/shared-exercise-config-publication-01.json`,
SHA `8631f091109442ff98197bd23d24b43db8458557d5ea5af42d42e7674b25d7bd`.

Fresh fetch still matches that checkpoint. Two separate avatar renderer/test
edits remain untouched; stash and index were empty at this turn's start. Keep
the hosted site offline. No deployment, submission or external message.

## Outcome and ownership

Connect source-linked measured calibration inputs to the **existing** camera
and ground contracts. A supplied calibration imports without solving again.
The planar-target route makes one bounded IPPE call and preserves both possible
poses. Neither path changes a human body, moves its floor, installs contacts or
establishes physical accuracy. No new simulator or task/scorer changes.

- Reconstruction: `floor_target_calibration.py`, its focused tests and
  [core/schema handoff](floor-target-calibration-core.md).
- Worker/reviewer: `floor_target_calibration_worker.py`, its focused tests and
  worker handoff; isolated frozen-code execution plus independent core review.
- UI: `floor_target_calibration_report.py`, its focused tests and report handoff;
  bytes-only source/alternative/frame inspection, no solver imports.
- Root: `floor_target_calibration_pipeline.py`, `cli.py`, package/CLI tests,
  this note/common guides, frozen numerical acceptance and scoped publication.

The [proposed contract](floor-target-calibration-contract.md) and
[bounded source search](exercise-floor-calibration-gap.md) remain explicit:
no independent calibrated unfamiliar exercise source is qualified yet. Existing
clips are exposed development evidence, not final holdouts. Analytical geometry
fixtures are not recordings or measured physical accuracy.

## Public path

```sh
.venv-lab/bin/python -m weft floor-calibrate \
  --observation SOURCE_OBSERVATIONS.json \
  --source-sha256 ORIGINAL_VIDEO_SHA256 \
  --measurement MEASUREMENT_RECORDS.json --request EXPLICIT_REQUEST.json \
  --out artifacts/my-floor-calibration-01
```

Repeat `--observation` / `--measurement` for at most eight documents each. Add
original `--camera` / `--ground` files for the request's supplied mode. Request
and measurement formats are in the core handoff; missing measurement support
must abstain. Camera/ground schema validity alone does not establish measurement.
Target evidence is required only for a target solve, not for an independently
calibrated supplied camera. Preserve raw quality labels and uncertainty.

The root package API is
`prepare_floor_target_calibration_package(observations, out, *, source_sha256,
request, measurements=(), camera=None, ground=None, max_seconds=30, progress=None)`.
Statuses mirror the core: `qualified_nominal`, `unavailable`, `rejected`,
`ambiguous`; malformed custody/execution yields unsealed `failed`. Only a sealed
qualified nominal pair returns CLI zero. Other terminal outcomes retain their
inspection page without admitted downloads. No body/build command executes.

## Custody and execution bounds

- At most 19 selected files / 4 MiB aggregate, request ≤64 KiB. Exact copied
  observation/measurement names use their raw SHA256 under `inputs/`; request
  and supplied sidecars retain separate aliases. Original input paths are
  rechecked with full-precision file identity before final publication.
- Parent saves a versioned `input-files.json`, exact frozen source closure and
  `worker-plan.json`. The child executes the verified cached module bytes,
  never the mutable working tree or an installed Weft fallback.
- One worker; ≤10-second parent hard wall plus ≤2 seconds cleanup, ≤8-second
  hard child CPU, ≤5-second cooperative core time, ≤256 KiB combined logs.
  Observed child peak RSS must be ≤512 MiB; this is **not a hard memory cap or
  OS sandbox**. No retry, nested process, network, decoder, model or physics.
- Core output ≤4 MiB, calibration metadata ≤1 MiB. Child files/receipt and
  parent logs/process record remain under `worker/`. Exact admitted files are
  copied to root for downloads; both copies count toward package bounds.
- Reporter receives only calibration metadata, exact request, input inventory
  and admitted sidecars—not all observation/measurement bodies again. All raw
  inputs stay downloadable. ≤4 MiB report input/output, 16 files, paged complete
  source frames and selected-only point drawings; existing brand CSS unchanged.
- Parent package ≤32 MiB / 96 files, including final seal. Maximum 30 seconds
  overall with report/sealing reserves. Exclusive writes, complete final file
  inventory and a monotonic terminal manifest. Late caller failures create a
  sibling receipt rather than relabeling a published package.

Raw camera bytes and canonical camera hashes remain separate. Existing surface
intervals do not propagate calibration uncertainty; no measured uncertainty is
converted to an authored interval or silently discarded. No `surfaces.json`
is emitted by this first adapter.

## Checks and next execution

Root's **21 package cases** pass in
`.local/floor-target-calibration-pipeline-tests-01.xml`; **16 CLI cases** pass in
`.local/floor-target-calibration-cli-tests-01.xml`. These use a mocked numerical
boundary, testing input/output custody, one-call dispatch, abstention, failure
preservation, exact style/source copies and terminal behavior. No solver ran.
Agent-owned checks and preserved failures are in their handoffs. Together,
**130 distinct checks pass**: 43 core, 31 worker, 19 report, 21 package and 16 CLI.
Root receipt `.local/floor-target-calibration-root-checks-01.json`, SHA
`ddf9c900f1cb53bb7c72b85b019b2dbe8dd07412802ff449ed12a24ae7845db4`,
reuses the existing passing results without rerunning them.

Review found a supplied-mode OpenCV guard gap, the report's incorrect canonical
hash encoding and missing projected-point rank checks. Targeted corrections
are fixed and independently reviewed; keep each red receipt and reuse unchanged
passing checks. Three reporter compatibility findings are also closed: request
hash order, pre-abstention point headers and explicit projection-frame order.

The frozen four-case authored public acceptance uses an oblique target
with skew/radial distortion/fractional clock; equal geometry expressed in mm;
far/near-affine ambiguity; supplied camera/floor exact-byte import with zero
solves. Pixels were constructed independently with analytical projection, before
any solver result. Fixture02 strengthens the unused fixture01 with twelve
corners, nontrivial target/world translation and rotation, and a floor at −0.2 m
separate from the tilted board. All twelve metadata/eleven decoded slots remain;
frames 0/1 are construction, frame2 evaluation and frame11 metadata-only. The
source identity is an authored test label, not a media digest. No source video
exists for these synthetic points. Do not tune against results or rerun passing cases.

- Inputs: `.local/floor-target-numerical-inputs-02/fixture.json`, SHA
  `6a3f3b7887b8806e5fc8b9fa8148b34923abe0c2437fc0ba7a11b47763165887`;
  18 files / 127,593 bytes. Fixture01 stays preserved and unexecuted.
- Runtime: `.local/floor-target-calibration-runtime-01`, exact `bf661cd` archive
  plus five reviewed overrides; 291 files / 6,292,000 bytes; seal
  `84da0762dd8ab5dd9bf71b1c8b7930c50821fb4385f900bc945ea418057f24e3`.
- Preserved unexecuted plan: `artifacts/floor-target-numerical-01/plan.json`, SHA
  `0f1b13ba3e91ed47c5cae25f568e998db3c9a3d0d517aa8c0e801bc33fd38c35`.
  Runner SHA `7e434af090af0e3b2620ebeae49fc22258702f2cb4bc63d939ca2b6d70e412b5`.
  Review caught a root-runner cleanup flaw: failed descendant lookup could skip
  killing/waiting for the known CLI. No execution occurred under plan01.
- Four public CLI calls, two concurrent, three IPPE calls maximum and zero for
  supplied mode. No retry; 35 seconds per CLI, 80 seconds aggregate execution,
  132 MiB / 430 files complete outer bound. Child limits above still apply.
  Expected unique nominal solutions in metre/mm cases, retained ambiguity in
  the far-camera case, exact supplied bytes in the fourth. Strict numerical
  expectations are frozen; failed expectations remain failures.

## Actual public numerical result

The root-only cleanup correction separates discovery/individual-child errors
from the known CLI's unconditional kill/wait attempt. Three fake-process cases
pass; receipt `.local/floor-target-root-cleanup-checks-01.json`, SHA
`f4ad6652fd76a5dde9b1461f7068a50841a42b8bedffb016ae30157f3ab03e9e`.
Independent delta review is clear. Product code, inputs and numerical criteria
are unchanged. Runtime84da0762 above is reused, not rebuilt.

Executed package: `artifacts/floor-target-numerical-02`; plan SHA
`b2c82a54c1929e5afc680240c6ea12d1a17e716b921e6a2966ea32a72fd22156`,
runner SHA `e824cc79b33edbe672a72c641ea4d9e2431e10f14dfc9bf60154499e6bde9a23`.
Preexecution review receipt SHA
`7310a448dec9e80e9a01f5bdc12d38f1bb0be38971204b744b703533aa8cfd2c`.
All four genuine public CLI paths passed once: **1.003870 seconds total**,
0.982349 seconds execution; **189 files / 2,431,160 bytes**. Outer seal
`4af92dec9c72b9b83fa08ea39d382478c3f8c18cc35dad1d7d32679bc06baf56`.
All CLI/worker processes are terminal and reaped; no live handle remains.

| Case | Verified outcome | Exact package seal |
| --- | --- | --- |
| Oblique target, skew13 px, k1=.15, fractional FPS | One distinct nominal solution; separate floor stays −0.2 m | `c4ede04d62c0275b064faedee615b48737f73816a33e3e625396572f3de10be8` |
| Same target in mm | Same selected geometry and decision | `ff75edeb44e735dab7fc66ad12695f76da9d66572be602541064676fec16331e` |
| Camera at30 m | Both distinct alternatives retained; no camera/ground pair; intended CLI exit1 | `fc615e0b6285d68d02d7c5d791dd1169fcb7b67bd47b1ea7c5e839f0727e0888` |
| Supplied calibration | Exact original sidecars/quality; zero solver/Rodrigues calls | `2802bed74aa4282e3177c439414cc12a438c7f40a5e09bd7f506117c30c19bf3` |

Root's single cached output pass verifies all files/seals, raw inputs, runtime
pins, nominal clocks and numerical composition. Maximum selected R/t absolute
error from analytical truth is `1.6514567491299204e-15`; maximum independently
recomputed projection difference is `1.1368683772161603e-13` pixels. These are
noise-free software checks, not measured camera accuracy. Each case retains all
twelve source slots/eleven decoded slots; excluded point payloads stay unused.
All three numerical workers keep both returned alternatives; total work is
three solves/six pose conversions. Largest observed child RSS is 74,612,736
bytes, not a hard process-tree memory guarantee.

[Inspect the nominal example](../../artifacts/floor-target-numerical-02/case-01/inspection/index.html)
or [the ambiguous result](../../artifacts/floor-target-numerical-02/case-03/inspection/index.html).
Independent saved UI audit passes: `.local/floor-target-numerical-saved-report-audit-02.json`,
SHA `aa81d8ef983166ad7acf1e8841562695f1a7e814c4589b4a4011b5c7ae3b72be`.
It read 25 selected files / 334,751 bytes in 0.156523 seconds: all 48 original
frame views, six alternatives, two rejection reasons, 144 displayed point
comparisons and 48 local links match. Evaluation frames contain no projected
points. Four thousand unchanged scripted UI events cause zero writes/nodes;
this is a Node harness result, not browser/device QA. No report regeneration.

Independent saved-process audit also passes: 15 selected files / 62,751 bytes,
all four public commands and nested frozen-worker bindings, all eight reaped
processes, exact versions and solver/conversion counts. Its receipt is
`.local/floor-target-numerical-process-saved-audit-01.json`; see the
[worker handoff](floor-target-calibration-worker.md). Receipt SHA
`51a982a0a898f28df015348abfe38d0587311b6505551287a53467f67ba24e7c`.
Its display-only TextEncoder checker error remains recorded; completion reused
cached selected bytes without reading the output again.
No full output scan, numerical recomputation or source stage was repeated.

## Publication and next safe work

Publish only the eighteen reviewed code/test/guide/handoff paths. Actual Git
outcome belongs in `.local/floor-target-calibration-publication-01.json`; a
missing receipt means publication is not complete. Fresh fetch still matches
`bf661cd`; no stash entries or staged changes existed before preparation. Two
independent avatar edits remain outside this checkpoint. Preserve every earlier
failure, unused plan and completed artifact. Do not rerun a solver, producer,
report or unchanged passing check.

The next calibration usability gap is a source-bound input worksheet: users
currently assemble request/measurement JSON explicitly. That can reuse existing
source/frame selection and saved inputs without acquiring another video or
running a solver. Real unfamiliar-source floor/equipment/contact accuracy still
needs independent licensed measurements; this remains a separate unresolved
workstream, not a reason to relabel synthetic inputs or stop the overall goal.

[Plain-language user guide](../floor-calibration.md) and the existing local-lab
guide now expose both routes. No calibration-input wizard exists yet.

No paid cost or recurring service is introduced. The shared $5 Meta ledger is
untouched. Native body/lab admission and real floor/contact accuracy remain
outside this analytical adapter check and must not be counted complete.
