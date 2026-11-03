# Hands-off avatar quality and creation studio

## Active outcome

Build a repeatable prompt-to-avatar workflow that preserves the requested look
and custom actions, automatically reviews and repairs quality, and exposes real
job progress from input through downloadable output. A form around the old
pipeline alone does not complete this goal.

Work stays in `tools/avatar-assets`; Weft simulation, physics, rewards and its
independent goal are unchanged. Existing assets and failed attempts remain
immutable. No deployment or persistent server starts automatically.

### Current user direction — November 2, 2026, PST

The user accepts the previous appearance and explicitly prioritizes the fully
hands-off Studio run. Stop new fur-density, shading and Muse-parity experiments.
Use that existing visual standard; retain checks for broken geometry, missing
requested features, detached fur and incorrect actions. Do not promote an old
manual asset as a new automated delivery. The next bounded acceptance attempt
should exercise one actual prompt through Studio to final downloadable assets;
fresh generation still needs its exact spending allowance. Broader generalization
claims remain limited to actually tested examples.

The reusable automatic attachment stage now passes on a complete saved fox;
[its evidence](portable-fur-attachment.md) is a component check, not the missing
Studio run. Its appearance is unchanged and is not another quality upgrade.

The latest accepted-source website Prepare passed, but its rigging API request
timed out after four minutes with no answer. No new rig/dance delivery resulted.
The uncertain charge remains reserved; no duplicate request is submitted. See
[the current handoff](../TODO.md) for exact ownership, accounting and next steps.
Optional [streaming transport and specific stopped-job status](api-streaming.md)
now pass local checks. They do not recover that answer or complete live acceptance.

## Completion evidence required

- [ ] A local creation screen accepts subject/style, an optional selected
  reference, and custom action descriptions such as dancing and flying.
- [ ] Durable jobs show actual planning, generation, rigging, finishing,
  quality review, repair and rendering states, including failures and outputs.
  Refresh/restart must not duplicate generation or mislabel stopped work.
- [ ] Action intent survives planning and repair. Actual weighted part motion
  and travel are checked; root travel alone cannot pass as wing articulation.
- [ ] Subject-aware finishing provides real fur/hair when requested and keeps
  native/exported representations honest. Final-resolution review must reject
  visible defects instead of accepting a small thumbnail or a movement count.
- [ ] New text-driven examples cover a furry dancing character, a flying
  creature and a substantially different articulated object. Final acceptance
  uses fresh examples, without hand-written per-example finishing recipes.
  Existing geometry is useful development evidence, not a fresh text-to-mesh test.
- [ ] At least one complete user-interface run follows an actual authorized
  job through its final assets. Browser/API fixtures are recorded separately.
- [ ] Bounds, costs, exact inputs, models, repairs, failures and source hashes
  remain reproducible. Unsupported outcomes do not count as quality successes.

## Current implementation ownership

- Parent: orchestration integration, local HTTP API,
  shared contract, end-to-end acceptance and save.
- `agent_runner`: completed exact-root source admission guard and focused tests.
- `pipeline_contracts`: completed independent area-weighted candidate audit.
- `pipeline_review`: completed analytic area-weighted allocation screen.

The latest local [fringe experiment](edge-fringe-experiment.md) improves root
allocation but still shows speckled, sparse fur and the existing contact defects.
The actual new candidate reloaded successfully; it is not an accepted finish or
fresh Studio delivery. The production sampler's math was already correct; its
new source-mode guard has 59 focused checks plus split verification of all 695
integration cases. Approved assets and production finishing defaults stay unchanged.

The fresh panda trial generated mesh/textures for 35 Meshy credits, then stopped
before rigging under the old personal-login backend. The server OpenAI backend
has since been exercised through actual browser Starts using the same paid mesh.
Typed transport and authenticated repair feedback now reach real native builds.
The latest completed API run passed skin/deformation, attachment, action and
native-media review, but exported fur failed visual review. A later optional
texture repair failed; no portable delivery was accepted. Persistent failure
memory and measured coat feedback are tested, with a new actual website run
now also terminal unsupported for exported coat artifacts. See the handoff
before starting any work.

Meshy totals remain 18 successful submissions / 237 credits against 245 authorized.
Settled API use is $2.777078 across 29 calls, under the effective **$15 total**
ceiling, including all prior usage. The medium-reasoning successor stopped before its first API call; see the
handoff for guarded same-job recovery and the corrected prompt-size check.
The user authorized bounded retries within that cap without repeated questions.
No new Meshy call occurs in these recoveries. Each job retains finite call/repair
bounds and unknown charges retain their full reservations.
Exact attempts, approvals and evidence are in [the current handoff](../TODO.md).

## Interface contract

The studio binds only to loopback, uses the existing browser/model-viewer
dependencies and returns same-origin opaque artifact URLs. Mutations require
JSON and a per-server `X-Avatar-Studio-Token`; reference upload is bounded multipart.
Cross-origin and unexpected Host requests are rejected. No arbitrary local path,
shell command or credential can be supplied through the browser.

| Route | Meaning |
| --- | --- |
| `GET /api/studio/session` | Token, limits, presets and planning-cost note |
| `GET /api/studio/jobs` | Bounded job summaries |
| `POST /api/studio/jobs` | Persist prompt, style, custom actions and selected reference |
| `GET /api/studio/jobs/:id` | Real lifecycle, estimate, review and output snapshot |
| `POST /api/studio/jobs/:id/plan` | Explicit bounded AI planning; no Meshy generation |
| `POST /api/studio/jobs/:id/start` | Execute the reviewed pinned plan within real allowance |
| `POST /api/studio/jobs/:id/cancel` | Stop at the next safe boundary; preserve current work |
| `POST /api/studio/references` | One selected PNG/JPEG/WebP, at most 8 MiB / 16 MP |

Creation JSON uses `prompt`, `style`, `custom_actions` (up to eight descriptions,
600 characters each), `preset`, and optional `reference_id`. A credit cap is not
a grant. Planning and generation are separate explicit controls because they use
different services and authorization. The UI never invents a global percentage.

Pipeline additions preserve existing CLI behavior: `plan_only=True` stops after
the planner/source preparation even for a reused mesh; `cancel_requested` is
cooperative at stage boundaries. Completed deliveries take precedence over
cancellation. Uncertain workers require inspection, not blind replay.

## Implementation checkpoint

The CLI, loopback HTTP/UI, durable queue, exact action contracts, real weighted
skin audit and final-media quality review are implemented. Final delivery now
requires decoded video evidence plus actual720px native/exported poses and four
views; native portrait/media are rendered before the independent critic. Every
repair keeps its own `media-N`, `media-portable-N` and `final-evidence-N`.

- `outputs/studio-ui-fixture-v1`: eight browser behavior checks passed,
  34 intercepted requests, no external requests/JS errors, owned browser exited.
  Mock API/renderer only; no real generation or server.
- The saved implementation checkpoint passed552 Python tests in13.475s; exact log:
  `outputs/studio-integration-v1/full-tests-v2.txt`. The earlier521-test run's
  oversized-upload exception remains preserved alongside its regression fix.
  These checks do not establish visual quality or a completed real Studio job.
  The current integration passes619 tests in14.771s:
  `outputs/studio-integration-v1/full-tests-v4.txt`, SHA256
  `a8c97ea107a3688cf62a9320eca0574e49e75f5f9163469abfacdd33c707f782`.
  The v3 run exposed a package/top-level groom import identity mismatch; its five
  failures remain saved. Package-relative resolution plus an import-order
  regression fixed that defect. `full-tests-v4-sources.json` binds136 source files.
- `outputs/studio-ui-fixture-v2`: nine grouped browser checks passed,
  36 intercepted requests, zero external requests/JavaScript errors, owned
  browser exited. It also verifies polling while the worker is launching.
  The API and renderer are fixtures, not actual generation.
- `outputs/studio-exported-motion-proof-v1`: existing panda GLB,
  four720px rest views plus eight actual exported `dance_flight` poses;
  23.280s /1,201,995,776B sampled direct-child RSS /72,545,104B output;
  exit0 and process exit verified. Former handle51559 is terminal.
  This proves renderer functionality, not hands-off quality: plastic/sparse
  exported fur and cheek/shoulder marks remain visible. A subsequent projected
  camera-fit correction passed separately on an analytic cube:
  `outputs/studio-framing-fixture-v1`,0.784s/226,836,480B sampled RSS.
  All four views retain at least7.6271% margin; the old three-quarter calculation
  had a negative7.3280% margin. No animation or avatar-quality claim.
- Action intent/audit/prompts:42 focused tests passed, including stationary
  body roles. A temporary99,999-vertex/
  eight-state benchmark measured0.783s/156,581,888B; its temporary receipt was
  not retained, so it is an observation, not a durable acceptance artifact.

## Prepared development job and next safe action

`outputs/studio-integration-v1/development-estimate.json` defines one actual-agent
robot greeting test on an existing generated source: zero new Meshy/Meta calls,
three expected AI calls, configured cap10, one local repair (at most five role
calls in the current orchestration). Account-specific AI dollar cost is unknown.
The existing input's SHA and exact prompt/action/style are preserved there.

Studio job `a7909fcea90e4df8940850ef` is **cancelled after rejected planning**;
its preserved output is `outputs/studio-a7909fcea90e4df8940850ef`, under canonical
private `tools/avatar-assets/.local/studio`. Do not create a second registry or
restart this job. The worker request-binding regression now passes (26 job
tests total); `development-plan-start.json` records the one explicit launch.
Planning completed in17.59s, but incorrectly classified stationary torso/feet as
articulation requirements. `development-plan-rejection.json` records the exact
failure and original plan hash. No author/native/provider stage followed.
Add typed stability plus world-space motion checks, then use a new immutable
development job and estimate; do not hand-edit this model's plan or reset its calls.
No persistent server is running.

Stability correction now passes42 focused intent/audit/prompt tests. A second,
unchanged-text development request is planned successfully as Studio job
`8a7e0d76b8b8427f93f52157` (`outputs/studio-8a7e0d76b8b8427f93f52157`): hand-wave
articulation plus separate torso/feet stability. Its pinned plan is
`b62fac2bd550e542995d8be894e8593e19fbab2a310eca6b906a99fc1932a7fe`.
`development-plan-reviewed-v2.json` records exact plan/usage and zero-credit
source-reuse authorization. The development study
keeps a ten-AI-call aggregate envelope, including the first rejected planner;
the new job's current one-repair flow can make at most five role calls total.

The actual browser sent one pinned Start. `outputs/studio-live-robot-v1` preserves
a verifier assertion error: it expected200 from the correctly asynchronous202
Start endpoint. Its browser/server exited without stopping the worker.
`outputs/studio-live-robot-v2` observed that same execution without another Start.
The pipeline built two real attempts and rendered final media. Attempt0 failed
torso stability; attempt1 fixed that but left obvious arm spikes. The real critic
requested repair, citing both supplied motion images and the supplied numeric
audit. A citation-validator mismatch rejected those real numeric IDs and stopped
the job as **needs_inspection**, without a delivery. Both attempts remain rejected.

All23 recorded controller/native/agent/utility PIDs were verified absent; the
observer's Chrome/server also exited. A reviewed one-time inspection release
marked the job **failed** and released its exact lease without dispatch or retry.
All480 files /382,622,259bytes and the spending ledger stayed unchanged.
`outputs/studio-inspection-release-v1/release-verification.json` records this
custody result, SHA256
`0a6fe840553ef94ac302298b97fd894d682993bf275c3adac569cedd7f90e454`.
No real UI-to-output success is claimed.
The development study has consumed five real AI calls so far; no Meshy calls.
Changes now address stage-terminal progress, known numeric citations while still
requiring visual evidence, and a gross-deformation gate before expensive media.
Source changes mean this old uncompleted job must not be resumed as a new run.
See [the exact arm-weight failure and fixed rejection criterion](rig-quality-diagnosis.md).

Cached report/artifact corruption now stops without spending another author call;
only a valid bound deformation failure enters geometric repair. The motion
sampler is pinned with the pipeline sources. Unfinished jobs cannot omit current
source pins; historical sealed deliveries remain read-only resumable. These
custody checks passed42 focused tests before the619-test integrated run.

New canonical job `c9ac0f48bd8044458a8390d4` preserves the exact original robot
prompt/style/action/source in `outputs/studio-c9ac0f48bd8044458a8390d4`.
`development-estimate-v3.json` retains the ten-call aggregate study cap, five
previous calls and at most five calls for this new one-repair flow. Its planner
was explicitly launched only after the integrated suite passed. The actual UI
then sent one pinned Start. The job ended **unsupported** after103.481s:
attempt0 failed the gross-spike gate; attempt1 passed that gate but31 directly
torso-weighted vertices still moved, up to4.828862mm against the fixed1.817729mm
stability limit. Hand-wave and foot-stability checks passed. No critic, complete
media or delivery followed; this is not an accepted UI-to-output example.
`outputs/studio-live-robot-v3/studio-live-test.json` records94 requests, one Start,
no external requests/browser errors, and verified browser/server exit without
signalling the generation worker. The worker exited and released its lease;
there is no active or queued Studio work. Tool12931 is terminal.
At that checkpoint the study totaled8 AI calls /196,763 input /10,061 output
tokens, with2 calls left under its then-current10-call envelope. No Meshy credits
were spent. The following different-route study explicitly expanded that finite
development estimate; old calls were never reset.

### Existing provider rig and weight-repair diagnostic

`outputs/studio-rigged-robot-preparation-v1` preserves a different hypothesis:
the exact same prompt/style/action on the already-paid24-joint robot skeleton,
source SHA256 `ee7477ccf173f480db7fe41924451e4483fa5fc369247fba679942b52df510a7`.
Its estimate records8 prior calls plus at most5 new ones,13 aggregate, with
unknown account-specific AI dollars and zero new Meshy/Meta calls. This is not
a new user-approved dollar cap or a fresh text-to-mesh example.

Actual Studio job `b184afa00c2f4c2d853de705` ended **unsupported**. One automatic
repair passed the unchanged numeric skin/action/stability checks and produced
full media (124.046s render), but the real critic rejected the low wrist swivel
as an unclear greeting. The browser followed one pinned Start for255.693s;
owned browser/server and worker exited, and its lease was released. Tool88829
is terminal. No accepted delivery is claimed. Exact evidence, including the
rejected clip, is preserved in `evidence/studio-rigged-robot-v1/manifest.json`,
SHA256 `cca42b76aa88c1bf9e85a47cc65c457112ef73fb4ce444b16556b553fd2efaa2`.

The author guide now distinguishes preserving an existing skeleton from keeping
faulty skin weights, and documents the existing ellipsoid/fallback control.
It explicitly forbids substituting a different action merely to pass numeric
checks.42 relevant contract tests pass in0.036s; the preceding integration
checkpoint passes629 tests in15.418s. Their source snapshots distinguish the
full-suite version from the subsequent prompt/report-label clarification.

The single remaining study call was used only for
`outputs/rigged-robot-weight-repair-probe-v1`, not to resume a terminal Studio job.
It proposed a bounded weight constraint and restored a raised gesture. One
22.758s local build passed the gross-spike gate but still failed torso stability
(35.527883mm), and independent image review found a raised chest-side wedge.
No exported preview, full media, critic or delivery followed. All13 calls are
now consumed:358,211 input /13,885 output tokens, unknown AI dollars, zero new
Meshy credits. No further AI call fits this study envelope. A separate opt-in
rigid-part weight-projection experiment is local-only and not automatic promotion.

The exact derivative SHA256
`78513a47f4232aaa5c5df3d2d520c4465f09a5d9bb772528f6b59f96607eff1b`
preserves all non-weight bytes and original majority role memberships. Its CPU
audit holds the torso/feet still, but the actual exported native check rejects
it: three severe edges at six of eight original action-key times, plus a visibly
distorted chest side under the raised arm. Native work completed in16.307s with
1,152,794,624B sampled direct-child RSS; tool50143 ended. Four rest views and
eight720px poses are preserved, not delivered as a completed animation pack.
See `outputs/robot-stable-projection-native-v1/manifest.json` and
`outputs/robot-stable-projection-review-v1.json`.

This exposed a diagnostic truncation defect: four longer extensions below the
8x ratio limit hid three shorter edges exceeding both severe thresholds.
The verified narrow fix prioritizes severe witnesses before both per-pose and
cross-pose truncation. The >8x and >3%-of-height limits and four-row cap stay
unchanged. All646 software tests pass in15.331s. Fresh numeric-only native replay
matches all eight prior pose/topology hashes and retains the three severe
witnesses across poses:1.820s /325,517,312B sampled direct-child RSS, verified
exit, no renders or new model/provider calls. Exact proof is
`outputs/deformation-retention-regression-v1/manifest.json`, SHA256
`013604be797dd87f7a403c445e0b230ede31958e6e2c6354e4e8c76220a0a637`.
This is detection proof, not a complete finisher rebuild or an accepted avatar.
The rejected projection remains outside automatic finishing.

The actual saved rejected media was rebuilt as HD review evidence in
`outputs/studio-hd-review-proof-v1/evidence`: six images, including unchanged
individual720px native/exported poses at1.2s. The worker completed in0.783s,
136,740,864B sampled direct-child RSS and7,133,675B output, with verified exit.
Manifest SHA256 `aea56aab79e595f0674468f47904d0e92fbfd05f75af5c70aff2b85248b457e9`.
This verifies image transport/assembly; the arm spike remains rejected, and
there was no new AI/provider/render call. An earlier wrapper failed before
process launch because its monitoring directory did not exist.

`evidence/studio-checkpoint-v1/manifest.json` preserves118 exact evidence files
(3,125,885bytes), including both planner prompts/receipts, the passing suite,
fixture results and the failed tuft trial. It is an incomplete-goal checkpoint,
not a delivery. Its manifest SHA256 is
`a081a1841af18f8b53844837ba45176bf63e67c8d12c98aa8a91f116935dfa78`.
Large diagnostic binaries remain at their original local paths with recorded
hashes; prior accepted assets remain unchanged.

## Portable fur experiment

The opt-in `tuft-v1` profile shares native regional grooming directions and samples
the source texture at each actual UV root. Existing profiles remain unchanged.
The first actual candidate, `outputs/panda-portable-tuft-v1`, **failed** the
unchanged2GiB sampled memory bound after27.428s. The runner verified process exit;
no automatic retry or exported preview ran. Retained views show the native coat,
not the new portable cards. The raw exported GLB remains diagnostic only:
87,333vertices/76,981triangles, SHA256
`3da9e96520c6fd47ee19f5160db6e24385217da2b999d63d3e1fc0362fcaa6da`.
No editable source/final finish report was completed. See
[the exact experiment and failure receipts](tuft-profile.md). Do not promote
the profile based on geometry counts or native stills.

A separate raw-GLB diagnostic completed in24.035s at1,207,336,960B sampled RSS:
`outputs/panda-portable-tuft-diagnostic-v1`, four720px views and eight actual poses.
The root reviewer inspected its front and flight pose and **rejected** the look:
larger tufts form conspicuous dashes/blocks and a spiky crest over a plastic-looking
base. This is not a quality improvement and is not promoted. The next hypothesis
is a source-colored short-pile undercoat with subtle silhouette fibers, prepared
as a separate opt-in experiment; no new generation or quality success is implied.

Undercoat trials `outputs/panda-portable-undercoat-v1` and `-v2` both stopped
before producing a texture/model. V1's correspondence tolerance was tighter than
Blender's measured import rounding. V2 retained exact face indices/UVs, measured
at most0.000000240274m positional rounding, then rejected aliased normal-map UVs.
Read-only attribution isolated all19,884 overlap samples to the authored repair
cap; the main body surface has no self-overlap. A separate bounded normal atlas
for only aliased regions now passes the tests and actual local build below.
No overlap threshold is relaxed and no new coat is accepted. Such a
UV1/normal-mapped result is HD-preview-only, not
Weft appearance-compatible; the existing approved geometry/motion stays unchanged.

The adaptive-undercoat integration passes 682 tests, a six-case native tangent
proof, an actual local finish and independent GLB reload. The v2 panda retains
75,548 vertices, one 19-bone skin and the original four-second authored action.
The correction changes only the selected tangent bytes; geometry, source color,
skin and animation bytes remain exact against the raw export. Native finish
122.283 s; four-view/eight-pose reload 22.214 s. Sampled review still finds smooth
portable fur and cheek/shoulder marks, so there is no default/gallery promotion,
new full-media claim or accepted Studio job. [Details and limits](adaptive-undercoat.md).

## Saved checkpoint

The tested implementation checkpoint was committed and privately pushed as
`87f224fb6e9e36c89df669bc3c8f2c00ee644a7a`. The private publication receipt is
Weft-root `.local/avatar-studio-checkpoint-v1/receipt.json`; a later independent
Weft commit advanced the remote while retaining this ancestor. Unrelated Weft
work was not staged. Subsequent defect fixes and diagnostic receipts are separate
work in progress; this save does not complete the goal.

`evidence/studio-reliability-v2/manifest.json` now preserves233 additional exact
evidence files /3,279,089bytes, including619-test source bindings, safe custody
recovery, actual rejected robot/UI attempts and both undercoat failures. SHA256:
`c893e370a636e5296be07f19248c4bbd9ae53175fbaa810b858aa1e190443b5c`.
Large originals remain at their manifest-bound paths. This is another incomplete
checkpoint, not a playable release or a quality pass.

`outputs/fresh-acceptance-preparation-v1` records three exact fresh inputs and a
105-credit /15-maximum-AI-call estimate. An actual user allowance question is
pending; no grants or fresh jobs/calls have been made. The original8-credit
remainder is unchanged. Execution also requires the local fixes to pass.
The append-only `outputs/fresh-acceptance-estimate-v2/estimate.json` revises the
input-token planning assumption to50,000 per call (750,000 at15calls), based on
measured15,814–42,341-token calls with selected evidence. These are estimates,
not token caps or known dollar costs; the old sealed estimate is preserved.

Use development failures to improve general code, not to hand-author final
recipes. Portable fur needs a real appearance improvement, not merely denser
numbers or a weaker reviewer. Fresh furry-dance, flying-creature and different
object examples plus actual UI-to-output acceptance remain required. Prepare
their concrete plans and aggregate credits before requesting added allowance.

The current save includes three separate compact rejected/component bundles and
`evidence/deformation-retention-v1` for the verified detection fix. Its exact
commit and private remote confirmation are recorded in tool-local
`.local/avatar-quality-publication-v3/receipt.json` after publication. This save
does not establish a successful Studio delivery or consistent Muse-quality output.
