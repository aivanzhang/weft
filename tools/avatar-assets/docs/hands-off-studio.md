# Hands-off avatar quality and creation studio

## Active outcome

Build a repeatable prompt-to-avatar workflow that preserves the requested look
and custom actions, automatically reviews and repairs quality, and exposes real
job progress from input through downloadable output. A form around the old
pipeline alone does not complete this goal.

Work stays in `tools/avatar-assets`; Weft simulation, physics, rewards and its
independent goal are unchanged. Existing assets and failed attempts remain
immutable. No deployment or persistent server starts automatically.

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
- `agent_runner`: completed durable jobs/worker; bounded opt-in portable tuft profile.
- `pipeline_contracts`: action stability intent/audit correction after live planner evidence.
- `pipeline_review`: completed browser/docs and analytic native framing regression.

No new Meshy generation has started. Two real AI planning calls produced one
rejected plan and one reviewed source-reuse plan. The renderer proof and temporary
HTTP/browser fixtures are local checks, not fresh generation.
Read-only canonical Meshy verification still shows
16 successful submissions / 202 credits against 210 authorized credits.
The remaining eight credits do not authorize a new 35-credit generation.
Fresh verification jobs require concrete prepared estimates and sufficient
user-authorized allowance; no automatic grants, purchases or paid retries.

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
- The latest full Python suite passes552 tests in13.475s; exact log:
  `outputs/studio-integration-v1/full-tests-v2.txt`. The earlier521-test run's
  oversized-upload exception remains preserved alongside its regression fix.
  These checks do not establish visual quality or a completed real Studio job.
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
source-reuse authorization. Execution has **not** started. The development study
keeps a ten-AI-call aggregate envelope, including the first rejected planner;
the new job's current one-repair flow can make at most five role calls total.
The tuft adapter is frozen and its native worker has exited. The browser verifier
owns the next heavy-worker slot for one explicit Start of this reviewed job.
Preserve both job histories; do not change source-pinned helper files while this
job needs execution. No live UI-to-output result is accepted yet.

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

Use development failures to improve general code, not to hand-author final
recipes. Portable fur needs a real appearance improvement, not merely denser
numbers or a weaker reviewer. Fresh furry-dance, flying-creature and different
object examples plus actual UI-to-output acceptance remain required. Prepare
their concrete plans and aggregate credits before requesting added allowance.
