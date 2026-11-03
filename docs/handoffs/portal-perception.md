# Unfamiliar-video perception in the local portal

Recorded November 2, 2026 (PST). Full video-to-qualified-world goal remains active.
This checkpoint connects new evidence preparation to the existing portal and
candidate constructors; it does not claim arbitrary-video or physical accuracy.

## Implemented user flow

For an unresolved exercise upload without registered prepared inputs:

1. **Check video locally** uses the normal background queue. No provider request
   or budget reservation occurs. It checks original MP4 eligibility, native body
   evidence and a source-bound initial person tracking region.
2. Inspect saved body/model/clock/preview artifacts and any failure. A usable
   local preflight produces a source-bound cost quote, not a simulation.
3. Separately consent to **OpenAI sampled images** and **Meta full video plus
   tracking regions**, then request preparation. Both controls start unchecked.
   The old upload's OpenAI checkbox never supplies Meta consent.
4. At most one analysis and one tracking request use the existing server-key
   adapters and separate lifetime ledgers. Exact OpenAI cache reuse remains
   possible. No ChatGPT login, agent runtime, new key, provider account or budget.
5. New sealed evidence automatically queues the same ordinary equipment builder.
   All admitted/rejected alternatives remain; the user explicitly selects one
   before opening the existing sandbox. Recordings remain locked to that world.

Prepared-source catalog matches keep their existing faster route. New evidence
uses a closed job-local descriptor, never writes the administrator catalog or
requires stopping/restarting the portal to register its own output. Existing
equipment attempts cannot trigger new paid perception in the same project.

The current path supports original uploaded MP4 bytes ≤37 MB, 12–500 frames,
≤60 seconds, portal-compatible dimensions, one usable person, fixed-camera
analysis and at most three equipment regions plus the inferred person anchor.
No silent conversion, crop, retiming, omitted object, inferred absence or lowered
confidence/geometry gate. Other source scenarios remain incomplete/unsupported.

The generic initial **Auto** upload may already have paid for classification;
that history remains separate. For the new no-paid-work-before-local-check path,
choose **Exercise**, **Equipment/obstacles** or **Unsure**, and **Interactive scene
simulation**. Body preview is still a distinct no-physics output.

## Implementation and boundaries

- `exercise_perception_pipeline.py`: local preflight and pure sealed reuse;
  native body/seed/anchor prepared once, exact source/child checks before each
  paid phase, small progress callbacks. Pure validation performs no inference,
  video decoding or seed reconstruction. Reused preflight implementation is
  pinned; changed code needs explicit compatibility rather than silent reuse.
- `portal_perception_jobs.py`: quote/consent/source/model checks, local and paid
  phases, immutable start markers, provider checks, retained failures and queue
  handoff. Source/model/cost changes invalidate old consent. Unexpected local
  integrity failures are `failed`, not mislabeled unsupported footage.
- `portal_equipment_candidates.py`: exact job-local bridge/base/equipment,
  OpenAI request/response/receipt, original tracking/export and person-anchor
  joins. External preflight body/seed stay inside the project. Generated body
  members count toward aggregate 256-MiB/1,536-file validation limits and final
  readback. All assignments still pass the ordinary source/geometry checks.
- `portal.py`/`portal_worker.py`: one supervised slot; phase-specific logs and
  receipts; 450-second parent wall deadline, inherited CPU ceiling, sampled RSS
  and storage limits. Preflight gets no provider environment keys. Paid mode
  receives only the existing relevant project-key environment variables.
- `shared_world_lab.py`: native-only seed propagation now retains the complete
  native/capture artifact set and exact body/observation clock summary. Legacy
  output semantics, source timestamps, dt 1/30 and 16 physics substeps remain.
- Portal UI: bounded status and cost metadata, separate consent, source artifacts,
  active cancellation, existing candidate cards, no fabricated progress percentage.

States distinguish local checking, awaiting consent, analysis, tracking, evidence
export and queued construction. Restarts interrupt active work without resending.
Duplicate actions cannot replace immutable consent or repeat a provider call.
Cancellation preserves unknown charges and already completed evidence; a queued
construction child published during cancellation is cancelled before it can run.
Original upload status/history and immutable lab runs are not repurposed.

HTTP actions:

- `POST /api/projects/{id}/perception/preflight`: zero-byte body, local only.
- `POST /api/projects/{id}/perception/prepare`: closed ≤2-KiB JSON containing
  `quote_sha256`, both explicit booleans in `consent`, exact quoted
  `openai_max_usd`/`meta_max_usd` and a 32-hex `request_id`; no paths/model/code.

Current conservative quote reservations are $0.3135 OpenAI and
`max(.15, max(50, frames) × 4 / 5000)` Meta, at most $.40 for tracking. These
are repository reservation rules, not measured future charges or provider hard
caps. Actual accepted region count sets the Meta reservation; invalid analysis
never triggers tracking. Both existing $5 budgets remain separate. Verify
provider pricing before a live acceptance request; none was made this checkpoint.

## Actual local acceptance and preserved failure

Source: the existing CC BY-SA 4.0 Danielflefil 68-frame squat GIF derivative,
`data/exercise/bodyweight-squats-danielflefil-500.mp4`, SHA
`ec636e28b4711887ecd9ff7014b64801d16055faf2f2b0bf6ed9221ea8280118`.
This is an attributed lossless video adaptation, not raw camera footage or a new
holdout. Source/provenance stay unchanged. No equipment absence is inferred.

First harness `.local/portal-perception-native-01` sent `{}` to the zero-body
preflight endpoint and stopped before inference. The same saved upload was reused
in corrected harness `-02`, not uploaded again. The real HTTP/queued local run
took **13.960 s**, worker **13.453 s /411,451,392 bytes sampled peak RSS**. It
served body XML, pose evidence, native clock and preview artifacts, preserved
parent/source/runtime bytes, and made **zero API calls or reservations**.
Server/queue/children were closed/reaped.

It exposed a genuine seed-adapter defect, not a physical-source qualification
failure: completed native body artifacts matched their own seals, but the seed
adapter omitted native/capture receipts before strict validation. The old
lifecycle output called this unsupported; preserve it alongside the corrected
classification. The native propagation defect and a second nominal-clock-label
defect are now fixed without changing source times, model confidence or task gates.

Project: `ba42b28738134686a10d2ada39eb0c6c` under
`.local/portal-perception-native-01/portal`; failed body remains at
`perception/preflight/body_source`. Do not overwrite or repeat its inference just
to inspect the already-preserved evidence.

Saved-only recovery `.local/portal-perception-native-seed-recovery-03` takes
**0.866 s /190,382,080 bytes peak RSS**, one source decode, no new inference/API
or dynamics. It successfully creates a native body seed, then correctly stops
at the unchanged .5 landmark-visibility gate: frame-zero right elbow .2622204
and right wrist .4848379. Presence scores do not override missing visibility.
This is **not** a successful real-source quote, paid preparation or playable world.
Do not repeat this clip under unchanged anchor rules or lower its threshold.

- Original body seal: `e064b5c3bc7501eb3b26d3b3f32850c6508af87a12c8b12e9aa5c9af2c26e92f`.
- Recovered native seed: `388f5648e7db17e77768a5f64ea2bc261c3239417377a7ba3b64f00151ccb2d8`.
- Recovery result: `1e017bc589a94439a955d9b4df53e505318f3bb7b65d1aa43bb62a4587a13d69`.
- Source audit: `1749261440333ca1a49eac2e1327421b72553a7a751cbc018795edce5ab8e818`.

Recovery attempts 01/02 stopped during harness imports (socket-class replacement,
then missing frozen palette resource), before decode or source work. Preserve
them. Root independently verifies 76 sealed body/seed members, 27,079,678 bytes,
exact native summaries, original failure and unchanged confidence refusal in
`.local/portal-perception-audit-02.json` (0.032 s). The earlier root audit used a
wrong JSON field path; its failure is retained separately, with no source mutation.

## Verification and next safe action

Three independently owned lanes implemented preflight, descriptor/construction
and portal UI/queue. Root owns orchestration, integration, docs and publication.
Analytical tests exercise real sealed body/seed reuse and one actual ordinary
constructor; provider/lifecycle fixtures are not live API or physical evidence.
**191 distinct scoped Python checks and 15 UI journeys pass.** Deduplicated
receipt: `.local/portal-perception-tests-final-01.json`. This includes 22 portal
route/lifecycle cases, 39 descriptor cases (one actual ordinary constructor),
preflight/reuse/native custody checks, orchestration and affected legacy cases.
Counts overlap across suites and are not summed naively. Five authenticated
loopback regressions passed separately with narrow sandbox permission. No browser
visual QA or new live-provider validation is claimed.

All three implementation lanes are released; no test server, queue or child
process remains. The previous turn was progress (`7847d98`); this turn implements
the portal flow and fixes defects found by actual source execution. It is not
blocked, and the overall goal is not complete.

Next obtain/qualify an independent front-visible equipment source, or develop a
separately evaluated source-supported tracking-anchor method without silently
lowering current gates. Exercise the paid complete bridge within existing budgets;
retain any identity/geometry rejection rather than relabeling evidence-ready as
a playable world. Connect newly qualified body evidence to the existing Muse
review contract; selected conditional candidates still lack a review attachment.
Full-task learning, unfamiliar ground/contact reconstruction,
broader independent evaluation and physical validation remain unfinished.

Both provider budgets are unchanged: Meta $0.282827 committed, 12 settled
reservations; OpenAI $0.003252, two settled calls; zero unsettled. Publication
receipt: `.local/portal-perception-publication-01.json` once the scoped main push
is independently verified. Preserve all separate avatar files/commits and local
studies. No stash cleanup is needed if the last empty inventory remains empty.

Separate avatar changes are untouched. No deployment, hosted-site restoration,
submission, browser interaction or persistent development server is authorized.
