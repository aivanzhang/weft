# Visible-person tracking and selected-world Muse

November 2, 2026, 5:15 PM PST. One integrated pipeline; full goal remains active.
Three parallel lanes delivered source acquisition, partial-pose tracking and
selected-world Muse custody. Root integrated portal behavior and actual local
acceptance. Separate avatar work is untouched. Hosted site remains offline.

## Delivered

- `person_tracking_anchor.py`: versioned tracking hint from the original frame-zero
  pose. Keep all 33 raw estimates; the same .5 visibility/presence threshold
  excludes unsupported landmarks. Require head/torso, one complete arm and leg,
  and a nondegenerate torso. Excluded landmarks are not imputed; full-person
  extent and physical identity remain unknown. This is not a body-quality gate.
- `exercise_perception_pipeline.py`: explicit optional v2 profile, helper code
  sealed with preflight, exact profile-aware revalidation and bridge reuse.
  General builders retain v1 default; **new portal preflights explicitly use v2**.
  Old immutable failures/seals are not rewritten or silently upgraded.
- `portal_perception_jobs.py` and `assets/portal.js`: inspect the tracking-region
  receipt and every excluded landmark alongside original pose evidence.
- `portal_perception_review.py`: a selected perception candidate can supply its
  exact generated preflight body for Muse review. Bounded metadata permits
  queueing; the worker verifies source/descriptor/selection/body/world joins
  before any request and again afterward. The existing strict attachment join
  is unchanged. The immutable receipt is `muse/perception-attachment.json`.
- `portal_equipment_candidates.py`: independently rederive the sealed anchor
  profile; optional absolute validation deadline remains capped at 30 seconds.
  Muse shares a 15-second source/body/world admission bound, leaving room for its
  existing 45-second provider limit and post-request revalidation.
- `portal_review.py`: selected role assignments remain explicitly conditional in
  Muse's existing question contract. No scores/policies/final evidence or new
  model/schema/budget. Portal API and UI keep checks/repairs unavailable for these
  candidates; review is not a repair or physical qualification.

## Verification

**209 distinct Python checks and 16 UI journeys pass.** Latest result per logical
case is recorded in `.local/parallel-portal-acceptance-01/test-union.json`.
This combines source/anchor/preflight tests, review/attachment/descriptor tests,
portal orchestration and route/candidate regressions; provider fixtures are not
live calls. The full integration suite exposed one stale error-message assertion;
its correction reran only that failing case. Failed receipts remain preserved.

Root independently checks 671 Muse snapshot members and 21 acquisition members,
34.51 MB total, plus all 458 native mapping rows in 0.118 seconds. It also verifies
all 33 raw anchor slots and original input/runtime hashes. Audit receipts:
`.local/parallel-portal-acceptance-01/{saved-evidence-audit,package-audit}.json`.

### Partial-pose hypothesis, saved real source

`.local/person-anchor-visible-01`: saved squat evidence yields ROI
`[195,29,124,471]`, supported by left arm/both legs. Raw indices 14,16,18,20,22
remain excluded/unknown. 0.013 seconds helper /0.047 process /27.7 MB peak RSS;
no decode, inference, tracking, dynamics or API. Result SHA256
`b1e9bef2989c3a0217297487863b58cb5a3ff81dcc55178d691218959be21983`.
This tests a new tracking hypothesis, not a successful source reconstruction.

### Selected-body Muse custody, analytical world

`.local/portal-perception-muse-01`: analytical 63-frame source and authored pose/
provider fixtures pass real preflight/bridge validation, ordinary conditional
construction, selected lab admission and unchanged exact body/world join.
Alternative 1 is selected; alternative 2 remains unsupported. Original 670 job
files remain unchanged. Altering body evidence stops the review worker before the
provider boundary; the saved final failure is that explicit negative test.

Actual integration takes 3.285 seconds. Final saved-only revalidation under the
shared 15-second deadline takes 0.542 seconds /121,880,576 bytes peak RSS, without
reconstruction, decoding, inference or API. New Muse identity-uncertainty question
is verified. Snapshot contains original absolute fixture paths: do not present it
as a relocated runnable portal. Final seal:
`18b681cbe4e0d4733f24b7b64a8ee798a8022505465d2bfd657431b21ec16bcc`.
No live Muse call or real-source physical qualification is claimed here.

## New real video: retained local rejection, no charge

[Source and rights](../../data/exercise/kettlebell-flip-juggling.source.json):
Taco Fleur / Cavemantraining, CC BY-SA 4.0, new recording but same performer as
existing kettlebell sources; development evidence, not a fresh final holdout.

- Original WebM SHA256 `62aa8e2eff77484a857a3d56b8e7c91d9448fe71d1a081862611b967a247c4e6`.
- MP4 SHA256 `c289fc1afe0b60a418cc6ed9e4554c8414a4d970d2dfcaea1f6f82839204dac3`;
  18,177,108 bytes, 458 frames, 640×480. Full decoded YUV pixels and native elapsed
  intervals preserved. Audio omitted. Source native time = derivative time +3 ms.
- Mapping SHA256 `8b4e960d58a86c21c03acc953adf3e1bfab0e43cab0d4f1515297f7aba9b6f55`.
  First exact-origin assertion failed; preserve it alongside the explicit mapping,
  not a second conversion or hidden clock normalization.
- Acquisition seal `f784c025a5149c5d63bc9d96c83f0e3130b95ae8bddccb1cde92304f93835e95`.

Actual ordinary HTTP upload/queue/local preflight:
`.local/portal-kettlebell-flip-01`, project `a85537e09e284facb10de22cf2a7dcbb`,
under that directory's `portal/`. **16.477 seconds**, worker15.271 seconds /
401,129,472 bytes sampled peak RSS. Failure/pose/overlay/resource artifacts served
through authenticated routes; parent/source/runtime unchanged. Server, queue and
children reaped. No quote, paid preparation, selected world or training.

Body construction refuses **competing person hypotheses** before reaching the
new anchor: 143/153 usable samples (93.46%), 7 low-confidence, 1 invalid geometry,
2 competing samples at source frames111/123 (3.7/4.1 seconds). Both alternatives
pass per-pose geometry; do not discard a convenient loser. Saved diagnosis
`.local/kettlebell-flip-pose-diagnosis-01` takes0.054 seconds, preserves all eight
original artifact hashes, makes no decode/inference/API call. Seal:
`f3e15e0a1fa86d9e15dca12c61d982209e919740d328474a14b9b8ec7050763e`.

## Spending, ownership and next step

Official pricing rechecked in `.local/parallel-portal-acceptance-01/cost-plan.json`:
[OpenAI](https://developers.openai.com/api/docs/models/gpt-5.4-mini) and
[Meta](https://dev.meta.ai/docs/pricing-rate-limits). No new provider call or
reservation. Last verified: Meta $0.282827 committed/$4.717173 remaining;
OpenAI $0.003252/$4.996748; zero unsettled. Budgets remain separate lifetime $5
allowances. The paid acceptance harness is prepared but **not executed**; local
source rejection prevents consent/request execution.

Do not repeat inference on this clip under unchanged rules. A new bounded lane
is assessing explicit ambiguous-frame abstention: preserve both hypotheses,
mark whole ambiguous frames unavailable, retain all slots and original coverage,
gap/floor/identity limits. Assessment confirms that merely skipping two ambiguous frames would bypass the
global identity veto and allow interpolation between potentially different
people. A separate pure source-window/identity-barrier helper is being
implemented; no constructor or native-clock change is authorized by this lane. The existing
selection/crop paths explicitly refuse competing hypotheses.

Runtime agent owns that assessment; root owns integration, docs and Git. Other
three-lane jobs are terminal. Publish only the verified files for this checkpoint,
with no avatar files or unrelated index entries. Main was fetched/current; stash
inventory empty. Publication identity is recorded separately in
`.local/parallel-portal-acceptance-01/publication.json` after remote verification.

Next qualify source identity/body evidence, run one bounded paid bridge only
once local checks pass, and test admitted-world interaction/recording/Muse.
Source-ground/equipment contact, useful complete-task learning, broader unseen
coverage and independent physical validation remain unfinished. Existing playable
portal examples remain available via `.venv-lab/bin/python -m weft portal`;
the user manages that server. No deployment, browser or submission.
