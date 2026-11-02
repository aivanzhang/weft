# One fresh Studio dance job — approval preparation

Status: **prepared only; no job created or planned, no allowance granted, no AI,
Meshy, native, server or browser execution**. The user accepts the previous
appearance quality. This proposal stops appearance-upgrade experiments and tests
one real hands-off text → fresh 3D model → authored dance → verified delivery.
It does not pursue Muse parity, companion separation, longer fur or new polish.

## Exact request

Use [`proposed-request.json`](proposed-request.json) unchanged with the canonical
`StudioJobs.create` API. It requests one original short-fur panda-like character,
one four-second rhythmic dance, no companion/props/reference, and no reused model.
The pose leaves clearance around moving limbs. The action requires three visible
beats, small hops and arm/head motion relative to the torso; root travel alone
does not satisfy it. This is authored toy motion, not physics.

Preset: `preview` (720×720 clip, 24 fps, 4 seconds/96 delivered frames, 1600px
portrait and static fallback). Studio passes `states=[]`; the planner supplies
one state ID and must retain the sole custom-action text verbatim as
`user_request`. Do not insert a handwritten rig, mask, weight recipe, motion or
edited agent answer. Native strands and portable fur remain separate outputs.

## Required narrow allowance

| Work | This one job |
| --- | ---: |
| Fresh Meshy geometry, 4K geometry setting | 25 credits |
| Texture refinement, 4K setting | 10 credits |
| **Meshy total** | **35 credits / 2 paid submissions** |
| Provider rigging, animation, paid regeneration/retries | 0 |
| AI normal flow | 3 calls: planner, author, final-media critic |
| AI maximum with one local repair | **5 calls**: planner + 2 authors + 2 critics |
| New hosted services, deployment or recurring cloud operations | 0 |

This is the current adapter estimate: `meshy-7.1`, triangle topology/remeshing,
target 35,000 triangles, no provider rig. The saved provider pricing check is
dated October 31, 2026; the acceptance-plan reference was checked November 1.
No network pricing or account-balance request was made during this preparation.
Account dollars per credit and AI account billing remain unknown. Generated
resolution settings are not proof of delivered texture detail or acceptance.

At **November 2, 2026, 10:25:42 AM PST**, the canonical ledger was unchanged:
16 successful submissions, **202 credits used/reserved out of 210 authorized**.
Base allowance 140, existing job grants 70, base usage 132: **8 base credits remain**.
The numerical shortfall is 27 credits, but `Ledger.authorize_job` only appends a
grant equal to the new immutable job's complete ceiling. The current executable
path therefore requires **one additional job-specific 35-credit grant**, leaving
the old 8 untouched. No credit purchase, recharge, account or subscription is
authorized by that grant.

The earlier study's **13 of 13 AI calls remain consumed** (recorded aggregate
358,211 input / 13,885 output tokens). This job needs a separately approved
maximum of five new calls; no old counter is reset. The broader proposed
105-credit/15-call three-case study remains unapproved and is not needed here.
At the five-call maximum, the recorded planning assumption is approximately
250,000 input and 5,000–40,000 output tokens; actual billing/token usage must be
retained. Bytes, deadlines and calls are bounded, not a promised token price.

## Five-call discrepancy resolved before any job

Read-only inspection found Studio persisted `max_agent_calls=10` despite a
one-repair flow with at most five role calls. Root authorized the narrow change:
new Studio requests now pin **5**, with `max_repairs=1`, `agent_timeout=240` and
the existing model selection unchanged. The general non-Studio pipeline default
remains 10; Studio supplies its explicit five-call override.

`avatar_assets/studio_worker.py` SHA256:
`b09cdea70a4c285bd69c2070b035f2199256b315fe7a6214fa58ccff35a9298e`.
The strict prepared-request validator also expects five; no budget coercion,
monkeypatch or handwritten request bypass is proposed. Historical completed
ten-call jobs remain readable without execution. Unfinished old requests are
not rewritten or restarted; incompatible preparation/backend pins reject them.

One focused bounded execution passed **29 Studio lifecycle methods** in 0.770s,
sampled direct-child peak **43,139,072 B**, PID 8944 exited. It includes new-cap
persistence, rejection of 10/5.0/boolean substitutions, historical completed
delivery preservation and an unfinished ten-call rejection before pipeline work.
Evidence: `outputs/studio-call-cap-tests-v2/manifest.json`, SHA256
`98c106db3366b68c21750627a8e66744e82fd07b5f2f7d38637ec6311b31ce8e`.
V1 preserves a source-inventory preparation failure (`tests/__init__.py` is absent);
**zero tests or bounded children ran in v1**. There was no actual Studio job,
provider, AI, native, server or browser call in the software tests.

## Canonical path and current registry

The class is `StudioJobs`, not `JobStore`. Canonical binding:
`tools/avatar-assets/.local/studio-registry.json` → `.local/studio`.
Registry database: `.local/studio/jobs.sqlite3`.
Creation writes the immutable request under `.local/studio/jobs/<24-hex-id>`;
the real worker creates `outputs/studio-<id>`. Do not use another registry,
fixture executor, direct SQLite edits or a prewritten pipeline request.

The read-only snapshot has **four jobs**, no active lease and no queued work:
one cancelled, one failed and two unsupported. Cached retained size is
1,050,856,884 B; a new 2 GiB + 64 MiB reservation fits the 8 GiB registry guard.
This is cached accounting, not a fresh disk-capacity or OS-process census.
Old terminal jobs and their evidence remain intact.

`StudioJobs()` can initialize metadata and dispatch queued work, so this
preparation **did not instantiate it**. `inspect.py` opened the existing databases
with SQLite `mode=ro&immutable=1`, refused journal/WAL/SHM files, and verified their
SHA256 before/after. It did not read credentials, balance, API responses or keys.
[`read-only-facts.json`](read-only-facts.json) binds the pre-correction sources and
registry/ledger hashes; its recorded ten-call value is historical, not the new cap.

## Future commands — not executed

Only after root records the user's narrow spending approval and freezes the
current source/test checkpoint, use the public Studio lifecycle. Recheck that
there is no conflicting active lease/queue before construction. Suggested unique
idempotency key: `studio-short-fur-dance-20261102-v1`.

1. Create the immutable draft (no AI or provider call by `create`):

   ```sh
   .venv/bin/python -B - <<'PY'
   import json
   from pathlib import Path
   from avatar_assets.studio_jobs import StudioJobs
   request = json.loads(Path('outputs/studio-completion-preparation-v1/proposed-request.json').read_text())
   jobs = StudioJobs()
   result = jobs.create(**request, idempotency_key='studio-short-fur-dance-20261102-v1')
   print(json.dumps({'job_id': result['id'], 'status': result['status']}))
   PY
   ```

2. Plan that exact job. This launches the real detached Studio worker and consumes
   the **first approved AI call**; it does not submit Meshy generation:

   ```python
   from avatar_assets.studio_jobs import StudioJobs
   result = StudioJobs().plan('<actual-job-id>')
   ```

   Retain the returned job ID, owner/invocation receipts and terminal plan ID.
   The planner and generation plan must preserve this request and the 35-credit
   ceiling. The expected stopped state is `awaiting_authorization` because the
   current eight credits do not cover the new job. An unsupported planner result
   is preserved, not rewritten into a different request.

3. Bind the existing user approval to the actual generated plan. Root must write
   a real approval record with the exact `job_id`, canonical `plan_digest`,
   `additional_credits:35` and verbatim user approval. Then use the established
   command, not `configure`, a reset or a purchase:

   ```sh
   .venv/bin/python -m avatar_assets meshy authorize \
     outputs/studio-<actual-job-id>/generation --approval-record <actual-approval-record.json>
   ```

   No approval record exists in this proposal. A preset, `max_credits` field or
   prior broad proposal is not spending authorization.

4. Start through the same Studio manager with the reviewed immutable plan ID:

   ```python
   from avatar_assets.studio_jobs import StudioJobs
   result = StudioJobs().start('<actual-job-id>', plan_id='<actual-plan-id>')
   ```

   This rechecks canonical allowance and plan/request/source bindings. The actual
   executor, not a fixture, owns generation, inspection, AI authoring, finishing,
   numeric attachment/action checks, exported views, final media, critic and
   delivery. Use saved status/owner receipts to monitor; never duplicate a running
   or uncertain operation. No manual intervention in the generated recipe is part
   of acceptance. A checkpoint may resume the same immutable job without resetting
   any allowance or submitting geometry again; an uncertain exit requires inspection.

The equivalent UI endpoints are `POST /api/studio/jobs`, then `/<id>/plan` and
`/<id>/start` (the last carries the reviewed `plan_id`). They require the existing
local session and creation idempotency header. No server or browser is started
by this preparation; a direct canonical-manager run alone is not a new browser
interaction verification.

## Bounded work and completion

- One active Studio lease; at most four queued jobs and eight invocations per job.
  Run no concurrent unrelated native job during this demonstration.
- Each role has 240 seconds, no automatic failed/uncertain retry and a persistent
  five-call cap. Normal completion is three calls; one repair can add two.
- Studio supervises each invocation for 5,400 seconds with a 1 GiB controller RSS
  and 64 MiB controller-output cap. Native/agent children have separate bounds;
  this is not whole-process-tree memory accounting.
- Pipeline stops starting new stages after 3,600 seconds per invocation and keeps
  a 2 GiB retained-output ceiling. A resume opens another watch window, not a new
  call/repair/generation allowance. Studio requests a cooperative stop if measured
  pipeline output exceeds that ceiling; it does not erase prior data.
- Meshy watch: 900 seconds, resumable existing task IDs, no automatic paid retry.
  At most two native builds/exported-review/video attempts (one repair). Native
  stages retain 4 CPU threads/2 GiB sampled direct-child RSS; build/view cap 600s.
  Final media cap 1,200s; only the existing eligible hash-bound continuation can
  add 600s. Root attachment audit is 60s/512 MiB/16 MiB; other existing stage bounds
  remain unchanged. These are ceilings, not a predicted end-to-end latency.
- Preserve accepted/rejected files, prompts, usage, source/model hashes, editable
  source, GLB, portrait, static fallback, clip and every native/agent/provider/job
  receipt. Reserve local headroom; do not delete evidence or increase caps to pass.

Success requires a real canonical Studio job ending `complete` with its sealed
delivery and all existing action, deformation, portable-root, media and critic
checks. The dance must remain the requested dance. An unsupported result is honest
but does not complete this demonstration. Existing appearance acceptance stops
further polishing work; it does not authorize pretending missing animation,
detached fur, corrupt evidence or failed jobs succeeded. No new Weft/physics,
continuous-motion, seamless-loop or Muse-equivalence claim is part of the goal.

**Next safe action:** root reviews this exact request, verifies the source cap
checkpoint, and requests the one-job **35-credit / maximum-five-AI-call** allowance.
This proposal does not grant it or consume the pending three-case allowance.
