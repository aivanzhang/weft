# Website and API backend

The user selected OpenAI API for the website backend. Website jobs must not
depend on a personal ChatGPT login, local CLI session or operator-written rig.
The first live API planner succeeded; no API-backed delivery has passed yet.

## Requested flow

1. The user enters a character, style, optional reference and custom actions.
2. The backend saves the immutable request and checks its spending authorization.
3. OpenAI returns structured planning and rig/motion proposals. It receives only
   this job's prompt and selected visual evidence, not credentials or filesystem access.
4. Meshy creates geometry/textures when needed. A recovery can reuse the exact
   already-paid model without submitting geometry again.
5. Bounded Python/Blender workers build the rig, weights, requested fur and motions.
6. Independent numeric checks and final-media review accept or reject the result.
7. The website shows durable progress, failures and model/media/source downloads.

The existing Studio HTTP service and detached job queue remain the integration
boundary. AI calls and native work belong in workers, never a browser or status
request. Retain exact action intent, static fallbacks, failed attempts and honest
loop/physics declarations. Accepting the earlier appearance does not excuse a
broken rig, missing requested features or an unrelated action.

## Credentials and spending

- Server-only OpenAI key: `/Users/ivan/kintsugi/weft/.local/credentials/openai-api-key`.
  The user supplied it; metadata-only inspection confirmed a regular owned file
  and permissions were tightened to0600. Its contents are never displayed.
- The existing Meshy key remains the geometry provider credential.
- OpenAI dollars and Meshy credits have separate persistent ledgers. Neither
  may use Weft's Meta key or its budget. A form submission cannot raise budgets.
- No automatic retry of an uncertain charged request; retain its reservation
  and evidence. Model availability, refusals and limits must produce explicit
  job states, not fallback to a personal CLI login.

Initial adapter work uses a priced, pinned GPT-5.4 snapshot with structured
JSON and image inputs. Model choice, price assumptions, request bounds and
reported usage must be recorded per job. The user approved a **$17 total maximum**
for the saved panda continuation, limited to three API calls and zero repairs.
The conservative reservation is $16.85592; the rough normal-usage estimate is
$0.55, not a measured bill. No new Meshy charge or automatic retry is authorized.
Approval and parent binding: `outputs/openai-panda-continuation-preparation-v1/`.
Continuation job: `dbd9606e33444ce9b69c2a3c`. The project dollar ledger is configured;
one live planner call succeeded in20.805s, with2,099input/1,178output tokens
and a conservative actual-cost estimate of$0.022918. It then failed local
validation because its state ID exceeded64characters. No rig/motion was produced.
The original API response and failure remain immutable; no retry was attempted.
The reusable schema now constrains state/role/requirement IDs and references to
`^[a-z][a-z0-9_-]{0,63}$`; local validation enforces the same grammar.
Nine focused regression checks pass, including the exact retained failed answer.
A new paid planner attempt needs explicit retry permission; the prior no-retry
condition remains in force.

## Existing panda recovery

Original job: `2fa07fbb61ef4da4bc8f7f70`. Both Meshy stages succeeded for35credits.
The textured model is preserved at:

`outputs/studio-2fa07fbb61ef4da4bc8f7f70/generation/stages/texture/downloads/model.glb`

SHA256: `f5f7074d3ffe85c5edb35a5f91ee29580a37b2f4fe26654e1cf850786f70594e`.
The planner succeeded; the original CLI author stopped at an account usage limit.
No rig/dance delivery was completed. Two of the approved five AI attempts count
as consumed; preserve them. A continuation needs an explicit parent binding and
no more than the remaining three calls, not a silently reset allowance.

The fresh-generation lease-release guard is being extended separately. It must
prove saved remote completion, downloaded hashes and local process custody before
releasing the failed job. No direct database edits or failed-folder replacement.

## Hosting boundary

This is website/backend implementation, not authorization to restore Weft's
offline hosted site or deploy a new service. The current web interface is local.
A hosted installation needs a real persistent Python/Blender worker, private
artifact storage, authentication and durable budgets; a browser-only wrapper or
serverless request that starts a laptop CLI is not that deliverable. Hosting and
recurring costs must be concretely prepared before deployment approval.

## Actual browser/backend outcome

Job `9e1cceb87c40437896c5a77f` is terminal **unsupported**, not complete. The
approved planner retry returned valid IDs. A local stationary-language false
positive was fixed, and the exact saved plan was revalidated without another API
call through an explicit append-only runtime revision. Original source snapshots,
response, failure events and spending records remain preserved.

One actual website Start then reused the paid model, completed geometry inspection
and called the API author. The outer structured response succeeded, but its
`design_json` string was malformed (expected delimiter at EOF). No native rig,
requested dance, final media or critic call followed. The browser test ended after
51.598s with browser/server exits verified, no worker cancellation, no retained
active lease and no queued job. `outputs/openai-panda-studio-live-v1/` records it.

Three API calls across both jobs used an estimated$0.141219total. The remaining
reviewer slot does not authorize an author repair. Preserve `unsupported.json`;
next implement a strict typed author contract instead of an opaque JSON string.
No new paid call or budget increase is implied by that local fix. The website/API
transport is live-tested; a successful hands-off avatar delivery is still unproven.
