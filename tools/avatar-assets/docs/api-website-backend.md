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

The adapter uses pinned GPT-5.4 structured JSON and image inputs. The user
approved a **$17 total maximum**, initially for three API calls without retries,
then explicitly approved one planner retry followed by authoring and review.
The first planner cost $0.022918 and failed local state-ID validation. Strict IDs
now enforce `^[a-z][a-z0-9_-]{0,63}$`. Its successor reached authoring and failed
as recorded below. Both jobs and their original grants remain preserved.
Those approvals did not allow another retry. The subsequent authorization below
explicitly covers the prepared typed-contract run.

Every call reserves $5.618640 before HTTP: the full model context at long-context
rates plus the configured maximum output. This is a conservative hold, not the
expected bill. New accounting policy `validated-settled-exposure-v1` counts
validated settled usage at its conservative actual cost and retains full holds
for uncertain charges. It never removes original reservation rows, resets call
counts or releases an uncertain charge. Transactional checks enforce both job
and project ceilings. Retired parent costs count exactly once. Each subsequent
call must fit its full hold; a five-call limit does not promise all five will fit.
The model, pricing, original holds and accounting policy remain pinned per job.

## Existing panda recovery

Original job: `2fa07fbb61ef4da4bc8f7f70`. Both Meshy stages succeeded for35credits.
The textured model is preserved at:

`outputs/studio-2fa07fbb61ef4da4bc8f7f70/generation/stages/texture/downloads/model.glb`

SHA256: `f5f7074d3ffe85c5edb35a5f91ee29580a37b2f4fe26654e1cf850786f70594e`.
The planner succeeded; the original CLI author stopped at an account usage limit.
No rig/dance delivery was completed. Two of the approved five AI attempts count
as consumed; preserve them. A continuation needs an explicit parent binding and
no more than the remaining three calls, not a silently reset allowance.

The guarded release verified remote completion, downloaded hashes and local
process custody, then released the failed job. Its evidence is preserved in
`outputs/studio-panda-inspection-v2/`. No direct database edits or failed-folder
replacement were used.

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
the strict typed author contract below replaces the opaque JSON string for new jobs.
No new paid call or budget increase is implied by that local fix. The website/API
transport is live-tested; a successful hands-off avatar delivery is still unproven.

## Typed author transport v2

New OpenAI jobs now request a strict nested design object, not an escaped
`design_json` string. The bounded schema covers all 26 existing author fields;
unused optional values use declared nulls, while a root bone's null parent is
retained. Bone transform/weight-region lists convert deterministically to the
existing maps, with duplicate rejection and no invented geometry or motion.
Legacy CLI answers remain readable. Existing rig, action, quality and native
validators still decide whether a design is usable.

Local tests round-trip existing furry-panda, mechanical-robot and dragon designs,
reject malformed nesting/unknown fields/duplicate transforms/oversized values,
and preserve the actual failed answer unchanged. These are transport fixtures,
not newly generated avatars or a live-provider test ofv2. The official
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
documents the supported schema constraints; local validation remains mandatory.

## Attachment policy v2

New furry jobs require source-colored `tuft-v1` cards with
`root_placement: "skin-consistent-v1"`, sharing the native groom direction and
regional flow. `material_fur` remains optional. Existing saved v1 policies retain
their behavior; old outputs are not rewritten. This resolves an attachment
policy conflict, not a fur-density or appearance upgrade. Counts, masks, native
strand checks, deformation limits and final-media acceptance remain unchanged.
The existing fox all-root audit passes 3,000 roots across 388 poses; this is
component evidence, not acceptance of a newly generated panda.

## Approved typed-contract run

The user subsequently approved up to five calls, including one bounded repair,
within the unchanged $17 aggregate ceiling (`call_ca345c2bab28466a8e7c54f137128603`).
Job `f628641c66e74ea09f3b8517` reuses the exact paid panda and unchanged user
prompt/style/action. Its $16.855920 grant plus $0.141219 prior settled costs totals
$16.997139 maximum allocation. The planner passed. The actual browser submitted one Start and exited after
71.471 s with clean browser/server shutdown. Typed author output parsed; its
first binding named missing hip bones. The one repair added them and reached
native rigging, which failed required leg-region vertex coverage. The run is
terminal unsupported, with no finished rig, dance or critic call. It used three
API calls for $0.221596; aggregate use is $0.362815. Two unused call slots do not
authorize another repair. Follow [TODO](../TODO.md)
for current handles and results; do not start a duplicate.
