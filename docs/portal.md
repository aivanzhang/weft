# Weft web portal

Upload a video, follow its reconstruction, open the interactive 3D sandbox, and
save your actions. OpenAI scene analysis and Muse physics review use **server-side project API keys**, never a
ChatGPT login, browser session, coding-agent CLI, or subscription token.

## Open it locally

From the repository root:

```sh
.venv-lab/bin/python -m weft portal
```

Open the complete `http://127.0.0.1:8765/#token=…` link printed in your terminal.
The user starts/manages this server. No browser opens automatically. Use
`--port 8766` if that port is occupied.

1. Choose a video, up to 60 seconds / 1,800 frames / 100 MB. Each builder also
   applies its own reconstruction and package limits.
2. Leave **Analyze with OpenAI** on and choose **Let AI identify the scene**.
   Confirm permission to process the video and send its sampled frames.
3. Watch **Understand → Build & check → Learn**. The original, sampled frames,
   scene proposal, API receipt, failure reasons and processing log are inspectable.
4. Select **Open interactive sandbox**. Inspect the source, tracks and geometry;
   rotate the view; choose manual, authored reference or learned controls where
   available; then **Record & start → Stop & save**.
5. Select **Send evidence to Muse for review** in the project. Inspect alternative
   physical explanations, cited observations, suggested tests and unresolved questions.
   This sends construction evidence and up to three thumbnails to Meta. Review
   runs in the background; the sandbox remains usable. Proposals are not applied.
6. Select **Run local comparisons · no API charge**. Exercise projects compare
   authored control, disabled root assistance and no control under the same task.
   Object projects compare opposite force interventions and no control. Inspect
   success/reward and download every action/state. These are local diagnostics,
   not execution of Muse's requested real-world measurements or automatic repairs.
7. For an exercise project, expand **Create a separate support experiment** to
   add an explicitly authored static box and load requirement. Enter position and
   size in the sandbox's body-scaled world units, not metres; select a body
   segment and half-open step interval. Optional training compares four controllers
   on three changed starts. This creates a separate project without new pose
   inference or AI calls; the original project remains unchanged.

Support experiments are advanced, manual assumptions—not automatic equipment
reconstruction or a measured correction. Their new task cannot establish an
improvement against the parent's different task. New Muse reviews include the
saved world's geometry, authored supports, load requirements and control assumptions,
separately from video observations. Older body-only reviews retain their narrower
scope. Custom avatars
are not yet qualified for this support-world profile. See the
[support guide](static-support.md) and [verified example](handoffs/portal-support-experiments.md).

Projects persist under `.local/portal/`. Closing the server preserves completed
runs and recordings. Interrupted work is not automatically resubmitted or billed
again. Existing projects reopen when the portal restarts. Saved runs stay separate
from new recordings.

The workspace includes a ready **single-leg-squat.webm** exercise project and its
verified recording, plus a clearly labeled **synthetic-rigid-objects.mp4** 3D
object project. Earlier rejected attempts remain visible with their reasons.
An additional authored-support experiment is ready: its deliberately distant box
cannot supply the requested forearm load, so its task fails even though the lab
can be played and recorded. Training retained the authored controller; it did
not discover a successful support policy.

## API configuration and spending

The key already configured for this workspace is read from
`.local/credentials/openai-api-key` (owner-only permissions), or `OPENAI_API_KEY`
in the server environment. Never put it in frontend code or a browser form.

The approved portal development allowance is **$5 total**, recorded in
`.local/portal-openai/ledger.sqlite3`. It is separate from the existing Meta
allowance and does not reset when a new project directory is chosen. To explicitly
configure an owner-approved total on another installation:

```sh
.venv-lab/bin/python -m weft openai-budget --total-usd 5
```

The planner uses `gpt-5.4-mini-2026-03-17` through the Responses API. Up to six
JPEG frames are sent; the full video and audio are not uploaded. A request reserves
**$0.3135** using the model's full context ceiling plus the bounded output. Actual
usage settles at the recorded price; uncertain requests keep their reservation.
There are no automatic API retries or model-generated commands. Identical source
and request identities can reuse an already settled analysis at no new API cost.
This is an application allowance, not a provider-account hard cap.

### Where Muse fits

Muse Spark is the physical-scene proposer/critic, not the avatar feature. The
portal now exposes the existing [Meta advisor](meta-advisor.md) on completed
exercise and object projects. It receives construction observations and authored
body assumptions, never evaluation scores or policies. New exercise reviews also
receive the [exact current world assumptions](handoffs/portal-world-review.md),
including static supports where present; these are not measured contact or anatomy.
Evidence IDs, alternative
explanations, required tests, the exact request/response and usage receipt are saved
under each project's `muse/` directory, outside immutable runs.

The existing Meta project key is already configured in this workspace. It uses
`META_API_KEY` / `MODEL_API_KEY` or `.local/credentials/meta-api-key`, and the
canonical `.local/meta-budget/ledger.json`. The separate **$5 total Meta budget**
is shared with other Weft Meta adapters. One review reserves an estimated
**$0.058368**, with no automatic retry. Repeated clicks return the existing review;
restart or cancellation preserves uncertain charges and never relabels the
completed simulation. The reviewed price was unchanged on November 2, 2026.

OpenAI currently provides initial scene planning; Muse provides inspectable
physical reasoning. An OpenAI-only run does not demonstrate Meta use. The full
world-repair loop remains unfinished. **Test Muse suggestions** now runs up to
three locally authored controller candidates tied to supported Muse hypotheses,
compares them against the unchanged task and records every acceptance/rejection.
It reuses the saved review, costs no new API call and never applies a controller.
Geometry/contact repairs and requested physical measurements remain unimplemented;
this does not establish hackathon readiness or physical correctness. Avatar
generation is separate; selecting a skin must not alter physics or task behavior.

## What the result means

- **Exercise:** the existing full-body shared-world workflow, assumed ground and
  explicit virtual root assistance, bounded training and separate changed-start
  evaluation. Muscles are illustrative; anatomy and contact are not measured.
- **Objects:** source-pixel color localization when qualified, otherwise bounded
  AI-region hypotheses subject to appearance/reconstruction gates. The 3D world
  retains front-plane evidence; depth, materials and parking controls are authored.
- **Unsupported input:** preserves its evidence and error instead of presenting
  an unchecked environment as ready. Moving cameras, missing joints, ambiguous
  identities and complex equipment contact remain important limitations.

This is a working **local, owner-only portal**, not a publicly deployed multi-user
service. The hosted website remains offline. Public accounts, tenant isolation,
remote job execution and production deployment are still required before sharing
it as a hosted product. See the [implementation handoff](handoffs/portal-api.md).

### Saved simulation compatibility

Existing exercise projects retain their original simulation and scoring versions.
If the current runtime differs, the portal can use the verified released runtime
`5819ba4` or `64f4bef` in an isolated local worker, without changing recordings or scores.
Unsupported versions still show an explicit error; saved project code is never
executed. [Automatic native timing](handoffs/portal-native-and-repair.md) now captures every
decoded exercise frame before reconstruction while retaining original detector
timestamps. Failed projects show pose coverage, uncertain intervals and a diagnostic
download when adequate evidence exists; they are not silently retried.
