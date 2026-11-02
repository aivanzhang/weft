# Weft web portal

Upload a video, follow its reconstruction, open the interactive 3D sandbox, and
save your actions. AI analysis uses a **server-side OpenAI API key**, never a
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

Projects persist under `.local/portal/`. Closing the server preserves completed
runs and recordings. Interrupted work is not automatically resubmitted or billed
again. Existing projects reopen when the portal restarts. Saved runs stay separate
from new recordings.

The workspace includes a ready **single-leg-squat.webm** exercise project and its
verified recording, plus a clearly labeled **synthetic-rigid-objects.mp4** 3D
object project. Earlier rejected attempts remain visible with their reasons.

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

Muse is not required for this workflow. Optional avatar asset generation is a
separate integration; selecting a skin must not alter the physics or task.

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
