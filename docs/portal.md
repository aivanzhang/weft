# Weft web portal

For the shortest tested exercise-and-avatar walkthrough, see [the local demo guide](hackathon-demo.md).

Upload a video, follow its reconstruction, open the interactive 3D sandbox, and
save your actions. OpenAI scene analysis and Muse physics review use **server-side project API keys**, never a
ChatGPT login, browser session, coding-agent CLI, or subscription token.

Failed local body extraction can offer a [separate detailed-body attempt](portal-body-detail.md), using the local Full pose model without an API charge. The original failure stays intact; later OpenAI and Meta processing still requires its own consent.

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
   For exercise, declare whether equipment/obstacles are present and choose an
   output. **Body evidence preview — no physics** is the default: it shows
   annotated source poses and downloadable body/joint models, not an interactive
   simulation. **Ground only** plus **Interactive scene simulation** enables the
   person-only sandbox; this is your scope declaration, not verified absence.
   If this exact video has registered prepared equipment evidence, **Build
   candidates** constructs the supported alternatives in the background.
   Otherwise, **Check video locally** starts new source preparation. Passing
   local checks offers separate consent for OpenAI images and Meta video tracking;
   passing those stages still does not guarantee a playable world.
   The downloadable **Person tracking region and excluded landmarks** explains
   the initial tracking hint. Hidden or low-confidence landmarks remain unknown;
   this region does not establish a complete silhouette or physical identity.
3. Watch the processing stages. The original, sampled frames,
   scene proposal, API receipt, failure reasons and processing log are inspectable.
4. For a completed simulation, select **Open interactive sandbox**. Inspect the source, tracks and geometry;
   rotate the view; choose manual, authored reference or learned controls where
   available; then **Record & start → Stop & save**.
5. Select **Send evidence to Muse for review** in the project. Inspect alternative
   physical explanations, cited observations, suggested tests and unresolved questions.
   This sends construction evidence and up to three thumbnails to Meta. Review
   runs in the background; the sandbox remains usable. Proposals are not applied.
   Selected worlds built from new perception evidence reuse their exact original
   body evidence. Muse revalidates that body and the selected world before any
   request; conditional object identities remain unresolved. Their local checks
   and repairs remain unavailable.
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

**Equipment candidate.webm** is also ready in this workspace: a source-bound
person and two equipment proxies share one world. Open it to inspect all three
actors, interact and record. Its original equipment-stability test still fails;
this is a playable candidate, not successful physical reconstruction. Six newly
recorded actions replay exactly. Its imported run and original task are unchanged.

### Import an existing verified exercise lab

With the portal stopped, copy a sealed lab into its project list:

```sh
.venv-lab/bin/python -m weft portal-import \
  --lab .local/working-equipment-cell-admission-01/lab \
  --name 'Equipment candidate.webm'
.venv-lab/bin/python -m weft portal
```

The example is already imported here; repeating the command returns that project.
Import validates the complete copied lab through the ordinary loader, preserves
original evidence/task/policy bytes and starts separate recordings. It does not
copy old recording sessions, infer poses, train or call an AI provider. A running
portal's exclusive lock prevents import from interfering with active work.
Unsupported runtime/source packages fail explicitly and retain private partial
evidence. Imported labs require an explicit exact original body package before
Muse review becomes available. With the portal stopped:

```sh
.venv-lab/bin/python -m weft portal-attach-evidence \
  --project-id 47dcca08e7a6461295993941fb9162bc \
  --body-source artifacts/pose-source-recovery-acceptance-01/kettlebell-pushup/body
```

The attachment is now present for this example. It preserves the original run,
checks the body/source/world correspondence and copies the sealed construction
package beside the run. Review revalidates it before sending anything to Meta.
The portal exposes the attachment receipt and exact request/context. Imported
local comparisons and repairs remain unavailable; an evidence attachment does
not qualify a repair. Ordinary video projects retain their existing Muse workflow.

### Propose equipment roles from saved evidence

For development, `shared-exercise-auto-build --run BASE_WORLD --evidence-run
OBJECT_EVIDENCE --out NEW_DIRECTORY` derives conditional body/equipment roles
and invokes the ordinary strict world/lab builder. Add `--source-mapping` for
each verified derivative mapping. It requires an existing source-qualified body
world and complete object observations, not just a video. Ambiguous identities
stop assembly with a saved explanation and exit code 1. Geometry, depth, mass,
floor support and task intent remain explicitly authored assumptions. This is
not automatic mixed-video upload support.

For ambiguous identities, `shared-exercise-alternatives` uses the same arguments
plus `--max-candidates 3`. It preserves the original ambiguity, constructs each
bounded assignment separately and retains every rejection. It never chooses a
winner. Different assignments may imply different tasks, so their scores cannot
be ranked as if they were the same task.

### Choose prepared equipment candidates in the portal

For an exact registered source, upload with **Equipment or obstacles** and
**Interactive scene simulation**, then choose **Build candidates**. Inspect the
assumed body/equipment identities and reasons before selecting a ready candidate.
Selection locks that project to one world so later recordings cannot switch
silently. Rejected candidates cannot open a sandbox. These new selected worlds
do not yet support Muse review, local comparisons or repairs; the existing
imported example's verified Muse attachment remains available.

**Equipment alternatives — saved rejections.webm** is already in the local
project list. It displays three failed alternatives from an existing real-video
study; inspection revalidated the source receipts without repeating construction.
It is not playable. **Equipment candidate.webm** remains the separate playable
example, with explicit assumptions and an unsuccessful full-task stability result.

An administrator can register prepared inputs while the portal is stopped:

```sh
.venv-lab/bin/python -m weft portal-equipment-register \
  --source-sha256 ORIGINAL_VIDEO_SHA256 \
  --run BASE_WORLD --evidence-run OBJECT_EVIDENCE \
  --source-mapping VERIFIED_MAPPING --label "Prepared exercise evidence"
```

Omit `--source-mapping` when both packages use the exact original video. The
catalog is bounded and requires complete source/body/mapping validation. Users
never need to enter these paths in the web interface. Registration and world
construction make no AI call. Unknown videos still need new evidence; matching
saved inputs does not establish arbitrary-video reconstruction.

[Current verification and remaining work](handoffs/conditional-equipment-and-constraints.md).

### Prepare an unfamiliar exercise video

If a failed project already has a separately verified sealed body, stop the
portal and reuse it without repeating pose inference:

```sh
.venv-lab/bin/python -m weft portal-reuse-body --root .local/portal \
  --parent-project PROJECT_ID --body-source PATH_TO_BODY \
  --body-manifest-sha256 EXACT_MANIFEST_SHA256
```

This creates a new child and preserves the failed parent. Start the portal again
to run its local checks. It does not make paid calls or guarantee a valid world.
Keep the returned child ID; it appears as a separate project.
This also accepts a completed equipment refusal with no selectable world. It
preserves that refusal; active, partial or already selected projects cannot use
this path to restart or replace their work.

Choose **Exercise**, **Equipment/obstacles** or **Unsure**, and **Interactive scene
simulation**. In the unresolved project, select **Check video locally**. Inspect
the estimated body, pose evidence, clock and preview before paying for new media
analysis. If local checks pass, inspect the separate provider reservations and
explicitly choose both image and video permissions, then **Prepare with the displayed AI
allowances**. The existing candidate cards handle selection and sandbox access.

OpenAI receives sampled images; Meta tracking receives the whole original video
and inferred regions. This tracking call is separate from the later **Muse review**
of a completed world. Old upload consent does not authorize a new Meta upload.
Duplicates, cancellation and restarts do not automatically repeat paid requests.

New equipment analysis can choose a clearer supplied image for each object,
rather than requiring equipment to be visible in the first image. The selected
original frame and image are retained. Tracking still receives the complete
video; earlier returned or missing slots keep their original labels. Existing
first-image requests remain readable and are not silently changed or retried.
This improves initialization options, not a guarantee of correct tracking or
reconstruction. [Actual result and current limits](handoffs/sampled-equipment-initialization.md).

The current path requires an MP4 ≤37 MB/500 frames/60 seconds and a usable initial
person pose, with at most three equipment objects. Missing/uncertain regions,
camera movement, source-quality failures and rejected geometry remain visible.
No person-only simulation is substituted for a failed mixed scene. If the pose
detector returns competing body estimates, the failed project shows their
timestamps, coverage and downloadable alternatives. This does not prove multiple
people were present. Screened time windows are diagnostic only, not playable
shortcuts, and no retry or paid call happens automatically.
[Verified scope and remaining reconstruction work](handoffs/dense-pose-and-contact-targets.md).

### Recover a completed perception export

If provider processing completed and the bridge was sealed, but publication
failed afterward, stop the portal and queue the saved evidence explicitly:

```sh
.venv-lab/bin/python -m weft portal-recover-perception --root .local/portal \
  --project-id PROJECT_ID --bridge-manifest-sha256 EXACT_BRIDGE_MANIFEST_SHA256
```

The manifest is `PROJECT_ID/perception/bridge/manifest.json` under the portal
root. Recovery verifies the complete saved source proof and queues ordinary
equipment construction in the **same project**. Restart the portal to process
that queue. The original perception failure and provider receipts remain
unchanged; recovery has a separate receipt and candidate status. It does not
repeat OpenAI, Meta, pose inference or consent. Missing or changed evidence,
existing candidate work and exhausted storage fail explicitly. A completed
export still does not guarantee an admitted world; candidate selection and
physical validation remain separate.

The project shows **Previous preparation failure retained** beside the current
equipment result. **Saved evidence queued for reuse** changes to **Saved evidence
reused** after recovery starts; neither sends media again. The original error
and **Saved-evidence recovery receipt** remain downloadable. An **abstained**
result with zero playable candidates means the remaining identity or geometry
uncertainty prevented construction—not that the recovery called the providers
again. No candidate or sandbox is substituted for that refusal.

### Prepare new equipment evidence offline

The developer command `exercise-perceive` binds a saved OpenAI analysis to the
exact original MP4 and prepares the existing tracking request. It verifies every
sampled image/time and adds a tracking-only person anchor from a sealed body.
It is offline by default; explicit execution uses the shared Meta budget.
The portal now uses this same bridge after local preflight and separate consent.
Evidence preparation does not guarantee world admission. See the
[command and supported limits](handoffs/conditional-equipment-and-constraints.md#source-pinned-perception-bridge).

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

- **Exercise simulation:** after an explicit ground-only declaration, the existing full-body shared-world workflow, assumed ground and
  explicit virtual root assistance, bounded training and separate changed-start
  evaluation. Muscles are illustrative; anatomy and contact are not measured.
- **Exercise preview:** annotated pose estimates and body/model files only; no
  dynamics, training, equipment reconstruction or interactive 3D renderer.
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
`5819ba4`, `64f4bef` or `b0917e7` in an isolated local worker, without changing recordings or scores.
Unsupported versions still show an explicit error; saved project code is never
executed. [Automatic native timing](handoffs/portal-native-and-repair.md) now captures every
decoded exercise frame before reconstruction while retaining original detector
timestamps. Failed projects show pose coverage, uncertain intervals and a diagnostic
download when adequate evidence exists; they are not silently retried.

[Native endpoint handling](handoffs/native-interval-and-association.md) preserves
the full retained reference for supported person-only clips whose last timestamp
falls between control steps. Interactive replay follows recorded times; exported
MP4/GIF files remain labeled constant-rate previews. This does not recover missing
poses, qualify ground contact or extend non-grid support to equipment scenes.
