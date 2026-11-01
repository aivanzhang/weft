# Prompt-driven avatar pipeline

`generate` connects AI planning, geometry generation, agent-authored rigging/motions, local validation, final-media review and asset export. The user supplies a text prompt and optional custom action descriptions; agents write the character-specific finishing configuration. The controller executes only validated data through the existing local tools.

This is a bounded workflow, not a guarantee that every prompt will produce an acceptable avatar. A run ends with a verified delivery, an explicit unsupported result, or a resumable checkpoint. It does not silently pass a broken model.

**Current status:** the revised studio, exact action contracts, material-aware
quality policy and final-media-before-critic path are implemented/in verification.
Fixture checks and an existing-model renderer proof are not a completed fresh
hands-off avatar run. The earlier dragon/robot deliveries below verify their
historical pipeline version. [Active acceptance gates](hands-off-studio.md) and
[latest handoff](../TODO.md) distinguish these milestones.

## Local creation screen

From Weft, start the local server yourself:

```sh
./tools/avatar-assets/run studio --port 8766
```

Open `http://127.0.0.1:8766/`. Enter a prompt/style, optional selected image and
one action per line. Saving only persists the request. **Prepare plan** explicitly
uses the configured AI login; **Start creation** requires the reviewed plan ID
and existing canonical-ledger allowance. The screen cannot raise a credit limit
or buy credits. Status comes from real jobs, with measured elapsed time and
stage progress only when available; there is no simulated overall percentage.

The studio serves same-origin opaque preview/download URLs, not arbitrary file
paths. Reference upload accepts one PNG/JPEG/WebP up to 8 MiB/16 MP. Durable jobs
retain inputs, stages, reviews, failed attempts and uncertain ownership across
page/server restarts; they do not silently replay uncertain generation. **Request
stop** is cooperative at safe boundaries, not instant cancellation or a refund.
The user controls this persistent server; no server/browser starts automatically.
[HTTP and UI contract](hands-off-studio.md), [durable job contract](studio-jobs.md).

## Run from a text prompt

From the Weft repository root, prepare a new job without any model or provider call:

```sh
./tools/avatar-assets/run generate \
  --prompt 'A friendly violet baby dragon with gold horns, small wings and a curled tail' \
  --action 'Stand at rest with a gentle breathing motion.' \
  --action 'Look left and right, then return to the starting pose.' \
  --action 'Flap both wings while standing still.' \
  --action 'Do a small rhythmic bounce and a happy tail sway.' \
  --preset preview --max-credits 35 \
  --out outputs/violet-dragon-pipeline-v1 --prepare-only
```

Read `tools/avatar-assets/outputs/violet-dragon-pipeline-v1/estimate.json`.
To run only the bounded AI planner, stopping before model generation or native
rendering even when reusing a mesh:

```sh
./tools/avatar-assets/run generate \
  --resume outputs/violet-dragon-pipeline-v1 --plan-only
```

That step uses the existing AI account and is not a free/no-call preparation.
Once the actual work is within user authorization and the canonical ledger has
sufficient allowance:

```sh
./tools/avatar-assets/run generate \
  --resume outputs/violet-dragon-pipeline-v1 --execute
```

The same authorized run can be started in one command by omitting `--prepare-only` and adding `--execute` to the first command. No manual rig JSON or separate platform operation is required between stages. A `--max-credits` flag is a job cap, not permission to raise the ledger or buy credits.

Without `--prepare-only`, the planner may make an AI call even when Meshy execution is not enabled. For a completely local, no-call preparation, use `--prepare-only` on a **new** job. The CLI rejects `--prepare-only` combined with `--resume`. Use `--plan-only` to stop after planning; a normal reused-model run without it can proceed into authoring/rendering despite requiring no Meshy call.

Prerequisites are the existing authenticated CLI login, this tool's `.venv` dependencies and `.tools/Blender.app/Contents/MacOS/Blender`. The current `generate` command uses that local Blender path; it has no `--blender` override. See [local setup](../README.md#local-setup) for the locked environment. Preparing a request does not create credentials or install these tools.

### Inputs

| Option | Behavior |
| --- | --- |
| `--prompt` | Exact subject/style/action request, up to 4,000 characters; preserved unchanged |
| `--style` | Optional explicit style, up to 800 characters; planner must preserve it |
| `--states` | Up to eight comma-separated safe IDs, in requested order; omit to let the planner choose four meaningful states |
| `--action` | Repeat up to eight times; each exact description is 1–600 characters. Planner/repair must preserve order and intent, not reduce it to a weaker label |
| `--reference` | One selected local PNG/JPEG/WebP, at most 8MiB and 16MP; original bytes and hash retained |
| `--preset` | `draft`, `preview` or `muse`; actual output specifications below |
| `--out` | New folder under `tools/avatar-assets/outputs/`; safe folder name becomes the job ID |
| `--reuse-model` | Explicit existing generated GLB, at most 64MiB and within local mesh bounds; no new Meshy generation |
| `--max-credits` | Meshy job cap, default 0; fresh generation needs at least 35 and `--execute` plus canonical-ledger allowance |
| `--max-repairs` | Local repair attempts after the first design, default 1, maximum 2 |
| `--max-agent-calls` | Persistent AI-call allowance, default 10, range 3–12; expected first attempt is three calls, each complete repair adds two |
| `--agent-timeout` | Deadline per agent invocation, default 240s, range 30–600s |
| `--agent-model` | Optional installed CLI-supported model identifier; otherwise uses the isolated CLI default |
| `--resume` | Continues the exact saved request, without replacing inputs or resetting allowances |
| `--plan-only` | Uses AI planning, then stops before generation/native work; valid with a new request or `--resume` |

If both actions and state IDs are supplied, their counts must match and IDs stay
in that order. Omit both for the four-state default. CLI action strings are
preserved exactly; the browser trims blank lines/outer whitespace when creating
its action list. Structured roles/requirements and actual motion evidence must
cover the request. A takeoff request cannot pass as a stationary wing wiggle;
root travel cannot substitute for articulation. Unsupported facial performance,
object interaction or fused-part separation must remain explicit unsupported
scope, not an easier replacement action.

For a selected reference, add `--reference /absolute/path/character.webp` when preparing the job. Only that image and the job's derived inspection/review images are supplied to the relevant models; unrelated files are not selected. Reference metadata is provenance, not independent rights verification.

## What runs

1. **Planner agent:** turns the prompt into a subject/style brief, generation prompt, material-specific surface intent and state meanings/roles/requirements. Explicit state IDs, custom action strings/order and style remain fixed.
2. **Geometry:** generates a Meshy7.1 textured model, or copies the explicitly selected existing GLB. The fresh route requests 4K geometry/textures, roughly 35,000 triangles and no provider rig/motion calls. Output limits still apply; a requested triangle count is not a guaranteed vertex count.
3. **Local inspection:** validates bounded GLB data, then renders actual views with native bounds and any existing bone names. Labeled front/side coordinate guides give the author the same normalized bounds used by the rig builder, including tail/horn extents.
4. **Rig-and-motion author agent:** sees actual geometry and proposes subject-specific bones, skin regions, role bindings, gestures and finishing. The quality policy requires native groom, regional flow, source-colored portable cards and measured non-fur masks when applicable; it must not put fur on a metal object by default. It cannot change identity, source hashes, limits or spending. New rigs and unseen surfaces are authored assumptions.
5. **Trusted local build and independent checks:** creates editable Blender source/animated GLB, validates attributes/skin, independently reloads the export, and audits motion of the actual bound vertices. Native and exported front/side/back/three-quarter views and bounded informative state poses use real 720px renders. Native strands and portable cards are separate representations; a passing structural policy is not appearance acceptance.
6. **Final media before review:** renders the selected-preset portrait, complete clips and static fallbacks for this attempt. Hash-bound evidence includes actual native/exported views, timed action poses and decoded samples of the final encoded clips. Small draft media stays labeled small; padding or a contact-sheet canvas does not add resolution. No delivered clip is accepted solely from an earlier still preview.
7. **Independent critic agent:** reviews the actual supplied evidence against identity, materials, framing, deformation and every action requirement. Selected poses retain timestamps; the trusted motion audit complements them but cannot decide semantic or visual success. This is sampled review, not every-frame human inspection. Visible defects can reject a structurally valid, moving model.
8. **Bounded repair and delivery:** failed designs, final media and findings stay on disk. The author receives the failed images, role-motion measurements, weight coverage and stretched-edge coordinates/influences. A new attempt cannot trigger paid regeneration or silently weaken a request. Only a passing attempt gets a sealed delivery and job-local catalog, with verified compact-media/source bindings. Exhausted repair/call limits produce no success claim.

The agent runner uses the existing authenticated local Codex CLI in an isolated temporary working directory, with tools disabled and structured JSON output. Returned tool activity, malformed data, failed/uncertain calls and changed cached inputs are rejected. Agents do not directly run Blender, modify source code, read credentials or authorize Meshy spend. The controller preserves exact prompts, schemas, selected image copies, invocation settings, usage receipts and outputs.

The AI calls use remote models through the existing login. “Local workflow” describes native build/storage and orchestration, not on-device neural generation. No new account, key, subscription or hosted service is created.

## Outputs and browser review

| Preset | Portrait | Clips | Sampling |
| --- | --- | --- | --- |
| `draft` | 512×512 | 256×256, 24fps, 2s | 4 native render samples |
| `preview` | 1600×1600 | 720×720, 24fps, 4s | 8 native render samples |
| `muse` | 1600×1600 | 720×720, 24fps, 8s | 8 native render samples |

Agent-authored source actions span four seconds; exports retime the complete curves to the selected duration. The `muse` preset specifies dimensions/timing, **not Muse-quality acceptance**. Portraits and silent H.264 videos are opaque; reduced-motion fallbacks are actual static frames. Matching endpoints do not certify visually seamless or velocity-continuous loops.

A completed job contains:

- `delivery.json` and `delivery-receipt.json`: selected attempt, exact model/source/gallery paths, hashes, costs, limits and review outcome.
- `model-N/model.glb`, `model-N/source.blend`: real skeleton/weights/actions and editable source, with manifests and trusted backend snapshots.
- `finish-N.json`, `review-N.json`, `evidence-N/`, `final-evidence-N/`: authored design, diagnostic/HD evidence and bound review; rejected attempts remain alongside the selected one.
- `reload-N/`, `exported-preview-N/`, `action-audit-N/`: independent exported-model and actual weighted-motion evidence.
- `media-N/` or `media-continuation-N/`, plus `media-portable-N/`: each attempt's full native frames, portrait, clips, contact sheets, static fallbacks and compact copies, rendered before its critic.
- `gallery.json`: a read-only browser collection for this job; it does not replace the default gallery.
- `request.json`, `estimate.json`, `plan.json`, `inputs/`, `agents/`, `events/`, `status.json`: resumable inputs, costs, agent evidence and lifecycle history.
- `quality-policy.json`, `sources/`: measured-input finishing/review policy and pinned backend source. Current pipeline snapshots include `pipeline.py`, `pipeline_prompts.py`, `agent_runner.py`, `action_intent.py`, `quality_policy.py`, `pipeline_evidence.py` and `action_audit.py`.

These are the revised layout, not a rename of old delivered files. Historical
dragon/robot jobs retain `media/`, optional `media-continuation/` and
`media-portable/`. Read the selected paths in that job's `delivery.json`;
never replace an old manifest or infer the chosen attempt from the largest N.

After completion:

```sh
./tools/avatar-assets/run gallery \
  --catalog outputs/violet-dragon-pipeline-v1/gallery.json
```

Open `http://127.0.0.1:8765/` yourself. No Blender window or provider call is needed to view the completed files. The pipeline validates the gallery catalog; that check is not a browser rendering or Weft runtime claim. Do not start a second persistent server on an occupied port.

## Cost and resource bounds

Fresh geometry costs an estimated **35 Meshy credits**: one image-conditioned request, or text geometry 25 plus texture 10. New Meshy calls require all three conditions: `--execute`, a saved cap of at least 35, and remaining canonical-ledger authorization. The pipeline never resets the ledger, spends an account balance as implied permission, borrows Weft's Meta budget, or buys credits. Approved grants are job-specific; a previous dragon allowance does not fund another job. Higher actual provider charges remain recorded and stop work for reconciliation.

The expected agent sequence is three calls: planner, author and critic. Each complete repair adds up to two calls; default call cap 10, maximum 12, default one repair/maximum two. Full-resolution media is now produced before each critic, so rejected attempts can consume additional bounded local render time/disk without a new Meshy submission. Account-specific AI usage pricing and dollar purchase cost per Meshy credit are unknown, not zero. `estimate.json` records the call envelope and token assumptions; call receipts retain reported usage. No recurring hosted service is started.

The last verified project ledger had 202/210 credits used/reserved and eight
remaining. That is insufficient for one new 35-credit job. A new CLI cap or
studio request does not grant the difference. Prepare the aggregate work estimate
before asking for additional allowance; reuse-development checks do not prove
fresh prompt-to-mesh generation.

- Whole job: at most 2GiB output, at most 128 lifecycle events. No new stage starts after one hour in an invocation; an in-flight stage retains its own deadline. Resume starts a new watch window, not a new call/repair allowance.
- Agent call: at most 128KiB prompt, 64KiB schema, eight images/32MiB total, 8MiB per image, 1MiB answer, 4MiB per output stream and 48MiB retained output. Sampled direct-child RSS is limited to 1GiB; cleanup is separately bounded.
- Native stages: four CPU threads and 2GiB sampled direct-child RSS. Inspection is bounded to 300s, finishing 600s, reload 180s and full media 1,200s. Only a hash-bound wall-time media failure can receive the existing single 600s frame continuation; memory/output failures are not automatically replayed.
- Existing GLB limits remain enforced: bounded embedded resources, explicit supported attributes, at most 100k vertices/100k triangles in this authoring profile, no arbitrary executable or remote/data-URL content.

These are monitored bounds, not an instantaneous OS-enforced whole-process-tree memory guarantee. Existing native macOS Blender may need permission outside the sandbox for Metal initialization even when rendering on CPU; the pipeline does not bypass that permission.

## Resume, failures and limitations

```sh
./tools/avatar-assets/run generate --resume outputs/violet-dragon-pipeline-v1 --execute
```

Resume reuses successful matching agent calls, provider task IDs and verified finished packs. It will not automatically retry failed/uncertain model submissions or overwrite an interrupted native stage. Read `status.json`, the last event, agent receipts and native handle/monitor files before intervening. Unresolved owned processes prevent automatic repair. Saved requests and the pinned pipeline backend are immutable; changed inputs/backend require a new job rather than relabeling old evidence.

Do not run a historical completed job under changed backend files and expect its
source-pin check to be bypassed. View/verify its delivered files instead, or use
its matching preserved implementation when explicitly reproducing that version.
The studio's stop control calls the pipeline's `cancel_requested` boundary check;
there is no `generate --cancel-requested` CLI flag. An in-flight bounded native
or provider operation can finish first, and a sealed completed delivery takes
precedence over a late stop request.

`unsupported.json` states the failed stage and reason. Possible outcomes include geometry outside the local profile, an unrepairable visual mismatch, unsupported task interaction, or no accepted result within the repair budget. Failed candidates and costs remain preserved. A `checkpointed` or awaiting-authorization result is unfinished, not an avatar.

The backend supports bounded skeletal rotations and authored meter translations. Those controls do not establish automatic believable flight/dance for arbitrary geometry. It does not provide arbitrary prop interaction, facial blendshapes, lip sync, recovered fingers, physics, cloth/hair simulation, flight dynamics or exact 3D recovery from reference media. A fused companion needs a separately reviewed geometry derivative; that earlier manual repair is not an automatically solved stage. For fur/hair, editable native strands and portable cards must both be considered; a texture alone is not a complete groom, and cards are not exported native strands. Agent review is sampled assessment, not human approval, anatomical truth or guaranteed Muse-level quality.

This pipeline does not create a Weft mapping or change Weft's importer, actions, rewards or physics. High-detail generated models can exceed its frozen 20k-vertex/diffuse-only profile; an optimized mapped derivative remains a separate explicit deliverable.

## Verification status

**Revised workflow:** the 19-case HTTP tests and eight-check browser fixture
exercise transport/UI guards, not real generation. The browser fixture used
34 intercepted requests, zero external requests/errors and verified owned-browser
exit (`outputs/studio-ui-fixture-v1/studio-ui-test.json`). Real exported-renderer
development proof `outputs/studio-exported-motion-proof-v1` renders the existing
panda at 720px in four views/eight poses: 23.280s, 1,201,995,776 B sampled RSS,
exit 0. Sparse exported fur and visible marks remain; this is functionality,
not new avatar or quality acceptance. A fresh hands-off run, real UI-to-download
acceptance and the revised full integrated suite remain pending.

**Historical checkpoint:** actual planner/author/critic runs completed on the dragon and robot, including automatic repairs, native reload, real media export, browser playback/fallback checks and completed-job resume without generation. They reuse preserved generated geometry, so that verification used **zero new Meshy credits**, not zero AI usage. Fresh Meshy generation remains the previously tested connected adapter; that wrapper test did not rerun the paid synthesis branch. [Exact results, commands, costs and limits](prompt-pipeline-results-v1.md).

That historical full suite passed 323 tests in 12.545s. The dragon's full preview took 757.04s; the robot's smaller draft took 195.42s on this host. These are not latency measurements or a full-suite pass for the revised studio path. User acceptance and Muse-quality equivalence remain unverified. Earlier failed jobs retain explicit unsupported/error results; none was relabeled as success.

The lower-level [Meshy workflow](meshy-workflow.md), [procedural scene route](scene-contract.md) and [reusable skill](../skills/avatar-asset-workflow/SKILL.md) remain available for focused authoring, inspection and repairs.
