# Prompt-driven avatar pipeline

`generate` connects AI planning, geometry generation, agent-authored rigging/motions, local validation, independent agent review and asset export. The user supplies a text prompt; agents write the character-specific finishing configuration. The controller executes only validated data through the existing local tools.

This is a bounded workflow, not a guarantee that every prompt will produce an acceptable avatar. A run ends with a verified delivery, an explicit unsupported result, or a resumable checkpoint. It does not silently pass a broken model.

## Run from a text prompt

From the Weft repository root, prepare a new job without any model or provider call:

```sh
./tools/avatar-assets/run generate \
  --prompt 'A friendly violet baby dragon with gold horns, small wings and a curled tail. Idle, look around, flap its wings while standing, and celebrate.' \
  --states idle,look_around,wing_flap,celebrate \
  --preset preview --max-credits 35 \
  --out outputs/violet-dragon-pipeline-v1 --prepare-only
```

Read `tools/avatar-assets/outputs/violet-dragon-pipeline-v1/estimate.json`. Once this work is within the user's actual spending authorization and the canonical ledger has sufficient allowance:

```sh
./tools/avatar-assets/run generate \
  --resume outputs/violet-dragon-pipeline-v1 --execute
```

The same authorized run can be started in one command by omitting `--prepare-only` and adding `--execute` to the first command. No manual rig JSON or separate platform operation is required between stages. A `--max-credits` flag is a job cap, not permission to raise the ledger or buy credits.

Without `--prepare-only`, the planner may make an AI call even when Meshy execution is not enabled. For a completely local, no-call preparation, use `--prepare-only` on a **new** job. The CLI rejects `--prepare-only` combined with `--resume`.

Prerequisites are the existing authenticated CLI login, this tool's `.venv` dependencies and `.tools/Blender.app/Contents/MacOS/Blender`. The current `generate` command uses that local Blender path; it has no `--blender` override. See [local setup](../README.md#local-setup) for the locked environment. Preparing a request does not create credentials or install these tools.

### Inputs

| Option | Behavior |
| --- | --- |
| `--prompt` | Exact subject/style/action request, up to 4,000 characters; preserved unchanged |
| `--style` | Optional explicit style, up to 800 characters; planner must preserve it |
| `--states` | Up to eight comma-separated safe IDs, in requested order; omit to let the planner choose four meaningful states |
| `--reference` | One selected local PNG/JPEG/WebP, at most 8MiB and 16MP; original bytes and hash retained |
| `--preset` | `draft`, `preview` or `muse`; actual output specifications below |
| `--out` | New folder under `tools/avatar-assets/outputs/`; safe folder name becomes the job ID |
| `--reuse-model` | Explicit existing generated GLB, at most 64MiB and within local mesh bounds; no new Meshy generation |
| `--max-credits` | Meshy job cap, default 0; fresh generation needs at least 35 and `--execute` plus canonical-ledger allowance |
| `--max-repairs` | Local repair attempts after the first design, default 1, maximum 2 |
| `--max-agent-calls` | Persistent AI-call allowance, default 5, range 3–8; allow 7 for a full two-repair sequence |
| `--agent-timeout` | Deadline per agent invocation, default 240s, range 30–600s |
| `--agent-model` | Optional installed CLI-supported model identifier; otherwise uses the isolated CLI default |
| `--resume` | Continues the exact saved request, without replacing inputs or resetting allowances |

For a selected reference, add `--reference /absolute/path/character.webp` when preparing the job. Only that image and the job's derived inspection/review images are supplied to the relevant models; unrelated files are not selected. Reference metadata is provenance, not independent rights verification.

## What runs

1. **Planner agent:** turns the prompt into a subject/style brief, generation prompt, observable appearance requirements and state meanings. Explicit state IDs and style remain fixed.
2. **Geometry:** generates a Meshy7.1 textured model, or copies the explicitly selected existing GLB. The fresh route requests 4K geometry/textures, roughly 35,000 triangles and no provider rig/motion calls. Output limits still apply; a requested triangle count is not a guaranteed vertex count.
3. **Local inspection:** validates bounded GLB data, then renders actual views with native bounds and any existing bone names. Labeled front/side coordinate guides give the author the same normalized bounds used by the rig builder, including tail/horn extents.
4. **Rig-and-motion author agent:** sees those images and proposes subject-specific bones, skin-weight regions, gestures and optional surface finishing. It cannot change identity, source hashes, output limits or spending. New rigs are authored assumptions; existing rigs are not silently replaced.
5. **Trusted local build and independent checks:** validates the proposal, creates the editable Blender source and animated GLB, checks actual weighted motion/attributes, independently reloads the exported model, and renders the exported appearance. Native strands and browser cards are evaluated as different representations.
6. **Independent critic agent:** reviews the actual evidence against the original brief, including each state's visible meaning. Contact sheets retain up to eight informative authored key poses, with timestamps, to avoid missing short gestures between fixed snapshots. Acceptance requires identity, appearance, framing, deformation and every state to pass with supplied evidence. Numerical movement alone does not establish a meaningful action.
7. **Bounded repair:** rejected designs and findings remain on disk. The author receives actual failed pose images, per-state motion measurements, weight coverage and bounded stretched-edge coordinates/influences. These measurements diagnose defects; they do not replace visual acceptance. A new attempt cannot trigger a paid regeneration or relabel an impossible action. Exhausted repair/call limits do not become successful delivery.
8. **Export and package:** renders full motion clips, portrait and static fallbacks; creates a byte-preserving compact media pack and job-local gallery catalog; verifies hashes and exact model/media parent binding.

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
- `finish-N.json`, `review-N.json`, `evidence-N/`: agent-authored design, hash-bound observations and sampled images; rejected attempts remain alongside the selected one.
- `reload-N/`, `exported-preview-N/`: independent exported-model proofs.
- `media/` or `media-continuation/`, plus `media-portable/`: full native frames, portrait, clips, contact sheets, static fallbacks and compact distribution copies.
- `gallery.json`: a read-only browser collection for this job; it does not replace the default gallery.
- `request.json`, `estimate.json`, `plan.json`, `inputs/`, `agents/`, `events/`, `status.json`: resumable inputs, costs, agent evidence and lifecycle history.

After completion:

```sh
./tools/avatar-assets/run gallery \
  --catalog outputs/violet-dragon-pipeline-v1/gallery.json
```

Open `http://127.0.0.1:8765/` yourself. No Blender window or provider call is needed to view the completed files. The pipeline validates the gallery catalog; that check is not a browser rendering or Weft runtime claim. Do not start a second persistent server on an occupied port.

## Cost and resource bounds

Fresh geometry costs an estimated **35 Meshy credits**: one image-conditioned request, or text geometry 25 plus texture 10. New Meshy calls require all three conditions: `--execute`, a saved cap of at least 35, and remaining canonical-ledger authorization. The pipeline never resets the ledger, spends an account balance as implied permission, borrows Weft's Meta budget, or buys credits. Approved grants are job-specific; a previous dragon allowance does not fund another job. Higher actual provider charges remain recorded and stop work for reconciliation.

The expected agent sequence is three calls: planner, author and critic. Each complete repair adds up to two calls. Account-specific AI usage pricing and dollar purchase cost per Meshy credit are unknown, not zero. `estimate.json` records the call envelope and token assumptions; call receipts retain reported usage. No recurring service is started.

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

`unsupported.json` states the failed stage and reason. Possible outcomes include geometry outside the local profile, an unrepairable visual mismatch, unsupported task interaction, or no accepted result within the repair budget. Failed candidates and costs remain preserved. A `checkpointed` or awaiting-authorization result is unfinished, not an avatar.

The current backend authors bounded skeletal rotation gestures. It does not provide arbitrary prop interaction, facial blendshapes, lip sync, recovered fingers, physics, cloth/hair simulation, flight dynamics or exact 3D recovery from reference media. For fur/hair, editable native strands and portable cards must both be considered; a texture alone is not a complete groom, and cards are not exported native strands. Agent review is sampled assessment, not human approval, anatomical truth or guaranteed Muse-level quality.

This pipeline does not create a Weft mapping or change Weft's importer, actions, rewards or physics. High-detail generated models can exceed its frozen 20k-vertex/diffuse-only profile; an optimized mapped derivative remains a separate explicit deliverable.

## Verification status

Actual planner/author/critic runs completed on the dragon and robot, including automatic repairs, native reload, real media export, browser playback/fallback checks and completed-job resume without generation. They reuse preserved generated geometry, so this verification used **zero new Meshy credits**, not zero AI usage. Fresh Meshy generation remains the previously tested connected adapter; this turn did not rerun the paid synthesis branch. [Exact results, commands, costs and limits](prompt-pipeline-results-v1.md).

The full local suite passed323 tests in12.545s. Analytic fixtures establish code/input guards, not visual quality. The dragon's full preview took757.04s; the robot's smaller draft took195.42s on this host. User acceptance and Muse-quality equivalence remain unverified. The earlier failed jobs retain explicit unsupported/error results; none was relabeled as success.

The lower-level [Meshy workflow](meshy-workflow.md), [procedural scene route](scene-contract.md) and [reusable skill](../skills/avatar-asset-workflow/SKILL.md) remain available for focused authoring, inspection and repairs.
