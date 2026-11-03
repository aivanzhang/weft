# Avatar assets

Local workflow for editable, rigged 3D avatars and their 2D portrait/animation exports, now maintained in Weft under `tools/avatar-assets/`. Authoring stays separate from Weft physics, actions, rewards and provider budget. See [relocation and preservation](docs/relocation.md).

**Budget:** standing authorization permits further bounded retries within **$15
total OpenAI spend**, including prior use, without repeated approval questions.
The append-only ledger cap is active; no new Meshy work.

**Current direction:** Studio uses the server OpenAI API, not a personal AI login;
Meshy remains the geometry/texture provider. The website now supports authenticated
**Retry saved work → Prepare → Start** under an owner-approved standing budget.
The recent fresh panda base repeatedly failed final appearance review, despite
real rigs, motion and media. Medium reasoning runs correctly, but did not make
that source acceptable. All rejected attempts remain preserved.
A new request uses the **unrigged generated base behind the previously accepted
panda appearance**, plus its accepted portrait reference. No old rig is copied;
the shoulder dragon stays perched in this trial. Its first planner exceeded a
text limit. The corrected website Prepare passed, but the rigging API call then
timed out after 240 seconds before returning a rig. No hands-off delivery is accepted.
Recorded use is **$3.999655 settled**, plus a **$5.618640 uncertain reservation**;
the latter is a conservative hold, not a known bill. The unchanged **$15 total cap**
blocks another full reservation until genuine provider evidence resolves the
unknown charge. The stopped job and earlier accepted assets remain preserved.
No new Meshy call or deployment.
The [handoff](TODO.md) contains exact ownership, source hashes and recovery status.
[Backend results](docs/api-website-backend.md).

**Status:** the core text-to-3D workflow and saved panda/dragon examples work.
The user has accepted the previous appearance; the active priority is now a
successful fully hands-off Studio delivery, not further appearance upgrades.
That real end-to-end delivery remains unverified. The latest [texture-preserving undercoat improvement](docs/undercoat-uv2-preservation.md)
passes component checks; it is not a newly accepted avatar.
The subsequent [actual-source control](docs/source-uv-control.md) builds a rigged
furred GLB and passes independent skin/texture-frame checks, but its eight rendered
views/poses still expose folded shoulder/cheek geometry. It remains unapproved.
The reusable pre-export guard now rejects unsupported normal-texture UV routes;
passing that check is not visual acceptance.
The [current local quality work](docs/quality-next-step.md) also removes an unused
shared-image requirement for non-color-driven finishing. An existing mixed-material
lamp passes the public finisher; this is an integration check, not a new generated
avatar or a Muse-quality result.
The new [separate-source composer](docs/multipart-sources.md) preserves complete
character parts, materials and textures in one unrigged GLB. It is tested on the
saved robot and dragon; automatic multipart generation and animation remain separate work.
The opt-in [material-owned fur profile](docs/material-owned-fur.md) now finishes
an existing three-texture fox without sharing one color image across its coat.
Its native portrait and exported views are verified locally; the coat still
reads as short plush, not dense Muse fur or a fresh accepted Studio delivery.
That first coat failed attachment by up to 2.50 mm. The opt-in
[skin-consistent root placement](docs/skin-consistent-fur.md) fixes this mechanism:
all 3,000 roots pass across 36 sampled poses, with maximum separation below
0.4 micrometres. Body, colors and existing motions are preserved. Appearance
still reads as short plush, with abrupt tail/leg boundaries and source deformation
defects; neither this fox nor the hands-off Studio is newly accepted.

The new [automatic fur-attachment gate](docs/portable-fur-attachment.md) now runs
before expensive media in the prompt pipeline. The actual saved fox passes
1,164,000 root-pose checks in1.3s; analytic detached-fur controls correctly fail.
This verifies attachment to the exported body, not denser fur or an accepted
end-to-end Studio result. No new paid generation was used.

The approved fresh dancing-panda Studio trial generated geometry and textures
(35 Meshy credits), then stopped at the AI account's usage limit before rigging.
The model and failure are saved; no automatic generation retry or purchase ran.
A completed hands-off Studio delivery is still unverified. [Current handoff](TODO.md).

## Create from a prompt — studio acceptance pending

The new local studio accepts a character/style prompt, an optional reference
image and up to eight custom action descriptions. It shows saved plans, actual
job stages, review/repair findings and available previews/downloads. From Weft,
start it yourself, then open `http://127.0.0.1:8766/`:

```sh
./tools/avatar-assets/run studio --port 8766
```

**Save request → Prepare plan → Review estimate → Start creation** are separate
steps. Saving is local and makes no model call. Planning uses the **server OpenAI
API key and a request-bound dollar grant**, never a personal AI login. It makes no
Meshy generation call. Starting requires the exact reviewed plan and sufficient
existing allowance; the screen cannot buy credits or grant more spending. The
Meshy ledger has **8 credits remaining**, below the estimated **35 credits** for
a new textured model. OpenAI dollars and Meshy credits are tracked separately.

The interface and HTTP/job contracts have fixture verification; the revised
hands-off quality pipeline and a complete real studio run are **not yet
accepted**. Existing-model renderer checks do not prove a new avatar meets its
prompt. Work now includes exact custom-action contracts, subject-specific
materials/fur, actual weighted-motion checks, 720px native/exported views and
review of final media before delivery. No universal prompt success or Muse-level
finish is promised. [Active goal and acceptance gates](docs/hands-off-studio.md),
[commands and bounds](docs/prompt-pipeline.md), [current handoff](TODO.md).

Earlier checkpoint **November 1, 2026**: the integrated suite passed **682 tests in
17.120s**, with nine browser fixture checks also preserved. The previous robot
run remains rejected for arm stretching; its reviewed recovery preserved all
480 files and spending history. Evidence corruption now halts instead of buying
another repair call. The new unchanged-text robot run ended **unsupported**:
its repair passed the gross-stretch gate but still moved part of the torso beyond
the requested stability limit. The actual UI verifier recorded that failure and
exited cleanly; no critic, full media or accepted delivery followed. The separate
already-paid-rig trial also ended **unsupported**: its repaired model passed
numeric motion/stability checks and rendered full media, but the critic rejected
the low wrist swivel as an unclear greeting. Its browser/server exited cleanly;
there is no accepted Studio delivery. The finite development estimate allows
**13 total AI calls**, now fully consumed. The separate last-call weight repair
also failed torso stability and visual review; it is not a delivered avatar.
Dollar cost remains unknown and no new Meshy credits are used. The three
fresh-example estimate requests 105 additional Meshy credits and up to 15 AI
calls, but the allowance reply/grants remain pending.
A separate UV1 texture diagnostic improves matte short-pile detail in sampled
views, not dense fur or Muse parity; no coat/default is promoted. Exact receipts,
ownership and remaining quality/UI acceptance are in the [handoff](TODO.md).
Private checkpoint `d259f26` is preserved. It saved the diagnostic
code and compact evidence; its exact commit/push confirmation is recorded in
`.local/avatar-quality-publication-v3/receipt.json`. A local-only rigid-part correction
passed stability measurements but failed the exported deformation/visual check;
it is not promoted. That trial also exposed a severe-stretch reporting gap now
fixed and verified on the same exported poses without new AI/provider calls.

The **opt-in adaptive undercoat** now passes analytic tangent-direction checks,
an actual panda finish and an independent exported-model reload. It preserves
color UVs and uses a separate normal-map UV where measured conflicts require it.
The 122.283-second finish and 22.214-second reload made no new paid calls.
Appearance review still finds smooth portable fur and exposed cheek/shoulder
marks, so this is a technical component pass—not a new accepted fur model,
Muse parity or a changed default. Weft's importer stays unchanged.
[Inputs, bounds and verification](docs/adaptive-undercoat.md).

A later **experimental upright-fringe pass** preserves the real rig/action and
improves sampled ear/tail/paw outlines. Its exported white coat is still too
smooth, and cheek/shoulder defects remain; it is not promoted into the automatic
finish. The build and independent reload passed within 1.279s/495 MB and
15.275s/1.04 GB sampled memory respectively, with no new generation calls.
[Exact experiment, previews and limits](docs/edge-fringe-experiment.md).

The area-weighted follow-up improves fur placement but still looks speckled,
not densely furry. It is also **not accepted**. A separate source-texture guard
now rejects unsupported filtering/UV modes before large allocations; sampler
math and approved assets are unchanged. Verification covers **695 tests across
two runs**: 668 passed initially, and all 27 sandbox-blocked localhost tests
passed with local permission. This verifies software, not hands-off visual quality.

Exported-model review now uses [verified EEVEE culling](docs/exported-preview-renderer.md),
while native hair rendering stays unchanged. A matched source-normal experiment
removes most white-body speckling; the separate color-atlas correction passes its
measured color gate. Reordering unchanged texture patches then removes most
dark-region flecks in matched actual previews without changing geometry or motion.
Cheek/shoulder defects and sparse fur remain; no experimental coat is promoted.
The renderer change passes six focused and 49 integration checks.

The experimental [portable edge-fringe workflow](docs/portable-fringe.md) has
reusable geometry/color libraries and a passing small Blender fixture. Real
panda/fox checks exposed exported normal/weight drift, and the panda exceeded
its memory limit. That option now returns **unsupported before native work**;
existing combed-card finishing and approved assets are unchanged. The final
guard and surrounding contracts pass 59 tests. This is not a new Studio delivery.

[Export fidelity and staged rendering](docs/export-fidelity-and-staging.md) now
prove a float-normal fix, exact full-source skin restoration and 13 native
stills below the same 2 GiB limit. The independent panda audit verifies nine
sampled poses and attached fringe. Its exported preview still has conspicuous
contour spikes and cheek/shoulder defects, so this experimental route remains
disabled and no approved asset is replaced. These repairs are not a fresh
hands-off text-prompt acceptance.

The latest [contour/contact diagnosis](docs/contour-contact-diagnosis.md) finds
that shorter regional fur improves outlines without changing the body, textures,
skin or actions. It remains short plush, not dense hair; cheek/shoulder defects
still prevent quality acceptance. The tested helper and diagnostic evidence are
preserved with compact evidence; the public fringe route and approved assets remain unchanged.

A [normal-transfer quality guard](docs/normal-transfer-guard.md) now rejects
inward-facing transfers before shading changes. The actual panda and Blender
checks reproduce 172 rejected corners; 23 focused software tests cover the guard
and its pinned, bounded repair diagnostics. This prevents a known defect but is
not a completed cheek repair or a new accepted Studio delivery.

A source UV repair now removes the pronounced green tint at all 16 fixed
shoulder witnesses in one matched color-only preview while preserving protected
white regions. The bright mark and cheek defect remain; the source derivative
is unrigged and still needs explicit finishing integration. All failures and
actual/analytic checks are preserved in `evidence/contour-contact-v1`. This is
local quality progress, not Muse parity or an accepted hands-off Studio run.

The studio is a local single-user workflow, not a hosted service. No server,
browser or generation job starts merely by reading this repository. The
existing read-only gallery remains separate on port 8765.

## Saved panda dance and shoulder-dragon flight

The **panda dance with its shoulder dragon taking off, flapping and landing**
works end to end as a tested authored-motion preview. It includes a real 19-joint
skin, editable source, four-second 720 px video, 1600 px portrait and static fallback.
Independent reload, 97-frame motion/framing audit, 444 checkpoint tests and browser playback
pass. The approved fur panda and standalone dragon remain unchanged.

[Watch the clip](outputs/muse-panda-dance-flight-media-v2-portable-v1/animations/dance_flight.mp4)
or use `./tools/avatar-assets/run gallery` from Weft; the new preview appears first.
For only this model, add `--catalog browser/gallery-compound-motion-v1.json`.
Open `http://127.0.0.1:8765/` yourself; no server starts automatically.

This is **not a polished/Muse-equivalent result**: cheek scuffs, mottled shoulder
shading and a small bright fleck remain. Browser fur also shows colored flecks;
the native video is the fuller fur representation. Physical flight, seamless
looping and unchanged-fidelity Weft import are not claimed.
[Files, controls, measured results and limitations](docs/compound-motion.md).
[Sealed delivery evidence](evidence/compound-motion-v1.json).

## Panda fur and detail upgrade

The latest fur work separates a clean generated undercoat from editable native
hair and lighter browser fur. Regional combing/length, eye and accessory masks,
source-colored tapered cards and bounded local color repairs are reusable
finishing controls, available to the AI author—not panda-specific code.
[Workflow, costs and preserved failures](docs/fur-workflow.md).

The new panda has an editable groom/rig, a1600px portrait and four720px/24fps/2s
motion previews: idle, look-around, head tilt and tail lift/dip. The complete4s
GLB actions are retimed, not cropped, for those videos. Wave/celebrate trials
remain rejected. [View the portrait](outputs/muse-panda-smooth-fur-media-v3-portable-v1/portrait.png).

From Weft, start or restart the gallery, then open `http://127.0.0.1:8765/`:

```sh
./tools/avatar-assets/run gallery
```

The default starts with the new dance/flight preview; this approved fur panda,
the standalone dragon, prior models and original2D reference remain available.
For just the fur panda and standalone dragon, add `--catalog browser/gallery-panda-fur-v1.json`. No server starts
automatically. Native hair looks much fuller than interactive3D's sparse cards;
the long generated tail and residual seam deformation remain limitations.
Muse-equivalent finish is not established.

One newly authorized Meshy generation used35 credits, with no paid retries.
That fur checkpoint passed 395 tests; its sampled native motion, actual export/reload
and artifact checks are recorded in [the handoff](TODO.md). Previous output files,
the approved dragon and the original reference are unchanged.

## Previous generated-model refinement

The finishing stage now supports bounded surface smoothing, separate material-response regions,
and native-hair sheen controls. A local comparison tool isolates geometry, texture/normal maps,
native hair and portable cards before spending on another generation. These are reusable,
configuration-driven controls, not panda-specific code. See [recipes, results and rejected trials](docs/meshy-quality-upgrade-v1.md).

The selected dragon refinement is `outputs/agent-dragon-polish-v1`; the two panda trials and
dragon material trial were rejected. Prior delivered packs remain unchanged. This pass does
not establish Muse-equivalent appearance, new texture detail or Weft compatibility.

Compare the previous and smoother dragon, including their matching 1600px portraits and four
720px motion previews:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-polish-v1.json
```

Then open `http://127.0.0.1:8765/`. The catalog also labels the original Muse pack as 2D-only.
No server starts automatically. Full results: 368 Python tests, native reload, and actual browser
playback/fallback checks passed; no new Meshy calls.

## Prompt-driven AI pipeline

`generate` connects an AI planner, rig/motion author and independent visual reviewer to the existing generation and local build tools. It accepts a prompt, optional reference/style, custom actions or state IDs, and an output preset. The coordinator enforces budgets, validates the generated configuration and preserves failed attempts. The revised route renders native/exported poses and final videos before the critic; only an accepted attempt receives a sealed delivery with real skinned GLB, editable Blender source, portrait, clips, static fallbacks and browser catalog. Unsupported requests return an explicit result. Revised quality/action acceptance remains under verification.

```sh
# From Weft: prepare the exact request and cost estimate without calling a model/provider.
./tools/avatar-assets/run generate \
  --prompt 'A friendly emerald dragon with amber eyes, small wings and a long tail' \
  --action 'Look left and right, then return to the starting pose.' \
  --action 'Flap both wings while standing still.' \
  --preset preview --max-credits 35 --prepare-only \
  --out outputs/my-dragon-001

# Explicit AI planning only; no Meshy generation or native render.
./tools/avatar-assets/run generate --resume outputs/my-dragon-001 --plan-only

# Only when covered by real authorization and the canonical provider ledger.
./tools/avatar-assets/run generate --resume outputs/my-dragon-001 --execute
```

Fresh Meshy geometry/textures are estimated at35 credits; AI-agent account charges are separate and currently not exposed by the authenticated CLI. The previous dragon and panda-specific allowances are spent: setting `--max-credits` does not create a new account allowance. `--reuse-model PATH --max-credits 0` tests new automatic rigging/motions on existing generated geometry with no new Meshy calls. Agent calls are still real model usage.

Custom action strings stay ordered and exact in the saved CLI request; each is
limited to 600 characters, with at most eight. Omit actions/state IDs for the
four-state default. The current AI-call allowance defaults to 10, capped at 12;
the expected first attempt uses three calls (planner, author, critic), and each
complete local repair adds two. These caps do not authorize new paid generation.

**Historical real-agent checkpoint on two different generated sources:** the dragon completed after two automatic repairs; the robot completed after one. Both exported real skins, four motions, portrait/videos/static fallbacks and passed actual browser playback checks. The robot test inferred style and state IDs from the text alone. These runs reused preserved Meshy geometry; fresh geometry generation was not purchased again. They verify the earlier bounded workflow, not the new final-media quality gate or studio. Full results, rejected attempts and measured limits are in [the verification report](docs/prompt-pipeline-results-v1.md).

View both completed examples with `./tools/avatar-assets/run gallery --catalog browser/gallery-agent-v1.json`, then open `http://127.0.0.1:8765/` yourself. Dragon media is1600px/720px/4s; the bounded robot test uses512px/256px/2s. Both retain their original4K texture maps.

See [commands, roles, outputs and limits](docs/prompt-pipeline.md) and the [current handoff](TODO.md). Agents cannot execute tools or spend directly; local scripts own execution. This is not a guarantee of Muse-equivalent quality, arbitrary interactions, hair physics or Weft compatibility. This release checkpoint includes source, selected editable models, compact media and verification evidence; full frame sequences and rejected binary intermediates remain local.

## Current scope

**Delivery target remains open:** a Muse-quality, controllable 3D avatar with convincing fur, matching proportions and expression, the shoulder dragon, meaningful actions and verified browser/Weft variants. The current models are prototypes below that visual target. A working compiler or a provider recommendation is not completion; a generation service would supply inputs to the workflow, not replace the remaining authoring and verification.

There are now **two implemented generation routes**:

- **Meshy7.1 text/reference-to-3D:** real paid generation using the user-designated project key, an independent non-resetting credit ledger, resumable tasks, original GLB/FBX downloads and local validation. [Repeatable commands, costs and limits](docs/meshy-workflow.md).
- **Local scene authoring:** agent-written JSON creates geometry/materials/skins/actions without paid generation. This remains useful for articulated objects, precise controls and repairs.

The new Meshy panda is closer in shape/identity than the procedural prototype, but **Muse-quality parity is not established**. An earlier source's auto-weights incorrectly assigned belly geometry to a hand; generated motions visibly distorted it and remain rejected. The new smooth source instead has a locally authored rig and four modest fur-demo motions. The finishing path supports explicit spatial weight corrections, accessory anchors, local replacement actions, native hair and separate browser fur cards. These are authored repairs, not secretly relabeled provider animation or recovered anatomy. Text-to-3D is automatic; final visual acceptance is not.

The independently text-generated blue/copper robot exercises the same finishing workflow. Earlier fox/lamp/HD models and all failed attempts remain available. Native hair and portable cutout cards have different appearance. No cloth/hair physics, recovered fingers or unchanged-fidelity Weft HD conversion is claimed. The original reference and hidden surfaces cannot establish the original 3D mesh or license.

**Standalone dragon:** Meshy generated the emerald mesh and scale textures; the local workflow authored its19-bone creature skeleton, skin weights and four motions: idle, look around, wing flap and celebration. This adds a reusable non-humanoid finishing route, not a universal automatic rig detector. [Dragon files, exact commands and limitations](docs/dragon-delivery-v1.md).

### What our code contributes

| Stage | Responsibility |
| --- | --- |
| Base shape and texture synthesis | Meshy, for the neural-generation route |
| Prompts, selected references, budgets and resumable jobs | Local workflow |
| Creature bones/weights, repairs, accessories, fur and explicit motions | AI author proposes subject-specific configuration; local tools validate/build |
| Structural checks, editable sources, renders and browser playback | Local workflow |
| Sampled visual review and bounded repair decisions | Separate AI reviewer; not human approval |

The procedural scene route creates geometry without Meshy, but has not matched the Muse target. Neural base generation currently uses Meshy only; another provider adapter is not implemented. Reuse of the downstream workflow is supported, while hands-off production quality for arbitrary prompts is not established.

The earlier [platform research](docs/generation-platform-research-2026-10-31.md) predates this authorized Meshy trial; it is not a report of tested alternatives. Current generation/results and costs supersede its previous “no backend connected” status.

## Migration verification

The relocated workflow passes 144 tests, both gallery catalogs, all four detailed browser models and the two legacy Weft importer checks. The complete local archive remains under this tool; selected portable assets are versioned. [Migration receipt](evidence/weft-migration-v1.json).

## Local setup

Python 3.14 on Apple silicon was used for this workspace. Dependencies are isolated in `.venv`. The lock contains exact hashes for the downloaded macOS wheels; other platforms require their own reviewed wheel hashes.

```sh
# From the Weft repository root:
cd tools/avatar-assets
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Blender 4.5.13 is installed only under `.tools/Blender.app`; use an explicit Blender executable on another machine. Official download SHA256: `663ce944257c61ff1d6aa09e15c8f57bbd8d59023adb2fa7edde33a9ed960b53`. No global app install or subscription. Local runtimes and intermediate outputs are ignored by Git. Selected complete model packs, portable media, source/configuration, tests and evidence records are tracked; see `.gitignore` for the explicit release allowlist.

The default, agent, HD, Meshy v1/v2/v3 and portable-quality catalogs include their required assets in Git. Historical `gallery-fur-comparison-v1/v2/v3.json`, `gallery-meshy-candidates-v1.json` and `gallery-quality-v2.json` depend on local-only experimental or full-frame packs; they are not fresh-checkout demos. Fur-comparison v2 also fails the current rig-facts validator. Use the current catalogs or `gallery-quality-v2-portable.json`; these historical records are retained without relabeling them as valid releases.

## View everything in the browser

**Dragon plus the previous generated candidates:** from Weft, use `./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v3.json`. The dragon is first; the panda, robot and original reference remain available. Open `http://127.0.0.1:8765/` yourself. The gallery makes no paid calls and does not drive physics.

**New Meshy candidates:** real generated panda/robot meshes, four locally authored clips each, and the original reference for comparison. These are review candidates, **not accepted Muse-equivalent assets**. From Weft:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v2.json
```

Use `http://127.0.0.1:8765/`. No Blender is needed. Current local bundles: [panda](outputs/muse-panda-meshy-refined-v6/manifest.json), [robot](outputs/workshop-robot-meshy-refined-v2/manifest.json), [workflow](docs/meshy-workflow.md). Each has a matching 1600px portrait and four 720px/24fps/four-second clips, with static fallbacks. Crafting/working remain weak pantomimes, not accepted task interactions. Generation used **132 Meshy credits**; local finishing used no further provider calls. Dollar purchase cost is unknown. Viewing the gallery makes no paid call.

The local gallery starts with the dance/shoulder-flight preview, then the approved soft-fur panda and smoother dragon, earlier HD panda, robot, fox, lamp and original 2D reference. From the Weft root, use `./tools/avatar-assets/run gallery`. No Blender or rendering job is needed to view them. Selected editable HD bundles and compact media are included in this release checkpoint alongside the earlier detailed release.

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -m avatar_assets gallery
```

Open **http://127.0.0.1:8765/** yourself. Choose a model, orbit/zoom, switch animations, play/pause, scrub the timeline, or view existing videos and portraits. Playback starts paused; reduced-motion settings default to a portrait. Ctrl+C stops the server. Nothing is uploaded by the gallery; it does not drive Weft physics.

Dependencies are already installed here. See [browser setup, controls and adding assets](docs/browser-gallery.md) for a fresh checkout or custom collection. The gallery is a read-only preview, not Weft physics or alignment verification.

If the gallery was already running, stop it with Ctrl+C and restart it to load the new catalog. The previous detailed collection is preserved in `browser/gallery-quality-v2-portable.json`; the original basic collection remains in `browser/gallery-basic-v1.json`. Select either with `gallery --catalog PATH --port 8766`.

## Use from a text request

Use [avatar-asset-workflow](skills/avatar-asset-workflow/SKILL.md) with a description such as “Create a blue workshop robot that idles, looks around, waves and gives a celebratory nod.” Use the [studio](docs/hands-off-studio.md) for local browser input/tracking, or [`generate`](docs/prompt-pipeline.md) for the CLI. Actual workpiece interaction is outside this backend; it is not silently replaced by a gesture. The commands below are the separate local scene-authoring route; they do not make paid calls.

```sh
.venv/bin/python -m avatar_assets request --subject 'Blue robot' --prompt 'A friendly blue workshop robot' --out outputs/request-new
.venv/bin/python -m avatar_assets build3d examples/scenes/workshop-robot-hd-v1.json --out outputs/robot-new --portrait-only --samples 32 --timeout 600
.venv/bin/python -m avatar_assets reload3d outputs/robot-new --out outputs/robot-new-reload
.venv/bin/python -m avatar_assets export2d outputs/robot-new --out outputs/robot-new-media --preset preview --size 720 --portrait 1600 --seconds 3 --samples 8 --denoise-portrait
.venv/bin/python -m avatar_assets verify outputs/robot-new-media
```

`request` captures text/reference hashes; it returns `needs_agent_authoring`, not a fabricated model. `build3d` consumes the scene JSON the agent authors. `--blender PATH` selects an existing local runtime. On this macOS host, Blender required execution outside the sandbox for Metal initialization even when rendering on CPU. It stays background-only; no viewers or servers open.

For raw media inspection:

```sh
.venv/bin/python -m avatar_assets inspect '/Users/ivan/Downloads/Poo avatar assets.zip' --sha256 cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb --out outputs/reference-new
```

See [scene schema/output overrides](docs/scene-contract.md), [examples](docs/scene-examples.md), [3D verification](docs/three-d.md), and [TODO/handoff](TODO.md). Existing output folders are never replaced. This tool uses Weft’s existing private Git remote; it has no nested repository.

## Verified artifacts

### Meshy-generated review candidates

| Subject | Model and editable source | Full media (original frames retained) |
| --- | --- | --- |
| Reference-based panda and dragon | `outputs/muse-panda-meshy-refined-v6` | `outputs/muse-panda-meshy-media-v2` |
| Text-only workshop robot | `outputs/workshop-robot-meshy-refined-v2` | `outputs/workshop-robot-meshy-media-v1` |
| Text-only emerald dragon | `outputs/emerald-dragon-finished-v3` | `outputs/emerald-dragon-media-v1` |

Both have real skeletons, skin weights, embedded textures and four locally authored gestures. Panda also has editable native hair and separate portable fur cards. Original provider models/motions and rejected attempts remain preserved. The gallery uses byte-identical compact media subsets; full frames stay in the folders above. No seamless loop or HD Weft compatibility is certified. The new workflow passed **255 Python tests** and the full browser check. See [delivery, verification and critical bone IDs](docs/meshy-delivery-v1.md) and the [current handoff](TODO.md). Fixture tests do not establish visual quality.

### HD browser editions

| Asset | Model/source bundle | Matching media bundle |
| --- | --- | --- |
| Panda + shoulder dragon | `outputs/muse-panda-hd-build-v3` | `outputs/muse-panda-hd-media-v1` |
| Workshop robot | `outputs/workshop-robot-hd-build-v2` | `outputs/workshop-robot-hd-media-v1` |
| Fox craftsperson | `outputs/fox-hd-build-v2` | `outputs/fox-hd-media-v1` |
| Hollow-shade articulated lamp | `outputs/lamp-hd-build-v2` | `outputs/lamp-hd-media-v1` |

All four include real skins, four skeletal actions, explicit normals/UVs/tangents, embedded textures, exact prompts, JSON sources and editable Blender files. Independent native reload and exported-attribute checks pass. Final recipes are `muse-panda-hd-v4.json`, `fox-hd-v3.json`, `workshop-robot-hd-v1.json` and `lamp-hd-v1.json` under `examples/scenes/`. Each reproduces exactly from its authoring recipe; GLB byte-for-byte rebuilds are not guaranteed.

Panda has7,688 skinned browser cards and238,328 configured native fibers; fox has5,000 cards and155,000 native fibers. Counts describe separate representations, not independently counted image features. Native hair is editable and follows the skeleton, without hair dynamics. The GLB uses alpha-masked ribbons, not the Blender hair system. Rebuild the GLB from the scene JSON; the saved native-render `.blend` hides the browser-card materials.

Each media pack has a native1600×1600 portrait, four silent720×720 H.264 previews at24fps, and exact first-frame PNG fallbacks. Panda/fox/lamp clips have48 frames/two seconds; robot clips72 frames/three seconds. They retime the complete four-second actions. All are opaque, every decoded thumbnail is distinct, and no seamless loop is certified. The optional isolated OIDN filter reduces portrait noise; raw renders and exact processing receipts remain preserved. Fine hair can soften during filtering.

These are **short HD previews**, not new eight-second Muse exports. The full preset remains available, but detailed fur rendering can exceed its1200-second/2GiB/512MiB bounds. Panda media rendered in967.158s and fox in800.671s; measured native peaks were1.65GB and1.42GB. At most two heavy native jobs ran concurrently, with four CPU threads each. Limits use sampled direct-child memory, not instantaneous OS allocation caps.

The [HD release index](evidence/hd-release-v1.json), [decisions/results](docs/hd-quality-pass-v1.md), and [visual review](outputs/hd-visual-review-v1/review.json) bind selected artifacts, source snapshots, costs and preserved failures. The [previous quality release](evidence/quality-release-v2.json) is unchanged. Higher-resolution textures, fiber counts and passing rig tests do not establish Muse-equivalent sculpting or appearance.

### Preserved original editions and Weft handoff

| Deliverable | Immutable local folder |
| --- | --- |
| Fox model, editable source, four skeletal actions | `outputs/fox-refined-v2` |
| Lamp model with tracing stylus, source, four actions | `outputs/lamp-refined-v1` |
| Fox full Muse-shaped media pack | `outputs/fox-muse-v1` |
| Lamp full Muse-shaped media pack | `outputs/lamp-muse-v2` |
| Muse-style panda with attached green dragon | `outputs/muse-panda-weft-build-v1` |
| Humanoid workshop robot | `outputs/humanoid-robot-weft-build-v1` |
| Panda model + explicit Weft metadata | `outputs/weft-muse-panda-v1` |
| Robot model + explicit Weft metadata | `outputs/weft-humanoid-robot-v1` |
| Frozen Weft importer acceptance evidence | `outputs/weft-import-proof-v1/results.json` |

The two full media packs each contain an opaque 1600×1600 WebP portrait, four silent 720×720 H.264 MP4s with 192 decoded frames at24fps/eight seconds, per-state PNG fallbacks, original rendered frames, exact source prompts/settings and hash manifests. Use the fallback images without autoplay for reduced motion. “Muse-shaped” means matching the supplied media dimensions/formats, not a verified proprietary Muse application import contract.

Full-size CPU renders took approximately 18–19 minutes per example while running two jobs together, peaking around 936MB sampled resident memory each. Refined 3D preview builds took roughly 16–21 seconds. These are measured host-specific timings, not service guarantees. No asset is certified as a seamless loop: exact endpoint images matched, but velocity continuity and continuous seam playback were not certified.

## Weft handoff

The models and metadata pass the **existing frozen Weft importer**. That code was read/executed without modification; sealed test imports were written only inside this workspace. There is no duplicate Weft importer, renderer, recording path or physics model here. [The delivery guide](docs/weft-delivery-v1.md) gives exact model/mapping paths, hashes, resource counts and critical node IDs.

The user authorized an [authored neutral convention](examples/weft/neutral-proposal-v1.json). This is not a measured Weft neutral state. [The alignment derivation](docs/weft-neutral-proposal.md) explains the exact transform equation and correction recipe. Each model/mapping folder includes actual glTF node IDs and global rest matrices in `node-table.json`. `AvatarRig` contains the mesh and all bones and maps to `mobile_root`; head/hands/accessories keep local rest transforms. Weft retains its19joint angles plus root translation. **Actual Weft pose alignment/display/replay remains for the integration owner to verify.**

Exact recovery of the original panda rig is explicitly unsupported in `outputs/reference-3d-unsupported-v1/unsupported.json`. The delivered panda is newly authored, not recovered fur, topology or anatomy. No independent license verification is claimed.

## Reproducibility and limits

Archive inspection reproduced the same manifest byte-for-byte. Input scenes, prompts, Blender/backend source and settings are preserved; finished artifacts are identified by hashes and never overwritten. Blender rebuilds can change triangle byte order and small floating-point normals even with identical scene/source/settings. Fixed triangulation preserves the tested topology, but **bit-for-bit GLB rebuilds are not guaranteed**; keep and integrate the hashed original model. See `evidence/reproducibility-analysis-v1.json` and the final handoff evidence.

The CLI bounds decoding, frames, image size, process time, sampled memory, log bytes and output. Native render jobs use four CPU threads; prefer at most two concurrent full renders on this ten-logical-CPU host. Packaging has a separate 180-second/768MiB bound. Current commands preserve failure manifests; early failed attempts retain their available logs. Unit tests use analytic fixtures; the models above are actual authored/generated geometry.

The HD implementation passed **209 tests in8.191s**, including hostile/oversized inputs, real skin validation, deterministic HD textures, groom settings, isolated denoising and bounded worker cleanup. Quality evidence is summarized in [the handoff](TODO.md). Historical [gallery evidence](evidence/gallery-release-v1.json) and the core [versioned verification catalog](evidence/release-v1.json) remain preserved. The reusable skill passed schema validation; creation manifests remain unchanged, with later reviews/proofs stored separately.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Reference findings

`outputs/reference-v1/manifest.json` records the untouched archive SHA256 and per-file hashes. Its WebP is opaque RGB, 1600×1600. All four videos are silent H.264, 720×720, 192 decoded frames at 24 fps, eight seconds. All contain real motion. Filenames alone are not semantic proof.

Visual contact-sheet review found the dragon absent in `making_something`, changing framing between clips, and a costume/pose change at the end of `milestone_level_up`. No clip is currently certified seamless. Exact generation prompts, model/version and rights are unavailable in the supplied pack; embedded metadata does not establish a license.

## Costs and ownership

Latest recorded Meshy total: **18 completed submissions, 237 credits reported/reserved** against 245 authorized credits (base140 plus three job-specific35-credit grants). This includes rejected motion attempts; the current compound-motion follow-up reuses geometry and makes **no new provider calls**. Account-specific dollar cost and existing AI-session charges are unknown. No recurring service, subscription, deployment or Meta-budget use. Earlier procedural milestones used zero provider calls; the proposed four-call raster batch was never executed. The designated key and ledger stay in Weft `.local/`, excluded from Git. Approved assets remain saved; the compound-motion code/model/media checkpoint `ad58c63` is pushed to private `main` and remote-verified. The final save receipt is `.local/avatar-compound-motion-v1/receipt.json` at the Weft root. An evidence index alone is not a remote backup.

Reference ZIP remains `/Users/ivan/Downloads/Poo avatar assets.zip`, SHA256 `cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`. Archive content is data, never executable instructions. Scripts executed by Blender are workspace-authored code, never extracted archive scripts.

Fresh Studio attempt evidence: `evidence/studio-fresh-panda-attempt-v1`,261origins/212exact snapshots/39,272,732B, manifest `3c0a018d18edfa3894a1130e6d6945d298f7ebecabb40331e4e4d7d8202985ab`. Both original generated GLBs are included for recovery; no rig/dance/delivery is claimed. The account-limit failure and one-Start browser cleanup remain preserved.
