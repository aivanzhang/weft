# Weft project guide

Current working guide, consolidated November 3, 2026 (PST). Older studies and
handoffs describe dated checkpoints, not the latest product status. Repository
instructions remain in [AGENTS.md](AGENTS.md).

## Goal and current scope

**One video → an inspectable world model → an interactive simulation → a
trainable reinforcement-learning environment.** Reinforcement learning (RL)
means an agent learns actions through trial, feedback and a separately defined
success test. A recorded animation or programmed controller is not learning.

Weft is a distinct project in `kintsugi-software/weft`, not a new implementation
of the neighboring Loom ingestion product. One pipeline serves object, ball and
exercise inputs. Exercise videos with a 3D body, joints and illustrative muscles
remain a priority; arbitrary-video physical reconstruction is not solved.

The immediate delivery target is a usable local demo: upload → processing and
artifacts → 3D sandbox → optional avatar → interaction → recording/replay → Meta
Muse review. Broader reconstruction research remains backlog rather than a
reason to prevent users from trying supported demonstrations.

The full product must eventually infer relevant objects, temporal identities,
camera/time/scale, interactions and uncertain parameters; construct one shared
world; test bounded repairs independently; and measure useful learning. Objects,
articulated bodies, animals and deformable materials need separate qualification.
An inferred alternate view is an inspection tool, not another physical observation.

## Run locally

Run these from the repository root in separate terminals:

| Environment | Command | Address / runtime |
| --- | --- | --- |
| Landing page | `npm run dev` | `http://127.0.0.1:3000/`; Node.js 22+ |
| Video-to-simulation sandbox | `npm run sandbox` | Open the complete authenticated link printed for port 8765; `.venv-lab` |
| Saved hackathon demo | `npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal` | Same sandbox, existing saved project collection |

The user starts and manages persistent servers. Do not start two sandbox
processes against the same project directory. Choose another landing port with
`npm run dev -- --port 3001`, or another sandbox port with
`npm run sandbox -- --port 8766`. Keep the printed sandbox token private.

The landing page has no npm dependencies and never launches Python, forwards
uploads, or calls AI APIs. Refresh after edits. Its allowlisted assets are
separate from the sandbox's code, credentials and private results. This is
process/dependency separation, not a container or OS security boundary.

Sandbox state defaults to `.local/portal`; quitting preserves completed projects
and recordings. The saved demo and other `.local` / `artifacts` examples are not
included in a fresh clone. Keep their exact saved locations where manifests
bind paths; moving and resealing them is not an evidence-preserving migration.

Reuse the working `.venv-lab`. A fresh environment uses
`requirements-lab.lock`; its tested platform is Python 3.14 / macOS arm64.
See [sandbox setup](sandbox/README.md). Preserve the original `.venv`; avatar
authoring has a separate `tools/avatar-assets/.venv` because its dependencies
conflict with the lab environment.

### Typical portal journey

1. Select a video. Portal intake allows up to 60 seconds, 1,800 frames and 100 MB;
   each reconstruction path also has stricter admission/resource checks.
2. Inspect the scene/output choices and consent before provider processing.
   Body evidence preview has no physics. Ground-only interactive construction
   is an explicit scope assumption, not proof that equipment is absent.
3. Inspect source frames, model observations, assumptions, stage logs and failures.
   Passing preflight or receiving an AI response does not guarantee a world.
4. Open an admitted sandbox. Choose manual, authored reference or learned control
   where available. Use **Record & start → Stop & save** for a fresh trial.
5. Select/import a compatible avatar if supported. Appearance follows the RL
   state and must not change actions, collisions, rewards or success rules.
6. Download recordings, offline replay, environment/evidence and verification
   reports. Muse review and local comparisons remain separate explicit actions.

Detailed usage: [portal](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/portal.md), [lab](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md),
[avatars](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/avatars.md), [demo walkthrough](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/hackathon-demo.md).

## Architecture and ownership

| Layer | Current implementation | Required boundary |
| --- | --- | --- |
| Product entry | Python portal, supervised jobs, static browser UI | Project API keys stay server-side; no coding-session or ChatGPT-login runtime |
| A · Evidence | Video decoding, native clocks, pose/tracking observations, explicit source adapters | Keep every source slot, ambiguity and missing estimate; preserve observed/inferred/assumed/unknown labels |
| B · World | Original rolling model, Pymunk planar scenes, MuJoCo rigid/human/shared-world profiles | One coordinate system and clock per world; contacts and controls obey that profile's declared assumptions |
| C · Proposals | Restricted parameter fitting, source-grounded Muse critique and bounded local candidate tests | A proposal cannot edit scorers, success gates, final evidence or spending history |
| D · Evaluation | Source comparisons, analytic checks, task evaluation, immutable receipts and replay | Separate reconstruction, simulated behavior, learning and physical validation |
| E · RL | Explicit task contracts, bounded learners, baselines and changed-start evaluation | Reward alone is not success; a retained authored controller is not learned improvement |
| Appearance | Rig import/mapping, retargeting, avatar catalog and appearance journal | Render geometry is not collision geometry; avatar substitutions leave physics unchanged |

`src/weft/` owns shared simulation and pipeline code. `src/weft/assets/` holds
the lab/portal viewer and avatar renderer. `scripts/` contains launchers and
bounded studies. `tests/` owns mechanism/contract verification. `website/`
contains the buildless landing page; `sandbox/` owns its separate launcher.
`hosting/` is the preserved, currently offline hosted path. `tools/avatar-assets/`
is the separate authoring package; Weft's consumer remains in the main pipeline.

Historical tracks A and B are ownership boundaries for parallel work, not
separate products. Assign non-overlapping files and output directories, reuse
prepared inputs, and coordinate native rendering/model jobs. Never duplicate
active jobs or assume an old handoff's worker ownership is still current.

## Verified status and limits

These are recorded results, not checks rerun by documentation consolidation.

| Capability | What exists | What remains unverified or incomplete |
| --- | --- | --- |
| Portal | Real upload/preflight, API-backed analysis, artifact inspection, admitted-world playback and recordings | Reliable reconstruction of unfamiliar videos and judge-ready hosting |
| Shared physics | Bodies and rigid objects in one MuJoCo world; manual controls and exact recorded-action replay | Source-supported depth, equipment grasp/contact, material accuracy and real balance |
| Training | Bounded task authoring, training and matched evaluation on supported profiles | Useful learned improvement on difficult exercise/equipment tasks; many preserved tasks fail |
| Meta | Real SAM tracking and Muse Spark API reviews with source/world binding and charge receipts | General autonomous reconstruction/repair and physical validation |
| Candidate testing | Local comparisons and a Muse-linked test path for bounded, locally authored controller alternatives | General execution of arbitrary Muse hypotheses; real measurements requested by Muse |
| Avatars | Built-in panda/robot, bounded compatible-rig import, state-driven motion, recording/export/replay | Arbitrary-model automatic rigging; Studio skin deformation; broad visual compatibility |
| Local environments | Separate landing/sandbox commands; eight launcher/HTTP checks passed | New browser visual QA or sustained-use performance qualification |

### Current demo evidence

- Primary saved project: `1a5b41db5aa34654a84c83a9dbf5e316` in
  `.local/portal-unfamiliar-source-acceptance-01/portal`. It uses the licensed
  single-leg-squat source and explicit conditional body/ground construction.
  Original scene inventory remains uncertain; omitted equipment is not absent.
- Continuous robot recording: `lab/sessions/session-fafeb2ffe374` under that
  project, with 234 controls / 235 frames and 7.8 seconds of simulation. Exact
  physical replay and appearance export pass. **Task success is false:** mean
  and peak body-position tracking fail; the fixed camera can crop the avatar.
- The earlier seven-frame, three-avatar clip is a 0.233-second integration test,
  not a useful animation demo. Paused wall-clock time adds no simulated frames.
- Studio panda import plumbing passes, but bad source skin weights cause severe
  deformation under RL poses. Do not describe this merely as reduced detail or
  claim visual qualification. Repair generic skin/rig compatibility, not one
  authored flight/hopping animation. See [motion findings](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/avatar-motion-quality.md).
- A real `muse-spark-1.3` review of the primary conditional world returned three
  hypotheses; no suggested physical change was applied. On a separate earlier
  exercise project, **Test Muse suggestions** tested three locally authored
  controller candidates and rejected all three. Both facts must remain visible.
  [Demo evidence](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/hackathon-demo.md),
  [candidate-test scope](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/portal-native-and-repair.md).

### Evidence that must not be overstated

- A separate ramp material study improved later image-center error from 61.66
  to 8.51 pixels, but still failed its frozen full accuracy check. That repeat is
  now exposed development evidence, not an untouched future holdout.
- A guided ball policy passed 8/12 changed starts while a fixed untrained
  controller passed 12/12. This does not establish a benefit from training.
- Body evidence, illustrative muscles, authored root assistance and programmed
  reference tracking are not recovered anatomy, measured activation, unassisted
  human balance, calibrated forces or validated biomechanics.
- Successful controls and exact replay establish software behavior, not accuracy
  of real-world changed-action predictions or arbitrary-video generalization.

## Remaining work and acceptance gates

### Immediate usable demo

1. [ ] Repair generic avatar skin/mapping compatibility and camera framing.
   Require continuous representative poses, visual inspection, no severe stretch,
   and unchanged seeded physics/reward. Preserve the failed Studio derivative.
2. [ ] Verify one complete user journey with an unfamiliar supported video and
   explicit assumptions. Retain full source coverage, failure states and a
   meaningful continuous recording; do not hide manual setup or shorten a failed
   source to claim whole-video success.
3. [ ] Make Meta's contribution visible and consequential. Keep existing critique
   and bounded rejection evidence; any accepted repair needs a supported
   hypothesis and an unchanged independent comparison, not only model approval.
4. [ ] Prepare a reproducible, concise demo and release package when requested.
   Public publication, hosting restoration and submission require their own
   authorization. Preserve source/asset rights and prior-work attribution.

### Broader product, not completed by a demo

1. [ ] Qualify unfamiliar-video identities, geometry, camera, ground and equipment
   contact across a frozen supported-input set. Report every failed/missing slot,
   identity alternative and manual intervention; retain uncertainty when evidence
   cannot identify hidden depth, mass, friction or actuator forces.
2. [ ] Extend bounded physical repair without changing the task, source evidence
   or scoring thresholds. Require positive/negative controls, retained rejected
   proposals and tests that distinguish plausible worlds using new evidence.
3. [ ] Demonstrate meaningful learning against simple baselines on matched seeds
   and changed initial conditions. Evaluate policies on a common independent
   world/task; success in different self-generated worlds is not a comparison.
4. [ ] Acquire independent physical intervention/contact evidence before claiming
   real transfer. Simulated counterfactuals and model-generated views cannot
   replace measured trials. Keep construction/development/final evidence separate.
5. [ ] Qualify robustness and resources across occlusion, missing limbs, camera
   motion, multiple candidates, longer clips and near-limit inputs; measure
   latency, peak memory, responsiveness, retained failures and total cost.

Full product completion requires a new supported video to generate an
interactive shared world and explicit task without clip-specific simulator edits,
pass fixed reconstruction/action checks, and demonstrate actual useful learning.
Every assumption, unsupported case and independent-evidence gap stays visible.

## Budget, data and authorization

- Meta: **$5 total**, shared across adapters and sessions; canonical ledger
  `.local/meta-budget/ledger.json`. OpenAI portal: a separate **$5 total**, ledger
  `.local/portal-openai/ledger.sqlite3`. Check the current ledgers before paid work;
  dated totals in old studies are not current remaining balances.
- Reserve before requests, preserve unknown charges and failed attempts, and
  never reset/copy a ledger to obtain another allowance. Existing avatar-service
  work has separate authorization; neither portal allowance covers it.
- Project credentials remain local/server-side. Never use ChatGPT login, agent
  subscription/OAuth credentials or a coding-session CLI as the product AI runtime.
- Hosted site stays offline. No public repository change, deployment, hackathon
  submission, credit application, social post or agreement acceptance is implied.
- Use `main` and the repository's current publication rules. Preserve unrelated
  dirty work, live job ownership, stashes and immutable experiment artifacts.
- Do not delete dot-prefixed directories just because they are hidden. `.local`
  contains private projects, budgets, credentials, models and evidence; `.git`
  is repository history; `.venv*` are runtimes. Inventory ownership and references
  before cleanup. Re-creatable caches are a separate category from unique data.

## Development and verification

### Repository cleanup and hidden folders

The November 3, 2026 (PST) cleanup retired 350 superseded Markdown files into
[the pinned Git history index](docs/HISTORY.md), preserving their
complete content and original section links. Use this file for current project
knowledge rather than creating new overlapping handoffs. Source-linked research
pages, agent instructions, source/license records and immutable evidence remain
separate. The root README retains concurrent product edits and points here.

Tracked Markdown falls from 669 files to 321, including this guide and the
history index. Verification checked all 350 historical file hashes, confirmed
16,317 protected source/evidence files unchanged, and found no active Markdown
links to retired local paths. No runtime code changed for this cleanup.

Hidden folders were inspected, **not deleted**. Sizes below are allocated disk
usage at the audit, not a promise of future space savings:

| Path | Size | Decision |
| --- | --- | --- |
| `.git` | 1.40 GiB | Keep repository history; the documentation archive depends on it |
| `.local` | 7.79 GiB | Keep unique projects, demos, recordings, models, credentials and ledgers |
| `.venv-lab` | 554 MiB | Keep; required by `npm run sandbox` |
| `.venv` | 206 MiB | Preserve the original Python environment |
| `tools/avatar-assets/.local` | 22 MiB | Keep authoring state and its separate budget history |
| `tools/avatar-assets/.venv` | 107 MiB | Keep the separate, dependency-incompatible avatar runtime |
| `tools/avatar-assets/.tools` | 1.13 GiB | Keep installed Blender; do not delete the whole folder |

Potentially removable caches total **about 258 MiB**: `.pytest_cache` (20.6 MiB),
`.local/swift-cache` (133.6 MiB), `.local/pip-cache` (57.8 MiB), and
`tools/avatar-assets/.tools/npm-cache` (46.3 MiB). Remove only after confirming
their test/build/installer processes are stopped; regeneration may need network
access and take longer. This audit is not authorization to remove them.

The Blender installer `.tools/blender-4.5.13-macos-arm64.dmg` under the avatar
package is another 297 MiB candidate after checking the installed app and active
mounts; keep its checksum. Installed `Blender.app` is used by authoring scripts.
Several nested `.cache` files are tracked evidence: never glob-delete dot folders.
Future `hosting/.wrangler` may contain local database state, not just cache.

Never remove `.local/credentials`, `.local/meta-budget/ledger.json`,
`.local/portal-openai/ledger.sqlite3`, `.local/meshy/ledger.sqlite3`, or
`tools/avatar-assets/.local/openai/ledger.sqlite3` to reclaim space. Reinstalling
dependencies cannot restore cumulative spending history or unique evidence.

### Change workflow

1. Choose one pending gate and state a falsifiable hypothesis before changing it.
2. Preserve source/model/task/scorer identities. Put each new study in a new
   directory; reuse unchanged prepared inputs rather than repeating paid inference.
3. Run tests proportionate to the change, then verify relevant real artifacts.
   Source fit, analytic simulation checks, interactive behavior, learning and
   measured physical results must be reported separately.
4. Record commands, versions, seeds, prompt/model receipts, costs, failures and
   artifact identities. Keep completed run manifests apart from mutable lifecycle
   metadata and new session recordings; never relabel a terminal result cancelled.
5. Update this guide when the current state changes. Historical evidence remains
   available without duplicating the same status into many new handoff files.

Useful checks:

```sh
npm run test:local
node sandbox/run.mjs --check
.venv-lab/bin/python -m pytest tests/<affected_test>.py
git diff --check
```

The original rolling-profile `weft compare` checks compatible runs only; it is
not a universal ranking tool for body/planar/shared-world studies. A fresh
holdout must be chosen before generation or tuning; repeated inspection makes
it development evidence. Hash manifests detect changes, not an author's honesty.

## Resume record

- Last published launch checkpoint before this documentation cleanup:
  `d0338eb` on `main`. It includes the separate landing/sandbox commands.
- The inherited avatar helper checkpoint `5caca13` passed 58 Python and eight
  Node checks; it does not qualify generated-source repair or new animations.
- `db33fa3` preserves `src/weft/moving_equipment_body_sweep.py` as an untested,
  unintegrated research draft. Do not call its presence a completed feature.
- At cleanup intake, `avatar_offscreen_bridge.cjs`, `avatar_retarget.js` and
  `avatar_worker.js` under `src/weft/assets/` had separate uncommitted edits.
  Coordinate their owner before staging or replacing them; current quality
  claims above do not include that unfinished work.
- Update this section and the affected status/backlog entry at meaningful
  milestones. Include exact commands, evidence paths, failures and live ownership
  when needed to resume; do not create another overlapping status document.
- Historical narratives are retained through the versioned index in
  [documentation history](docs/HISTORY.md). Their old pending tasks and ownership
  assignments are dated evidence, not current instructions.

Keep scoped `AGENTS.md` files, source/license records, model/data provenance and
immutable evidence inventories separate where their function requires it. This
guide consolidates current project knowledge; it does not replace those records.
