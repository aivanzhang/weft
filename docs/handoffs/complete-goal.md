# Complete Weft goal — video to an interactive, trainable world

Build and verify Weft end to end: provide a video, inspect what the system observes and infers at every step, generate one executable world, interact with it through an RL environment, train and evaluate an agent, and record/replay actions and results. Continue implementation, not just planning. Improve the existing system rather than replacing it with a disconnected prototype.

## Scope and ownership

Own BOTH former tracks in this session:

- **Track A:** video understanding, evidence, tracking, camera/geometry/body reconstruction and evidence-backed world candidates.
- **Track B:** simulation, coupled physics, RL tasks and learning, the unified local lab/CLI, interaction, recording and replay.

The user closed agy's **[CURRENT] Integrate Spatial Rigid 3D** session and transferred its work here. The old A-only objective, old agy assignment, cross-session no-edit restrictions and historical pause no longer define the scope. Preserve their useful architecture and evidence boundaries. Do not wait for the closed session or duplicate its implementation. Review and finish its existing uncommitted work.

Build ONE generalizable system. Balls, tabletop objects, exercise videos, and broader physical scenes are validation cases, not separate products. Exercise remains a priority: generate an inspectable 3D body with joints and illustrative muscles, including interactions with equipment. A person and objects in the same scene must share coordinates, time, contacts and physical dynamics, not be rendered from unrelated simulators and spliced together.

Generalizable does not mean claiming arbitrary-video recovery is already solved. Develop reusable methods, test unfamiliar inputs, document the supported envelope and abstain honestly when evidence is insufficient. Do not redefine the complete goal as passing the current small set of profiles, unit tests, or authored fixtures.

## Repository and starting context

- Workspace: `/Users/ivan/kintsugi/weft`; private repository: `kintsugi-software/weft`; branch: `main`.
- Consolidated on **October 29, 2026, 7:04 PM PDT**, America/Los_Angeles. Originating thread: `01a18139-c198-70d3-8cfe-e8e79b56fa2a`.
- Read `AGENTS.md`, this prompt, `goal.md`, `TODO.md`, `docs/architecture.md`, `docs/iteration.md`, and `docs/parallel-workstreams.md`. Read `docs/local-lab.md` and the relevant simulation/learning guides before changing those paths.
- `docs/handoffs/track-a.md`, `docs/handoffs/agy.md`, `AGY_HANDOFF.md`, `AGY_GOAL_PROMPT.md` and `docs/handoffs/track-a-goal.md` contain historical implementation context. Their old ownership/pause instructions are superseded by this combined assignment. Do not treat their snapshots as current process or remote state.
- Last published checkpoint: `66490b925a69ce63f78b0a2be8970e2a0d7cf831` (`Add source-bound tracking and shared-world construction checks`). Publication independently verified the exact remote hash. Local `main` and cached `origin/main` still matched when this prompt was prepared; refresh remote state before the next integration. Stash list was empty. The worktree is deliberately NOT clean because inherited implementation remains unfinished.
- Full lab environment: `.venv-lab`, `requirements-lab.lock`; preserve the original `.venv`. Inspect existing dependencies before installing anything. Saved videos and studies under `data/`, `artifacts/` and `.local/` are local evidence, not necessarily available from a fresh clone.

## Required end-to-end product

### 1. Ingest and understand video as evidence

Accept a local video and optional explicit task/calibration inputs. Provide a documented CLI path and an understandable local lab; a notebook may supplement them but must not become a separate pipeline. Find appropriately licensed example videos when needed and preserve attribution, source URLs, retrieval metadata and exact hashes.

Preserve the original file, decoded frame count, native timestamps, frame identity, dimensions and every relevant entity-by-frame slot. Show missing, unobserved, occluded and pre-initialization slots; do not renumber away failures or present interpolation as measurement. Validate derivative mappings when conversion is necessary. Container frame-count metadata is not automatically authoritative.

Detect and maintain identities for relevant people, objects, supports and obstacles across the sequence. Track masks, features, poses and contact candidates with confidence and source references. Handle camera motion and visibility changes explicitly. Distinguish observed, inferred, assumed and unknown values. Label manual prompts and authored assets; do not conceal per-clip intervention behind an automatic-video-only claim.

Produce a versioned physical scene graph, not just labels. Represent support, collision, rolling/sliding, joints, actuation, containment and occlusion with evidence and uncertainty. Proximity and temporal coincidence are not proof of contact or causation. Use interaction neighborhoods rather than uncontrolled all-pairs analysis.

### 2. Reconstruct one coherent executable world

Separate appearance, physical geometry/dynamics, and behavior/control. Estimate supported camera, geometry, pose, joints, scale, initial state, materials and controls. Preserve alternative explanations and unknown depth, mass, friction or force when a video cannot identify them. Relative units must not silently become metres; physical defaults must remain assumptions.

Reconcile entities in one explicit coordinate system and timeline. Source bindings must identify exact evidence documents and entities, not ambiguous bare IDs or unverified counts. Any spatial or clock transform must declare its direction, units, method, uncertainty and evidence. Reject incompatible or unknown transformations instead of silently shifting floors, rescaling bodies or retiming recordings.

For mixed articulated/rigid scenes, compile and step one coupled world with real simulated contacts and shared state. Complete the existing MuJoCo implementation rather than stitching simulations. Validate shape, material, gravity, inertia, actuator and task migrations separately; transforming positions alone does not preserve physics.

Provide reconstruction mode, which compares simulated motion against source evidence, and interaction mode, which accepts changed actions/starts. Passive bodies must evolve under the physics engine, not follow hidden recorded-position scripts. Show source-camera and alternate views; unseen surfaces remain inferred rather than newly observed.

Exercise support must cover useful whole-body motion, root movement, joint constraints, supports, ground and equipment interaction. Preserve differences between anchored bodies, mobile bodies, virtual root assistance, authored reference controllers and learned policies. Illustrative muscle bands are a visualization, not recovered anatomy, measured activation, EMG or validated biomechanics.

### 3. Improve candidates without weakening evaluation

Implement a bounded observe → propose → simulate → compare → repair loop. Use deterministic numerical fitting for numerical changes; generate code only where model structure requires it. Preserve rejected candidates, failures, changes and reasons for acceptance. Later stages may challenge an interpretation, but must not overwrite raw evidence.

Use meaningful bounded Meta reasoning when useful for scene relationships or hypotheses; it is a proposer/critic, not a ground-truth oracle. Keep local fallbacks and clearly expose unavailable capabilities. Bound candidates, tokens, requests, retries, compute, runtime, memory and output size. Run untrusted generated code outside the host process with no secrets or network by default.

The generator/repair process cannot modify independent scorers, final evidence, task-success conditions, protected thresholds or the evaluation ledger. Do not select on final holdouts and later call them untouched. Compare alternative worlds and policies under the same frozen independent evaluation protocol.

### 4. Make the sandbox understandable and useful

Provide a clear start-to-finish local workflow: choose video, inspect processing progress, inspect detections/tracks/poses, inspect the world and assumptions, run comparisons, interact/train, then record/export/replay. Reuse the existing lab and CLI.

Use plain-language stage names and explanations. Show what each artifact means, why a stage failed, what is uncertain, which inputs/settings were used, what controls do, and what to try next. Distinguish real progress from decorative animation. Provide useful defaults without hiding assumptions or advanced details.

Show original video and evidence beside source-fit replay and live simulation. Provide 3D camera controls, relevant entity/joint/contact inspection, actor selection where supported, reset/step/play/pause, control modes, visible recording state and accessible exports. Never imply an unavailable or untrained control mode works.

Keep heavy inference, parsing, construction and export off UI/request threads. Bound queues and concurrency, cancel stale jobs, avoid overlapping polling, and release resources. Measure representative end-to-end latency and sustained resource use before claiming responsiveness; tiny function timings or mocked UI tests alone are insufficient.

Keep completed `run/` manifests immutable and separate from lifecycle metadata and new `sessions/` recordings. Preserve actions, seeds, state, actor/controller selection, reset events, source identity and runtime versions for exact replay. Recovery must write a new output and preserve the interrupted original. Never relabel a published terminal result cancelled during worker cleanup.

### 5. Expose a real RL task and verify learning

Provide standard seeded reset/step semantics and an inspectable task contract: controlled entity, bounded action space, observation/hidden-state boundary, reward, independent success, reset distribution, termination and time limits. Identify inferred versus authored task choices; a video does not uniquely define intent or reward.

Train a real bounded learner, compare with appropriate coast/zero/random/authored baselines, freeze policy artifacts, and evaluate on independent seeds and changed initial conditions. Preserve training budgets and failures. Do not call scripted reference playback, one-joint learning, privileged-state assistance or higher reward in a policy's own world proof of full-task learning or physical transfer.

Test changed actions, support removal, different starts, occlusion, longer rollouts, timestep sensitivity and reward shortcuts. Camera/appearance-only changes must not change physical trajectories. Physical-transfer claims require independent recorded or measured outcomes under a declared protocol, not simulator agreement with itself.

## Evaluation and completion evidence

Maintain a score vector with sample counts, evidence partitions, uncertainty and failure examples. Report unavailable metrics instead of inventing zero error. Cover all fourteen dimensions from `goal.md`:

| Dimension | Required evidence |
| --- | --- |
| Execution and actions | Valid load/reset/step, legal bounded actions, finite states, termination and seeded replay |
| Perception and identity | Missed/extra entities, switches and visibility errors against independent checks |
| Geometry and camera | Projection/pose/shape errors, calibration or explicitly relative units |
| Temporal dynamics | Trajectories, velocities, event order and contact timing across the rollout |
| Physical relationships | Constraints, collision tolerances, force budgets and analytic controls |
| Hidden-state continuity | Persistent identity/state through occlusion, with predictions labeled |
| Parameter validity | Independent measurements, identifiability and uncertainty; no inferred truth by default |
| Causal response | Matched changed-action tests and rendering/physics separation |
| Long-run stability | Drift, solver failures, reset and timestep stress checks |
| Robustness and coverage | Unfamiliar clips, scenes, seeds, starts and visibility conditions; declared limits |
| Uncertainty and abstention | Unsupported claims, retained alternatives and clear failure/unsupported output |
| Task and learning utility | Independent success, matched baselines, multiple seeds and exploit checks |
| Reproducibility and integrity | Rights, hashes, partitions, models, prompts, patches, seeds and verified manifests |
| Practical efficiency | Human intervention, stage latency, throughput, memory and total failed/successful-run cost |

Maintain three evidence boundaries: construction/source fit, exposed development validation, and fresh final evaluation. Previously inspected Cam3 data, downsampled equivalents and failed repeat clips remain development evidence. Source fit, synthetic controls, simulated interventions, learned-policy evaluation and measured physical validation must remain visibly separate.

Do not mark the overall goal complete merely because current profiles run. Completion requires the integrated local user journey, source-qualified coupled worlds where the evidence supports them, working interaction and recording/replay, meaningful independent RL evaluation, diversity/failure coverage, reproducible acceptance artifacts, and an honest report of physical accuracy and remaining unsupported scenarios. Unsupported cases must be explicit; they cannot be silently counted as successes or used to narrow away the requested generalized capability. Do not manufacture independent physical evidence if it is unavailable: continue useful authorized work and record the exact remaining gap.

## Verified foundation — reuse it

### Track A

- `world_evidence.py` binds exact JSON hashes and qualified entity IDs, preserves complete slots and exports only authorized construction partitions. `world_alignment.py` validates explicit spatial/clock mappings, units, proper rotations and source/target direction. These are construction boundaries, not a completed shared simulator.
- Latest focused checkpoint: **69 tests** (17 original-point flow, 23 evidence registry, 29 alignment). `artifacts/world-binding-01` retains five documents, eight qualified entities, 860 source frames and all 1,266 slots; 106 withheld slots stay withheld. Seal: `7d1a3ea836c53afe80581cf37a039670b2cc2cfa8234ca8bbc0fe4f0e1252c87`. Independent audit: `.local/world-binding-verification-01.json`.
- The 684-frame original-point-flow study regressed real coverage; all 179 shape windows rejected. Earlier broader track-shape studies also qualified zero real windows. Preserve those negative results and keep defaults unchanged. See `docs/point-flow-diversity.md` and `docs/world-construction-boundary.md`. Do not rerun unchanged failures.
- Camera/depth, human joint/ground/contact geometry and independent changed-action accuracy remain unresolved. A source-fit improvement or convergent optimizer is not physical validation.

### Track B inherited from agy

- Published milestones: `f7b5f2f` spatial 3D lab integration; `8c501de` capability contracts and multi-object actor controls; `f914a31` bounded spatial RL and whole-body changed-start evaluation; `82ce979` non-destructive recording recovery, provenance and B handoff. All are ancestors of the published checkpoint above.
- Existing capabilities include planar/3D rigid and articulated environments, manual/authored/learned modes where supported, 3D inspection, actor selection, recordings and exact replay. CLI entry points include `spatial-fixture`, `spatial-lab`, `spatial-train`, `body-evaluate`, `capabilities`, and `replay --recover-incomplete --out`.
- B reported 136 non-HTTP tests, 58 HTTP tests, 85 human/foot regressions, 38 mocked UI checks and a spatial replay smoke pass before subsequent mixed-world edits. These are historical scoped results, not validation of today's entire dirty tree or live-browser/device QA.
- Saved studies: `artifacts/track-b-spatial-lab-01` and `artifacts/track-b-learning-changed-init-02`. Six manifests / 179 files were independently checked in `.local/track-b-publication-manifests-01.json`; do not retrain just to rediscover them.
- Retain actual failures: authored spatial learning had **0/12 changed-start successes**, ramp **1/12**; single-leg whole-body passed **30/30 jitter and 12/12 explicit perturbation trials**, while squat retained **6/9 floor failures**. Virtual root actuators and privileged state remain explicit. Rigid distances use `scene_width`, not measured metres. Correct squat-evaluation manifest: `b431173761bd42a49fc5e41f801b9487d1ad9c086c821c54261e044c4bd8d1d9`; the earlier B handoff contains a stale hash.

## Inherited unfinished implementation — preserve and finish

At preparation, these B changes were uncommitted and require review plus relevant verification before publication:

- New `src/weft/shared_world.py`, `shared_world_lab.py`, `shared_world_learning.py`, `src/weft/assets/shared-world-replay.html`, and `tests/test_shared_world.py`.
- Modified `src/weft/assets/lab.js`, `capabilities.py`, `cli.py`, `lab.py`, `recording.py`, `spatial_learning.py`, `whole_body_learning.py`, and `tests/test_whole_body_changed_init.py`.
- Also preserve the local source-preparation script `scripts/studies/prepare_kettlebell_tracking.py` and its failed study described below.

The mixed implementation proposes `shared-world-v1` using one MuJoCo model/data/clock. Its current `adapt_to_shared_world` accepts a sealed mobile `exercise-v3` body and adds an **authored ball and pedestal**. That is useful runtime scaffolding, not reconstructed equipment from the recording.

Review these integration risks before claiming source-qualified mixed support: exact evidence-document/ID resolution; full original frame coverage rather than counts inferred from sampled poses; native versus assumed pose clocks; scale/rotation/origin alignment; evidence-backed calibration flags; truthful units and authored force/material labels; initial penetration; migrated dynamics; task/action/actor state and recording compatibility. Current bodies with `camera_reference` are not accepted by that adapter. Do not loosen validators to disguise unsupported cases.

Require focused coupled-world checks: momentum transfer between rigid body and articulated limb with a non-contact control; rejected initial penetration without silently moving bodies/floor; passive bodies not source-driven; exact joint/rigid reset and recorded replay; actor/control transitions; preserved legacy behavior where compatibility is claimed. These checks establish software behavior, not real-world physical accuracy.

## Immediate source-backed integration case and preserved failure

Use the already acquired kettlebell push-up as a concrete person-plus-equipment case, not a new permanent product restriction:

- Original: `data/exercise/kettlebell-pushup.webm`, SHA-256 `4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`; attribution at `artifacts/exercise-diversity-kettlebell-pushup-02/source.json` (Taco Fleur/Cavemantraining, CC BY-SA 4.0).
- Useful sealed mobile-body input: `artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run`. It has 121 references, root/mobility metadata and no camera-reference field. Preserve all 257 source slots, including 129 sampled and 128 unsampled pose slots. The older anchored `exercise-v1` result is not the mobile body.
- `observations.adapt_exercise` currently emits `anchored-pelvis` / `anchored_support` when a scene is provided, including mobile input. Correct this with profile-aware topology/support and focused regressions; do not rewrite old sealed exports.
- **Source preparation failed safely before encoding.** `artifacts/kettlebell-tracking-input-01` preserves the original, attribution, script snapshot, commands, logs, failure and all 257 native frame hashes. FFmpeg `-copyts` gives **3–8536 ms**, while the prior decoder-relative inspection gave **0–8533 ms**. Both clocks must be represented and reconciled explicitly. No derivative, successful mapping, report or seal exists. Do not rerun the unchanged expectation or silently remove the 3 ms origin.
- The preparation process is terminal; its worker reported no live handle. No API request, payload preparation, media upload, key read or ledger change occurred. Recheck other process/lock state before launching shared jobs.

Next safe sequence: inventory the inherited diff and current processes; finish/review the shared-world runtime using existing analytic fixtures; fix profile-aware observation export and exact evidence binding; resolve the clock origin in a new source-preparation output; then obtain/verify source-linked person/equipment evidence and construct one uncertainty-labeled coupled candidate. Preserve unsampled/missing slots and all rejected attempts. Do not replace absent equipment evidence with authored props while calling them recovered.

If Meta video tracking is useful, use the existing guarded `segmentation_lab.track_video` path, explicit initialization prompts and verified original/derivative mapping. Confirm current pricing and a bounded per-request allowance first. A prior conservative plan for 257 frames and three objects estimated up to **$0.1542**; it is not an executed request or new budget. Never submit this failed/unmapped preparation.

## Operating constraints and spending

- Total Meta development authorization remains **$5 for the entire project**, shared by both adapters and all workers. Last verified ledger: eight settled calls, **$0.186152 committed**, **$4.813848 remaining**, zero unsettled. Recheck before execution; reserve before requests, retain unknown charges, avoid duplicate requests and never reset/copy the ledger to obtain more spend. Do not expose keys in prompts, artifacts, Git or subprocess logs.
- The separately rejected jumping-jacks media upload remains unapproved. Do not retry or reroute it without that specific authorization. A general budget is not permission to bypass an automatic review rejection.
- Estimate new paid work and aggregate retries/verification/storage/compute before execution. Prefer reusable local evidence and small bounded screens. No unapproved service subscription, provider-account/key creation or new recurring infrastructure spend.
- Keep the hosted website offline. No deployment, public-repository publication, hackathon submission, application, external message/post, credit form or agreement acceptance is authorized.
- No browser/computer interaction unless the user requests it. Never run `npm run dev`; the user manages development servers. CLI/API verification is preferred. Do not rename the session unless asked.
- Use least privilege. Do not borrow sibling Loom credentials, data or implementation without explicit authorization and attribution. Weft is a distinct product; preserve its identity and existing approved assets.
- Work on `main` unless the user requests another branch. Pull current remote changes safely after inspecting dirty state; never overwrite inherited work. Follow applicable merge/approval instructions. Periodically checkpoint tested, scoped changes and publish authorized changes; verify the remote result. Do not use blanket staging, stashing, `git clean`, force pushes or history rewrites to obtain a clean status.
- Review stashes individually and preserve unique contents before authorized cleanup. Retain saved runs, original videos, failures, manifests, local ledgers and verified stash backups. Do not publish secrets, environment directories, caches or generated research artifacts. Never mention Codex in commit messages.

## Execution, durable handoff and definition of done

Use safe bounded parallelism where tasks are genuinely independent. Assign non-overlapping files and unique job/output directories; one integrator coordinates shared schemas, dependencies, evaluator boundaries, budget and Git publication. Do not restart an existing live job or race mutable outputs. A blocked experiment must not stop useful work elsewhere in the combined goal.

For each meaningful milestone:

1. State the concrete hypothesis or user-visible outcome, inputs, owner, resource/cost bounds and acceptance checks.
2. Implement the smallest coherent improvement across the pipeline; verify relevant regressions, exact artifact integrity and representative performance. Reuse unchanged passing checks; do not retrain or rescan without new inputs or a justified hypothesis.
3. Preserve both successes and failures with exact commands/settings, source/model/provider versions, prompts/tool calls where applicable, seeds, hashes, timestamps, latency and costs. Use new output directories and keep source/scored runs immutable.
4. Update `TODO.md` and the current handoff with completed versus pending work, changed files, evidence identities, known limits, live job handles/locks, budget, Git state and next safe action. Mark old track notes historical where needed. Critical context must live in the repository so another developer can continue without chat memory.
5. Keep the local workflow usable and provide a short reproducible command path for testing it. Report what is verified, what is simulated/authored/inferred, and what still lacks independent evidence. Use America/Los_Angeles timestamps and plain language.

Carry this combined goal through implementation and relevant verification. Do not stop at a split plan, a single successful clip, a polished replay, or isolated simulator tests. The intended outcome is one understandable, evidence-grounded video-to-world-to-RL system whose interactions and recordings work end to end, whose learning and limitations are independently inspectable, and whose remaining physical uncertainties are never disguised as completed reconstruction.
