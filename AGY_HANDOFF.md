> **Historical split assignment — superseded October 29, 2026.** The user closed agy and assigned both tracks to this session. Use the [combined goal/current handoff](docs/handoffs/combined.md); preserve the implementation evidence below, not its obsolete ownership restrictions.

# Weft: teammate handoff for agy

## Your assignment

Use the complete [Markdown /goal prompt](AGY_GOAL_PROMPT.md) to start the other session. The temporary text copy was removed at the user's request. This handoff supplies the detailed project map and first-task acceptance evidence.

You own **Track B — simulation, RL and the unified local lab**. Codex owns **Track A — video understanding and scene reconstruction** in another session. Work in parallel on **one generalizable system**, not an exercise-only fork or a separate objects product. Follow [the ownership and interface contract](docs/parallel-workstreams.md).

Track A has resumed under the user-installed scoped reconstruction goal. You can progress independently with existing simulators, contracts and analytic fixtures. Check the current A handoff for ownership and active work; do not spawn another copy of Track A or repeat its research.

**Mandatory:** document critical information in [your handoff note](docs/handoffs/agy.md) throughout the work. Another developer must be able to resume without your chat. Record decisions, files, commands, verified results, artifacts, failures, active jobs, blockers, Git state and the next action. Update before ending the session. Do not merely say “done” or leave the explanation only in conversation.

## What Weft is

Private repository: `kintsugi-software/weft`. Canonical local checkout: `/Users/ivan/kintsugi/weft`. Weft is its own Kintsugi product, separate from sibling Loom infrastructure. Its goal is:

**Provide a video → inspect every processing step → construct an executable shared world → interact and train through an RL interface → record/replay actions and results → independently test and improve reconstruction.**

The user prioritizes exercise videos with a 3D body, joints and illustrative muscles, while also requiring balls, tabletop objects and broader videos. One observed video cannot uniquely identify every hidden depth, material, force or intention. Preserve those uncertainties instead of inventing physical truth.

Working local capabilities include:

- Python CLI and loopback lab, stepwise artifacts, source/pose inspection and visible failures.
- Pymunk planar rigid scenes; MuJoCo 3D primitive scenes and articulated exercise bodies.
- Manual controls, authored reference-following controllers, bounded learning tasks and action/state recording with offline replay/video export.
- Source hashes, evidence partitions, immutable completed-run manifests and retained rejected hypotheses.
- Meta reasoning/segmentation adapters with one persistent shared spend ledger.

Still incomplete: reliable arbitrary-video reconstruction, calibrated camera/depth/shape, stable physical point/body identity, reliable floor/contact geometry, broad whole-exercise control, and independent generalization/physical transfer. A working replay or engine is not proof the source world was recovered accurately. Muscle bands illustrate modeled drive, not measured activation.

Rough engineering estimate: about **50% of the full goal**, about **85% of a usable local demo**. These are judgment estimates, not measured accuracy. Track A is roughly 35–45% complete; Track B roughly 70–80%. Accuracy and generalization remain the major uncertainty.

## Read first, in this order

1. [AGENTS.md](AGENTS.md): authorization, preservation, Git and runtime rules.
2. [Parallel workstreams](docs/parallel-workstreams.md): ownership, interfaces and coordination.
3. [TODO.md, session handoff](TODO.md): latest verified state, active/paused status and remaining gates. Historical sections describe facts at their original dates.
4. [Architecture](docs/architecture.md) and [iteration workflow](docs/iteration.md): actual implementation versus intended design.
5. [Local lab](docs/local-lab.md), [3D rigid sandbox](docs/spatial-rigid.md), [whole-body control](docs/whole-body-learning.md), and [scene observations](docs/scene-observations.md).
6. [Full product goal](goal.md) and [v1 milestone](v1.md). Hackathon materials are reference evidence, not permission to deploy or submit.

Do not spend your first session rereading every historical study. Use the map below and relevant tests to start the assigned milestone.

## First independent milestone: one lab for existing 3D worlds

**Bring `spatial-rigid-v1` into the existing local lab's inspect/control/record workflow, without changing perception or claiming a new reconstruction capability.**

Why this is ready: `SpatialRigidEnv`, `scene-sandbox`, recording dispatch and offline 3D replay already exist and have analytic tests. However, `Lab.load_session` and `prepare_lab` currently dispatch only rolling/planar/appearance/exercise, so the 3D rigid workflow is separate. You do not need a new model, new dataset, API key or research result to close that integration gap.

Start with a validated authored scene fixture, then a copied source-linked candidate if the local saved artifacts are available. Keep assumptions/failures visible. Proposed work:

1. Add an explicit adapter/import path for an existing `spatial-rigid-v1` scene/run into a **new** lab directory; do not overwrite or relabel the source run.
2. Expose the existing eight controls and complete 3D state/contact details. Reuse the engine and recording implementation rather than building another simulator or journal.
3. Support reset, pause, stepping, bounded continuous interaction, record/stop/export and reopening the recording. Distinguish viewing a recorded state from advancing physics.
4. Make existing rolling/planar/exercise modes keep working. Show the actual model's capabilities and limitations; do not pretend every profile has identical actions.
5. Keep expensive parsing, export and simulation work off request/UI hot paths where needed; bound queues and cancel stale work. Preserve loopback/token/Host/Origin protections.
6. Document the exact user command, supported controls, verification and limits in your own note and the relevant lab guide. Request coordinator edits to shared README/TODO when ready.

Acceptance evidence:

- A deterministic authored 3D scene opens in the same lab; lateral/vertical forces, rotation, contacts and reset behave consistently with `SpatialRigidEnv`.
- A recorded sequence, including reset, replays exactly; all source run hashes remain unchanged and recordings live under new `sessions/` directories.
- Existing planar/exercise interaction and recording checks still pass. New tests cover profile dispatch, action bounds, lifecycle and failures, not just copied implementation details.
- Mocked UI checks and bounded HTTP tests pass. Browser interaction is not authorized unless the user requests it; do not call mocks real browser QA.
- Measure representative interaction/serialization/export workload before claiming responsiveness. Preserve hidden-page pause, pending-request cancellation and resource cleanup.
- No paid calls, source relabeling, provider configuration, deployment or scorer threshold changes.

This milestone is integration progress, **not** completion of generalization or of Track B. Subsequent B work includes a stable environment capability/task interface, whole-body control/learning and complete multi-object inspection, coordinated with A's world candidates.

## Code map and ownership

| Area | Key files | Owner |
| --- | --- | --- |
| Source/pose/mask/feature evidence and camera/shape inference | `observations.py`, `*vision.py`, `feature_tracks.py`, `surface_features.py`, pose/camera/segmentation modules | A; read, do not reimplement |
| Planar and 3D rigid execution | `scene.py`, `spatial_rigid.py`, `scene_console.py` | B |
| Articulated execution and learning | `human.py`, `human_task.py`, `human_learning.py`, `whole_body*.py` | B; coordinate public schema changes |
| Lab lifecycle and API | `lab.py`, `lab_worker.py`, `cli.py` | B |
| Viewer and replay UI | `assets/lab.html`, `lab.js`, `lab.css`, `spatial-replay.html` under `src/weft/` | B |
| Recording and export | `recording.py` | B; preserve format/integrity |
| Cross-track contracts, dependencies, independent graders, provider ledger | See shared-file list in `docs/parallel-workstreams.md` | Coordinator; request exact changes |

`human.py` currently mixes body construction with execution. Do not make this a reason for both agents to edit it: B owns the file, A adds external construction adapters against its frozen validator. Apply the same rule to other mixed-responsibility files.

## Getting started without private artifacts

The [collaboration checkpoint](docs/checkpoints/2026-10-29-collaboration.md) passed 2,027 distinct Python checks and all three scripted UI suites on an isolated candidate-file snapshot. Tests requiring loopback sockets passed after the sandbox restriction was lifted; this was not an application defect. Verify your own checkout and relevant changed inputs rather than treating this historical result as permanent coverage.

The full lab uses `.venv-lab` and `requirements-lab.lock`. Do not modify the original `.venv`. Python 3.14.7/macOS arm64 is the verified platform; portability to another platform is not established merely by the lockfile. Follow [environment setup](docs/local-lab.md#reproduce-the-environment) for a fresh checkout; do not reinstall a working environment.

These checks construct their own scenes and need no provider key or private run directory:

```sh
.venv-lab/bin/python -m pytest tests/test_spatial_rigid.py tests/test_spatial_recording.py tests/test_scene_console.py
.venv-lab/bin/python -m pytest tests/test_lab_http.py tests/test_lab_recording.py
node tests/lab_ui_smoke.mjs
node tests/spatial_replay_smoke.mjs
```

Read the fixture constructors in `tests/test_spatial_rigid.py` and `tests/test_scene_console.py` for complete valid scenes. Do not make production code depend on the test modules. Build a small separately labeled authored fixture for your integration demonstration.

If working on the canonical machine, existing optional examples are:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/heavy-squat-body-01/lab
.venv-lab/bin/python -m weft scene-sandbox --run artifacts/ramp-conditional-depth-01/candidate --record artifacts/track-b-manual-01
```

The user starts/manages the lab server. Do not run `npm run dev`. `artifacts/`, `.local/`, environments and model caches are local-only and **will not arrive in a Git clone**. Do not repeat acquisition/inference/training just because a named example is absent; use analytic fixtures for this milestone or request a specifically needed evidence bundle. Do not claim the fresh clone has the prepared exercise demo.

## Current research evidence you must preserve

- Fixed-reference feature study: 684 source frames across five views of four already exposed scenes. Missing matches and tiny-ball failure remain. `docs/surface-feature-diversity.md`.
- Adjacent appearance histories: 17,283 links / 35,596 raw current-point slots, 14 explicit original-reference conflicts, all 1,090 object/frame slots retained. Side-view box links 36/36 later frames; van 352/353. These are image hypotheses, not verified physical point identities or unbroken tracks from initialization. Tiny balls still have no persistent feature tracks.
- Known-warp control: 75 synthetic frames; adjacent point-error p95 0.147 px, original-anchor p95 0.743 px. This proves the controlled software case, not real 3D accuracy. Twelve new tracker tests pass. `.local/feature-track-diversity-verification-01.json` verifies the saved graph/arithmetic and 684 video frames.
- A conditional box-depth change improved one camera's center error but worsened both side-view alternatives. No promotion. `docs/conditional-rigid-camera.md`.
- Human multi-view/length candidates retain missing joints, camera and ground conflicts; no anatomical/contact accuracy claim. `docs/freemocap-three-view.md`, `docs/freemocap-fixed-lengths.md`.
- Exposed development clips, the failed repeat trial and Cam3 must not become fresh holdouts again. Independent evaluators, held-out data and success definitions remain outside a repair agent's write authority.

Do not restart A's completed studies or discard their failures. New B experiments get a unique `artifacts/track-b-...` directory with hypothesis, resolved settings, versions, input identities, commands and terminal report.

## Budget, safety and publishing boundaries

- **$5 total Meta development budget shared across all sessions**, eight settled calls / $0.186152 last verified. Do not initialize another $5 ledger or copy the key. Your first task is entirely local and needs no model calls.
- A specific jumping-jacks media upload to Meta was rejected before execution and remains separately unapproved. Do not retry or reroute it. This does not block local implementation or the assigned task.
- Hosted Weft remains intentionally offline. No deployment, submission, post, message or agreement acceptance is authorized by this handoff.
- Generated code remains untrusted; keep resource limits, no default network/secrets, and independent evaluation boundaries.
- Follow AGENTS Git/approval rules, preserve other work and immutable evidence. No destructive cleanup or force push. Work in a separate checkout when possible; do not race the coordinator on shared files or main pushes.

## What to return at each checkpoint

Update `docs/handoffs/agy.md`, then report: **what changed; exact tests and measured results; what remains wrong/unverified; artifact paths and hashes; interface requests; commit/push/stash state; and the next safe task**. If interrupted, document partial work and live handles before stopping when possible. Never leave essential knowledge only in your final chat message.
