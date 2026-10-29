Own and complete Track B of Weft: the shared simulation, reinforcement-learning and inspection/recording runtime for one generalizable video-to-environment system. Implement and verify the work, not just a plan. Continue beyond the first milestone until the Track B acceptance criteria are satisfied, while preserving the broader Weft goal and coordinating cross-track dependencies.

## Project and collaboration context

Weft is a private project at `kintsugi-software/weft`; the canonical checkout is `/Users/ivan/kintsugi/weft`. The product goal is: provide a video, inspect every processing step and its artifacts, construct one executable world, interact/train through an RL interface, and record/replay actions and results. Balls, tabletop scenes, exercise videos and broader videos are test cases for ONE system—not separate products. A person and objects in the same scene must ultimately share coordinates, time, contacts and physical rules; do not stitch independent simulations together.

Codex owns Track A: video understanding and scene reconstruction—source provenance, segmentation/poses, tracking/identity, camera/shape estimation and evidence-backed world candidates. That session is currently paused. Do not restart its studies or duplicate its algorithms. Your Track B can progress independently using existing scene contracts and analytic fixtures; do not wait for perfect perception.

Read `AGENTS.md`, `AGY_HANDOFF.md`, `docs/parallel-workstreams.md`, the current handoff at the top of `TODO.md`, `docs/architecture.md`, `docs/iteration.md`, and `goal.md`. Then read the lab/runtime guides relevant to your first task. Verify the current checkout and remote state instead of relying on this prompt's snapshot.

## Current implementation and limits

Existing code includes a Python CLI/local lab, Pymunk planar scenes, MuJoCo 3D rigid and articulated-human environments, manual controls, authored reference controllers, bounded RL learning tasks, exact state/action recordings and offline replay/video exports. Source evidence, assumptions, missing observations and rejected candidates are preserved. There are Meta adapters with a shared persistent budget ledger.

This is not yet reliable arbitrary-video reconstruction or validated physical transfer. Camera/depth/shape, identity, human floor/contact geometry and full-exercise control remain incomplete. Muscle bands are illustrative—not measured activation. Kinematic pose playback, authored servo, learned policy and physical simulation must stay distinct. Current profile-specific schemas are implementation adapters, not a finished universal world model. Do not turn the current supported types into a permanent restriction of the product goal.

## Ownership: avoid redundant work

You own runtime/lab/recording work in `src/weft/lab.py`, `lab_worker.py`, `cli.py`, `recording.py`, `scene_console.py`, `scene.py`, `spatial_rigid.py`, `human.py`, runtime/control/learning adapters, `whole_body*.py`, and the lab/replay assets. You own their focused tests and your own handoff note.

Do not edit or rebuild Track A's vision, pose, mask, tracking, camera, triangulation, feature-history or reconstruction algorithms. Do not reacquire its datasets or rerun completed experiments merely to orient yourself. `human.py` mixes construction and execution; you own that file while A adds external construction adapters against its agreed validator.

Public schemas/signatures, independent graders, provider/budget logic, dependency locks and shared top-level documents are coordinated interfaces. Propose exact changes, migrations and contract tests in your handoff before crossing the ownership boundary. Do not silently change action meanings, units, evidence statuses, evaluator thresholds or success definitions. The repair proposer cannot alter its independent judge.

## First milestone: existing 3D scenes in the unified lab

Integrate `spatial-rigid-v1` into the existing local lab's inspect/control/record workflow. `SpatialRigidEnv`, `scene-sandbox`, recording dispatch and offline 3D replay already exist; the lab currently dispatches only rolling/planar/appearance/exercise. Reuse the existing implementations instead of creating another simulator, UI application or recording format.

1. Add an explicit opt-in way to import/open a validated existing 3D scene/run in a NEW lab directory. Keep the original run immutable and preserve its provenance and limitations.
2. Expose the existing eight controls, full 3D state/orientation/contact inspection, reset/pause/step and bounded continuous control. Viewing a replay must not advance physics.
3. Support record, stop, export, reopen and exact replay, including resets. New recordings belong under `sessions/`, outside completed `run/` seals.
4. Preserve existing rolling, planar and exercise workflows. Present profile capabilities honestly instead of pretending all modes have identical actions or trained policies.
5. Preserve loopback authentication and Host/Origin checks, bounded work, hidden-page pause, cancellation and cleanup. Keep expensive parsing/export off request/UI hot paths. Measure representative responsiveness before claiming optimization.

Use authored fixtures from the existing test constructors to start; production code must not depend on test modules. A source-linked saved candidate is optional additional evidence, not a prerequisite. No new model/API call is needed.

## Continue toward Track B completion

After that milestone, continue toward a stable shared environment/task capability interface across the supported rigid and articulated representations; useful multi-object/body controls; whole-body learning with matched baselines and changed initial conditions; coherent inspection of source evidence versus simulation; and reliable bounded recording/export/recovery. Plan mixed rigid/articulated-world requirements with A rather than creating disconnected products. Document unsupported cases and propose required shared-contract changes without blocking unrelated ready work.

Completion requires reproducible evidence: valid scenes execute through the shared workflow; controls alter the intended physics; reset and observations/actions/rewards/termination are explicit; recordings replay exactly; original artifacts remain intact; failed or unsupported inputs are inspectable; existing modes retain coverage; and meaningful learning comparisons do not imply unverified real-world accuracy. A polished viewer, synthetic fixture or one-joint controller alone does not complete Track B or all of Weft.

## Verification and environment

Use `.venv-lab` and `requirements-lab.lock`; preserve the original `.venv`. Python 3.14.7/macOS arm64 is the verified platform. Do not reinstall a valid environment or assume the lockfile proves portability.

Useful starting checks:

    .venv-lab/bin/python -m pytest tests/test_spatial_rigid.py tests/test_spatial_recording.py tests/test_scene_console.py
    .venv-lab/bin/python -m pytest tests/test_lab_http.py tests/test_lab_recording.py
    node tests/lab_ui_smoke.mjs
    node tests/spatial_replay_smoke.mjs

Run checks appropriate to changed inputs; do not rerun unchanged passing work unnecessarily. Some HTTP tests require temporary localhost socket access. Mocked UI tests are not browser/device QA. Do not use browser interaction unless the user requests it, and never run `npm run dev`; the user manages development servers.

`artifacts/`, `.local/`, environments, model caches and credentials are LOCAL ONLY and absent from a Git clone. Do not assume the prepared exercise example or research exports are present. Use analytic fixtures for your first milestone or request a specifically necessary evidence bundle. Preserve all source/evidence IDs, original clocks, complete object/frame coverage, missing observations and uncertainty. Previously exposed clips, Cam3 and the failed object repeat are development data, not fresh holdouts.

## Durable documentation is mandatory

Maintain `docs/handoffs/agy.md` so another person can continue without your chat. Update it at meaningful milestones, before a long-running job, when blocked and before ending the session. Record:

- Current objective, checkout/branch/revision, owner and America/Los_Angeles timestamp.
- Completed versus pending work and exact changed files.
- Critical design decisions, rejected approaches, failures and unresolved limitations.
- Exact commands/settings/dependencies and reproducible artifact/source/manifest identities.
- Test outcomes and performance measurements, clearly separated from physical-accuracy claims.
- Live process/job handles, ownership/locks, output directories and stop/resume procedures; verify liveness rather than trusting a status file.
- Cross-track requests, blockers, budget use and the next safe action.
- Commit/push/stash state and any uncommitted or local-only work.

Keep notes concise and current. Never include secrets or whole transcripts. Promote shared decisions through the designated integrator rather than racing edits to README/TODO/contracts. Do not leave critical information only in a final chat message.

## Operational boundaries

Work in a separate checkout where possible and follow the repository's current branch/approval rules. Do not create named branches without the user authorization required by AGENTS. Coordinate publication to shared main; do not race another agent's push. Checkpoint relevant tested work periodically and before handoff. Preserve other work and review stashes before any cleanup; no force-push, blanket stash clearing or deletion of evidence to obtain a clean tree.

All sessions share ONE $5 Meta development budget. Last verified use was eight settled calls / $0.186152. A second checkout must not create a fresh $5 ledger or copy/reuse the key independently. Your initial work is local and needs no paid calls. A specific jumping-jacks upload to Meta remains unapproved: do not retry or reroute it.

Keep the hosted website offline. Do not deploy, submit to the hackathon, post, message others, accept agreements or create paid resources. Generated code remains untrusted and resource-bounded, without secrets or network by default.

Begin by checking repository state, confirming your B-owned files and updating `docs/handoffs/agy.md` with your first concrete task. Then implement and verify it. Continue useful independent work when another task waits. Report what changed, what was actually verified, remaining gaps, artifact/commit references and the next checkpoint; never claim the broader generalization goal is complete merely because this track makes progress.
