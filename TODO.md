# Weft iteration backlog

## Session handoff

- Workspace: `/Users/ivan/kintsugi/weft-v1`, branch `feat/weft-v1`; base revision `a35070c25ec3e00a063e711a0958497d14c6676f`.
- Thread: `01a17bd5-caa3-76a0-ac47-89f4ebb0b005`.
- Last verified: **October 27, 2026, 12:29 a.m. PDT**.
- Status: local iteration foundation is verified; the full physical-video-to-world product is not complete.
- Resume guard: implementation and documentation live on `feat/weft-v1`; check `git status` and remote tracking before resuming. Preserve local changes and all saved runs. The separate `/Users/ivan/kintsugi/weft` main checkout still contains the research archive; do not mistake it for this implementation branch.
- Both setup experiment processes completed with exit code 0. No service, cloud deployment, background trainer, or recurring job was started by this setup.
- Next independent action: specify the general multi-object scene/evidence contract and its fixtures (A1). Final scene choice and physical validation evidence remain undecided; neither blocks contract work.

## Completed and verified setup

- [x] Preserve A–E scope, current component boundaries, and unknowns in [architecture](docs/architecture.md), without equating the single-ball fixture with the full product.
- [x] Add hash-pinned experiment presets, strict bounded settings, explicit hypotheses, and independent training/evaluation seeds. Unknown settings and mismatched source hashes fail before output creation.
- [x] Run both repair-disabled and repair-enabled paths end to end. Each preserved 45 verified artifacts, including 21 implementation/build-file snapshots. See [verification evidence](docs/verification.md).
- [x] Provide matched-evidence run comparison, with explicit refusal to rank RL results from different worlds or changed benchmarks.
- [x] Verify 46 tests, all eight restricted-engine physics checks, dependency consistency, three terminal action steps, viewer asset references, and JavaScript syntax/physics parity. Browser interaction/visual QA was not performed.
- [x] Document reproducible commands, source/license provenance, cost bounds, and acceptance gates. No paid model call, public release, deployment, submission, or merge.

## Preserved baseline and resume commands

- `artifacts/iteration-baseline-02`: no repairs; manifest `a1f1238dd0447cbe761b9945ce8b5686906b955e6cf7297cd8cd567fbb3984aa`.
- `artifacts/iteration-repaired-02`: two accepted repairs; manifest `0b21d7582de846a13cd6f00d093b649d073ac404f70e1ad0df575ad6a93a9300`.
- `artifacts/comparison-02.json`: reconstruction is comparable; RL is not, because each policy was evaluated in its own fitted world.
- `.local/iteration-setup-tests.xml`: 46 tests, zero failures/errors at the observation above.
- `artifacts/run-01` and `iteration-*-01`: historical runs without source-file copies; preserve but do not claim their hashes alone restore overwritten source.

```sh
.venv/bin/python -m weft verify artifacts/iteration-repaired-02
.venv/bin/python -m weft compare artifacts/iteration-baseline-02 artifacts/iteration-repaired-02
.venv/bin/python -m weft experiment experiments/rolling-ball.json --out artifacts/next-unique-run
```

Do not rerun an unchanged passing experiment just to refresh a timestamp. Reverify before relying on edited artifacts or changed implementation. Use [iteration workflow](docs/iteration.md) for changes and comparisons.

## Still required for the product

These are future product milestones, not claims that the iteration setup implements the entire A–E vision. Completion requires the evidence stated for each item.

### A — Video understanding and evidence

- [ ] **A1: General scene contract.** Version object IDs/roles, units/cameras/time, pose/state/visibility, relationships/events, constraints, evidence references, and observed/inferred/assumed/unknown labels. Require fixtures with static, passive, and active objects; reject dangling relationships and inconsistent units/times. Preserve the current fixture through an explicit adapter or migration.
- [ ] **A2: Multi-object perception and context.** After A1, implement primary/secondary-object extraction, persistent tracking through brief occlusion, event order, and contextual interpretation. Require independently annotated tracks/events on multiple supported videos; report missed objects and identity switches, not only a prose summary.
- [ ] **A3: Meaningful Meta perception/reasoning.** After access and a bounded cost estimate are established, run the selected perception/reasoning tools on real input and preserve model IDs, requests, outputs, costs, failures, and material decisions. The existing optional parameter-proposal adapter is not completion. No keys, account setup, or paid calls are authorized merely by this checkbox.

### B — Shared physical world

- [ ] **B1: Multi-object rigid-body simulation.** After A1, generate one scene with a moving object, supporting surface, obstacle, and controlled actor in an established physics engine. Require consistent shared coordinates/contact events and changed-action rollouts; no independent videos spliced together.
- [ ] **B2: Geometry, camera, and dynamics uncertainty.** Separate appearance, collision shapes, physical parameters, and behavior. Require original-view replay, alternate-view inspection, a calibration/assumption record, and at least two plausible hypotheses where evidence is ambiguous. Do not label invented hidden surfaces or mass as observed truth.
- [ ] **B3: Object-focused refinement.** After B1/B2, allow passes focused on individual objects while validating global constraints. Require a test where conflicting object/contact estimates are reconciled or explicitly rejected, not averaged into an impossible scene.

### C — Candidate improvement

- [ ] **C1: Evidence-grounded diagnosis and proposals.** Expand beyond fitting two scalar parameters. Require several bounded hypotheses for a real mismatch, generation/development evidence separation, and recorded accepted/rejected changes under fixed checks. Keep intentionally injected faults labeled as synthetic tests.
- [ ] **C2: Disagreement-driven experiments.** After B1/C1, propose scenarios that distinguish plausible worlds. Require predicted outcomes and a record of whether real evidence exists. Without observed outcomes, retain disagreement instead of declaring the winning simulator physically correct.

### D — Grading and independent evidence

- [ ] **D1: Freeze independent perception/reconstruction benchmarks before optimizing A2.** Require annotations and generation/development/final splits established before exposure, separate visual/motion/contact/physics scores, and fixed tolerances. The current development-inspected second pass is not a final holdout.
- [ ] **D2: Validate changed actions physically.** Choose accessible real evidence (not yet decided), perform/obtain authorized intervention observations, and compare predictions against those outcomes. New simulated scenarios or an AI judge's approval alone leave this unchecked.
- [ ] **D3: Shared RL evaluation world/task.** Evaluate policies trained in alternative generated worlds against the same independent benchmark and seeds. Require matched trials and uncertainty across multiple training seeds. Current 100% parking results do not demonstrate a repair benefit or physical transfer.
- [ ] **D4: Supported-input and resource envelope.** Test missing detections, occlusion, stationary objects, multiple candidates, moving cameras, and near-boundary input sizes. Require explicit unsupported-input failures, bounded process time/memory, representative resource measurements, and no hidden fallbacks. Current frame/episode bounds are not whole-process isolation.

### E — RL task generation and usefulness

- [ ] **E1: Evidence-linked task proposal.** Define controlled actor, observations/actions, resets, goal/reward/success, and termination. Require provenance for inferred versus authored decisions and working `reset`/`step` semantics on B1. The fixture's added parking goal is not task extraction.
- [ ] **E2: Learn a meaningful interaction.** After B1/E1/D3, train and evaluate an actual policy against simple baselines on varied starting conditions. Require task success measured independently of reward; record training time, failures, and scope. Do not substitute prerecorded trajectories or scripted expert actions for learning.

### Later expansion and release

- [ ] Select the first product interaction family (tabletop pushing/sorting or rigid-body collisions) and validation access. Keep the choice separate from the existing ball fixture.
- [ ] Extend to articulated animals/humans, deformable objects, richer environments, and learned actor behavior only after the rigid-body milestone; each extension needs its own evidence and action-validity checks.
- [ ] Explore website workflows later with explicit UI state/transitions/notifications and tasks, not physical-engine assumptions.
- [ ] Prepare hackathon/public-release artifacts only when requested; preserve prior-work attribution and explicit scope. Public visibility, deployment, submissions, and merging remain separate authorization gates.

## Completion gates

**Iteration setup:** a new contributor can locate the scope, run a pinned local baseline/candidate, inspect the shared state and provenance, verify artifacts, compare only compatible results, and pick the next task with explicit acceptance evidence. This is verified by the setup evidence above.

**First product milestone:** one supported physical video produces a multi-object interactive world and task without clip-specific simulator edits; replay and changed-action behavior pass the relevant independent checks; an actual policy learns the interaction. All assumptions and unverified physical claims remain visible. This milestone is **not yet achieved**.
