# Weft workstreams and integration

## Current ownership — October 29, 2026, 7:04 PM PDT

The user closed agy's **[CURRENT] Integrate Spatial Rigid 3D** session and transferred both tracks to this session. Codex now owns A and B, including agy's uncommitted shared-world/runtime work. The [complete combined goal](handoffs/complete-goal.md) supersedes the A-only objective, old cross-session edit prohibitions and earlier pause. Review and finish existing work; do not duplicate it or wait for the closed session. Evidence partitions, independent scoring, immutable artifacts, budget and source/runtime interface constraints still apply.

The sections below retain the **historical parallel assignment and interface contract**. Their file split remains useful for future non-overlapping delegation, but it no longer assigns work to an active agy session. The latest complete goal records the inherited diff and next safe integration steps.

Use the [combined implementation handoff](handoffs/combined.md) for current verified work, costs, live-handle status and remaining integration; historical snapshots below are retained for provenance.

## Objective and assignments

Build one generalizable **video → evidence → executable shared world → RL task → inspection/recording** system. Balls, tabletop scenes and exercise videos are validation cases, not separate products. Existing profile-specific engines are implementation adapters; they must not become disconnected pipelines or independent worlds for objects in the same scene.

| Track | Owner | Owns | Does not own |
| --- | --- | --- | --- |
| A — Video understanding and scene reconstruction | **Codex, this session** | Source acquisition/provenance, segmentation/pose observations, tracking/identity, camera and shape inference, evidence-backed world candidates | Lab UI, runtime/session dispatch, recording implementation, reward definitions or independent graders |
| B — Simulation, RL and the unified lab | **agy, teammate/session** | Executable environment adapters, constraints/contacts, RL task/control interfaces, lab/session integration, inspection and recordings | New tracking/pose/camera algorithms, re-labeling source evidence, new source acquisitions or duplicating A's experiments |

Track A is **active under the user-installed scoped goal**. Track B has committed runtime milestones 1–4; its mixed rigid/articulated implementation is now in progress, with unfinished files kept outside A checkpoints. Both sessions use the canonical checkout, so commit/stage only explicitly owned paths and preserve the other session's unfinished files. Do not use a blanket add, stash or cleanup to obtain a clean tree. Their implementation work remains independent across agreed interfaces. Start with the [self-contained agy handoff](../AGY_HANDOFF.md).

## The dependency boundary

```mermaid
flowchart LR
    V[Videos and source evidence] --> A[Track A: observe and reconstruct]
    A --> C[Versioned evidence and world candidate]
    C --> B[Track B: execute, learn and inspect]
    F[Frozen analytic fixtures] --> B
    B --> D[Independent checks and recorded discrepancies]
    D --> A
```

**B starts now against existing validated scene formats and analytic fixtures.** A improves reconstruction against the same boundary. Neither waits for perfect perception, invents another scene format privately, or claims synthetic fixtures prove physical accuracy.

Current interfaces are real but **not yet one universal world schema**:

- `observations.py`: `scene-observations-v1`, schema versions 1–3. Preserve source hashes, original clocks, units, object IDs, visibility, evidence partitions and observed/inferred/assumed/unknown claims. Complete observation exports include evaluation evidence; filter before supplying a reconstruction/repair model.
- `scene.py`: `planar-rigid-v1/v2`; `spatial_rigid.py`: `spatial-rigid-v1`; `human.py`: `exercise-v1/v2/v3`. Consume their validators and explicit units; do not reinterpret normalized lengths or model coordinates as calibrated meters.
- Gym-compatible environments expose `reset`/`step`; saved snapshots, action definitions, termination and task semantics remain explicit. Kinematic replay, authored servo, learned policy and physical simulation remain distinct.
- Completed `run/` artifacts are immutable. Mutable lifecycle status and new `sessions/` recordings remain outside their seals. A new evidence/model/profile version requires explicit migration, not mutation of old artifacts.

A cross-track API/schema change requires a short proposal in this file: exact fields/signatures, owner, migration and contract tests. Coordinate it before editing. Until agreed, use existing contracts and opt-in adapters.

### Shared-world boundary review — October 29, 2026

Integrator review accepts B's **one `MjModel` / `MjData` and one simulation clock** direction. Its original example is not a verified runtime contract; B now owns the in-progress compiler and schema. A owns the following construction-only boundary and does not alter B's scene/action/recording schemas:

1. `world_evidence.load_world_evidence({observation_document_sha256: raw_json_bytes})` validates exact hashes, the existing observation format, and complete decoded object/frame coverage. Qualified identities are `(observation_document_sha256, object_id)`, not bare `object-0`. `.summary()` returns an inspection snapshot; `.construction_bytes(allowed_partitions=...)` preserves all slots with explicit withheld stubs and excludes unauthorized evidence/relationships/properties. Default construction scope; only construction/generation/development-validation may be requested, never audit. Raw documents remain unchanged. Maximum eight documents / 64 MiB aggregate; no network or model calls.
2. `world_alignment.load_world_alignment(document, evidence_ids_by_document=...)` validates the separately versioned `world-alignment-v1` proposal. Required fields are `schema_version`, `profile`, `from_frame`, `to_frame`, `spatial`, `clock`, and `claims`. Frame descriptors carry `id`, `length_unit`, `axes`; spatial values carry positive uniform `scale`, proper `rotation_wxyz`, and `translation`; clock values carry `rate`, `source_origin_s`, `target_origin_s`. Each spatial/clock claim carries `status`, `method`, `uncertainty`, and qualified `evidence_refs`. Unknown mappings remain null and cannot transform points. Source/target frames and units must match at use time.
3. Coordinate law: `p_target = scale * R * p_source + translation`; clock law: `t_target = rate * (t_source - source_origin_s) + target_origin_s`. Velocity scales by `scale/rate`, acceleration by `scale/rate²`, angular velocity by `1/rate`, orientation by left multiplication with the frame rotation. No shear, reflection, automatic floor shift, origin removal or implicit resynchronization is permitted. A supplied mapping is a hypothesis, not recovered calibration.

A implements these helpers and focused tests in separate owned files; it does not compile or step a world. B consumes approved bindings when extending its proposed shared-world compiler. Migration remains explicit and opt-in: A leaves old profiles/validators untouched; B must verify its separate mixed-world capability before claiming support. Body-local origin removal, shape dimensions, gravity, materials, actuator parameters, task tolerances and newly coupled contacts require separate migration evidence; position conversion alone cannot claim preserved dynamics.

**Required A checks:** exact hash/clock/partition retention; complete missing slots; no aliasing across documents; no audit payload escape; known position/orientation/velocity/acceleration transforms; wrong units/direction and unknown alignment rejected; immutable inputs. **Required B checks:** one-model ball/forearm momentum coupling with a non-contact control; initial-penetration rejection without moving bodies/floor; passive bodies not driven by source positions; exact combined replay/reset/actor state; preserved pure-profile behavior where claimed. These analytic checks do not establish physical accuracy.

Integration corrections for B's current note: rigid distances use **`scene_width`**, not measured metres; masses are relative, and human dimensions are uncalibrated model estimates, not necessarily normalized body height. The verified `squat-changed-eval/manifest.json` SHA is `b431173761bd42a49fc5e41f801b9487d1ad9c086c821c54261e044c4bd8d1d9` (the earlier handoff line differs). Six output manifests / 179 files verify in `.local/track-b-publication-manifests-01.json`; no training was repeated. Preserve B's actual success failures and floor-penetration results.

## File ownership

### Integrated camera-body extension — November 2, 2026

The combined-track integrator extends existing shared-world-v6 admission with
`camera-shared-coupling-v2` for existing validated multi-view root/body methods.
The original two-view-root coupling v1 remains unchanged. The same body, source
clock, cameras, original parent, construction audit and fit settings are pinned;
no scorer, floor gate, force or geometry is modified by import. Shared-lab
runtime snapshots add the reconstructed-body validator dependencies. Existing
saved labs retain their original runtime; new import is opt-in. Qualification
covers two/three/four views, input tampering, missing observations, native
parameters/geometry, floor rejection and exact recording/replay. Root owns
both sides of this interface. [Results and remaining work](handoffs/reconstructed-camera-coupling.md).

The same reconstructed bodies now support the existing external joint-feedback
learner and policy lab/recordings. Pure receipt and policy admission inventories
include the exact v2 body validators; v1 requirements and all actor/scorer/physics
arithmetic remain unchanged. Root owns both sides. [Verification, playable
command and next unfamiliar clip](handoffs/reconstructed-camera-learning.md).

### A: exclusive implementation area

`src/weft/*vision.py`, `observations.py`, `segmentation_evidence.py`, `segmentation_masks.py`, `segmentation_observations.py`, `surface_features.py`, `feature_tracks.py`, `regional_features.py`, `joint_tracking.py`, `pose_*`, camera/triangulation/correspondence modules, and source-derived reconstruction/fit studies. Corresponding source-label metadata, focused tests and study docs belong to A.

Do not change B-owned `human.py` to add an inference experiment: place construction adapters in A-owned modules and use the existing body validator. Likewise, scene-parameter inference must not rewrite B's simulator or independent evaluator.

### B: exclusive implementation area

`src/weft/lab.py`, `lab_worker.py`, `cli.py`, `recording.py`, `scene_console.py`, `scene.py`, `spatial_rigid.py`, `human.py`, environment/control/learning adapters, `whole_body*.py`, and `src/weft/assets/lab.*` / `spatial-replay.html`. Corresponding runtime, HTTP, recording and mocked-UI tests belong to B.

Public schemas/signatures in these files are frozen across tracks until coordinated. B may improve implementation without reclassifying inferred source values as measured facts or changing fixed accuracy gates.

### Coordinated files, one writer at a time

`AGENTS.md`, `README.md`, `TODO.md`, `goal.md`, `v1.md`, `docs/architecture.md`, `docs/iteration.md`, this workstream contract, `pyproject.toml`, dependency locks, `pipeline.py`, shared I/O, provider/budget adapters and independent scoring code. The integrator owns these edits; request the specific change instead of racing another session. Do not silently edit another track's files under a broad glob above; declare any ambiguous ownership first.

Each track keeps its own handoff note and uses **new** experiment directories, e.g. `artifacts/track-a-<study>-01` and `artifacts/track-b-<study>-01`. Never share a mutable output directory, reuse another job's lease, rerun a live job, or install dependencies while another session is using that environment.

## Parallel operation and Git checkpoints

- Prefer separate local checkouts. This repository currently requires work on `main` and no new named branches without an explicit user request; this handoff does not silently override that rule. Do not create branches/worktrees here as part of reading it.
- Assign exact files before editing. Only the designated integrator publishes to shared `main`; other sessions deliver a reviewable diff/commit and test evidence. Observe the repository's approval rules. No force-push, history rewrite, deployment or unrelated cleanup.
- At a meaningful completed/tested milestone, and before a session handoff, commit/push the authorized checkpoint rather than accumulating days of dirty work. Reuse passing checks whose inputs are unchanged. Check remote state before integration and verify it afterward.
- Keep secrets, environments, model caches and generated runs out of Git. Retain source attribution and required fixtures. Do not delete local evidence to obtain a clean status.
- Inventory stashes before cleanup. Preserve unique changes in a commit or verified local backup, then remove only the reviewed stash under the user's cleanup authorization. Never use blanket `stash clear` or `git clean` to hide unfinished work.
- Meta spend is **one project-wide $5 budget**, not $5 per session/checkout. Last verified total is eight settled calls / $0.186152; the canonical ledger is local. B's first task needs no key or paid call. A second checkout must not create a fresh budget ledger and reuse the key. Coordinate all paid execution through the canonical reservation path; never copy keys into a handoff or Git.

## Mandatory durable handoff

**Both sessions must document critical context as they work so another person can continue without this chat.** A owns `docs/handoffs/track-a.md`; B owns `docs/handoffs/agy.md`. Update at each completed milestone, before a long-running job, when blocked, and before ending a session. Record:

1. Objective, owner, current checkout/branch/revision and timestamp in America/Los_Angeles.
2. Completed versus pending work and exact changed files; distinguish implementation, test verification and real-world accuracy.
3. Decisions and rejected approaches that would otherwise cause repeated work; preserve failures and reasons.
4. Exact commands, configuration, dependencies and artifact/source/manifest identities needed to reproduce the result.
5. Tests/checks with results and remaining limits. Link immutable evidence; never treat a green unit test as a real-world guarantee.
6. Any live process/job handle, ownership/lock, output directory, progress and stop/resume procedure. Re-check liveness; a status file is not proof a job is running.
7. Blockers, cross-track interface requests, budget use and the next safe action.
8. Commit/push/stash state. List any uncommitted work and its owner; do not imply another person can fetch local-only artifacts.

Never include credentials, tokens, private auth state or whole chat transcripts. Keep notes concise and current; archive historical detail by reference rather than repeating it. Promote cross-track decisions into `TODO.md` and this contract through the integrator.

## Completion means the original goal

Neither track independently declares Weft complete. Integration must show the same workflow handling diverse previously untuned videos, complete source/evidence coverage, coherent multi-object/articulated worlds, usable RL controls and learning, recording/replay, and independent accuracy checks—including observed changed-action outcomes for physical-transfer claims. A feature demo, synthetic score or one improved camera view is insufficient. Illustrative muscles need not become medical measurements; their illustrative status must remain clear.
