# Moving and handheld equipment in the existing coupled world

## Active implementation scope — November 2, 2026 (PST)

- Source lane owns only new `moving_equipment_evidence.py` and its tests first: complete bounded frame/identity/clock ledger, no inferred depth or physics.
- Runtime lane owns `dynamic_native_interval.py`, analytic tests, exact-profile dispatch in `native_interval.py`, conditional helper custody in generic `shared_world_lab.py`, and explicit rejection of unqualified new-marker training in old task/generic learning entry points. No physics-mechanics rewrite or legacy weakening is approved.
- Portal lane owns a new explicit generic-world action replay helper and analytic tests first. Existing generic recording lacks qualified exact action replay, and portal import requires the old exercise profile; importer/UI/task wiring remains pending a closed exploration contract. No stationary scorer, parking-reward success or meaningful-learning claim may be substituted.
- Root integrates and owns actual saved-source/native/portal acceptance, budgets, handoff and publication. No new paid request is planned.

The following reviewed architecture is a plan, not implemented capability. Initial depth/shape, source constructor, new task and optional attachment control remain separate unfinished milestones. Do not treat a ledger or partial-action helper as a playable or trained world.

## Decision

Add a distinct **conditional moving-equipment** construction contract, not an exception to the static floor-box constructor. Reuse SharedWorldEnv, exact source body, native clocks, shared contacts, inspector and recording. Keep every source frame/reference, initialization phase, unavailable slot and identity alternative. First deliver an honestly labelled free-body simulation candidate; then add an explicit assisted-attachment candidate if needed. Neither is recovered grip or verified exercise physics.

This is a read-only design. No implementation, API, source processing or native physics was run. Parent owns selection of the next milestone and its acceptance plan. Runtime agent separately diagnoses the current image-consensus refusal.

## Existing capabilities and hard boundaries

- `shared_world.py:_build_world_xml` already gives every non-static object a free joint. `validate_shared_world` admits bounded position, quaternion, linear/angular velocity, mass, friction and box/sphere/compound proxies. `SharedWorldEnv.reset` sets these exact initial states; contacts, substep observations, actions and recording already share the same simulator.
- `action_contract_for_scene` exposes bounded object force channels and states `hidden_assistance=False`. Existing exercise task wrappers forbid those forces. Do not secretly drive equipment along an image trace.
- `equipment_proposal.py:floor_box_from_bounds` and `shared_world_reconstruction.py:_construct_equipment_world` deliberately require a floor-supported box, authored camera/depth, zero initial motion and no equipment reference forces. `equipment_box_consensus.py` deliberately checks consistency across all eligible boxes; it does not promise stationarity or physical rigidity. A moving object should not enter this profile by widening its tolerances or choosing one convenient frame.
- `equipment_root_binding.py` now transfers the original qualified scale and all body roots exactly. Reuse it; do not refit the body, remove the two interpolated references or retime the video.
- `exercise_support_evidence.py:estimate_exercise_support` already proposes **projected wrist/equipment co-motion**, under a fixed-camera prior, using the original construction generation partition. `equipment_hand_evidence.py` and `equipment_hand_constraint.py` expose original wrist/pinky/index/thumb versus mask image intersections/exclusions. These are conditional image relations, not contact/grip/force measurements. Their current adapters and bounds must be checked before reuse on this source; possession of saved points is not an admission bypass.
- `source_contact_trajectory.py` adjusts authored human targets against fixed equipment with source-image/floor constraints. It is not moving-object reconstruction or a grasp controller; do not relabel it as either.
- `perspective_root_depth.py` fits body-root depth under an authored camera, explicitly excluding equipment/contact/task reward. It is not an object-depth estimator. Camera-coupled scene profiles currently reject dynamic equipment, so switching to those profiles is not a shortcut.
- `shared_exercise_task.py` scores reset-relative equipment stability and explicitly has `grasp_supported=False`. A moving-object task requires a separately versioned, frozen task and score protocol. Existing tasks, gates, scores and old laboratories remain unchanged and are not directly comparable to the new task.
- `native_interval.py:validate_interval` currently admits only person-only `shared-world-v3`. Equipment full-span v2 requires an exact 30 Hz endpoint (`full_reference_horizon`). The original source ends at 15.233 s: full moving-equipment coverage requires qualifying the existing shorter terminal-action mechanism for dynamic objects, including task integration/export/replay. Do not truncate, stretch, silently append a hold, or use a six-second window as full-clip success.

## Smallest useful new contract

Proposed pure preparer: `prepare_moving_equipment_hypotheses(base_run, evidence_run, out, *, source_mappings=(), settings, max_seconds=30, max_candidates=3)`. New module and profile, not additional permissive fields on the floor-box profile.

1. Independently load sealed base/body/root/native proof and construction-only equipment evidence. Keep canonical source/frame/time joins and original identity groups. One bounded parse of shared inputs serves every candidate.
2. Produce a complete per-frame **image track ledger**: exact source frame, equipment and body clocks, mask phase/status, partition, raw track identity, admitted box/mask/hand-point references or unavailable reason. Never change pre-initialization/withheld/occluded slots into observations. Returned pre-initialization provider output retains its actual provider phase and uncertainty; do not claim it is later-initialization evidence or verified identity.
3. Fit nothing to simulator scores. Preserve the current frozen body and authored camera. Form image-center rays using that camera. Enumerate an explicitly configured, bounded depth/shape hypothesis family; e.g. a constant-depth spherical or box proxy, with size derived conditionally from the same observed image support. Depth, orientation, constant shape, camera, mass and friction remain authored priors. Shape is not a recovered kettlebell handle or cavity. A moving spherical proxy can avoid pretending box orientation was recovered; a tight enclosing sphere is not proof of collision geometry.
4. Store source-conditioned 3D **proposal** positions at available timestamps alongside original pixels. Do not feed them directly into physics state at every frame. Missing intervals stay null. If a derived comparison/interpolation view is used, version it, retain endpoint IDs, and bound each bridge explicitly; it never creates source observations or extends the original body interpolation policy.
5. Initialization must be at the unchanged original source start, not the first clear equipment image. If qualified evidence exists at the start, derive a conditional ray/shape initialization. If not, record `initial_state_status=authored_prefix_hypothesis`, the exact unknown prefix and a declared initial-state model. Options are a body-relative left/right wrist offset or an explicit free-state prior. A later observation may constrain a stated back-extrapolation model, but must not be presented as a measured initial state. Never reset/teleport/spawn the object at frame 91 and call the original full clip reconstructed. Unsupported initial state can remain an inspectable unadmitted proposal.
6. Preserve complete role/contact/depth alternatives. Materialize at most three closed candidates under the same work budget. If the complete required set exceeds the bound, keep a factorized inventory and require explicit selection or abstain; do not silently discard alternatives based on later task success. Existing portal explicit candidate selection can be reused. A selected lab remains immutable for recordings.

Suggested closed receipt sections: `source_binding`, `full_body_sha256`, `root_proof_sha256`, `image_track_sha256`, `camera_prior`, `shape_prior`, `depth_prior`, `initial_state_hypothesis`, `contact_hypothesis`, `identity_assignment`, `unavailable_slots`, `derivation_sources`, `scope`, `runtime_sha256`. Large full rows remain exact sealed sidecars, not duplicated in scene metadata. No arbitrary file paths or executable/generated code.

## Runtime and contact milestones

### A. Free-body candidate — smallest honest interactive result

Construct the object once at t=0 with its declared state, then use the existing free-body integration, gravity, contacts and human controls. Zero velocity is allowed only as a labelled authored prior; a source-supported image derivative remains a conditional model-space velocity, not a measured physical speed. Existing position/speed/spin/mass/shape/world bounds and initial-overlap checks still apply. The object may fall or lose contact: this is a visible model failure, not a reason to add hidden forces or overwrite its trajectory.

This can be an inspectable coupled candidate even when no exercise-success claim is available. It is not automatically a full-clip reconstruction. Do not reuse the old stationary-equipment scorer to call intentional movement a failed or successful moving exercise.

### B. Explicit assisted attachment — separate dynamics and control profile

There is no implemented grasp primitive or articulated hand. If a useful interactive holding action is needed, add a closed **force/torque-limited compliant wrist attachment** hypothesis with visible attach/release control. It must expose the selected wrist, authored anchor/offset, limits, power/effort and assistance state in actions, observations, snapshots, replay and UI. Compare against an unassisted free-body baseline. Never install a hidden weld, unlimited constraint or direct object pose playback. Image co-motion/intersection only proposes an attachment candidate; it does not establish grip, contact normal force or measured load. True finger/handle grasp remains additional geometry and control work.

## Task and independent comparison

Version a moving-equipment task only after freezing semantics. Keep original body tracking/floor gates unchanged; add a separately declared source-image consistency diagnostic or conditional trajectory task evaluated only where evidence is admitted, with coverage/unavailable slots reported. Image agreement alone is not physical success. Report original body tracking, object image errors, task failures and assistance effort separately. Do not choose a candidate using a final holdout or rewrite existing gates to make this clip pass. Compare alternative priors under the same frozen task only when target identity and task semantics actually match.

## Implementation split and bounded checks

1. New pure motion/initialization proposal helper + typed input/receipt reader: no MuJoCo, no inference/API, <=30 s / 1 GiB / 3 candidates, reuse current sealed inputs. Tests: full source slots and prefix, distinct clocks, withheld leakage, role alternatives, malformed/nonfinite depth, initial state absent, no nearest-frame substitution, exact body/root proof, immutable input pins.
2. New moving-equipment constructor profile in the ordinary shared-world path, preserving existing floor-box profile. Reuse common primitive validation, source packing, initialization, collision rejection and no-hidden-assistance rule. Qualify dynamic-object native terminal intervals through reset/actions/task/substeps/export/replay tests before any full-clip claim.
3. Extend portal candidate presentation/explicit selection only after the new ordinary package is admitted. Existing manual action/recording path is reusable; do not build another simulator or manual-only CLI workaround.
4. Optional attachment/control and task profiles are separate reviewed milestones, not implied by A. Analytic mechanics tests should prove capped attachment force/torque, release behavior, no hidden channels and exact recording replay.
5. Parent may then approve ONE frozen saved-source proposal/construction acceptance under current local resource limits, no new providers. Success is admission/interaction/evidence custody, not recovered physics. If the source cannot constrain initialization or identity sufficiently under the declared family, preserve the complete refusal rather than changing a gate.

## Concrete parallel file/API split and dependency order

**Lane A — source-conditioned hypotheses (new files only first):**

- `moving_equipment_evidence.py`: `prepare_motion_evidence(construction_bytes, source_binding, body_reference_ids, *, max_seconds=10) -> immutable receipt bytes`. Complete typed rows, raw track phases/identities, source and body timestamps, unavailable reasons; never imputes missing observations or filters statistical outliers.
- `moving_equipment_proposal.py`: `derive_moving_proposals(base_scene, full_human_bytes, root_proof, motion_receipt, settings, *, max_candidates=3, deadline) -> proposal bundle`. Fixed reviewed depth/shape/initial-state family, conditional positions/velocities with original row IDs and explicit prefix assumptions; no physics, API or task score access. Independently rederive source receipt and root transfer; not an arbitrary user JSON bypass.
- `moving_equipment_inputs.py`: exact sealed receipt/proof/camera/body join and fixed-path source collector. Tests in `test_moving_equipment_evidence.py` / `test_moving_equipment_proposal.py` / `test_moving_equipment_inputs.py`. Large full rows remain sidecars.

**Lane B — coupled runtime and full-interval custody (analytic fixtures can proceed independently):**

- `moving_equipment_world.py`: `construct_moving_equipment_world(base_scene, proposal, *, motion_evidence, full_human_bytes, root_proof, source_binding, deadline) -> scene`. New explicitly dispatched source-constructor profile; reuse existing physical primitive checks and common SharedWorldEnv, never call the static floor inversion with relaxed fields.
- Narrow reviewed additions to `native_interval.py`, `shared_world.py` and packaging/runtime-closure consumers to qualify dynamic equipment for the existing partial final action. Tests first: exact final source time, unchanged object integration over the shorter action, early failure, all substep durations and exact replay/export. Existing person-only and unmarked scene bytes stay unchanged.
- New task adapter `moving_equipment_task.py` and narrow dispatch in lab/recording only after a task contract is frozen. It must not masquerade as existing reset-stability task. Shared source checks and human floor/tracking safety remain. Ordinary generic manual-world recording can be qualified before a new learned-policy path; no learning is implied by constructor readiness.
- Tests `test_moving_equipment_world.py`, `test_dynamic_native_interval.py`, `test_moving_equipment_recording.py`. Existing physics constants, floor tolerance and legacy task gates are not edited.

**Dependent integration:** A's frozen proposal/receipt schema and B's analytic admission/clock tests must both pass before their single saved-source construction. Then parent-owned portal routing reuses ordinary candidate rows, source artifact inspector, explicit one-time selection and immutable session runtime. No portal changes during the schema design. Attachment controls, assistance telemetry and their task semantics are a later independent profile, not an implicit dependency hidden inside initial object construction.

**Learning admission:** no new training until full source/custody/initial-geometry/native-clock/recording checks and the chosen task's existing preflight gates pass. A moved task definition is a new task, not an improvement over an old failed score. New contact/attachment learning must expose assistance channels and frozen scorer identity; old policy runtime pins never accept the new dynamics implicitly.

## Actual clip limits reported by the independent diagnostic lane

`.local/sampled-init-consensus-diagnosis-01/result.json` reports all 458 slots and 442 eligible references over 0–15.233 s. The 16 excluded slots are only missing original body references; there is no count/span/coverage failure. Both selections remain null:

- sam-1 exceeds image edge range, fitted drift and cohort limits; edge ranges [49,210,37,197] pixels versus [12.8,9.6,12.8,9.6] limits. Vertical cohort gaps 44/35 pixels versus 4.8. Eligible box area varies from 2 to 2838 pixels², so a tracking/visibility collapse is also plausible; movement alone is not established.
- sam-2 exceeds range and drift; edge ranges [27,80,6,3] pixels and right-edge fitted drift 6.876 pixels versus 6.4.
- Before selected initialization frame 91 there are 91 original slots, 90 eligible body-reference slots. Returned provider proposals retain their pre-initialization phase; they are not verified backward measurements or a known physical initial state.

Therefore no claim that the proposed new dynamic family will admit this clip. A source-bound initial state may remain unresolved; small/collapsed inferred boxes may violate unchanged shape bounds; body-floor feasibility and contact remain separate; the full native endpoint still needs dynamic qualification; reset-stability task is unsuitable; grasp and learned full-exercise success remain unsupported. Do not substitute a per-clip manually chosen box, a later start, held-object teleportation or a gate change to remove these limits.
