# Explicit planar candidates from segmentation evidence

The bridge can construct and replay an **authored planar sandbox**, not promote segmentation into verified physics. It accepts sealed `segmentation-evidence-v1` or decoder-timed shared observations v3. It does not call a provider, consult manual labels, train a policy, optimize physical parameters, or silently change the source interval.

## Interfaces

- `generate_candidate(evidence, assumptions)` creates one validated `planar-rigid-v1` scene and its provenance/partition envelope.
- `evaluate_candidate(candidate, evidence)` coasts that unchanged world through the exact source timestamp schedule and reports separate fixed-ID and anonymous-location diagnostics.
- `export_candidate(evidence_run, assumptions, out)` seals the inputs before construction, then seals either the candidate/replay or an explicit unsupported outcome. The output directory must be new.

The CLI uses the same path:

```sh
.venv-lab/bin/weft tracking-scene \
  --run artifacts/segmentation-evidence-01/three-ball \
  --config artifacts/segmentation-scene-three-ball-config-01/config.json \
  --out artifacts/my-new-planar-attempt
```

That particular saved source is expected to be rejected for camera motion. Do not change its known motion to `unknown` to obtain a candidate.

## Required authored assumptions

The complete [three-ball specimen configuration](../artifacts/segmentation-scene-three-ball-config-01/config.json) shows every field. The caller must explicitly specify:

- A fixed-orthographic camera hypothesis, declared known camera-motion status, common 2D plane, image-width units rather than meters, and closed image-edge walls.
- Per-shape friction 0.5, matching the existing engine; gravity, damping, restitution and simulation timestep.
- Every object's role, mass, circle radius or box dimensions, and display color. Model boxes are not automatically collision shapes or materials.
- One controlled object, target, task tolerances, action acceleration and horizon. These controls are authored, not recovered forces or demonstrated intent.
- An unforced-motion hypothesis for the fitting prefix and one generation/development split boundary.

Unknown camera motion remains an unverified hypothesis. Declared moving camera in raw evidence, metric-3D units, open boundaries, omitted objects, incompatible geometry, and other unsupported assumptions are rejected. A complete validated [stabilized wrapper](camera-stabilization.md) can instead use the explicit `similarity_stabilized_orthographic_assumed` hypothesis while retaining the original moving-camera declaration. No geometry is shrunk, object removed, source frame searched around, or initial speed clamped to obtain acceptance.

## Temporal and identity boundaries

All frames from the latest object initialization through the final evidence frame belong to the attempt. The caller chooses one generation-end boundary; the full following suffix is development evidence. The preceding prefix is reported separately as coverage before **candidate initialization**; original per-object phases remain in the frozen input.

Only the first five complete generation samples determine initial position and velocity. Actual decoder timestamps are used, with authored gravity/damping and no inferred external control. Development positions and future ambiguity groups cannot change the generated scene or its generation-only identity alternatives. The complete input hash still identifies the evaluation evidence.

Close-object groups remain unresolved identity hypotheses. At most six correspondence alternatives are retained **overall**, including the product of independent groups; excess ambiguity is rejected rather than truncated. These alternatives describe correspondence within the same authored physical world—not independently recovered dynamical worlds. Differing authored masses, shapes or roles do not establish which physical object follows a model ID. No alternative is selected by replay error.

Replay uses the existing `SceneEnv` coast action, walls, contacts and speed guards. A copied runtime scene subdivides each source-time gap so steps end at the exact timestamp; no source frame is rounded onto a fixed simulator grid. Reported time is the source schedule, not the engine's fixed-step snapshot clock. Contact lists combine simulated events within each source interval and are never called observed contacts.

## Scores do not authorize promotion

Fixed-ID replay compares each model-ID hypothesis with its corresponding simulated object. Anonymous replay uses bounded one-to-one matching of the returned location set, without changing model IDs. Missing observations, unmatched simulated objects, and unavailable simulation frames are counted separately. Initialization receives no score; missing geometry never receives zero error.

Both errors are conditional on available model proposals and successful simulation. They are not independent physical truth or task-learning results. An engine clamp, escaped state or simulation failure retains partial valid snapshots and leaves remaining errors unavailable. No passing threshold, candidate selection, repair promotion or automatic `physics_ready` change exists.

The first implementation is bounded to 500 frames, eight objects, six seconds after initialization, 1,800 physics steps, six identity correspondences and 256 anonymous-assignment states. Source gaps above 0.25 seconds or fitting gaps above 0.1 seconds are unsupported. Generation-prefix residual is bounded at 0.01 image widths; authored static-body displacement at 0.005 image widths. These controls were frozen before the saved real-source attempt.

## Verified outcomes

### Real three-ball source: unsupported, retained

The [sealed real attempt](../artifacts/segmentation-scene-three-ball-01/report.json) used the existing 543-observation export without another tracking call. It returned `unsupported_retained`: **known moving camera is unsupported; no camera correction or override**. No candidate, scene or replay was created. The camera gate runs before velocity fitting or initial-overlap checks, so this outcome does not establish that later gates would pass.

The [sealed configuration provenance](../artifacts/segmentation-scene-three-ball-config-01/report.json) preserves the original source-only description, `unknown; observed drift`. The older affine screen reports `not_verified_fixed` with `no_large_motion_detected` samples; that is not proof of a fixed camera. The attempt kept the drift limitation instead of reclassifying it as unknown. Its authored 3.5 px radii were not altered or tested around this rejection.

### Synthetic fixture: executable collision sandbox

The [separate synthetic input](../artifacts/segmentation-scene-synthetic-input-01/report.json) contains an analytically generated two-body elastic collision, a rendered synthetic video, the exact generated positions, an explicit configuration and a reproducible generator. **It contains no physical video or SAM inference.**

The same CLI produced a [sealed executable candidate](../artifacts/segmentation-scene-synthetic-01/report.json), including `scene.json`, from 13 frames. Coast replay used 24 engine steps and preserved the collision. On its 16 development object slots, fixed-ID and anonymous conditional RMSE were both **0.351 px**, with maximum error **0.497 px**. This checks executable plumbing against an analytic fixture, not real-world reconstruction or transfer. `physics_ready` remains false and physical validation remains unverified.

The synthetic scene is available for manual control. It is not a substitute for the real source's rejected reconstruction.

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/segmentation-scene-synthetic-01 \
  --record artifacts/my-collision-recording
```

Enter `right 3`, `coast 6`, `brake`, or another direction to advance explicit steps. `status` refreshes the current image without advancing; `reset` starts the authored scene again; `quit` saves an MP4 and offline replay. Nothing advances automatically. The printed `current-frame.png` path shows the current simulation. Use a new recording directory each time.

The [interaction check](../artifacts/segmentation-scene-console-01/report.json) exercised all six actions over 30 steps plus one reset, exported 32 decoded frames, and reproduced every state/reward/termination flag exactly from the saved actions. Thirty-two focused console tests pass. A separate actual CLI smoke check recorded two actions and verified three exported frames in `artifacts/scene-console-cli-smoke-01`. Neither check trained a policy or established physical transfer.

## Verification and remaining work

Twenty-six analytic tests cover exact irregular timing, executable coast replay and collisions, static bodies under gravity, generation/development separation, shared-v3 interchange, identity-hypothesis limits, absent geometry, initial overlap, speed/gap rejection, altered-candidate rejection, failure accounting, sealed success/rejection exports, and rejection of output paths inside immutable evidence inputs. Both retained CLI attempts were offline and added no provider calls or cost.

Twenty-five tests and the generation/scoring controls were frozen before those attempts. The immutable-input output-path guard and its additional test were added afterward; it does not change candidate generation or replay. The already sealed runs retain their original implementation copies and were not rerun or rewritten.

A defensible real reconstruction still needs supported camera/plane evidence, suitable physical/task assumptions, and independent validation. This bridge supplies neither camera correction nor recovered contacts, forces, materials, 3D state, or general video-to-RL capability.
