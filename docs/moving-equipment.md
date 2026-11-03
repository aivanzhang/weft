# Moving-equipment exploration

This opt-in path builds conditional moving objects beside the original video-derived body in the existing simulator. It does not recover grip, measure muscle activity, or provide a scored exercise task. The existing stationary-equipment builder is unchanged.

**Integration status:** saved-source admission is checked, and an analytic fixture passes construction, portal import, six manual actions, recording and exact replay. Complete real-source-to-portal acceptance is still pending. The first actual exercise clip retained all source frames but rejected every proposed initial world: one sphere overlapped a knee, and the competing track produced an oversized sphere.

## Inputs

Use a sealed native-clock body seed and the original sealed equipment-tracking package for the **same exact video**. This first input profile does not accept arbitrary edited JSON, nonnative bodies, camera-body variants or different-media mappings. It never uploads media or makes API calls.

Save an explicit priors file, for example:

```json
{
  "schema_version": 1,
  "profile": "moving-equipment-priors-v1",
  "camera": {"distance_model_units": 2.0},
  "depth_offsets_model_units": [0.0],
  "shape": {"method": "sphere_from_median_image_box_diagonal"},
  "mass_kg": 1.0,
  "friction": [0.5, 0.005, 0.0001],
  "initialization": {"method": "first_initialized_box_zero_velocity"}
}
```

These values are assumptions, not measurements. The sphere approximates the tracked image extent; it does not recover a kettlebell handle or a collision mesh. Camera distance and depth use model units, not measured metres.

The initialization method places the object once at the original video start. When its first usable image occurs later, it explicitly assumes the same position and zero velocity over the unknown prefix. It preserves those unknown frames and the later evidence reference; it does not start the simulation later or secretly animate the object along the video.

## Build and inspect

```sh
.venv-lab/bin/python -m weft moving-equipment-build \
  --run PATH_TO_SEALED_BODY_SEED \
  --evidence-run PATH_TO_SEALED_EQUIPMENT \
  --settings priors.json \
  --out artifacts/moving-exploration-01 \
  --max-candidates 3 --max-seconds 120
```

The output directory must be new. Source preparation is shared across candidates. The command retains the complete identity/depth family or explicitly abstains when that family exceeds the bound; it does not rank alternatives by simulator success or silently discard them.

- `moving-proposals.json`: image-linked positions, all raw-track alternatives, missing rows and the physical assumptions. Proposed positions are evidence for inspection, not physics controls.
- `result.json`: each admitted, rejected or deferred candidate and its reason. A rejection can be the correct result—for example, a proposed initial object overlaps the body or floor.
- An admitted candidate's `lab_directory`: the ordinary interactive lab with sealed sources, runtime and assumptions. Nothing is selected automatically.

The process runs no learner and starts no server. It exits nonzero when no candidate is ready or when the overall build fails. Preserve failed outputs before changing inputs or settings.

## Portal use

Once a candidate passes admission, import its exact `lab_directory` using the existing stopped-portal workflow:

```sh
.venv-lab/bin/python -m weft portal-import \
  --lab PATH_FROM_RESULT_JSON --root .local/portal
```

Then start the portal yourself:

```sh
.venv-lab/bin/python -m weft portal
```

Exploration is manual and unscored. Use the named body and equipment-force controls; actor switching is unavailable in this profile. Recording starts from a seeded reset. Full action vectors, seeds, actor, scene, source ledger and runtime are pinned for exact replay. Raw simulator rewards or terminal flags are diagnostics, not a successful reconstructed exercise. Unsupported or changed runtimes refuse rather than silently replaying a different world.

Useful moving-equipment tasks, assisted attachment where justified, learning comparisons and broader unfamiliar-video reconstruction remain separate unfinished work. See the [current foundation and source-admission evidence](handoffs/moving-equipment-foundation.md).
