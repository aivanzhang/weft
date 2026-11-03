# Moving-equipment exploration

This opt-in path builds conditional moving objects beside the original video-derived body in the existing simulator. It does not recover grip, measure muscle activity, or provide a scored exercise task. The existing stationary-equipment builder is unchanged.

**Integration status:** the saved kettlebell-flip video now passes the actual portal journey: two explicitly chosen depth hypotheses build, one opens, six manual actions record, and export replays exactly. The initial fixed-depth attempt remains rejected and preserved. The newer path retains complete depth alternatives and uses explicit authored choices; it does not establish correct physical depth or grasp.

To explore the verified local clip, start the portal yourself:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-kettlebell-flip-01/portal
```

Open project `cb3ea4169eae4804bd0aa6f99d3453cb`, then its sandbox. It is bound to `candidate-1`; recording `session-015bbe72b3db` contains the verified six-action example. The full second hypothesis and previous failures remain saved. This command starts no paid processing. These local video/run assets are not included in a fresh Git clone.

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

### Continue a saved exercise upload

For an unresolved exercise upload with completed body and equipment evidence, the portal offers **moving-equipment exploration** without another AI request:

1. Confirm the camera, depth range, sphere shape, material and unknown-prefix assumptions. These are authored settings, not measurements.
2. Prepare alternatives. The background worker retains every raw track and computes the allowed initial-depth ranges under the original collision limits. An allowed range does **not** establish correct physical depth, identity or grip.
3. Choose the track, range number and a depth inside that range. Add up to three hypotheses explicitly; the portal does not select a winner or hide other alternatives.
4. Build those hypotheses. Inspect each admitted world or refusal, then choose one world to open. That project stays bound to the chosen world so its controls and recordings cannot switch underneath you.
5. Open the sandbox, reset, use manual controls, and record/export the interaction. Original video, body references, assumptions and rejected alternatives remain saved.

The complete family and construction report are downloadable. Work runs through the existing bounded local worker with no provider credentials. Failed attempts remain available; they are not automatically retried. This workflow is manual and unscored: task learning, gripping equipment and physical accuracy are not established by opening a world.

### Import a CLI-built exploration

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
