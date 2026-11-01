# Camera and floor calibration

Use `floor-calibrate` to check measurements relating an exercise video's pixels
to a camera and floor in metres. It saves an offline report; it does **not**
discover the floor or move a body to make its feet touch.

Calibration is optional for [trying the local sandbox](local-lab.md#start-here).
Use it to replace camera/floor assumptions with source-linked measurements.

## Choose a route

| Route | What you supply | What Weft does |
| --- | --- | --- |
| **Supplied calibration** | Existing camera and ground files, plus supporting measurements and uncertainty for both | Checks the matching pair without solving a target pose. Accepted files retain their exact bytes and quality labels. |
| **Planar target** | Camera internal calibration, measured target corners, their image locations, the target-to-world relationship and a separate floor measurement | Solves the target pose once, retains every alternative and checks explicitly chosen construction frames. Outputs are labeled reconstructed. |

A target board can be tilted: **its plane is not automatically the floor**.
Supply the measured relationship between the target and the world, and a separate
horizontal floor height. Tilted floors are currently unsupported. Setting the
board's local height to zero does not measure the floor.

## Prepare the inputs

- **Source observations:** use matching saved observation JSON from a completed
  Weft run, or export a supported run with `weft observations --run RUN --out NEW_JSON`.
  This does not detect a board, annotate corners or measure the floor. The video
  SHA-256 identifies the exact original file. Required references must exist in
  the matching construction evidence.
- **Measurement JSON:** prepare it from your capture calibration, physical
  measurements and explicit image annotations. Include the method, provenance,
  units and uncertainty. A `measured` label is the supplier's claim, not independent
  authentication. Unknown uncertainty must remain unknown.
- **Request JSON:** choose `supplied` or `planar_target`, pin the exact input
  files and select the measurement records. For the target route, explicitly
  choose the anchor and construction-check frames, keep evaluation frames separate,
  and document that the camera and target stayed fixed over the complete interval.

There is no calibration-input wizard or automatic measurement command in this
workflow. Use the [current request and measurement schemas](handoffs/floor-target-calibration-core.md#public-api-and-exact-serialized-contract).
The target route requires at least four usable, non-collinear named corners;
metres and millimetres are explicit, not inferred.

## Run locally

For an existing camera/floor pair, with a request whose mode is `supplied`:

```sh
.venv-lab/bin/python -m weft floor-calibrate \
  --observation path/to/source-observations.json \
  --source-sha256 ORIGINAL_VIDEO_SHA256 \
  --measurement path/to/measurements.json \
  --request path/to/supplied-request.json \
  --camera path/to/camera.json --ground path/to/ground.json \
  --out artifacts/my-supplied-calibration-01
```

For a target solve, with a request whose mode is `planar_target`:

```sh
.venv-lab/bin/python -m weft floor-calibrate \
  --observation path/to/source-observations.json \
  --source-sha256 ORIGINAL_VIDEO_SHA256 \
  --measurement path/to/target-measurements.json \
  --measurement path/to/floor-measurements.json \
  --request path/to/target-request.json \
  --out artifacts/my-target-calibration-01
```

Repeat `--observation` and `--measurement` for up to eight files each. Do not pass
`--camera` or `--ground` for a target solve. `--max-seconds` optionally sets the
overall packaging limit to 10–30 seconds; the default is 30. Use a new output
directory whose parent already exists. Inputs are limited to four MiB combined,
1,800 original frame slots and 60 seconds.

This bounded local command performs no video decoding, pose inference, training,
server startup or paid API call.

## Read the result

Open the printed `inspection/index.html` directly. Inspect uncertainty, every pose
alternative/rejection and all original frames, including missing and unused ones.
Point comparisons check image geometry, not physical contact.

| Status | Meaning |
| --- | --- |
| `qualified_nominal` | The supplied pair passes, or one distinct target pose passes the fixed checks. Camera and ground downloads are available. |
| `unavailable` | Required support is missing or unusable; the report names the reason. |
| `rejected` | A solve ran, but no alternative passed. |
| `ambiguous` | Multiple distinct alternatives pass; none is silently chosen. |

Only a sealed `qualified_nominal` result returns CLI success. Other completed
statuses return nonzero but retain their report and evidence. A packaging or
execution failure preserves diagnostics and may have no completed report or seal.

The package retains `inputs/`, complete `calibration.json` and worker diagnostics.
Qualified output adds root `camera.json` and `ground.json`. Keep `manifest.json`
with the files; it records their exact identities. No `surfaces.json` is generated.

**Nominal does not mean uncertainty-certified.** Declared bounds and standard
deviations are recorded, not propagated through the camera/floor calculation.
Contact, grasp and load remain unknown.

The local [nominal example](../artifacts/floor-target-numerical-02/case-01/inspection/index.html)
and [ambiguous example](../artifacts/floor-target-numerical-02/case-03/inspection/index.html)
use synthetic analytical points, not exercise recordings or physical measurements.

## Use the pair in the existing body workflow

After inspecting a qualified pair, explicitly create a **new** camera-aware
candidate from a compatible completed mobile exercise run for the same source:

```sh
.venv-lab/bin/python -m weft camera-body \
  --run path/to/completed-mobile-exercise-run \
  --camera artifacts/my-target-calibration-01/camera.json \
  --ground artifacts/my-target-calibration-01/ground.json \
  --out artifacts/my-grounded-body-01
```

This separate step checks source, geometry and ground. Calibration alone does not
guarantee body admission, fix shape or establish contact. Existing labs stay unchanged.
See [camera-aware bodies](camera-aware-body.md), [supplied ground](camera-ground.md)
and the [local workflow guide](local-lab.md) for the next supported steps.
