# Ground from two calibrated images

`stereo-ground` estimates a **conditional ground plane** from matching points you
mark in two images. It saves source images, labels, uncertainty assumptions,
3D estimates, a readable report, and matching camera/ground files for the existing
`camera-body` workflow. It does not automatically discover ground or measure
gravity, contact, anatomy, or camera accuracy.

Use this when two supplied cameras already share metric world coordinates and
you can identify at least three well-spread points on the same static ground
surface. A single uncalibrated video is insufficient for this command. Mark the
same physical point in both images; a cone's apex is not its ground-contact point.

## Run it

```sh
.venv-lab/bin/python -m weft stereo-ground \
  --camera path/to/first-camera.json --camera path/to/second-camera.json \
  --image path/to/first-frame.jpg --image path/to/second-frame.jpg \
  --annotations path/to/ground-labels.json --out artifacts/my-ground
```

Use a new output directory. Camera files follow the existing source-bound
[camera contract](camera-aware-body.md), including every original frame, source
SHA-256, dimensions, timebase, intrinsics and transforms. The two images must be
original-raster JPEG/PNG files, in the same order as the cameras and annotations.
No source-video decoding, inference, API, training or server is started.

Open the printed `index.html` yourself. It shows the labels over both images,
each pair's reprojection error, geometry checks, uncertainty sensitivity and
downloadable outputs. Rejection keeps the report and inputs but emits no new
camera/ground files; the CLI exits unsuccessfully.

## Annotation format

The top-level JSON has exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `profile` | `stereo-ground-annotations-v1` |
| `world_provenance` | Why the cameras share metric coordinates and the marked surface is considered static/planar; state assumptions |
| `views` | Two image/source bindings, in camera order |
| `points` | 3–16 distinct named point pairs, including at least three construction points |

Each view contains `name`, `source_sha256`, `calibration_sha256`, `image_sha256`,
`frame`, `time_s` and `provenance`. Use `weft.io.canonical_hash(camera_object)` for
the calibration hash and `weft.io.digest(image_path)` for the exact image bytes.
The frame index and time must match the original camera stream. State the image's
association with that source frame: this command verifies supplied bytes and
clocks, not the original video's decoded pixel identity.

Each point contains:

```json
{
  "id": "near-marker-base",
  "partition": "construction",
  "pixels": [[1725.0, 1566.0], [160.0, 1448.0]],
  "uncertainty_px": [6.0, 6.0],
  "provenance": "Manually estimated apparent base center; footprint and camera errors remain uncertain"
}
```

Coordinates and pixel ranges refer to each original image, not a resized preview.
Ranges must be positive, no larger than 32 pixels, and remain inside the image.
Use `assessment` instead of `construction` for additional points that should
check the frozen plane rather than fit it. Three construction points always
define a plane when non-collinear; their near-zero fit error is not validation.
With no assessment points, the report explicitly says the independent source-label
check is unavailable. Even consistent assessment labels are not measured truth.

## What changes, and what does not

All construction pairs must pass the existing stereo geometry guards and a fixed
3-pixel reprojection limit. The fit retains every construction point, rejects
poor spread, excessive slope and more than 2 cm nonplanarity, and checks any
assessment points separately against a fixed 4 cm limit. No score-based point
selection, threshold adjustment, body movement or floor fitting occurs.

`camera-0.json` / `ground-0.json` belong together, as do the `1` files. One proper
rotation makes the estimated plane horizontal while preserving projected pixels,
source clocks, camera translations and scale. Installing gravity along that
normal is a separate authored physical assumption. Original calibrations,
labels and implementation sources remain in the sealed package.

The sensitivity report perturbs one image coordinate at a time by its supplied
pixel range. All outcomes, including rejected estimates, remain visible. These
samples are **not confidence intervals or worst-case bounds**: combined label
errors, incorrect point identity, raised marker bases, uneven ground and camera
calibration errors remain unmeasured.

## Verified example and limits

The saved [ASPset ground report](../artifacts/aspset-manual-ground-01/index.html)
uses three manually marked colored-marker bases. The far left-view marker is
partly obscured; that inference is explicit. Largest nominal reprojection error:
0.974 pixels. Its 24 single-coordinate perturbations vary the normal by at most
0.1462° and the plane at its center by 0.00864 m. There are no independent ground
assessment points or gravity measurements.

The command took 0.219 seconds and peaked at 121.9 MB process RSS on this local
4K example. A separate analytic maximum-size numerical call—16 points, 1,800
camera frames per view, 128 perturbations—took 0.155 seconds. These are scoped
measurements, not sustained-load or browser responsiveness claims.

The plane was then frozen before constructing a two-view body with 47 retained
source poses. The native floor check **fails at source frame 80**: maximum
penetration is 0.06666 model units versus the unchanged 0.025 limit; three
interpolated control-grid poses also fail. The body was not shifted to hide this.
No learner, default, physical-contact claim or general-purpose accuracy promotion
follows. [Exact evidence and next work](handoffs/aspset-ground-construction.md).

To inspect this unpromoted manual body candidate, start the viewer yourself:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/aspset-camera-body-lab-02
```

The established interactive equipment example remains available through
`scripts/play_sandbox.py --example auto-equipment --console`; this calibration
does not replace it.
