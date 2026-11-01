# Exercise floor calibration gap

## Current status and ownership

Updated November 1, 2026, 1:51:36 AM PDT. Reconstruction lane completed a bounded primary-source metadata search; **no acquisition-ready recording met all requested conditions**. This is not an exhaustive search. No media, model, native geometry, physics, forms, account operations or paid calls ran. No product files changed; no live job remains. Root owns publication and any subsequent acquisition authorization. Previously released guided-config files remain frozen.

The target was one licensed fixed-camera exercise recording with visible feet/floor and hand/equipment, exact camera intrinsics, and an independently measured floor reference. A complete recording no longer than 20 seconds and no larger than 32 MiB was preferred; no crop was acquired as a substitute.

## Search evidence and limits

Six responses: two official-domain search queries and four opened official pages. The conservative returned-text upper bound was 256,084 bytes, below 1 MiB. Full returned text and decisions are retained in `.local/exercise-floor-source-research-01.json` (70,847 bytes), SHA-256 `0edf6cdc90063ebc6fafeffd4459f64b8ae96dcb3c320be920b9673187479796`.

The response set was captured by November 1, 2026, 1:50:27 AM PDT. Individual server retrieval times were not exposed; do not invent them. No external request remains authorized by this completed six-response budget.

| Primary source inspected | Supported fact | Unresolved dependency |
| --- | --- | --- |
| [MM-Fit project](https://mmfit.github.io/) | Describes RGB-D recordings and dumbbell exercises. | No numeric intrinsics or independently measured floor sidecar established by the inspected page; no visual inspection performed. |
| [MM-Fit video release v2](https://zenodo.org/records/7672767) | Smallest listed complete video is `w19_rgb.mp4`, 415.4 MB, MD5 `5d5f8dd6434deffba8842cfefe0101db`. | Exceeds preferred size; duration unverified. Rendered License field contained no license text, so dataset rights remain unverified. |
| [MM-Fit starter repository](https://github.com/KDMStromback/mm-fit) | Repository identifies starter software under MIT. | Software MIT is not proof of the separately hosted video's data license. No qualifying floor calibration established. |
| [UCOPhyRehab++ release](https://zenodo.org/records/17935737) | Describes OptiTrack joint truth and derived silhouettes/flow/segmentation. `samples.zip` is 326.8 MB; joint JSON is 230.3 MB. | No small original video package, numeric intrinsics or measured floor reference established. Rendered data-license text absent. |
| [Seven fitness exercises dataset](https://data.mendeley.com/datasets/kgbb3yn47p/1), official search-returned text only | Raw smartphone dumbbell/bodyweight exercises; explicit dataset CC BY 4.0. | Front/side/diagonal and form labels do not establish calibration. Individual size/duration, camera intrinsics and measured floor are unverified. No additional page opened. |

The two search queries were `exercise dataset camera calibration ground reaction force video dataset license squat chair` restricted to Zenodo/Mendeley/Figshare/Nature, and `exercise human object interaction dataset calibrated cameras floor intrinsics license` restricted to GitHub/Max Planck/Tübingen/CVF. Other returned leads were not individually qualified. Do not treat search snippets as a complete dataset inventory.

## Existing local evidence is exposed, not a new holdout

- `data/exercise/lunge-twist.webm`: recorded source SHA `afe0c808e05c6b178c90295f9b9deeeda16322a5b6e1b10d8ebeac31c5eb8a34`, 3,005,172 bytes, 20.6 seconds, Taco Fleur/Cavemantraining, [CC BY-SA 4.0 source](https://commons.wikimedia.org/wiki/File:Kettlebell_Lunge_and_Twist.webm). Prior source/pose/body failures are exposed. Fixed camera/full feet and independent floor/equipment measurements remain unverified. The hash is from existing custody metadata, not a new media read.
- Interval push-ups, source SHA `e93ea83a78d47c27972f245a61d114118793598cfd059c318e120c453fd40bbe`: visible handheld supports and partial hand occlusion were already inspected; pose, stationary-ground and projection attempts are development evidence. No independent floor calibration. Do not repeat rejected fits unchanged.
- FreeMoCap: existing 58 mm board calibration and study-only corner checks do not establish a floor-fixed/gravity-aligned world. Separate data rights are not established by the software AGPL license. Earlier cross-camera/board/carpet studies failed; third view is exposed.
- Existing jogging cameras/ground are supplied estimated priors; all relevant views and reconstruction alternatives have been exposed. Successful conditional fits are not independent floor/contact truth.
- Prior OpenCap metadata research already found unavailable SimTK access and unverified data rights/calibration; do not repeat that unchanged blocked request. See `.local/fresh-source-readiness-01.json` and `.local/opencap-metadata-rights-01/receipt.json`.

## Reuse boundary: no new geometry subsystem

Existing production functions already accept the desired supplied geometry:

- `camera_pose.validate_camera`: exact source/dimensions/clock, K, radial coefficient and per-frame proper rotation/translation; explicit quality/units.
- `camera_ground.validate_ground` / `validate_ground_binding`: source/calibration-pinned horizontal world height, preserving the existing body origin and floor placement.
- `source_surface_anchors._receipt_bound` and `source_surface_envelopes.prepare_source_surface_envelopes`: source-bound camera-calibration and surface-height measurement receipts, exact uncertainty/provenance, complete slots, missing-value abstention and interval geometry.
- `world_alignment.load_world_alignment`: explicit spatial and clock mappings rather than equality inferred from frame numbers.

`scene_vision.findHomography` is image registration, not measured floor calibration. Existing study scripts `diagnose_freemocap_board.py` and `check_freemocap_calibration.py` already use OpenCV planar pose/Charuco primitives. They must not be rerun on their failed inputs or duplicated as another solver.

The missing reusable seam is **measured planar-target evidence to the existing camera/ground contracts**, if that evidence is supplied. A proposed new `floor_target_calibration.py` should reuse these primitives/validators, retain every planar-pose alternative, and abstain on ambiguous solutions, insufficient rank/conditioning, invalid depth or missing target-to-floor measurement. Supplied K/distortion, source/frame/corner identities, units and uncertainties must remain exact. A photographed board is not automatically the floor; setting board coordinates to z=0 does not measure the floor relation. If a sidecar already supplies valid R/t and floor coordinates, import it directly without solving again.

## Next concrete action

Obtain one consented/licensed complete fixed-camera exercise recording with matching capture-resolution K/distortion, visible feet and hand/equipment, at least four identified non-collinear measured floor-target corners, and documented target-to-floor transform/uncertainty. Freeze independent check points/frames before fitting; preserve all original slots and every rejected calibration. A 10–20 second, at most 32 MiB package with small JSON sidecars avoids the large dataset acquisitions above. No contact/load labels may be inferred from those measurements.

Before implementation, root should approve the exact adapter contract and a bounded analytical test plan: known-transform recovery, nonzero supported distortion, mirrored/ambiguous planar alternatives, degenerate corners, source/clock mismatch and absent floor relation. A new source study needs separate frozen inputs/settings/limits. Until the measurements exist, the honest result is unavailable—not an authored floor relabeled measured, a shifted floor, or a fresh holdout claim.
