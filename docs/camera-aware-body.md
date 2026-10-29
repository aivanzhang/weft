# Camera-aware exercise sandbox

The new jogging sandbox uses supplied camera positions to separate camera movement from body movement. It is playable and recordable, but **not an accurate human reconstruction**: some poses are missing and the body still intersects the floor. This optional mode does not replace the ordinary video-only lab.

**Latest follow-up:** [supply the ground height explicitly](camera-ground.md) to keep it in the same world coordinates as the cameras. That version removes the ground-offset mismatch but exposes floating and inverted pose estimates; it is still unvalidated. The original first-pose-floor example below remains preserved.

## Try the completed example

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/camera-aware-jog-lab-01
```

Open the printed local URL. Inspect the source and inferred body, choose the manual sandbox, select a joint, then use **Record & start** and **Stop & save**. Joint controls, release, pushes, support and authored reference following are available. The view follows the body while preserving its full world position. No trained policy is available for this candidate; the learning buttons are disabled.

Without starting a server:

- [Source/body comparison](../artifacts/camera-jog-interaction-01/index.html).
- [Saved action replay](../artifacts/camera-jog-interaction-01/session/replay.html): rotate and scrub the recorded simulation; this is playback, not live control.
- [Recorded video](../artifacts/camera-jog-interaction-01/session/video.mp4).
- [Body-travel plot](../artifacts/camera-jog-root-reproduction-01/trajectory.png).

The recording contains **39 actions, one reset and 41 states**. A fresh simulator reproduces every state, reward and termination flag exactly. Both the new model and the original video run remain unchanged. Muscle bands are illustrative motor-command indicators, not measured muscle activation or force.

## How it works

1. Reuse the original video and saved 2D/3D pose estimates; no new model call.
2. Use the supplied camera geometry to find the body position that best agrees with the same-frame image joints. Body size remains the pose model's estimate.
3. Remove the camera's movement and express body positions and orientations in one shared, upward-facing world coordinate system.
4. Build a separate articulated-body candidate, preserving missing poses, source times, authored forces and the independent floor check.
5. Inspect or control the simulation and save the action/state journal, MP4 and offline replay.

The camera file is **additional input**, not something this mode automatically recovers from an arbitrary video. The Autelica sample's cameras and reference joints were themselves reconstructed from multiple views. Agreement with them is not independent physical validation. [Source, attribution and original baseline](multiview-jog.md).

Only image joints, monocular model joints and camera transforms enter construction. Provider body joints are read afterward for the comparison. No body scale, reference alignment, ground contact, temporal smoothing or missing-pose interpolation is fitted to improve the score. Incorrect assumed body size can still produce a convincing image match at the wrong depth.

## What improved, and what failed

Both travel scores use exactly the same 112 available frames. Each trajectory subtracts its own first body position; the baseline is rotated into world coordinates with the first supplied camera orientation. No scale or alignment is fitted to the reference.

| Check | Original image-motion body | Camera-aware candidate |
| --- | ---: | ---: |
| Body-travel root-mean-square error against reconstructed reference | 10.511 m | 0.246 m |
| Absolute body-position root-mean-square error | Not evaluated | 0.392 m |
| Source poses exceeding the floor limit | 5/112 | 17/112 |
| Control-grid poses exceeding the floor limit | 5/239 | 28/239 |
| Maximum floor penetration, model units | 0.041491 | 0.237743 |

Travel agreement improves, but floor consistency worsens. The candidate is **not promoted** to a physically validated model. The floor threshold remains 0.025 model units; placement still uses only the first pose. Later intersections are not removed by lifting poses or moving the floor. Source and control-grid counts can overlap in time.

All **240 source frames** remain accounted for: 112 estimated poses, 120 unsampled frames and eight unavailable sampled poses. The longest gap between available references is 0.2002 seconds. The last source frame is explicitly shown without an inferred body. No interval is silently shortened.

The original five-unit travel envelope cannot represent this clip's 14.075-unit maximum displacement component. A separate `camera-root-body-v1` mode supports ±50 units without compressing the trajectory. This is a declared capability expansion, not a pass under the old limit. Body geometry, forces and the recording-coordinate envelope remain unchanged. The existing whole-body trainer and raw-camera foot adapter explicitly reject this new coordinate mode until separately integrated and validated.

## Build another camera-aware candidate

Start with a completed, sealed mobile exercise `run/` and matching supplied calibration:

```sh
.venv-lab/bin/python -m weft camera-body \
  --run artifacts/multiview-jog-image-01/run \
  --camera artifacts/camera-jog-root-01/camera.json \
  --out artifacts/camera-aware-jog-new
```

Use a new output directory. This command does not start a server, train a policy or make a paid request. Original policies/results are preserved under `inputs/original/`, not presented as policies for the new body.

Optional `--ground <ground.json>` uses a source/calibration-pinned horizontal height instead of first-pose floor placement and reports both penetration and signed clearance. [Ground schema, example and remaining errors](camera-ground.md).

The [example camera file](../artifacts/camera-jog-root-01/camera.json) is source-pinned. Schema 1 requires source SHA-256, exact width/height/fps/frame count, metre units, `right_handed_z_up`, provenance, calibration quality, pinhole matrix `K`, radial coefficient `radial_k1`, and one `{frame,time_s,R,t}` record per original frame. The world-to-camera convention is `camera_point = R @ world_point + t`; camera image axes are right/down/forward. Bounds are 60 seconds, 1,800 source frames and 900 sampled poses. Mismatched identities, clocks, reflections, invalid matrices and ambiguous estimates are rejected or explicitly unavailable.

Construction stores camera provenance, every root estimate and unavailable frame, the generated model, original inputs, implementation hashes and floor results. [Candidate result](../artifacts/camera-aware-jog-lab-01/status.json).

## Verification and reproducibility

There are 317 distinct resolved Python checks across camera geometry and affected simulation/recording paths, plus 27 deterministic UI checks. The initial outlier-fixture failure and hardcoded long-travel snapshot-bound failure remain saved; the corrected cases pass. These are scoped results, not a newly repeated full-repository suite. [Verification evidence](../artifacts/camera-body-jog-plan-01/report.json).

The camera estimates generated by the packaged CLI match the original frozen estimates byte for byte. A later, explicitly post-hoc comparison script reproduces all 240 original rows and every metric exactly; this is reproducibility, not a fresh holdout test.

```sh
.venv-lab/bin/python scripts/studies/check_jog_camera_root.py \
  --study artifacts/camera-jog-root-01 \
  --source artifacts/multiview-jog-source-01 \
  --run artifacts/multiview-jog-image-01/run \
  --out artifacts/camera-root-check-new

.venv-lab/bin/python scripts/studies/check_camera_body_lab.py \
  --lab artifacts/camera-aware-jog-lab-01 \
  --recording artifacts/camera-aware-jog-lab-01/sessions/session-a3b323117432 \
  --out artifacts/camera-interaction-check-new
```

Omit `--recording` to create a new bounded test interaction instead of reusing the existing export. The checker never starts a server.

The actual build took 0.344 seconds internally; root estimation alone took 0.020 seconds. Across 120 in-process single-step calls, including state construction and JSON serialization, median latency was 0.817 ms, p95 0.885 ms, maximum 0.964 ms; the largest response was 35,150 bytes. This excludes HTTP, browser painting and sustained idle behavior. Native `/usr/bin/time -l` could not read sandbox-restricted clock metadata after the successful build; no RSS result is claimed. Saved image panels and recorded frames were visually inspected; no browser/device interaction occurred.

| Saved evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| [Root construction and comparison](../artifacts/camera-jog-root-01/report.json) | 11 | `63b198d95b4438347b1b86ea910fd6d14d1327c9bc77a8c4554ae936e99a1b9f` |
| [Manual camera-body run](../artifacts/camera-aware-jog-lab-01/run/manifest.json) | 239 | `b5d8d0f6227d29d4e890f152086146350e6b73557edea4b8e3f61cf5b837b62e` |
| [Interaction and local performance](../artifacts/camera-jog-interaction-01/report.json) | 21 | `8a0320166e4a91eade2ffeeff09167e5a6969c01c1919fc8ed92e10c70f4365f` |
| [Exact comparison reproduction](../artifacts/camera-jog-root-reproduction-01/report.json) | 5 | `93954bbdd4eab0d94b8064478ce5e99d1e94877d0845d630ff5a64896bfda012` |

No paid request, deployment or new inference was needed. The six settled Meta calls still commit $0.166152 of the shared $5 application-side budget, leaving $4.833848 with no unsettled reservation.

Next work: separate remaining pose, body-shape and root-height errors without fitting against the floor score; retain the full interval and failed candidate. Independent physical observations are still needed before claims about human anatomy, contact or changed-action behavior.
