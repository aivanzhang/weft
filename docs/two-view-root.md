# Two-view body position, unchanged monocular shape

The new manual sandbox uses two camera views to position the body's hip midpoint. It improves position agreement on the jogging sample without changing limb shape, joint angles, muscles, forces or the supplied ground. **The body still floats; this is not recovered anatomy, validated contact or learned jogging.**

The later [partial two-view body experiment](two-view-body.md) separately tests dimensions, available joint directions and their combination. It retains this original root-only candidate and its results; mixed improvements and regressions prevent a blanket accuracy claim.

## Try the saved candidate

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/two-view-root-jog-lab-01
```

Open the printed local URL. Inspect the source/body poses, use manual joint/release/push/support controls, and record with **Record & start** / **Stop & save**. No learned policy is available. Muscle bands illustrate motor commands, not measured muscle activity.

- [Source/body panels](../artifacts/two-view-root-jog-interaction-01/index.html), including [frame 200](../artifacts/two-view-root-jog-interaction-01/reference-0200.png) and [the largest ground gap](../artifacts/two-view-root-jog-interaction-01/reference-0022.png).
- [Exact action replay](../artifacts/two-view-root-jog-interaction-01/session/replay.html) and [recorded video](../artifacts/two-view-root-jog-interaction-01/session/video.mp4).
- [Comparison report](../artifacts/two-view-root-jog-study-01/report.json) and [all 2,880 source/joint slots](../artifacts/two-view-root-jog-study-01/comparison.json).

This opt-in candidate requires two pose sequences, supplied camera calibration, declared synchronization and shared world coordinates, plus ground height. It does not change the usual single-video workflow or its Lite default.

## What changed

`two-view-root-body-v1` takes the midpoint of hips 23/24 from the existing guarded two-ray reconstruction. Both hips must be usable for **every original retained body pose**; otherwise the entire candidate is rejected. No pose is dropped, filled or replaced with a fallback root. The original Full-model angles, dimensions, body orientation, motor limits, support gains and missing samples stay unchanged. Ground is not fitted to the body.

The new origin and root translations are pinned to per-frame hip coordinates, original source identities/times, both camera/evidence hashes and explicit world/clock provenance. A shape/support hash rejects changes beyond root position. Repeated refinement, first-pose floor fitting, unsupported whole-body learning and the raw-camera foot adapter remain prohibited.

`camera-reference.json` remains the **original monocular estimate**, byte for byte. The lab separately saves `two-view-root.json`, every slot in `two-view-triangulation.json`, copied second-view inputs and `inputs/monocular-camera-body.json`. Nothing relabels the old estimated depth as a two-view measurement.

The primary camera is the exact original calibration JSON, preserving its ground binding. Its numerical transforms match the earlier Full two-view experiment, whose different metadata string produced a different canonical hash. No guard was weakened to accept that mismatch.

## Results on the same source slots

All 240 frames remain accounted for: **112 retained body poses, 120 unsampled frames and eight unavailable body poses**. Body comparisons use exactly 1,344 common major-joint slots. RMSE is root mean square error, which gives larger errors more weight.

| Reconstructed-reference comparison | Original Full body | Two-view root |
| --- | ---: | ---: |
| Hip-midpoint position RMSE, 112 poses | 0.3020 m | 0.0481 m |
| Root height RMSE | 0.1182 m | 0.0301 m |
| Absolute body-joint RMSE, 1,344 slots | 0.3220 m | 0.1393 m |
| Body shape relative to root, RMSE | 0.1363 m | 0.1363 m |
| Median body/ground gap | 0.2954 m | 0.1884 m |
| Minimum / maximum body/ground gap | 0.1336 / 0.4676 m | 0.0462 / 0.3348 m |

Every native body point changes by precisely the root translation, within 3.57×10⁻¹⁵ model metres. The unchanged shape metric confirms that this does not fix limb geometry. All 112 poses still have positive ground gaps. Passing the unchanged 112-source/239-grid penetration checks does not establish correct foot contacts or airborne phases.

Provider joints, camera scale and ground are reconstructed, not laboratory truth. Original-video synchronization remains assumed, as documented in [the two-view study](multiview-pose.md). This root candidate was motivated by prior descriptive comparison, so it is development evidence, not a fresh final holdout. Provider body data is excluded from construction and opened only after the candidate is sealed. Drone3 remains outside construction; this study does not repeat its projection comparison.

## Build a separate candidate

Create a JSON config like this, resolving file paths relative to the config file:

```json
{
  "schema_version": 1,
  "first_name": "camera1",
  "second_name": "camera2",
  "second_evidence": "second-run/evidence.json",
  "second_camera": "second-camera.json",
  "clock_provenance": "Describe why the exact packaged frame indices are aligned, including any uncertainty.",
  "world_provenance": "Describe the shared metre/z-up camera basis and scale, including any uncertainty."
}
```

```sh
.venv-lab/bin/python -m weft camera-body \
  --run /path/to/original-mobile-run \
  --camera /path/to/first-camera.json \
  --ground /path/to/ground.json \
  --root-view /path/to/root-view.json \
  --out artifacts/my-two-view-root-lab
```

The source must be the original sealed mobile run, not an already camera-bound candidate. The primary evidence/camera must match exactly. Inputs are bounded: 64 KiB config, 32 MiB second evidence, 8 MiB second camera; no automatic clock fitting, inference, training, network call or server startup. Second-view source identity is supplied in its evidence/calibration, not independently authenticated by downloading another video.

To reproduce this saved development comparison with new outputs:

```sh
.venv-lab/bin/python scripts/studies/check_two_view_root_jog.py \
  --out artifacts/two-view-root-study-new --lab artifacts/two-view-root-lab-new
```

## Verification and cost

**73 distinct scoped checks pass:** 33 new root/packaging checks and 40 existing camera-body/ground checks. One initial missing-frame fixture kept stale reference rows; only that fixture was corrected and its failed test rerun. Both records remain: `.local/two-view-root-tests.xml` (72 pass, one fixture failure) and `.local/two-view-root-missing-tests.xml` (one pass). Production code did not change after those checks. Analytic tests verify known metric roots despite biased monocular scale, unchanged native geometry, explicit missing slots, invalid inputs/tampering, source-change rejection and exact recording.

The actual CLI build took **1.047 s wall time**, **0.691 s internally**. A new recording contains **39 actions, one reset and 41 states/video frames**, with every state, reward and completion flag replayed exactly. Across 120 in-process action/state/JSON calls: median **0.806 ms**, p95 **0.822 ms**, maximum **0.829 ms**, maximum response **35,116 bytes**. This excludes browser/HTTP painting and sustained idle behavior. Panels for frame 200, the largest gap and the unsampled final frame were visually inspected without browser interaction.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Manual lab run | 242 | `325fd37a2a0d856949c4d63513769b318f2af79558d02a65123f6543d3201eed` |
| Construction/comparison study | 9 | `64ac3834334ed7da05526a587ac824f4e3c88b7c5121a6a824b9e77b2be68664` |
| Interaction study | 22 | `3923ac1f0076d03f1b4f94ef55038b7ccd7f1407ab66d8f999d4e89717295e63` |

Session `session-fc3da48ef032` has recording seal `1c03b51be231fb16acece920af9c904136ec3997fe26828f1b0bb740f415e553`. Reuse this export via `check_camera_body_lab.py --recording` rather than generating another verification recording. Original inputs, candidates and studies remain unchanged.

This work uses saved inputs and local geometry only: **zero downloads, inference calls, training or paid requests**. Meta spending remains $0.166152 of $5, with $4.833848 remaining and no unsettled reservation, last checked **October 29, 2026, 12:34 AM PDT**. No deployment, browser or server was started. Remaining work is source-supported limb/foot geometry, timing/contact validation, broader ball/general-video coverage and independent changed-action validation.
