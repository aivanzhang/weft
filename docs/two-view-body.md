# Partial two-view body geometry

For three or four supplied cameras, use the [multi-view extension](multi-view-body.md). It preserves this two-view path and reports missing/conflicting observations explicitly.

The manual sandbox can now use two camera views for body dimensions and available joint directions, not just body position. Missing limb directions remain explicitly labeled single-view estimates. **Agreement improves in some respects and regresses in others; this is not validated anatomy, contact, muscle activation or learned exercise behavior.**

The later [upper-segment v2 experiment](two-view-segments.md) uses supported hip/knee or shoulder/elbow pairs when distal endpoints are missing. It improves aggregate agreement but worsens individual poses; this page preserves the original v1 candidates and results.

## Try the combined candidate

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/two-view-body-jog-lab-01-both
```

Open the printed local URL. The **3D body** description shows how many poses have usable two-view estimates for each limb. Manual joint/release/push/support controls and **Record & start** / **Stop & save** remain available; no learned policy is present. The original single-video workflow and Lite default are unchanged.

- [Source/body panels](../artifacts/two-view-body-jog-interaction-01/index.html), including [frame 200](../artifacts/two-view-body-jog-interaction-01/reference-0200.png), [largest gap at frame 174](../artifacts/two-view-body-jog-interaction-01/reference-0174.png) and [worst joint error at frame 148](../artifacts/two-view-body-jog-interaction-01/reference-0148.png).
- [Exact action replay](../artifacts/two-view-body-jog-interaction-01/session/replay.html) and [recorded video](../artifacts/two-view-body-jog-interaction-01/session/video.mp4).
- [Three-candidate comparison](../artifacts/two-view-body-jog-study-01/report.json), [all 2,880 comparison slots](../artifacts/two-view-body-jog-study-01/comparison.json) and [per-frame geometry sources](../artifacts/two-view-body-jog-lab-01-both/run/two-view-body.json).

The separately saved `artifacts/two-view-body-jog-lab-01-directions` and `...-dimensions` labs isolate direction and length changes. All three remain unpromoted development candidates.

## What the two cameras support

The same 240-frame domain retains 112 body poses, 120 unsampled frames and eight unavailable body poses. No frame is dropped or filled. Among the 112 body poses:

| Geometry | Complete two-view estimates | Missing complete estimate |
| --- | ---: | ---: |
| Torso, both shoulders and hips | 112 | 0 |
| Left arm, shoulder/elbow/wrist | 59 | 53 |
| Right arm, shoulder/elbow/wrist | 0 | 112 |
| Left leg, hip/knee/ankle | 90 | 22 |
| Right leg, hip/knee/ankle | 80 | 32 |

`two-view-body-v1` builds torso orientation from the four supported torso points and limb directions from complete three-point limbs. Missing/degenerate groups keep their **original local joint angles**; they can still move in world coordinates when the torso or dimensions change. Straight-limb twist remains assumed, singular angle charts have zero coupled-coordinate confidence, and flexion above 2.6 radians remains clamped and audited. Confidence reflects image-model visibility, not independent anatomical accuracy.

Dimensions are median available segment lengths, requiring at least five distinct frames and the existing body-length bounds. Left/right lengths remain shared: symmetry is an authored assumption, especially for the hidden right arm. No unavailable endpoint is declared observed. The new shins are 0.4402 m versus 0.3774 m originally, and the torso is 0.5683 m versus 0.4936 m; these use supplied reconstructed camera scale, not measured anatomy.

Root trajectory, ground, body anchor, motor limits and support-gain settings stay unchanged. **Changing dimensions changes authored mass/inertia at fixed density**, including mass-scaled support behavior; this does not recover muscle or contact forces. Ground height is never fitted to improve a score.

## Paired results, including regressions

The study freezes all three candidate labs before opening provider body/third-camera comparison inputs. It follows an already-observed coverage/length diagnosis, so it is development evidence, not a new final holdout. Original-video synchronization and shared metric scale remain supplied assumptions. Provider bodies/cameras are reconstructed using multiple views; the comparison is not independent laboratory truth.

All body rows below use the same **1,344 major-joint slots**. Third-camera rows use the same **1,117 confident in-image slots**; the other 1,763 remain unavailable. RMSE is root mean square error, which gives larger errors more weight.

| Comparison | Root-only baseline | Directions only | Dimensions only | Combined |
| --- | ---: | ---: | ---: | ---: |
| Absolute body-joint RMSE, m | 0.1393 | 0.1312 | **0.1440** | 0.1346 |
| Root-relative body RMSE, m | 0.1363 | 0.1300 | **0.1407** | 0.1331 |
| Third-camera projection RMSE, px | 32.97 | 28.95 | 29.38 | 25.19 |
| Maximum absolute joint error, m | 0.6425 | **0.6779** | **0.6814** | **0.7207** |
| Median body/ground gap, m | 0.1884 | 0.1891 | 0.1273 | 0.1278 |

Combined shoulder RMSE improves substantially (left 0.1620→0.0425 m, right 0.1493→0.0501 m), while wrist errors worsen (left 0.1252→0.1539 m, right 0.1798→0.2301 m). There is no automatic winning-model selection or claim that every body part improves.

The combined body still floats in all 112 poses: minimum 0.0229 m, median 0.1278 m, maximum 0.2556 m. It has no penetration at the unchanged 112-source/239-grid checks. Dimensions alone introduce one 0.0156-m intersection, below the existing 0.025-m failure threshold; that intersection is still reported. Passing this threshold does not identify physical foot contacts or airborne phases.

The worst combined error is the left ankle at frame 148. Both hip and knee are estimated, but ankle rays disagree, so this version retains the original whole-leg local angles. Its longer shin then increases endpoint error. Across the clip, usable proximal-only pairs remain unused for 42 left-arm, 22 left-leg and 15 right-leg poses. The separately versioned [v2 candidate](two-view-segments.md) now uses those supported directions while preserving explicit distal/twist uncertainty; it does not fill or relabel the missing endpoints.

## Build with your supplied views

Use the original sealed mobile run and the [root-view configuration](two-view-root.md#build-a-separate-candidate), then choose an explicit geometry mode:

```sh
.venv-lab/bin/python -m weft camera-body \
  --run /path/to/original-mobile-run \
  --camera /path/to/first-camera.json \
  --ground /path/to/ground.json \
  --root-view /path/to/root-view.json \
  --body-view both \
  --out artifacts/my-partial-two-view-lab
```

`--body-view directions` preserves dimensions; `--body-view dimensions` preserves angles and their provenance. Omitting it preserves the root-only behavior. All require exact source/camera/clock bindings; no automatic calibration, synchronization, inference, training or server startup is performed.

The new model embeds its original root-only parent and pins the output angles, dimensions, evidence and audit. Root-only validation remains unchanged: geometry changes use a different method, not a bypass of its shape invariant. Repeated candidate chains and mismatched inputs are rejected. The lab preserves raw monocular estimates, both-view triangulation, the original root-only model under `inputs/two-view-root-body.json`, and the new `two-view-body.json` audit separately.

The complete development study can be reproduced with new outputs:

```sh
.venv-lab/bin/python scripts/studies/check_two_view_body_jog.py \
  --out artifacts/two-view-body-study-new \
  --lab-prefix artifacts/two-view-body-lab-new
```

## Verification, performance and cost

**100 distinct scoped Python checks pass:** 27 new body tests and 73 root/body/ground checks. The first run's new analytic fixture used NumPy scalar coordinates where the input contract requires JSON-native numbers, causing 26 fixture failures before body construction. Converting fixture coordinates to ordinary floats fixed them; one representative case and then the remaining 25 failed cases were run, without repeating the 74 passing checks. Records: `.local/two-view-body-tests.xml`, `two-view-body-fixture-tests.xml`, `two-view-body-corrected-tests.xml`. An initial study-helper import typo failed before creating any run; `.local/two-view-body-jog-study.log` preserves it.

**29 deterministic UI checks pass**, including partial-limb disclosure, no extra geometry fetches and continued manual-only controls. The small scene-label addition after the initial Python run is covered by the later package/recording and UI checks; no dynamics code changed. Browser/device interaction remains unperformed.

Two bounded local workers build the three candidates in **2.756 s total**; individual CLI wall times are 1.291–1.404 s. No inference is repeated. The combined model JSON is 1,555,451 bytes; its geometry audit is 583,155 bytes, and its compact scene metadata is 7,799 bytes. The UI receives counts in scene metadata, not the full geometry audit on status polls.

The combined candidate's actual **39 actions, one reset and 41 states/video frames** replay exactly, including every reward and completion flag. Across 120 in-process action/state/JSON calls, median latency is **0.771 ms**, p95 **0.786 ms**, maximum **0.801 ms**, maximum response 35,172 bytes. This is not HTTP/browser or sustained-idle evidence, and it is not a controlled speed comparison with earlier runs. Frame 200, largest-gap and worst-error panels were visually inspected.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Directions-only lab run | 245 | `d3e6b3704d3f00f2ee8e8c6a63c521354358777501b63e804c34abfed3865102` |
| Dimensions-only lab run | 245 | `3e696a9afea31ee94758c7d82d5f87e0eec53812d5258965798126e051b432ee` |
| Combined lab run | 245 | `d451c3057f930c48c73b6a0f6f71082e2cb3f2257df21cfc64863807060e7e15` |
| Paired comparison study | 12 | `31fa27a418490e8651f8d10e7dea9e5e7dd688839558d286158004b1cf7a54a5` |
| Interaction study | 23 | `72e12287d01525f06141beaf668d1ae7046c8ede6bb0d20ceed248e808ca0ffa` |

Session `session-d99bde1d27bb` has recording seal `d7613180bd6e686e2ee5f790a78c7da75a5b8cea27f47f0fd46e577b7ca07d38`. Reuse its saved export via `check_camera_body_lab.py --recording`; all previous studies and inputs remain unchanged.

At **October 29, 2026, 1:01 AM PDT**, the Meta ledger remains six settled requests, **$0.166152 of $5**, $4.833848 remaining and no unsettled reservation. This work used saved local inputs: no downloads, paid calls, training, browser interaction, server startup or deployment. The broader diverse-video, physical-accuracy and changed-action-validation goal remains incomplete.
