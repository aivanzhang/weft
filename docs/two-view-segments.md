# Partial limbs with explicit uncertainty

The opt-in `two-view-body-v2` candidate uses a supported shoulder-to-elbow or hip-to-knee direction even when the wrist or ankle is unavailable. It retains the original bend and an assumed twist around the bone. **Average agreement improves, but some individual poses worsen substantially. This remains an experimental manual sandbox, not validated anatomy or exercise physics.**

The later [correspondence diagnosis](two-view-correspondence.md) finds confident left/right label disagreement in the problematic frames. Most comparable pairs support both label assignments under the existing ray checks; no automatic swap or threshold change has been applied.

A separate [every-frame crop-recovery sandbox](temporal-crop-recovery.md) now retains 224 poses instead of 112, with exact recording. It fixes the later dense run's hip-ray failure using fresh image inference, not left/right permutation. Accuracy on these original poses remains nearly unchanged, and ground contact is still unverified.

## Try it locally

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/two-view-segment-jog-lab-01-both
```

Open the printed local URL. The **3D body** description separates complete limbs from upper-only estimates. Use the joint, release, push and support controls; **Record & start** / **Stop & save** preserve your actions and results. There is no learned exercise policy. This command starts the server only when you run it.

- [Source/body panels](../artifacts/two-view-segment-jog-interaction-01/index.html), [recorded actions](../artifacts/two-view-segment-jog-interaction-01/session/replay.html) and [recorded video](../artifacts/two-view-segment-jog-interaction-01/session/video.mp4).
- [Largest regression, frame 104](../artifacts/two-view-segment-regression-02/regression-0104.png), [regression counts](../artifacts/two-view-segment-regression-02/report.json) and [frame 148](../artifacts/two-view-segment-jog-interaction-01/reference-0148.png), which still has the largest joint error.
- [Paired comparison](../artifacts/two-view-segment-jog-study-01/report.json), [all 2,880 source/joint slots](../artifacts/two-view-segment-jog-study-01/comparison.json), [unchanged-geometry checks](../artifacts/two-view-segment-jog-study-01/segment-invariants.json) and [per-frame geometry audit](../artifacts/two-view-segment-jog-lab-01-both/run/two-view-body.json).

The [complete-limb v1 candidates](two-view-body.md) and [root-only candidate](two-view-root.md) remain separately saved and playable. The original single-video workflow and Lite default are unchanged.

## What changes

First, the existing v1 adapter constructs all fully supported limbs. A separate pass rotates an incomplete limb's original orientation by the smallest rotation that matches the available upper segment. It rejects a degenerate segment or an ambiguous opposite direction. It does not change a complete limb, infer the missing endpoint into the evidence, or claim the twist is measured.

The retained bend is numerically unchanged. The twist is a **transported single-view assumption**, not necessarily an unchanged Euler angle. All three coupled rotation-coordinate confidence values are zero; the observed pair's image confidence and component-level provenance are stored separately. High image confidence alone is not proof of correct 3D anatomy.

The same 240-frame interval retains 112 body poses, 120 unsampled frames and eight unavailable poses:

| Limb | Complete | Upper-only correction | Entire direction remains single-view |
| --- | ---: | ---: | ---: |
| Left arm | 59 | 42 | 11 |
| Right arm | 0 | 0 | 112 |
| Left leg | 90 | 22 | 0 |
| Right leg | 80 | 15 | 17 |

Both new candidates apply 79 limb/pose corrections. Compared with their corresponding frozen v1 candidate, all dimensions, root motion, ground, body anchor, forces and other angles remain identical. Native simulator points outside the targeted limbs are exactly unchanged; supported unit directions agree within 1.83e-14. The complete-limb reference hash matches v1 before the extra pass.

The combined candidate still uses v1's shared left/right dimensions, including its symmetry assumption and dimension-dependent authored mass/inertia. The correction does not recover muscle forces, foot contact or unknown anatomy.

## Paired results and regressions

All body errors use the same 1,344 available joints. Third-camera errors use the same 1,117 confident image slots; 1,763 slots remain unavailable. RMSE is root mean square error, which weights larger errors more strongly.

| Comparison | v1 directions | v2 directions | v1 combined | v2 combined |
| --- | ---: | ---: | ---: | ---: |
| Body-joint RMSE, m | 0.1312 | 0.1282 | 0.1346 | 0.1281 |
| Root-relative RMSE, m | 0.1300 | 0.1272 | 0.1331 | 0.1266 |
| Third-camera RMSE, px | 28.95 | 28.56 | 25.19 | 24.44 |
| Maximum joint error, m | 0.6779 | 0.5719 | 0.7207 | 0.6053 |

For the combined candidate, 112 joint/frame errors improve, **46 worsen** and 1,186 remain unchanged. Directions alone improve 97 and worsen 61. At frame 104, combined left-ankle error worsens **0.1448→0.5427 m**, and its third-camera error worsens **13.63→96.25 px**. The hip/knee pair is available, but ankle rays disagree; transporting the upper-leg direction while keeping the old bend/twist gives a worse ankle. No post-hoc rejection threshold was introduced to hide this result.

The maximum combined joint error remains at frame 148: 0.7207→0.6053 m. Its third-camera error nevertheless worsens 34.46→40.58 px. The entirely prior-driven right arm is unchanged, with right-wrist RMSE 0.2301 m. These outcomes prevent a blanket improvement or full-body accuracy claim.

Every retained combined pose still has a positive ground gap: minimum 0.0229 m, median 0.1278 m, maximum 0.2539 m. No penetration is found at the unchanged 112-source/239-grid checks. This does not identify foot contact or distinguish true airborne phases from reconstruction error.

The candidates were sealed before comparison inputs were opened. The experiment follows an already-observed failure, so it is development evidence, not a fresh final holdout. Provider bodies, camera scale and ground are reconstructed, not laboratory truth; original-video synchronization remains assumed. Drone3 stays outside construction, but the provider's calibration used multiple views. No additional real-video diversity or physical intervention was tested.

## Build a separate candidate

Use the original sealed mobile run and the [source-bound root-view inputs](two-view-root.md#build-a-separate-candidate):

```sh
.venv-lab/bin/python -m weft camera-body \
  --run /path/to/original-mobile-run \
  --camera /path/to/first-camera.json \
  --ground /path/to/ground.json \
  --root-view /path/to/root-view.json \
  --body-view both --partial-segments \
  --out artifacts/my-upper-segment-lab
```

`--partial-segments` also works with `--body-view directions`; dimensions-only and missing root-view inputs are rejected. Omitting the flag retains v1 behavior. Repeated candidate chains are rejected; the root-only parent and strict source/camera/clock bindings remain intact.

Reproduce the bounded comparison with new output directories:

```sh
.venv-lab/bin/python scripts/studies/check_two_view_body_jog.py \
  --partial-segments \
  --out artifacts/two-view-segment-study-new \
  --lab-prefix artifacts/two-view-segment-lab-new
```

## Verification and cost

**128 scoped Python checks pass in 6.769 s**, including synthetic known-direction cases, preserved unobserved bend, complete-limb invariants, rejected ambiguity, strict bindings and actual package/recording checks. **30 deterministic UI checks pass**, including complete versus upper-only coverage and no extra full-audit requests. Records: `.local/two-view-segment-tests.xml` and `.local/two-view-segment-ui-tests.log`. No browser/device interaction was performed.

Two bounded local workers built both candidates in **1.434 s total**, without repeated inference. The combined model JSON is 1,574,978 bytes, geometry audit 670,834 bytes and compact scene metadata 8,076 bytes. Status polling does not download the full audit.

The combined sandbox's **39 actions, one reset and 41 states/video frames replay exactly**, including every reward and completion flag. The interaction study took 1.372 s. Across 120 in-process action/state/JSON calls, median latency was 0.795 ms, p95 0.810 ms, maximum 0.822 ms and maximum response 35,170 bytes. These are not HTTP/browser or sustained-idle measurements, nor a controlled speed comparison. Source/body frames 148 and 200 and the frame-104 regression panel were visually inspected.

| Evidence | Files | Manifest SHA-256 |
| --- | ---: | --- |
| Directions lab run | 246 | `fc8235fcc3ee806fefe796693a7ec5859b2a0859212b3b489204f46e43f53c98` |
| Combined lab run | 246 | `a6e491a187bfeb9c4385b1a94f34c1dbc5d160102ea90282d3029fe051453406` |
| Paired comparison | 11 | `24ccc9b769f28465b5d5988298ff5b502355d4e8bd48de70482138aba6d7908d` |
| Interaction study | 23 | `051f8264386c20d7fd2351c8e4855f0718bf27318b310cf99f1c9db2bbca11d1` |

Session `session-9c879c044923` has recording seal `12d0f754c25bcd937bf087120357785527d672de2a161cd8dd65e1e0fe35aaf9`. Reuse its export with `check_camera_body_lab.py --recording` rather than recording again. The first regression panel had overlapping footnotes; `two-view-segment-regression-01` remains preserved, and `...-02` corrects only that layout using the same sealed inputs.

At **October 29, 2026, 1:31 AM PDT**, the local Meta ledger still has six settled requests, **$0.166152 of $5** committed, $4.833848 remaining and no unsettled reservation. This work made no paid calls, downloads, new inference, training, browser interaction, server startup or deployment. The next work must address partial-limb failures and independently supported contact/geometry while preserving the broader ball/general-video and changed-action-validation requirements.
