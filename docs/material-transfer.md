# Contact materials and a reserved repeat

**The fitted 3D scene predicts a second trial substantially better, but still fails the frozen accuracy check.** Later image-center error falls from 61.66 to 8.51 pixels; five of six source-only points pass instead of two. The green box still travels too far, and projected object widths remain wrong.

**[Open the repeat comparison and playable sandbox](../artifacts/spatial-ramp-repeat-01/index.html).** It includes the complete source comparison video, original and fitted 3D replays, recorded controls, every source slot and the failed checks. The [first-trial construction comparison](../artifacts/spatial-material-fit-01/index.html) shows all fitted alternatives.

## Try the candidate

From the repository root, use a new recording directory:

```sh
.venv-lab/bin/python -m weft scene-sandbox \
  --run artifacts/spatial-ramp-repeat-01/candidate \
  --record artifacts/my-repeat-ramp-session
```

Try `right 6`, `forward 6`, `back 6`, `brake 6`, `reset` and `quit`. The recording folder contains the current image, complete actions/state, exported video and offline replay. Nothing advances while typing. Controls are virtual forces; their predictions have not been tested against real interventions. Existing exercise labs and defaults are unchanged.

## What changed

The earlier scene follows the ramp descent, then loses too much speed on the table before impact. It assumes one sliding-friction value for the ramp, table and both objects. The new construction study separates the two table-contact coefficients and target mass while keeping camera, geometry, gravity, opening motion, lateral offset, ramp friction and contact-solver assumptions fixed.

| Parameter | Original hypothesis | Construction-selected hypothesis |
| --- | ---: | ---: |
| Wood/table sliding friction | .25 | .1211 |
| Green box/table sliding friction | .25 | .0750 |
| Green-to-wood relative mass | 1 | .9173 |
| Wood/ramp sliding friction | .25 | .25 |

These are conditional fit parameters, **not measured material properties**. Equal-priority MuJoCo contacts take the maximum of the two geometry coefficients. Setting the table coefficient to zero and using each moving object's coefficient permits distinct table contacts without changing the wood/ramp pair. The wood/green collision coefficient remains the maximum of those moving-object coefficients. [MuJoCo contact rules](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters).

`material_fit.py` provides bounded material alternatives, source-bound construction-slot validation, complete coast simulation and source-image scoring. It rejects evaluation/holdout rows, missing slot accounting, mismatched clocks and incomplete trajectories. It is an explicit study helper, not automatic lab repair.

The already-exposed first trial is now entirely construction data. A frozen coarse grid followed by five bounded local refinements evaluates 68 new alternatives, reusing the original baseline. Every alternative retains 88 object/frame slots and its recording. All complete; no failed alternative is silently omitted. The selected first-trial error is 6.01 px versus 52.47 px. Changing only wood/table friction gives 30.98 px; changing only target friction gives 50.99 px; changing only mass gives 52.36 px. The combined result is not evidence that the three parameters are uniquely identified: ten alternatives lie within three construction-error pixels of the selected result. That range is not a statistical confidence interval.

The 514-file [construction study](../artifacts/spatial-material-fit-01/report.json) is sealed as `7e2bd2d7b40eb9883c05efc1b2bd10f15223eea0cc723ffcd2a54a8668159b07`. Search and exports take 3.831 s with two bounded local workers. No model inference, API call or policy training is repeated for this fit.

## How the prospective repeat was protected

1. Freeze the chosen scene, initialization algorithm, comparison rules and source-only labeling protocol **before inspecting repeat pixels**. [Protocol](../artifacts/phys101-repeat-protocol-01/protocol.json), seal `b17e73f2edf3de706df2bcd7d5ae4dabb0a36c4fd2ebca9707763d35a188fc0a`.
2. Inspect only source frame 8 to initialize two tracking boxes. A fixed background-camera check passes: correlation .99934, maximum linear-transform difference .000157 and translation .501 source pixels. No correction is applied; this is not metric calibration.
3. Make one Meta SAM3.1 request using the unchanged original 47-frame public clip. Both objects return all 39 frames from initialization onward; the sixteen earlier object/frame slots remain unavailable. Original IDs and the unresolved proximity group remain explicit.
4. Save six approximate source-only centers at frames 26, 36 and 46, each with twelve-pixel uncertainty, before viewing repeat predictions or simulation overlays. [Labels](../artifacts/phys101-repeat-labels-01/labels.json), seal `2c4d475f8b467bb8b6fdaedda357baf2b688ecd164eeaaa4ee4dd229f4adfc1e`. These are nonexpert image annotations, not measured 3D truth.
5. `ramp_transfer.py` receives only the complete prefix through frame 18. Fixed gravity/friction determine downhill acceleration; a small least-squares fit initializes wood position/speed and the target's initial horizontal position. No geometry, camera, gravity, mass or friction changes. All eleven actor and eleven target samples are required; maximum prefix error must remain below twelve pixels. Actual maximum is 2.17 px, RMSE 1.01 px.
6. Freeze both repeat scenes before grading. The original-material baseline shares the candidate's exact initial state. Predict every later frame, then compare frames 19–46. No settings, identities or thresholds change after the result.

The repeat is **now exposed**. It cannot be reused as an untouched future holdout. Improving a later model against this repeat would make it development evidence; a new prospective claim needs separate reserved evidence.

## Repeat results

**October 29, 2026, 10:21 AM PDT.** Each condition retains all 94 object/frame slots: sixteen pre-initialization, twenty-two prefix-initialization and fifty-six later comparisons.

| Check | Original materials | Fitted materials | Frozen requirement |
| --- | ---: | ---: | --- |
| Later center RMSE | 61.66 px | 8.51 px | Reported, not the sole gate |
| Later 95th-percentile error | 105.16 px | 19.26 px | ≤12 px |
| Later maximum error | 105.16 px | 19.76 px | ≤24 px |
| Source-only points within uncertainty | 2/6 | 5/6 | 6/6 |
| Overall center-motion check | Fail | **Fail** | All requirements together |

The wood's later RMSE improves 45.77→2.95 px; the green box improves 74.23→11.67 px. At the last source-only point, the green box still misses by 17.76 px. All six Meta centers agree with the approximate labels within their uncertainty; this does not verify every intermediate identity or a physical center of mass.

The additional [geometry diagnostic](../artifacts/spatial-material-geometry-01/report.json) reuses saved states and source boxes. At the repeat's final frame, projected widths are 1.274× the observed wood box and 1.227× the green box. Height ratios are close to one. Improved center motion does not establish correct cuboid depth, orientation or collision shape. This diagnostic does not replace or relax the frozen center-motion check.

## Verification, cost and limits

- Twenty-five new distinct analytic checks pass: eighteen material-fit/slot-contract checks and seven prefix-transfer checks. They cover recovery against analytic motion, unchanged inputs, withheld-row rejection, missing observations and complete episode boundaries. Saved results are in `.local/material-fit-tests-01.xml`, `.local/material-fit-collision-test-02.xml`, `.local/material-fit-boundary-test-01.xml` and `.local/ramp-transfer-tests-01.xml`. The first file preserves one superseded failure: soft contact does not exactly reproduce an instantaneous perfectly inelastic collision. The corrected check verifies the true mass ranks distinctly ahead of alternatives.
- The first-trial selected recording replays 71 exact states. The repeat replays 77 baseline, 77 candidate and 26 control/reset states exactly: **251 distinct states across these checks**. The repeat's simulation, grading and exports take 2.567 s locally. No browser/device responsiveness claim follows.
- All seven new/exported videos decode, totaling 413 frames. Inspection MP4 metadata rounds 30000/1001 to 29.97 fps, introducing at most 1.535 microseconds of display-clock difference. Original videos, source clocks and state journals are unchanged. The initially overstrict metadata assertion and its diagnosis are retained; successful recording checks were reused.
- One new Meta call takes 5.395 s and bills fifty frames for **$0.01**. The prepared and executed request hashes match; no retry, Contributor mode or Files upload is used. Eight settled calls total **$0.186152 of $5**, leaving **$4.813848**, with no unsettled reservation. [Official SAM pricing](https://dev.meta.ai/docs/pricing-rate-limits). [Saved live result](../artifacts/meta-phys101-repeat-live-01/report.json).
- The 51-file [repeat study](../artifacts/spatial-ramp-repeat-01/report.json) seal is `93dcb80c998e42419a0bf9179c8a1063bd765940b13d1ce36821ff8efbafcbb9`. Final audit `.local/material-transfer-final-verification-01.json` verifies source IDs/clocks, all 5,984 new construction slots, complete repeat slots, 113 runtime hashes, frozen constructor bindings and recorded spending.

This is one object pair, camera and ramp. No default is promoted. Next work must constrain source-supported 3D geometry and impact behavior, preserve this failed prospective result and reserve new evidence before another generalization claim. Real-ball diversity, exercise floor/anatomy errors, learned whole-exercise control and independently verified changed actions remain open.

## Initial-shape follow-up

**[Inspect the initial silhouettes](../artifacts/initial-ramp-geometry-02/index.html).** The original width code already projects mask pixels onto the rotated object axis: a suspected axis-aligned-width bookkeeping bug is not present. A cuboid under an uncertain camera still need not match that enclosing extent.

The explicit `primitive_geometry.py` helper tests 31 local-width alternatives per object against only the saved first-trial frame-8 masks. Camera, centers, heights, depths, orientations and material settings stay fixed. It chooses the best discrete mask intersection-over-union, with ties favoring the least width change. All alternatives remain available; no later trajectory or repeat observation drives the fit.

| Object | Original mask overlap | Fitted overlap | Selected width change |
| --- | ---: | ---: | ---: |
| Wood block | .8803 | .9022 | −8% |
| Green box | .9431 | .9431 | None |

The camera projection's depth column is zero, so this opening silhouette cannot identify depth. A separate wood-depth candidate combines the provider-reported 33 mL volume with a uniform cuboid and the conditional scale implied by assumed physical gravity 9.81. It gives depth .02814 instead of .04 scene widths; this is **not independent metric calibration or a measured depth**. The green depth stays unmeasured.

Thirteen distinct checks pass across the saved `.local/primitive-geometry-*.xml` reports. They cover analytic overlap, known-width recovery, unchanged fields, construction-only scope, conditional volume, invalid inputs and allocation bounds. The earlier reports retain NumPy 2D-cross-product and scalar-type failures; only affected checks were rerun. The first study attempt also retains a PNG-file-versus-decoded-pixel hash guard failure before any fitting: nine-file seal `1f0e8662cc19482bfaacda62757ebe7f24d8ce4c9e7d66471a34b4e7ab207d70`. Recovery verifies both representations correctly.

The successful [17-file study](../artifacts/initial-ramp-geometry-02/report.json) seal is `290943f75aac438d3d24ff82d05373b1ab1e6cd0684e9378987a93350f990b54`. Local fitting takes .0126 s, excluding source decoding and export. The source comparison has been visually inspected. There is no new simulation, model inference, policy training, API charge or default change. These shapes still need complete trajectory/contact checks before another material search or prospective trial; the earlier failed repeat remains unchanged.

## Complete-motion check rejects both shape changes

**[Inspect the complete comparison and recorded controls](../artifacts/initial-geometry-motion-01/index.html).** Both opening-shape candidates now have complete first-trial rollouts with all material, camera and starting-motion settings fixed. The original recording is reused. Neither candidate is fitted or selected against this comparison, and the separate failed repeat is not reused.

| Shape | Center RMSE | Mean wood mask overlap | Mean green mask overlap |
| --- | ---: | ---: | ---: |
| Original | 6.01 px | .7810 | .8778 |
| Narrower wood | 8.44 px | .8273 | .8647 |
| Narrower wood + conditional depth | 8.49 px | .8421 | .8650 |

Both changes improve eight center comparisons, worsen forty-two and leave twenty-two unchanged. Maximum individual regressions are 9.90 and 10.29 pixels. The simulated impact appears on source-grid frame 30 instead of 29; these are simulator contacts, not independently labeled physical impact times. The conditional-depth candidate's final wood width ratio improves 1.300→1.138, but the green ratio remains 1.210 and its motion gets worse. **Better silhouettes alone do not qualify either shape as a repair.** Both remain inspectable, unpromoted experiments.

All 264 mask slots across three conditions and 176 paired center slots across the two changes remain. The two new coasts and two control/reset sessions replay **170 states exactly**. Five exported videos decode **214 frames**, including all 44 source-comparison frames. Simulation and coast exports take 1.081 s with two workers; this excludes later grading, controls and source-comparison rendering. The 56-file bundle seal is `d81a01b360c1935a7e1cc2740e7b4531da9c3a4d8fedc62dcb9a402c1130300f`. Audit: `.local/initial-geometry-motion-verification-01.json`.

No API call, model inference, policy training, default change, browser or server is involved. Current evidence calls for better camera/depth/contact constraints rather than promoting an opening-outline fit. The material-only candidate and its failed prospective result stay unchanged.

The subsequent [second-view camera/depth study](conditional-rigid-camera.md) fits approximate opening corners and tests one explicit orthographic-depth hypothesis. Front center RMS improves only 6.006→5.968 px while both side-view alternatives worsen. It preserves all observations, materials and the failed repeat, and does not promote the new playable candidate.
