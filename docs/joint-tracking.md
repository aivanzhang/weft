# Experimental joint object tracking

The opt-in `"association_mode": "joint"` now uses **experimental v2**. It changes the suspected incoming-ball wrong branch at frame 2646, but **does not improve aggregate frozen-label accuracy or joint coverage over v1, and does not qualify the three-ball video for reconstruction**. The default remains `independent`.

## V2: broader search, mixed development result

V2 preserves the largest observed displacement as bounded search memory instead of letting repeated or near-stationary frames shrink the window. Sources classified as mutually confusable search the full configured radius. It also records the exact search rectangle, the best visual hypothesis even below threshold, and separate `proposal_status` / `final_status`. These audit coordinates are not accepted measurements. The score floor, configured assignment margin, immutable templates and resource bounds are unchanged.

Twenty-seven analytic tests passed before v2 real-source scoring. In synthetic controls, a 28-pixel reversal after repeated frames changes from missing to an exact observed match; a previously excluded same-appearance contact alternative changes two confident matches to explicit ambiguity. These controls do not establish real-video accuracy.

The saved six-case comparison was then independently checked without decoding video or rerunning tracking. Both 48-file input seals verified; initialized boxes, intervals, source identities and all existing labels are unchanged. Recomputed label arithmetic exactly matches both saved grade reports.

| Three-ball measure | V1 | V2 |
| --- | ---: | ---: |
| Accepted visible labels | 9/22 | 9/22 |
| Accepted-center RMSE / maximum | 1.05 / 1.41 px | 1.05 / 1.41 px |
| All-three tracked coverage | 47/180 (26.1%) | 47/180 (26.1%) |
| Accepted tracked rows | 267 | 264 |
| False-visible labels in retained sample | 0 | 0 |

The unchanged aggregate grade hides a tradeoff: incoming at **2650 gains a graded accepted center with 1.41-pixel error**, while incoming at **2670 loses its previously accepted 1.41-pixel-error center and becomes ambiguous**. Of eight newly accepted rows, only 2650 has a frozen label; seven remain ungraded. Eleven rows lose acceptance, and two accepted coordinates change at unlabelled frames 2646 and 2669. More or different accepted rows are not evidence of accuracy by themselves.

At **2646**, incoming changes from center **(196,124)** to **(184,118)**, consistent with the previously inspected left-moving branch. V2 considers two candidates and explicitly rejects the old box; its assignment margin is 0.1232, not v1's misleading single-candidate 0.7554. The v2 trace uses a 27-pixel window and a different preceding accepted position, so this is not a causal isolation of window width alone. No new truth label or quantitative error is assigned to that frame; physical sphere identity after contact remains inferred continuity. [Detailed diagnosis](joint-loss-diagnosis.md).

All five other cases preserve **every status and accepted position**, not just sparse grades: disks 10/10 at 2.02 px, pool 7/7 at 3.02 px, electrostatic 8/11 at 2.18 px, pendulum 0/11, and all 21 grayscale positions. Pendulum failure remains unresolved. The lower, incoming and upper tracks still become permanently lost at **2708, 2739 and 2754**, respectively. Broader search does not solve immutable-reference appearance loss, boundary changes or calibrated physical reconstruction.

Artifacts: `artifacts/joint-appearance-02` (48-file seal `850e48349eff1dd874358080bd700c6aa071edb55a119d5716b7e7cabcc5d202`) and `artifacts/joint-v2-comparison-01` (four-file seal `d5d45c04c193dac2553e42b693660edd41a1e98201050d4db837662395655af6`). The comparison preserves all changed observations, unchanged-label arithmetic, focused traces and input identities. Existing v1 artifacts remain untouched.

## What changes

The tracker proposes at most six immutable-template matches per object and searches for a conflict-free assignment across all objects. Every accepted match still needs a visual score of at least max(0.70, configured minimum). A bounded local-motion preference can separate candidates; it is an assumption, not a measured position or calibrated probability. That preference is disabled during detected similar-appearance contact and detection gaps. These contact checks are heuristics, not measured physical contacts.

The joint assignment must beat its best alternative by the configured margin (default 0.06). This is **not** the independent tracker's raw visual peak-margin rule. Both values remain separately named in the audit. For nonconfusable sources, the search uses 1.5 object sizes plus bounded maximum observed-hop memory and gap expansion, capped by the configured radius; confusable sources use that full configured radius. This comparison evaluates the complete experimental method, not a causal isolation of the assignment solver.

Missing/ambiguous results contain no accepted box or position. Rejected candidate boxes are retained only as hypotheses under `association_evidence.candidate_hypotheses`. Predictions and optical flow never become measurements. Original templates and their hashes stay immutable. Long losses still require new initialization.

Limits: eight objects, six hypotheses each, at most 20,000 solver nodes shared across the best assignment and alternatives, maximum analysis dimension 960. Exhausting proposals or solver capacity causes abstention rather than an approximate confident match. No new dependencies, downloads, paid calls or recurring services were added. The separate flow prototype explicitly rejects this mode, selected-source intervals and delayed initialization instead of silently misreading their configuration.

## Historical v1 frozen development comparison

Twenty analytic tests were frozen before the implementation agent saw real-source scores or labels. `artifacts/joint-appearance-01/pre-evaluation-freeze.json` pins the integrated sources, contracts and six cases before candidate processing. Source videos, intervals, initialized boxes, labels and reconstruction gates were unchanged; labels are never supplied to tracking.

| Source | Accepted visible labels: independent → joint | Accepted center RMSE: independent → joint |
| --- | ---: | ---: |
| Three-ball collision | 6/22 → 9/22 | 9.64 → 1.05 px |
| Air-table disks | 10/10 → 10/10 | 2.02 → 2.02 px |
| Pool break | 7/7 → 7/7 | 3.02 → 3.02 px |
| Electrostatic balls | 8/11 → 8/11 | 2.18 → 2.18 px |
| Metallic pendulum | 0/11 → 0/11 | Undefined: no accepted labels |
| Grayscale rolling | All 21 accepted positions identical | No new independent manual labels |

All accepted positions on the five non-three-ball cases are exactly unchanged, not merely their sampled grades. No false-visible label was introduced, but the small absent-label set does not establish general visibility accuracy. Pendulum failure remains unresolved.

At three-ball frame 2650, the upper track now matches the frozen upper center exactly rather than the lower center 23.54 pixels away. The lower track remains undetected and incoming track ambiguous at that frame; neither gets an invented position. Maximum accepted-label error falls to 1.41 pixels. All 24 visibility slots remain, including five visible authored/pre-initialization slots that are not detections. Post-contact identities remain inferred continuity, not unique physical-sphere ground truth.

All-three-object coverage rises from 12/180 (6.7%) to 47/180 (26.1%), still below the unchanged 80% requirement. More unlabeled tracked rows (138 → 267) do not establish their accuracy. Lower-ball appearance loss and later exits need further work. Camera calibration and physical parameters remain unverified.

Study seal: 48 files, `b8442871c648443307d27fc3453d95ea4006de995300c809348a02f3fb50bc60`.

## Historical v1 end-to-end verification

`benchmarks/joint-association-v8.json` preserves the prior interval suite and changes only the explicit tracking mode. `artifacts/joint-association-e2e-09` completed in 4.301 seconds with two local workers:

- Analytic long interval: reconstruction error approximately 1.5e-12 px; learned/random/coast success 100%/22%/0%.
- Real-source grayscale interval: reconstruction error 3.12 px; learned/random/coast success 100%/42%/88%.
- Three-ball interval: desired-case execution failure at the unchanged coverage gate; no reconstructed world claimed.

Both successful environments passed changed-action checks and 12-transition/13-frame recording verification, including unchanged nested-run manifests. Suite seal: 227 files, `c7e66b17cb41b65afe63a8e321f5fd69b44bdce28b1b325cc9080133fc47c94f`.

All 487 Python tests passed with no failures/errors/skips in 16.041 seconds (`.local/joint-appearance-tests.xml`). No UI code changed; the prior 16 UI checks remain the last UI verification, and browser interaction remains unperformed. Real tracking timers in this study ranged from 0.043 to 0.998 seconds, excluding preview serialization. These concurrent local measurements are not a speedup or sustained UI responsiveness claim.

## Try it

```sh
# Open a verified interactive environment and record actions.
.venv-lab/bin/python -m weft lab --resume artifacts/joint-association-e2e-09/rolling-grayscale-interval

# Reproduce the six-source comparison in a new directory.
.venv-lab/bin/python scripts/studies/check_joint_appearance.py --out artifacts/my-joint-comparison
```

For another appearance run, add `"association_mode": "joint"` inside its `tracking` configuration. Use a new output directory. Inspect accepted tracks and rejected hypotheses separately; do not treat this option as automatic object discovery or recovered physics.
