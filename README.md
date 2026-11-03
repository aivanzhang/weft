# Weft

**Current guide:** [PROJECT.md](PROJECT.md) contains setup, architecture, status and remaining work. Older detailed notes are indexed in [documentation history](docs/HISTORY.md).

Generate executable training environments from evidence. Test them against observed outcomes. Repair the environment when the evidence disagrees.

Weft generates and repairs reinforcement-learning environments using real-world evidence. It is a Loom-adjacent project with its own identity. This **separate hackathon repository** is adjacent to, not a fork of, the existing `../loom` data-ingestion product.

The original application used the name Loom. Historical application text and source snapshots retain that name; the project is now called **Weft**.

## Product direction

**One video → an object-based world model → an interactive simulator → a trainable RL environment.**

- [Product goal](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/goal.md): the full A–E design, physics-grounded relationships, and independent evaluation informed by WorldBench.
- [First-version scope](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/v1.md): the bounded product milestone, deliverables, build order, and completion criteria.
- [Hackathon pitch](hackathon/pitch.md): polished vision and prototype copy, with evidence boundaries and a demo storyboard.

These describe the target and its presentation. The implementation status below remains separate.

## Start locally

The landing page and interactive sandbox run as separate local environments:

| Environment | Run from this repository | Address |
| --- | --- | --- |
| Landing page | `npm run dev` | `http://127.0.0.1:3000/` |
| Video upload / 3D sandbox | `npm run sandbox` | Open the authenticated link printed for port `8765` |

Use separate terminals. The landing page needs only Node.js 22+; no npm install
is required. It serves only public site assets and never starts Python, uploads
videos or calls AI APIs. Refresh the browser after edits. The sandbox uses the
existing `.venv-lab` Python environment and `.local/portal` project storage.
This is process/dependency separation, not container isolation. Neither command
deploys anything; the hosted site remains offline.

To reopen the saved hackathon demo rather than the default project collection:

```sh
npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal
```

Existing projects, recordings and spending ledgers are preserved. Do not run two
sandbox processes against the same project directory. See [sandbox setup and
options](sandbox/README.md) and [landing-page development](website/README.md).

These saved examples are ready in this workspace; their local artifacts are not
included in a fresh clone. Run one command, then open its printed local link:

| Try | Command |
| --- | --- |
| Recommended exercise body | `.venv-lab/bin/python scripts/play_sandbox.py --example exercise` |
| Body resting on a static support | `.venv-lab/bin/python -m weft lab --resume artifacts/static-support-play-acceptance-01/present/lab` |
| Guided video → 3D ball with learned control | `.venv-lab/bin/python -m weft lab --resume artifacts/spatial-guided-play-acceptance-02/workflow/lab` |
| Trained 3D tabletop objects | `.venv-lab/bin/python -m weft lab --resume artifacts/spatial-v2-play-01/authored` |
| Trained 3D ramp hypothesis | `.venv-lab/bin/python -m weft lab --resume artifacts/spatial-v2-play-01/ramp` |
| Manual object-video lab, no training | `.venv-lab/bin/python -m weft lab --resume artifacts/manual-video-acceptance-01/lab` |
| 3D ball hypothesis with source video, manual only | `.venv-lab/bin/python -m weft lab --resume artifacts/spatial-scene-lab-acceptance-01/lab` |

Choose manual or learned control where available, start movement, and use
**Record & start → Stop & save** to save actions, video and interactive replay.
**Recording evidence** shows what was actually verified. For terminal-only
control, add `--console` to the exercise launcher or replace `lab` with
`lab-console` in the other commands. [Controls and your own video](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md).

The [three-camera reconstructed-body lab](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/reconstructed-camera-learning.md#learning-comparison-and-playable-result)
also supports manual and trained joint control, matched baseline comparisons,
recording and exact replay. It uses explicit body-position assistance; it does
not establish unassisted balance or physical contact accuracy.

To compare original and fitted joint targets, use the
[fitted-control sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#compare-a-fitted-reference-without-changing-the-task).
It keeps the same body and task, shows the saved comparison, and records your
actions. Fitted targets are programmed controls, not a learned policy.

The [separate arm/leg feedback lab](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#try-separate-arm-and-leg-feedback)
adds an experimental learned controller for the person-and-equipment example.
Controls, recording, exact replay and fresh reopening are checked; equipment
stability still fails. The recommended exercise preset is unchanged.

The exercise body has illustrative muscles and virtual root assistance. The
tabletop scene is authored; the ramp geometry remains physically unverified.
Successful simulated tasks and exact replay do not establish accurate reconstruction.

The new [static-support example](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#inspect-static-support-loads)
shows actual simulated contact loads, coverage and gaps. Its authored body rests
on a support with translational root assistance disabled; lowering the support
causes the unchanged task to fail. Both recorded trials replay exactly. This is
a controlled simulator example, not a recovered exercise or free-floating human
balance. [Build from saved video-body evidence](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/static-support.md).

For a guided object-video workflow, use
[`build_sandbox.py --profile spatial`](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-scene-construction.md).
It connects explicit tracking, 3D construction, optional training or saved-policy
reuse, and lab preparation, with a step-by-step artifact index. The lower-level
`spatial-scene` command also accepts saved planar runs. Both preserve the original
evidence and require explicit depth, material and task assumptions; neither
recovers those missing properties automatically.
The new ball example has authored zero gravity and no floor; it demonstrates
construction, controls and verified recording, not recovered rolling physics.
Its preselected learned policy succeeds on 8/12 changed starts; a fixed untrained
feedback controller succeeds on 12/12. The complete guided workflow and recording
pass, but these results do not show a benefit from training.

To inspect the remaining body-fitting errors, open the
[saved comparison page](artifacts/whole-body-reference-views-01/image-first/index.html).
It preserves missing estimates and rejected results; it does not replace the working sandbox.
For your own saved body, [run a bounded proposal and generate its comparison](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-model-options.md#propose-a-correction-from-a-saved-body).

To see whether a fitted reference helps the **unchanged simulation task**, open
the [recorded controller comparison](artifacts/reference-controller-trial-cli-01/comparison/index.html).
It includes both 3D replays, videos, actions and exact replay checks. Both trials
finish but still fail equipment stability; the working sandbox is unchanged.
[Run the same workflow on another saved proposal](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#compare-a-fitted-reference-without-changing-the-task).

## Brand assets

The selected logo, deep-clay palette, and mascot animation sources are in [assets/brand](assets/brand/README.md). Open [the mascot preview](assets/brand/preview.html) locally to watch the on-load weave formation and try the blink and wave sprites. The local evidence viewer uses the approved logo and palette; saved experiment folders remain unchanged.

## Website

The [hosted Weft site](https://weft.ivanzhangofficial.workers.dev/) is temporarily offline at the user’s request. Its paired source-video/reconstructed-environment replay and private submission service are preserved for later restoration. Open [website/index.html](website/index.html) directly for the offline gallery and hidden animated footer; uploads require the hosted service. All website assets and Cloudflare deployment code live in this repository. See [website/README.md](website/README.md) and [hosting/README.md](hosting/README.md).

## Current state

**Joining the project:** read the [current combined handoff](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/combined.md) and [parallel workstream boundaries](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/parallel-workstreams.md). Video reconstruction and simulation/RL build one Weft system under the current integrator; the closed agy session's [handoff](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/AGY_HANDOFF.md) is historical context. Local `artifacts/`, `.local/` and model caches are not included in a Git clone; the handoffs include fixture-based work that does not depend on them.

A CLI-launched **local lab** now shows video processing step by step, exposes the evidence and results, runs an interactive Python simulation, and records actions/state with video and offline replay exports.

**Try the person-only 3D example:** `.venv-lab/bin/python scripts/play_sandbox.py --example person`. [Report and current limits](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#try-the-person-only-3d-example).

**Bring a person-only exercise video:** run `.venv-lab/bin/python scripts/build_sandbox.py --video /path/to/video.mp4 --person-only --root-assistance --train --serve`. It builds with the existing local pipeline, saves the step-by-step report, and starts the lab only after success. A rejected input keeps its report and does not become a playable world. `--person-only` is an explicit no-equipment declaration; root assistance is a virtual control, not recovered balance. [Options, failure inspection and recording controls](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#build-from-your-person-only-exercise-video).

For a person and equipment in one world, use the [human-only exercise sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/shared-exercise.md). The latest lab covers the **complete available 8.47-second body reference**, with 254 continuous actions, passive equipment and recording/replay. Its Source vs simulation card separates video coverage, reference time and live simulation time.

```sh
.venv-lab/bin/python scripts/play_sandbox.py
```

Run from the repository root, open the complete printed link and keep the terminal running. Choose **Sandbox → Searched joint corrections → Start**. To save a fresh trial, choose **Record & start → Stop & save**; replay and video links appear below. For manual control, choose **Move it yourself**, select a human motor and tap **Motor +0.25 / −0.25**. The launcher keeps this lab on its verified runtime while development continues; recordings stay in the same lab folder. [Launch details and source inspection](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md).

The current task-priority controller completes the full interval but still fails equipment stability on **3/3 fresh starts**. Its worst equipment displacement is **44.3% lower** than the previous reward-selected policy, but still exceeds the unchanged limit. Exact full-interval recording/replay passes. The earlier two-second task's 3/3 success does not transfer; neither result proves grasp or physical accuracy. The [shape comparison](artifacts/silhouette-shared-world-03/shape-comparison.html) retains geometry assumptions and overlap failures.

Under saved recordings, choose **Explain this trial** to inspect limit values and the first recorded crossing, with each reset kept separate. The person example adds body tracking, including averages that recovered after an earlier crossing; older frozen examples retain equipment-only explanations. This reads saved data without changing the live simulation or running another trial.

The [equipment assembly command](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/equipment-proposals.md) accepts a saved mobile-body run or lab directly through `--human-run`; no separate shared-world adaptation is needed. With saved equipment evidence and explicit camera/shape choices, `--full-reference` creates one continuous task and one lab. The bounded construction-only body snapshot runs no physics or new inference; no old policy is attached. Existing prepared worlds still use `--run`. The [conditional shape command](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/shared-world.md#propose-a-shape-from-saved-silhouettes) adds bounded mask-based proposals and an offline comparison. Camera, depth and unobserved properties remain assumptions; unsupported inputs stop with inspectable reasons.

- **Exercise videos (current focus):** local pose estimation → a source-bound mobile 3D body in the shared MuJoCo world, with 19 joint controls and three explicitly optional virtual root controls. Inspect poses and assumptions, compare controllers, move the body, and record/export/replay. Muscle bands are illustrative, not measured anatomy, activation or force. The older eight-joint anchored-body lab remains available separately.
- **Tabletop objects:** bounded distinct-color tracking → a shared Pymunk 2D rigid-body scene, bounded damping/initial-speed proposals, reconstruction checks, and actual Q-learning. The included multi-object demo is clearly labeled synthetic.
- **Original rolling profile:** the existing real-yellow-ball fixture and its evaluation/repair behavior remain available.
- **Appearance regions:** explicitly selected video regions can drive the same planar sandbox without color masks. Shapes and physics remain authored assumptions; [tracking scope and example](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/general-tracking.md).

**This is not arbitrary-video reconstruction, accurate anatomical recovery, or validated physical transfer.** Controls, goals, masses, and other unobservable parameters remain authored assumptions. Automated physical human-model repair and learning the complete exercise are not implemented.

The optional **19-axis spatial body** adds sideways limb motion, twist and moving torso orientation, with verified interactive recordings. [Try the completed spatial exercise run](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-human.md). Pelvis position remains anchored; source identity errors and physical-contact validation remain open.

The optional **mobile body** adds three translation axes, simulated floor collisions, release/push/support controls and complete recordings. Three real exercise clips pass the current checks; jumping jacks and lunge retain explicit failures. [Open the mobile squat sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/mobile-human.md). Floor placement and support forces are authored assumptions, not recovered balance or measured muscle activity.

The experimental [whole-body sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/whole-body-learning.md) trains explicit 22-motor feedback, compares it with zero/random/hand-written controllers and records manual actions with visual replay. Packaged whole-body labs now expose all 22 motors in the visual interface; ordinary exercise labs retain their separate single-joint learner. Full-reference floor conflicts are shown explicitly. Known reference motion and virtual root forces do not establish a recovered human exercise policy.

Optional `--pose-identity temporal` corrects short, strongly supported left/right swaps while preserving raw detections and an audit. [Open the corrected exercise example and inspect its checks](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-identity.md). This does not establish anatomical or biomechanical accuracy.

### Diverse-video checks and Meta access

The [15-video development suite](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/diversity-verification.md) has 10 environments passing declared gates, two expected synthetic rejections, and three unresolved real-video failures. Passing execution/servo checks is not comprehensive physical accuracy. The [shared observations contract](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/scene-observations.md) is the foundation for broader perception.

```sh
.venv-lab/bin/python -m weft benchmark benchmarks/video-diversity-v1.json --out artifacts/diversity-new
# Fresh setup only: enter a Meta key privately; do not overwrite the configured project key.
.venv-lab/bin/python -m weft meta-configure --budget 5 --key
```

[Meta advice](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-advisor.md) prepares source-grounded interaction questions without network by default. Live calls require a configured key, explicit execution and both per-request and shared-total budget checks. Suggestions never alter the independent evaluator or automatically become physics facts.

Live reasoning and [SAM tracking comparisons](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-tracking-results.md) are verified. The toy-van comparison matches four frozen noninitialization center labels, but the [three-ball comparison](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-ball-results.md) exposes identity disagreements despite complete returned coverage; camera/edit/physics limitations remain. Anonymous locations match closely, and one frozen absence label conflicts with source imagery; neither finding changes the original identity grade. The [current handoff](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/combined.md) records the latest verified spend within the approved $5 total; the persistent project ledger must be checked before another paid request. Exercise advice proposes camera, geometry and support checks, not verified physics. The [image-direction body candidate](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-image-refinement.md) improves independent visual checks from 18/28 to 24/28 but remains unpromoted because floor regressions persist. A [fresh body-shape diagnostic](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/fullbody-geometry-diagnosis.md) rejects all three static shape alternatives on later poses.

Use the [explicit tracking CLI and offline viewer](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/meta-tracking-cli.md) to inspect up to eight objects frame by frame. Preparation and viewing are free/local; `--execute` is required for a bounded paid request. The new [silhouette viewer](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/decoded-masks.md) displays decoded masks on exact source frames. Open the [68-frame exercise result](artifacts/meta-exercise-silhouette-masks-02/index.html) or the [joint/silhouette comparison](artifacts/exercise-silhouette-agreement-01/index.html); disagreements remain visible and do not alter the 3D body.

New [full-interval checks](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/full-interval-reconstruction.md) keep later motion and missing frames visible instead of silently shortening the planar reconstruction. The sandbox shows exact coverage. The follow-up [initial-speed repair](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/initial-motion.md) reduces the two long controls from 15.72/14.68 px to 3.22/1.64 px, now passing the stricter study target. Three cases complete interaction/recording; the expanded real-ball interval and later-direction-change controls remain rejected. Original failures are preserved, and these scoped passes are not generalized reconstruction success.

The offline [`tracking-evidence` bridge](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/segmentation-evidence.md) exports saved proposals with exact timestamps and persistent identity-uncertainty groups into the shared scene schema. Unknown camera, support, force, and actor properties remain unknown; it does not automatically promote tracking into physics.

[`tracking-scene` and `scene-sandbox`](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/segmentation-planar-candidates.md) add explicit planar hypotheses, separate location/identity replay diagnostics, terminal controls and recording. A synthetic collision completes this path; the real three-ball source remains rejected for unresolved camera drift.

[`tracking-stabilize`](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-stabilization.md) adds bounded fixed-reference camera correction with unchanged raw evidence. The latest image-appearance checks qualify 43/141 required three-ball frames (22 via seeded flow) and 2/354 toy-van frames; neither generates a corrected scene. Accuracy and spatial-support gates remain unchanged.

The [four-video body-length experiment](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/pixel-body-shape.md) includes an [offline side-by-side viewer](artifacts/pixel-body-shape-view-01/index.html). Inspect source images, original bodies and rejected candidates. Better image agreement does not guarantee valid floor contact; no candidate replaces the working models.

A [new real cylinder test](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/cylinder-source.md) retains depth motion and hand occlusion: the fixed-size appearance baseline misses three of four scored centers and exceeds tolerance on the remaining one. It is preserved as a failure, not a generated environment. The [opt-in scale experiment](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/scale-tracking.md) recovers one additional scored center but worsens another; it does not generate physics. [Foot-reference overlays](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-reference.md) preserve missing feet and expose projection errors. The separate [ankle/foot sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-sandbox.md) now provides 28 direct motor controls and close-up recordings; its capsule-foot candidates still fail floor checks and do not replace the default body.

Use [`feet-build` on your completed mobile video run](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-sandbox.md#build-from-your-own-video) to create a separate candidate, control it and record your actions. An optional [image-direction correction](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/foot-image-refinement.md) resolves 16 single-leg floor failures, but all tested models still fail the full floor check. [Inspect the verified CLI recording](artifacts/feet-build-cli-01/session/replay.html).

A new [jogging-video sandbox and 3D reference comparison](docs/multiview-jog.md) retains all 240 source frames. IMAGE mode creates an interactive body and an exactly replayed twelve-action recording; default VIDEO mode rejects competing poses. Eight sampled poses are missing, relative 3D disagreement averages 12.4 cm where comparable, and floor checks fail. The multiview reference is itself reconstructed, not measured anatomical truth.

The optional [camera-aware jogging sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-aware-body.md) uses supplied calibration to separate camera movement from body travel. It preserves the full trajectory, follows the body in the viewer, and exactly replays 39 actions plus a reset. Paired travel error drops from 10.51 m to 0.25 m against the reconstructed reference, but floor failures worsen from 5 to 17 of 112 poses. It remains an untrained, unvalidated manual candidate; this is not video-only recovery.

The latest [supplied-ground example](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/camera-ground.md) fixes a 31-cm ground-coordinate mismatch without changing the estimated body. Penetration checks pass, but the body still floats and one source pose is visibly inverted. The UI now shows body/ground gaps, and the new 39-action recording replays exactly. Correct coordinates and passing collision checks do not establish accurate anatomy or contact.

A separate [two-view pose experiment](docs/multiview-pose.md) uses actual image evidence from two cameras, reserving a third for comparison. Paired third-camera error improves, but hidden joints and the bad torso pose remain unavailable. It has not replaced the sandbox body; supplied calibration and assumed synchronization are additional inputs, not single-video recovery.

The optional [Full pose model and new jogging sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/full-pose-model.md) correct the inverted frame and preserve exact interaction/recording. Four complete inputs expose visibility regressions as well as improvements, so Lite stays the default. The new manual body still floats; it has no learned policy or physical-accuracy approval.

The newer [two-view-root sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-root.md) improves body position while keeping Full's limb shape and missing poses unchanged. Root/reference RMSE drops from 30.2 cm to 4.8 cm, but every pose still floats (median gap 18.8 cm). Its 39-action recording replays exactly; supplied calibration and assumed synchronization remain additional inputs, not proof of physical accuracy.

The [partial two-view body sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-body.md) adds camera-supported lengths and directions with visible per-limb coverage. Three separate candidates retain missing directions as single-view estimates. The combined model improves comparison-camera agreement and reduces the median gap to 12.8 cm, but wrist and worst-joint errors worsen. Recording remains exact; no candidate is promoted as validated anatomy or learned exercise behavior.

The opt-in [upper-segment sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/two-view-segments.md) also uses visible hip/knee and shoulder/elbow pairs when an ankle or wrist is missing. Combined body-joint RMSE improves 13.5→12.8 cm, but 46 individual joint/frame errors worsen, including a large ankle regression. Complete limbs, ground and unobserved bends remain unchanged; exact recording and explicit uncertainty labels are verified. This is still experimental, not accurate contact or recovered muscles.

The latest [every-frame exercise study](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/dense-exercise-poses.md) adds 240 local pose estimates while reusing 240 unchanged observations. The proposed limb-label repair still lacks enough evidence, and a newly observed bad hip pose prevents building the denser sandbox. The existing playable body stays unchanged; the new source images, failure diagnostics and 99 passing software checks are saved separately.

The follow-up [crop-recovery sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-crop-recovery.md) re-estimates four inconsistent poses from original image regions and now builds **224 poses across all 240 frames**. Its 39-action/reset recording replays exactly. The remaining 16 missing poses stay explicit; accuracy on the earlier poses is nearly unchanged and every body still floats above the supplied ground. This experiment uses two supplied calibrated views, not single-video recovery of camera calibration or anatomy.

The [complete five-exercise crop study](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/selective-crop-exercises.md) retains 1,340 sampled poses across 2,677 source frames and produces playable **squat and push-up** candidates with exact recording/replay. Push-up gains five body poses; two other clips retain competing hypotheses, and jumping-jacks retains movement gaps. Squat floor conflicts and unverified body/contact accuracy remain explicit. The guide includes launch commands and a source/body comparison page; no default changes.

The optional [Heavy-model squat sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/heavy-pose-model.md) improves checked body directions from 11/16 to 15/16 and verifies controls plus exact recording/replay. Five complete videos expose substantial coverage losses on push-ups and jumping-jacks, so it is not a default upgrade. Squat floor conflicts remain; the guide includes a five-step walkthrough and saved comparisons.

The follow-up [image-position comparison](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-pixel-refinement.md) includes an offline source/body viewer across five existing bodies. Better image fit still produces new floor and limb-direction failures; these candidates are retained for diagnosis without replacing the working sandboxes.

A new [three-camera human-motion source](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/calibrated-human-sample.md) provides physical board dimensions and supplied camera parameters. The original monocular board check fails; separate [stereo predictions and full motion inspection](docs/freemocap-body-evidence.md) expose incomplete joints. The [frozen third-camera comparison](docs/freemocap-fixed-lengths.md) passes its board check but rejects the fixed-length body correction. Cam3 is now exposed development evidence, not an untouched holdout. The floor remains unqualified; no new body or policy is admitted.

The [three-view follow-up](docs/freemocap-three-view.md) reuses all saved observations without new inference. Two-camera pixel fitting adds 88 supported major-joint estimates but slightly worsens comparison-view accuracy; requiring all three cameras exposes additional conflicts. Inspect the source/skeleton panels and retained failures. Neither method replaces the working exercise labs.

The latest [joint and ground diagnosis](docs/freemocap-joint-ground-diagnosis.md) rules out simple global swaps and clock shifts as general repairs. A regional feature correction recovers previously missed carpet features, but not a floor; six crop proposals improve some joints while retaining torso failures. Inspect the original/cropped source panels and unchanged limits. No candidate replaces the working exercise labs.

The new [real ramp comparison](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/phys101-ramps.md) adds stationary and sliding/contact wooden-block videos. One $0.01 Meta request tracks both objects throughout the moving clip, covering all six frozen comparison points where local tracking covers two. An offline silhouette viewer shows every frame. The new [tilted-support sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/oriented-supports.md) adds analytically checked ramp physics, terminal controls and exact recordings. Its geometry, scale and friction remain authored assumptions; an accurate video-derived environment is still unverified.

The [source-derived ramp follow-up](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/source-ramp-reconstruction.md) fits opening tracked motion, creates nine playable friction/mass hypotheses and overlays simulated boxes on the source. Later collision errors remain visible; no hypothesis is promoted as accurate. It reuses the saved Meta results without new API calls.

The experimental [3D rigid-object sandbox](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-rigid.md) adds box/ball rotation, sideways and vertical forces, complete state/contact recordings, and front/top/angled inspection views with zoom and following controls. [Open its recorded examples](artifacts/spatial-ramp-views-01/index.html). Twelve-object execution is checked; the source-linked ramp collision still mismatches, so this is not a validated 3D reconstruction. Exercise labs are unchanged.

The [contact-material follow-up](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/material-transfer.md) fits separate table contacts on the first trial, freezes the method, then tests a reserved repeat. Later center error improves **61.66→8.51 px**, and source-only points improve **2/6→5/6**, but the green box overshoots and the frozen accuracy check still fails. [Open the prospective comparison and playable candidate](artifacts/spatial-ramp-repeat-01/index.html). One new Meta tracking call costs $0.01; no thresholds or materials are changed after the repeat result.

The [second-view camera/depth check](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/conditional-rigid-camera.md) adds reusable conditional camera and shape fitting, then compares one opening-derived box-depth change against both views. Front center error slightly improves, but both side-view alternatives worsen; no model is promoted. [Inspect the comparison and interactive candidate](artifacts/ramp-conditional-depth-01/index.html). Manual corner hypotheses are explicit, all source slots remain, and no new API call occurs.

The [automatic image-feature check](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/surface-feature-diversity.md) applies one unchanged method to 684 frames across ramp views, a toy van, three balls and an exercise clip. [Inspect all five videos](artifacts/surface-feature-diversity-01/index.html). Missing and ambiguous matches remain visible: tiny balls provide no features and some rigid-object motion loses its matches. This adds reusable observations, not a new environment or proof of generalization; the working labs are unchanged.

The [adjacent-frame history follow-up](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/feature-track-diversity.md) retains local appearance links through more motion while splitting fourteen original-reference conflicts. [Inspect its videos](artifacts/feature-track-diversity-01/index.html). Missing opening anchors, featureless-ball failure and source edits remain explicit; improved link coverage is not verified 3D motion or physical identity.

The later [shape-and-collision comparison](artifacts/initial-geometry-motion-01/index.html) improves wood silhouettes but worsens first-trial center error **6.01→8.44/8.49 px** with materials fixed. Both geometry changes remain unpromoted. Their complete 3D replays, interactive commands and 170 exactly replayed coast/control states are available without another API call.

The optional [inclined-floor implementation](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/ground-orientation.md) keeps body physics and recordings consistent for angled ground hypotheses. Four exercise sources failed source-foot orientation screening; no inferred correction was applied.

The initial two reasoning requests used approximately $0.03951 of the $5 application budget; the later tracking calls are included in the total above. The first truncated response is retained; the second produced hypotheses that were independently checked, including rejected coordinate/confidence claims. No model suggestion was automatically applied.

### Original single-joint demo

This older anchored demo controls one joint. For the current full-body workflow,
use [Start locally](#start-locally) and the
[person-only video guide](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#build-from-your-person-only-exercise-video).
The original pose model, licensed clip and completed run remain available:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/exercise-lab-03
```

Open the printed local URL. To process a video through this **legacy single-joint** workflow:

```sh
.venv-lab/bin/python -m weft lab --video /path/to/exercise.mp4 --out artifacts/my-exercise-01
```

The **Start here** shortcuts open the captured movement, manual sandbox, or learned-controller demo. Choose **Record & start**, then **Stop & save** to export a trial.

Read the [local lab guide](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md) for stage descriptions, recording controls, tabletop inputs, fresh setup, and limits. See [verified lab results](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/lab-verification.md) for actual evidence. The website is a separate presentation; it is not the processing lab.

The larger physical-first A–E scope remains the target. The original rolling-ball clip and synthetic tabletop scene are engineering fixtures; exercise videos are now an explicit user priority. No existing Loom code, data, credentials, or Git history was copied. This local lab expansion was not deployed or submitted; the separately hosted website remains intentionally offline.

- [Scope and architecture](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/architecture.md): target capabilities, component contracts, and current limits.
- [Iteration workflow](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/iteration.md): change one hypothesis, run, inspect, and compare.
- [TODO and acceptance gates](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/TODO.md): verified state and the next independent work items.
- [Execution cost and bounds](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/v1-cost-and-scope.md).

## Run locally

The commands below retain the original restricted rolling workflow. For the full exercise/tabletop lab use [.venv-lab setup](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/local-lab.md#reproduce-the-environment). Tested on Python 3.14.7 / macOS arm64. Node.js is needed for physics parity and deterministic UI checks. Other Python versions/platforms are not yet verified against the lockfile.

For a fresh checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m weft fetch-sample
```

Reuse an existing valid environment. The sample is hash-pinned and already included locally; fetching verifies it without downloading again when present.

```sh
.venv/bin/python -m pytest
.venv/bin/python -m weft experiment experiments/rolling-ball-no-repair.json --out artifacts/baseline-01
.venv/bin/python -m weft experiment experiments/rolling-ball.json --out artifacts/candidate-01
.venv/bin/python -m weft compare artifacts/baseline-01 artifacts/candidate-01 --out artifacts/comparison-01.json
.venv/bin/python -m weft verify artifacts/candidate-01
```

Use a new output name for each run. Open `artifacts/candidate-01/index.html` directly for the local evidence viewer; no development server is needed. Alternatively, step the Python environment:

```sh
.venv/bin/python -m weft interact --world artifacts/candidate-01/world.json
```

Default runs use local CPU computation with **$0 external API/cloud charges**. Meta is not used by default. Optional live reasoning and segmentation are verified on saved examples, with explicit execution and the shared budget ledger. No credential presence alone authorizes a paid call; this development run has an explicit $5 total authorization.

## What a run preserves

Source attribution and hash; frame-level evidence and partitions; the object graph; initial/selected worlds; task definition; every repair decision; changed-action trials; training records and policy; evaluation trials; replay/video viewer; resolved experiment settings; implementation source snapshots; and a file-hash manifest. Failures preserve a `failure.json` instead of silently falling back.

Comparison requires matching evidence and scoring implementations for reconstruction. RL comparisons additionally require the same fitted world, task, implementation, and evaluation seeds. Results from different simulated worlds are descriptive—not proof of improved real-world performance.

## Hackathon reference archive

Intended entry: **Indie Developer / Push the Frontier** at the Meta Global AI Developer Hackathon. The original reference analyses below predate the one-video A–E scope; use [architecture](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/architecture.md) and [TODO](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/TODO.md) for current implementation planning. Official snapshots remain unchanged.

1. [Hackathon reference hub](hackathon/README.md)
2. [What the judges want and how Loom fits](hackathon/analysis/fit-and-strategy.md)
3. [Requirements, dates, and unresolved conflicts](hackathon/analysis/requirements.md)
4. [Proposed demo and evaluation design](hackathon/analysis/demo-plan.md)
5. [Prior work versus hackathon work](hackathon/analysis/originality.md)
6. [Application context](hackathon/analysis/application-context.md)

The submission deadline is **November 4, 2026, at 6:00 p.m. PST**. Source snapshots retain the organizers' original timezone wording; planning documents use America/Los_Angeles.

## Avatar authoring

The [avatar workflow and browser library](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/avatar-authoring.md) now live in `tools/avatar-assets/`, with editable panda/dragon, robot, fox and lamp models, portable previews, a reusable skill and verification records. Its environment stays separate from Weft's simulator dependencies.

```sh
tools/avatar-assets/run gallery
```

Open `http://127.0.0.1:8765/` yourself. These are appearance previews; detailed fur/PBR assets are not silently substituted into Weft physics or its constrained renderer.

## Repository boundaries

- Private GitHub repository: `kintsugi-software/weft`. No public release or hackathon submission. The separately hosted website remains intentionally offline.
- `hackathon/sources/`: source snapshots and explicit unavailable-content records.
- `hackathon/analysis/`: interpretation and proposed project direction, not official rules.
- `website/`: canonical product site, evidence-backed environment gallery, styles, and interactive artwork.
- `hosting/`: Cloudflare Worker, private upload storage, bounded generation container, and release configuration.
- `src/weft/`: local A–E profile, experiment runner, and comparison tools.
- `experiments/`: versioned run configurations; input paths resolve relative to each config.
- `tests/`: contract, physics, provenance, repair, and comparison checks.
- `data/`: attributed source fixture; see [source provenance](data/README.md).
- `artifacts/`: ignored local run directories. Preserve runs; never edit scores to improve a result.
- `scripts/`: reference archival tooling.
- `.local/`: ignored raw HTTP responses and supporting artifacts; never publish blindly.
- Source documents retain their original ownership and terms. A public-code license is not selected yet; do not apply a future project license to third-party reference material.

The event requires a public repository at submission. This repository has **not** been made public. Publication and submission remain separate future actions.
