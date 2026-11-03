# Video diversity development suite

Run the complete local pipeline on the pinned inputs:

```sh
.venv-lab/bin/python -m weft benchmark benchmarks/video-diversity-v1.json \
  --out artifacts/video-diversity-new --workers 2
```

Each case creates a new lab, runs inference/tracking, reconstructs a world, trains/evaluates its controller, tries changed actions, records those actions, exports and decodes the recording, and verifies the unchanged run manifest. Open any successful case with `weft lab --resume artifacts/video-diversity-new/CASE-ID`.

The manifest contains **15 cases**: eight independent analytic ball scenes, five real exercise clips, a real rolling-ball clip, and a suspended electrostatic-ball scene. Source files, rights, SHA-256 identities, conditions, expected outcomes, training budgets and gates are explicit. Hash mismatches fail before work starts. The pose model is a separate pinned local prerequisite; use `weft fetch-pose-model` if missing.

## What a result means

- **gates_passed:** the environment ran and met every declared gate. This is not a universal physical-accuracy claim.
- **gates_failed:** it ran, but one or more declared measurements failed or were unavailable.
- **execution_failed:** a desired environment could not be created, verified or recorded. Failure artifacts remain available.
- **expected_rejection:** a deliberately unsupported input failed with the specified perception error type and message. Crashes, timeouts, integrity failures and exporter failures never count as successful abstention.
- **execution_only:** it ran without declared acceptance gates. This does not count as a passing accuracy test.

The two synthetic negatives deliberately contain ambiguous colors and camera translation. Lunge/twist, jumping-jacks/burpees and the electrostatic clip remain **desired environment cases** even where the current implementation fails them; they are not relabeled successful rejections to improve the score.

The analytic sources come from closed-form motion equations, not Pymunk output. Their separate `truth.json` is never supplied to reconstruction. [Ball study](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/ball-diversity.md) compares tracks and simulated motion with that independent truth. The end-to-end suite additionally measures same-video reconstruction, executable actions and recording integrity. Real exercise coverage is measured model output, **not independent pose accuracy**. Servo error measures how well programmed motors follow estimated joint angles, not biomechanical realism. RL rates are printed separately from fidelity and compared against random/no-control baselines.

## Bounds and evidence

At most two local workers; each has the lab's 300 CPU-second / 360 wall-second bounds. Each interactive recording has 12 or fewer action steps and a separate export subprocess killed after 90 seconds. The initial suite uses 1,000 training episodes and 50 matched evaluation episodes per policy/case; budget exhaustion is preserved by the worker. Upper worker CPU budget: 4,500 seconds across 15 cases, not an expected duration. Actual case and total elapsed times are reported. Local CPU/storage only; $0 external API/cloud calls, no recurring services. Electricity/storage are not represented as zero-cost resources.

`report.json` and `report.md` summarize outcomes; each case has its own result, worker log, pipeline artifacts and recording. Benchmark results and recorded states must agree exactly, and before/after run-manifest hashes must match. Existing output directories are never overwritten. A rerun is a new development study, not a fresh final holdout.

General appearance-independent tracking, calibrated contacts/materials, grounded human dynamics, changed-action physical validation, and comprehensive arbitrary-video reconstruction remain open requirements.

## Affected-path follow-up

[explicit-modes-v3.json](explicit-modes-v3.json) selects four exercise cases with explicit `pose_mode` and one grayscale appearance-region case with required `config`. It preserves the original thresholds; jumping jacks still fails its servo gate. The grayscale source is an existing real clip transformed, not independent evidence. [Results](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/iteration-followup.md) include the earlier IMAGE-mode regression and all exclusions.

[spatial-human-v4.json](spatial-human-v4.json) exercises the explicit 19-axis model on four real clips. `spatial_reference_rmse_rad` is separate from legacy angle coordinates; both use a 0.35-rad declared threshold, but the scores are not directly comparable. Source-view limb fidelity is evaluated with a separately frozen grader. [Results](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/spatial-human.md).

[temporal-identity-v5.json](temporal-identity-v5.json) adds explicit short-span left/right correction to the same four spatial cases and retains lunge/twist as a desired case. It preserves raw detections, fixed gates and unresolved multiple-pose failures. [Identity checks](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-identity.md) use frozen source-visible shoulder/hip directions separately from simulation performance.

[appearance-guards-v6.json](appearance-guards-v6.json) checks source-color compatibility and stronger shared camera alignment. Seven affected cases preserve three working environments, one expected camera negative and three desired real collision failures. [Results and cost](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/appearance-guards.md) distinguish better tracking abstention from successful physical reconstruction.

[source-interval-v7.json](source-interval-v7.json) checks explicit sections of longer files, absolute provenance, timestamp cadence, pre-initialization coverage and recordings. Two environments pass; the newly ingestible real three-ball interval still fails tracking coverage. [Results and completeness limits](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/source-intervals.md).

[joint-association-v8.json](joint-association-v8.json) repeats those three cases with explicit experimental joint assignment and unchanged gates. Two environments pass controls/recording; three-ball tracking improves but still fails coverage. [Frozen six-source comparison and limitations](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/joint-tracking.md).

[mobile-exercise-v9.json](mobile-exercise-v9.json) adds image-root translation, actual simulated floor contacts and explicit virtual support. Three of five desired exercise cases pass; missing root evidence and competing-person hypotheses remain failures. [Independent image-position checks, recording and limits](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/mobile-human.md).
