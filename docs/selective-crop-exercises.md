# Complete-video exercise crop results

**Two new single-camera exercise candidates are playable: squat and kettlebell push-up. Three other clips remain rejected.** The push-up candidate gains five usable body poses. These are inferred bodies with authored physics and illustrative muscle bands, not verified anatomy or muscle activity.

## Try it

Start the push-up sandbox yourself:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab
```

For squat, replace `kettlebell-pushup` with `squat`. Open the printed local URL, select a joint, try its controls or reference playback, and use **Record & start** / **Stop & save**. Squat has known body/floor conflicts; a working control or learned policy does not resolve them.

No server is needed to inspect the [comparison page](../artifacts/selective-crop-bodies-comparison-01/index.html), including source/body panels, launch commands and saved interaction replays.

## Results across all five clips

The study retained all **2,677 source frames** and **1,340 original sampled pose slots**. It used the pinned Full IMAGE-mode pose model throughout. Unlike the earlier [selected-frame screen](temporal-crop-recovery.md#follow-up-five-single-view-exercises), crop decisions now use the complete original Full-model sequence as their context.

| Video | Usable poses before → after / sampled | Accepted crops | Mobile body result |
| --- | ---: | ---: | --- |
| Single-leg squat | 115 → 115 / 118 | 0 | Rejected: competing hypotheses in three samples |
| Squat | 107 → 107 / 107 | 0 | Playable; 55/107 reference poses violate the floor limit |
| Kettlebell push-up | 116 → 121 / 129 | 5 | Playable; 0/121 reference poses violate the floor limit |
| Jumping-jacks/burpees | 581 → 584 / 678 | 5 | Rejected: missing body movement across long gaps |
| Lunge/twist | 297 → 300 / 308 | 3 | Rejected: competing hypotheses in five samples remain |

The 13 selected replacements add 11 usable poses; two replace already usable but inconsistent estimates. All other **1,327 observations remain unchanged**. There are 31 crop attempts, with all rejected proposals retained. Competing hypotheses are never collapsed into a chosen person.

Jumping-jacks still has **13 reference intervals longer than the fixed 0.25-second limit**, reaching **1.201 seconds**, plus one unfilled root-position gap. Both original and corrected bodies fail. The clip was not trimmed, gaps were not fabricated, and an anchored body was not substituted for a mobile body.

### Follow-up: jumping-jacks recovery

A later local pass uses one fixed crop derived from the original poses across the complete video. It retries all 94 remaining unusable samples, accepts 37 new estimates and preserves the other 641 observations exactly. Three usable control samples are inspected but never replaced. The new module passes 17 analytic checks; 97 local model calls and complete source decoding take 6.375 seconds.

| Stage | Usable / sampled poses | Gaps over 0.25 seconds | Longest gap | Unfilled body-position gaps |
| --- | ---: | ---: | ---: | ---: |
| Prior selective crops | 584 / 678 | 13 | 1.201 s | 1 |
| Fixed whole-video crop | 621 / 678 | 8 | 0.400 s | 0 |
| Add unused gap frames, whole image | 625 / 717 | 6 | 0.400 s | 0 |
| Add unused gap frames, same crop | 634 / 717 | 4 | 0.334 s | 1 |

The last two rows are separate, prespecified diagnostic conditions, not combined repairs. Both infer the same 39 previously unsampled source frames inside the eight remaining gaps and preserve all 678 existing observations exactly. All failed estimates remain saved. Together they require 78 local calls and 2.960 seconds, decoding all 1,356 source frames. No body was constructed: both still fail the unchanged movement-continuity checks, and the cropped condition introduces an unfilled body-position gap.

For the **621-pose** candidate, fixed development-label limb passes improve **63→67/84** with four newly scored arm segments at frame 200; torso-side passes remain **33/39**. Previously scored errors are unchanged and no scored regression occurs. The additional odd-frame probes are not independently labeled or body-validated. These are coverage gains, not verified anatomy or physical accuracy; no pipeline default changes.

Inspect the [fixed-crop report](../artifacts/jacks-pose-envelope-01/report.json), [frame-200 pose overlay](../artifacts/jacks-pose-envelope-01/proposal-0200.png), [all crop decisions](../artifacts/jacks-pose-envelope-01/decisions.json) and [unused-frame comparison](../artifacts/jacks-gap-samples-01/report.json). The fixed-crop bundle has 331 files (manifest `c53e8b77c7a442ce7861faacd84bbcdeb4a4e1d6c334ca1a2b29f844fddf27ba`); the unused-frame bundle has 211 files (manifest `5d8da02f7e98ea7febd5cb79ff76d85b9c85656aa4658e251c205b7b5f36cd97`). Original observations and rejected proposals remain intact.

One separate Meta SAM3.1 request is prepared but **not sent**: an exact-pixel, 350-frame/11.65-second prefix of the public clip, estimated $0.07 with a $0.15 reservation inside the authorized $5 total. Automatic approval review requires consent for this specific media upload; approval is pending. No upload, reservation or charge occurred. Local recovery does not depend on that approval.

### Full-sequence tracking did not resolve the gaps

The next comparison applied the same fixed crop to **every one of the original 678 samples**, using independent IMAGE inference and temporally tracked VIDEO inference as separate conditions. Both decoded all 1,356 source frames. IMAGE reused the 97 exact cached crop results and made 581 new calls; VIDEO made 678 new calls. Two bounded local workers took 11.035 and 12.040 seconds respectively.

| Condition | Usable / sampled | Competing samples | Long movement gaps | Longest gap | Unfilled body-position gaps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole-sequence IMAGE crop | 619 / 678 | 0 | 5 | 0.734 s | 0 |
| Whole-sequence VIDEO crop | 595 / 678 | 48 | 10 | 0.934 s | 2 |

Neither supports a complete mobile body. The fixed development labels show two newly passing torso-side checks and **two regressions** at frame 334 in each condition; total torso-side passes stay 33/39. Limb passes rise 67→68/84 without a scored limb regression, but that aggregate does not erase the missing poses, torso regressions or long gaps. No condition becomes a default or a per-frame automatic winner. [Complete report](../artifacts/jacks-fixed-crop-tracking-01/report.json); 174-file manifest `de300b71392fecc54ab8ae189a8ead2064480fa12021783e3b679f83f9e25743`.

### Inspect tracking failures in the sandbox

The **Video** tab now includes **What got tracked?**, a source-frame timeline and **Next tracking issue**. It distinguishes partial poses, unusable/competing hypotheses and frames that were not analyzed. The button seeks the original video without playing or changing the simulated body. Coverage is not an anatomical-accuracy score. Existing squat and push-up labs receive this view without changing their saved runs.

To inspect the rejected 634/717-sample jumping-jacks candidate:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/exercise-coverage-inspection-01/jacks
```

This inspection package deliberately has **no interactive body**: four long movement gaps and one missing body position remain. Video, timeline, failure messages and source files are available; simulation controls remain disabled. No new inference, training or recording was needed to package it.

Verification: 51 distinct Python checks and 33 deterministic UI checks pass, including authenticated access, bounded input, background loading, failed-run inspection and source seeking without simulation changes. The initial Python run retained one floating-point test-fixture failure; only that corrected test was rerun. A later stopped-preview wording change was checked with its single affected UI case. No browser visual inspection was performed.

Three real saved runs produce compact summaries of 3,745–27,182 bytes. Cold preparation takes 8.8–64.6 ms off the request/control thread; 120 cached calls per run, including JSON serialization, have p95 0.098–0.255 ms. The 900-sample analytic case produces 78,666 bytes in 1.25 ms. These are in-process measurements, not browser, HTTP or sustained-idle performance. The [inspection report](../artifacts/exercise-coverage-inspection-01/report.json) has 131 sealed files, manifest `7067d6150cb2ecf885b220a079222149150a109fecc06eba5e9d02c00d37fe94`; its runtime snapshot predates only the final stopped-preview message.

## What improved—and what did not

- Fixed image labels: torso-side passes improve **29→33/39**, limb passes **55→63/84**, with no scored regression. The added scores come from jumping-jacks frame 40; adjacent labels reuse that one pose. Previously scored errors remain unchanged.
- Native-body labels: **21/84** source-observable limb checks pass before and after. Only 28 are scored; the other 56 remain missing because three bodies are rejected. The five new push-up poses have no fixed independent labels in this comparison. Better image coverage does not prove better body geometry.
- Squat is unchanged: **55/107** source-pose and **111/213** control-grid floor checks fail; maximum penetration is **0.17885 model units**. The body is interactive but not a physically accurate squat reconstruction.
- Push-up gains five poses with **0/121** source-pose and **0/255** grid floor-limit failures. This checks penetration only; correct foot/hand contact, the kettlebells as physical objects, forces and body depth remain unverified.

These are already-exposed development clips with approximate assistant-made image annotations, not an independent anatomical benchmark. Source-only image consistency checks do not establish measured movement or physical transfer. No correction becomes a default.

## Controls, learning and recording

Both candidates retain the existing mobile body, gravity, support controls and joint-target task. Each trains for 100 episodes and evaluates the learned, random and no-control policies on the same 20 starting conditions:

| Candidate / authored target | Learned | Random | No control |
| --- | ---: | ---: | ---: |
| Squat / right hip | 20/20 | 0/20 | 0/20 |
| Push-up / left elbow | 20/20 | 13/20 | 0/20 |

This is one-joint control with virtual support, **not learned execution of the whole exercise**. Success requires angle error at most 0.08 radians and angular speed at most 0.15 radians/second. It does not measure anatomical accuracy.

Each sandbox records **12 actions and 13 states/video frames**, then reproduces every state, reward and completion flag exactly. Changing the action changes simulated joint states. Construction, bounded learning, reference checks, recording and inspection panels take 4.481 seconds for squat and 2.914 seconds for push-up, excluding the preceding pose studies. No browser, server or HTTP performance was tested.

Source/body panels for push-up frames 62 and 182 and original/cropped image estimates at push-up 62 and jumping-jacks 314 were visually inspected. The render remains an inferred simplified body; no panel is a physical validation.

## Inspect each stage

For push-up:

1. [Complete original Full-model observations](../artifacts/selective-crop-videos-03/kettlebell-pushup/whole_image/evidence.json).
2. [Source-only crop plan](../artifacts/selective-crop-videos-03/kettlebell-pushup/crop-plan.json) and [all acceptance/rejection decisions](../artifacts/selective-crop-videos-03/kettlebell-pushup/decisions.json).
3. [Selected observations with originals retained](../artifacts/selective-crop-videos-03/kettlebell-pushup/selected/evidence.json).
4. [Executable body specification](../artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run/human.json) and [floor checks](../artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run/reference-feasibility.json).
5. [Matched RL trials](../artifacts/selective-crop-bodies-01/kettlebell-pushup/selected_lab/run/rl-evaluation.json) and [recording/replay verification](../artifacts/selective-crop-bodies-01/kettlebell-pushup/report.json).

The [five-clip image report](../artifacts/selective-crop-videos-03/report.json) and [complete body outcome report](../artifacts/selective-crop-bodies-comparison-01/report.json) preserve missing slots, rejected clips and unchanged comparisons.

## Reproducibility and failures

The first attempt stopped before inference because its input bundle lacked a required report. The second completed all 1,340 whole-frame observations but stopped at the same report requirement for nested outputs. Those estimates were sealed and reused in the corrected run; **no successful whole-frame inference was repeated**. Recovery made 31 new local crop calls and took 8.479 seconds. Including the saved first pass, this follow-up made 1,343 new local calls and reused 28 earlier exact-image estimates. There is no single uninterrupted end-to-end latency claim.

The body study completed both playable candidates and recordings, but stopped the jumping-jacks worker on its original mobile-root rejection. A separate recovery attempted the corrected body and retained the same rejection without repeating successful builds, training or recordings. The runner now records both such rejections explicitly; saved evidence predates only that reporting fix.

| Bundle under `artifacts/` | Files | Manifest SHA-256 |
| --- | ---: | --- |
| `selective-crop-videos-01` | 116 | `74dff70800270f9442a7a8417bb4cfc45e18d56420b1a847ac8a65cc83f198e4` |
| `selective-crop-videos-02` | 169 | `141280a65beef658c1610757735d0c90bab51e064f7e215391458d6f7c1f3ce2` |
| `selective-crop-videos-03` | 382 | `3917bdc315552e379c85eabe3601ed63aa2c756fc9040d523fc8f42d34d6c25e` |
| `selective-crop-bodies-01` | 353 | `568d5f5ed471de3996447eff1ed563f636fc450ed60ed8f1fe94b8e4246af3bc` |
| `selective-crop-bodies-comparison-01` | 13 | `34eec61107db9358121414504f03663b705c6c2de6d8a74cdde70bfabe6a51ea` |

All work is local; no paid API call, model download, deployment or recurring job occurred. Meta's existing budget and saved results are unchanged. The broader balls/exercise/general-video goal remains open, including independent changed-action evidence. Next useful work is image-grounded recovery across the remaining gaps and competing poses, plus camera/contact constraints—not promoting these image gains as physical accuracy.
