# Conditional finite-support reconstruction

## Current state — November 1, 2026, 10:19 AM PST

Implemented and reviewed locally; root owns integration and publication. This
lane owns `source_surface_envelopes.py`, `exercise_surface_report.py`, the new
`test_conditional_surface_domains.py`, the added public collection test in
`test_exercise_surface_pipeline.py`, and this note. No Git mutation, active job,
source acquisition, video/model/native-physics run, API call or ledger change.
Unrelated avatar and root control/diagnostic edits remain untouched.

The missing behavior was all-or-nothing finite support admission: an otherwise
useful pixel/height uncertainty region became unavailable whenever any part fell
outside a supplied bench/equipment top. **Legacy behavior stays the default.**
An explicit static-surface declaration can now include:

```json
"domain_mode": "condition_on_declared_support"
```

Use the existing `surface-inputs --resume … --request … --out …` workflow with
the edited collection request. No new command, simulator, scorer or data format
family is introduced. The option survives the sealed surface-config round trip
into the existing fitter-input objective. The ordinary offline collection report
explains the option and labels every opted-in surface; frame pages retain its
full original and conditional bounds. Root owns the local-lab guide addition.

## Meaning and geometry

The mode explicitly assumes the annotated point lies on the supplied finite
static support. It does **not** infer that assumption from image overlap or prove
contact, load, grasp, anatomy, calibration accuracy or physical feasibility.

For the existing constant-sign, forward ray region, intersect the exact-rational
eight-vertex pixel/height frustum with the support's four XY half-planes. Retain
height/pixel correlations during clipping, then round the resulting enclosing
box outward. Empty intersections stay unavailable, including cases where the
unrestricted axis-aligned box overlaps the support but the correlated region
does not. Closed boundary, line and point intersections remain conditional.
Available results retain the unrestricted enclosure and unchanged original
pixel/height ranges; no midpoint is selected.

Positive radial distortion retains its existing conservative outer ray rectangle
and certificate. Even a nonempty clipped radial region can be infeasible for
the actual correlated radial inverse. A point inside any final box need not
satisfy the original correlated pixel/height constraints. The report states
these limits. Unknown heights, occluded/missing annotations, excluded partitions,
unsupported distortion and invalid clocks remain unavailable. Moving/handheld
supports remain unsupported; no source, body, floor, task or gate is changed.

## Verification

**134 distinct focused checks pass** across four receipts:

- `checks-01`: 90 geometry/public-component/default/radial checks passed.
- `checks-02`: 41 pipeline/report checks passed; three existing scripted-DOM
  tests stopped at the runner's blanket child-process guard, not product code.
- `checks-03`: only those three cases reran with exactly three bounded Node DOM
  children allowed; all passed. Network/native-physics operations stayed blocked.
- `checks-04`: two affected cases passed after narrowing the metadata/report
  wording to say that **available** enclosures retain the original enclosure.

Receipts and JUnit files are in `.local/contact-reconstruction-resume-01/`.
The all-process-blocked analytical runner is
`.local/split-limb-feedback-actor-01/run_pure.py`; the narrow Node correction is
`run_report_dom.py` in this lane's local directory. No failing receipt was edited.
The known MuJoCo import-time process probe is denied by both runners; zero native
physics calls occurred. Scripted DOM checks are not browser visual verification.

The 1,200-slot authored workload prepared in **0.969395 seconds**, retaining
**1,940,106 cached row bytes**. A further 1,000 cached frame reads required no
clipping. This is a bounded analytical workload, not maximum-input or sustained
UI/RSS verification. Existing time, input, frame and output limits are unchanged.

Seven authored default cases matched the frozen
`.local/split-limb-feedback-play-04/runtime` byte-for-byte, including ground,
complete/partial supports, positive radial distortion, unknown height, occlusion
and excluded evidence. Receipt `defaults-01.json` SHA256:
`39910629e1ef58c648e63be78828deb1be21e1a22223e7b181477afb12d30df5`.
The later edit only narrows opt-in metadata wording; the default paths are unchanged.

## Saved public-component workflow

The existing test drives actual discovery → selection → request qualification →
sealed configuration and real offline report, then loads that config into the
actual surface fitter-input objective. **Only runtime-archive capture is mocked
by the established test fixture.** It is not a frozen real CLI/native or real-video
acceptance. Four original source slots survive; original source/body bytes and
both earlier package seals remain exact. No fitting or simulation is performed.

Completed test outputs are copied, not rerun, under
`.local/contact-reconstruction-resume-01/public-workflow-01/`:

| Package | Files including manifest | Manifest SHA256 |
| --- | ---: | --- |
| `discovery` | 13 | `02fcc2c457a16e0719a13d213522bee0b9bfab00c628eff4b5b4152868d2430a` |
| `draft` | 19 | `36910b2ed46fe26077d7e6a3da4144e60f0103945a182853582ef930f975857a` |
| `conditioned` | 23 | `003faadb8db0308f723f6d0b31423957a36bf0bcd92d1eacf5425acb88192791` |

The copied fixture has 60 files /169,163 bytes including inputs/edits, with every
package member verified after copying. `public-workflow-custody-01.json` SHA256:
`e2ce9b363b5d90e0b41cf373cdc5917a95752c332371a675b313089531151f6f`.
Preserve `copy-preflight-failure-01.json`: an inferred pytest directory did not
exist, so copying stopped before output; directory discovery found the completed
fixture. It is a copying-preflight error, not a reconstruction result.

## Remaining work

- [ ] Root: integrate the reviewed scoped files and preserve unrelated work.
- [ ] Apply this opt-in mode only when a new source supplies justified camera,
  pixel/height ranges and finite static-support bounds. Keep unknown inputs
  unknown; do not rerun old failed pose/control searches to manufacture admission.
- [ ] Establish unfamiliar-video ground/equipment and changed-action accuracy
  with independent evidence. This numerical construction improvement does not
  complete general reconstruction or fix the saved kettlebell control failure.

Completion of this lane means the bounded opt-in geometry, normal collection and
report path work with unchanged defaults and truthful uncertainty. The full
video-to-world-to-RL goal remains active.
