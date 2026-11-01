# Surface-interval fit input readiness

## Status — October 31, 2026 (PDT)

No complete real-source interval-fit bundle was found in the targeted local
bundles inspected. This was a bounded metadata/code review, not an exhaustive
archive search, source study or fresh model admission. No video/image decode,
inference, geometry execution, native fitting, tests, network or paid calls ran.

The reconstruction lane owns the optional objective inside
`whole_body_reference_refinement.py`. Root owns input packaging, worker, CLI,
tests and the one separately planned native acceptance. This lane owns only this
note and read-only integration review. No live handles or Git mutations.

## Available real inputs and exact joins

Closest camera/source bundle:
`artifacts/camera-shared-admission-jog-01/candidate/`, seal
`29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`.

| Input | Existing identity / scope |
| --- | --- |
| `human.json` | Raw SHA `849ece268cc6b091cf85460f3e22ecee172aa5323c4db88dac3e221e367d0867`; 112 references, original body timestep 1/30, interval 0–7.9412666667 seconds |
| `evidence/5483442b26725f532aaa91fc2d46fa0464b1a6b5f1c5f5629d7ab4071bc997dd.json` | Exact canonical observation document with that SHA; entity `person-0`; 240 source slots, 120 construction and 120 unsampled |
| `camera.json` | Source-bound reconstructed camera; 1920×1080, FPS 30000/1001, 240 frames; radial coefficient `0.00047528394168603385` |
| `ground.json` | Raw SHA `90768933ec0fb8e51ecbc1075745532496d61f68e331b82a54e265231ca8d86e`; reconstructed height 0, not measured ground/contact or an uncertainty interval |

All share source SHA
`b1b362fa7516879c8ae37c48c40972b0090680e6bdcb6933fcf9ccb8c2e2b0cd`.
For frame zero the observation is
`<source>:observation:0:person-0`, with evidence `<source>:frame:0`.
Body evidence must resolve to this exact document/entity and original time;
sharing a video hash alone is insufficient.

The bundle is **not fit-ready**:

1. Interval envelopes explicitly abstain on its nonzero radial distortion.
2. The existing whole-body fitter rejects `camera_reference`; do not strip it.
3. No authored static-surface height interval, compatible endpoint pixel-range
   annotations or explicit body-to-surface-world binding accompanies it.

The existing non-camera fitter's original real inputs remain at
`artifacts/whole-body-reference-candidate-01/inputs/human.json` (manifest pin
`6db9a3598cf8efa4d0d3aae458671e293d33849ff9f24a610558c8d45c6bd9fb`)
and `inputs/evidence-derived.json` (pin
`2e1c851cd41c83c9458d37b58696fdfeb723b2ac24cb6615810554657064506e`).
These paths/pins were read from seal
`42f43355225785365baffbfa77cf559a7e94aca951ed75d19176df0d4c2f235a`,
not re-admitted or refitted. They do not supply the missing surface-input bundle.
Never stack a previously fitted candidate as the original body.

## Preserved unsupported evidence

`artifacts/interval-foot-projection-consistency-01` remains rejected: its fixed
640-pixel focal hypothesis produced 33.9294° foot-direction error against the
frozen 5° criterion. The earlier 73.57° stationary foot-plane ambiguity is not
physical ground. Original Interval/kettlebell source
`e93ea83a78d47c27972f245a61d114118793598cfd059c318e120c453fd40bbe`
has 1,543 decoded /1,544 declared frames. Reuse the saved rejection; do not retry
heel fits, alter focal lengths, or move the floor to manufacture a usable case.
See [projection evidence](exercise-projection.md).

`.local/source-surface-cli-fixture-01` is a saved **point-height** analytical
fixture, not an interval/body package. Its source is a synthetic descriptor, and
it lacks a physics-valid body plus interval/body binding. The minimal body in
`tests/test_source_surface_interval_construction.py` likewise proves pure joins,
not native anatomy or a complete physical model.

## Smallest meaningful acceptance

Use the existing authored fixture in
`tests/test_whole_body_reference_refinement.py::fixture(count=3)`. Its
`ToyGeometry` exercises the real preparation/optimization path analytically;
root may separately use the existing native geometry adapter with a validated
authored body for one worker/CLI acceptance after focused tests pass.

- Three original references at frames 0/2/4, with all ten declared source slots
  retained; first reference stays fixed.
- One explicitly authored core-body endpoint channel, such as an ankle—not an
  inferred toe-to-ankle substitution.
- Freeze an undistorted synthetic camera, pixel range, static surface-height
  interval and explicit body/model-unit→world alignment **before fitting**.
  Every value must remain labelled an authored assumption, not a real-video
  reconstruction. Do not auto-generate configuration for the real source.
- Use one bounded fitter call, optionally a two-joint mask. Preserve original
  roots, floor, lengths, reference clocks, evidence and existing gates. Require
  a genuinely active outside-box residual and complete before/after diagnostics;
  retain rejection if independent image/floor criteria do not pass.
- Preserve jogging as an explicit unsupported preflight case. Missing optional
  inputs must not invoke a hidden default camera, interval or coordinate map.

The objective's documented bounds are body≤8 MiB, binding≤64 KiB, up to eight
observation documents/64 MiB, 1,800 source slots, 900 references, 12 endpoint
channels and five-second cooperative preparation. Queries admit only the named
19 joint derivatives; unknown mappings stay unavailable, not zero error. Root
must additionally freeze worker time/output limits before any native acceptance.

This acceptance can establish optional-workflow execution and custody, not
unfamiliar-video reconstruction, foot/contact calibration, physical accuracy or
generalized success.

## Workflow review and next safe action

Read-only review covered `source_surface_fit_inputs.py`,
`whole_body_reference_pipeline.py`, `whole_body_reference_worker.py` and the
`cli.py` dispatch delta. No further custody/bounds blocker was found:

- Only explicit, hash-pinned files beneath the config directory are copied;
  traversal, symlinks, missing files and unexpected plan members are rejected.
- Image/floor objectives retain their original worker input/runtime inventory.
  Surface mode adds nine named modules; inspected transitive imports are covered.
  The worker checks copied input/runtime hashes before and after its single fit.
- The wrapper is stricter than the pure objective: surface files total≤16 MiB,
  each≤8 MiB, config/binding≤64 KiB, and camera/surface/annotations total≤4 MiB.
  Non-recovery initial snapshots remain≤32 MiB; outputs remain≤64 MiB/96 files,
  with the existing worker wall/CPU/native-call limits and no physics stepping.
- Root applied the reviewed isolation improvement: `_runtime` imports the
  optional helper only for `surface=True`; default preparation does not depend
  on it or enlarge the frozen default worker closure.

The old report initially rejected the new method. The UI lane owns explicit
surface-profile/report validation; native acceptance must wait for that and the
fitter to freeze. Root reported 87 focused orchestration checks passed; this lane
did not rerun or independently audit those tests. No new native result exists
from this review.

Next safe action: freeze the fitter, report and one explicitly authored
input/runtime plan after focused checks, then let root run that single worker
acceptance. Preserve unsupported jogging/kettlebell evidence and all failed
proposals. Do not rerun previous real-source studies or automatically invent a
real-source surface configuration.
