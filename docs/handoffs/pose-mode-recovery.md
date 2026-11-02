# Source-only temporal pose recovery

## Active — November 1, 2026, PST

Previous goal turn made progress and published `a874a1e`: explicit single-person
capacity, 234 scoped passing checks, actual Full VIDEO body construction and a
complete floor rejection. No scope reduction or completion claim. Root owns this
new source-only recovery work; preserve independent avatar work and all previous
inputs. No peer worker is live; their known usage limit remains, so root proceeds.

Saved diagnosis of `artifacts/full-single-person-kettlebell-body-02` finds a
specific observation inconsistency at source frame132: inferred right-thigh
length is about0.317 of its complete-clip median. The saved Full IMAGE estimate
does not show that collapse. Full VIDEO has similar low ratios at180 and224;
do not treat132 as the only possible target. Original model confidence at132
is0.537, so a confidence-only rejection does not identify the glitch.

The existing quantized annotated preview at source132 visibly puts the inferred
right lower leg below the person's actual legs. It is retained at
`.local/pose-rigidity-diagnosis-01/source-132-annotated-preview.png` with its GIF
hash/index/source-frame provenance. This is visual diagnosis, not a new frozen
anatomical label or calibrated measurement. No new video decode or inference.

## Hypothesis and bounded implementation

`src/weft/pose_mode_recovery.py` proposes retained alternative observations using
only the two complete original source-pose documents. It cannot read a body,
floor, task, scorer, development annotation or training result. All original
neighbors are used before selection; selected poses never become new anchors.
The alternative must use the same model/version/runtime/source/dimensions/clock,
with IMAGE capacity2 versus explicit single-person VIDEO capacity1.

Frozen candidate criteria: adjacent original anchors span at most0.2s and agree
in torso-normalized limb length within25%; a target limb halves/doubles in3D
and has at least0.25 image-torso-length endpoint error. The alternative must
restore length within a factor1.25, halve the image error to at most0.25, retain
all previously visible major joints, limit other-joint worsening to0.15 and
remain within0.25 torso lengths of the source-context pelvis. Confidence0.5
only establishes model visibility, not truth. Competing hypotheses abstain.
These are exposed development proposal bounds, not changed evaluation gates.

Scan all original sample rows and all eight segment slots. At most900 samples
and32 eligible targets; exceeding the candidate bound accepts none. Selection
uses an actual retained IMAGE estimate, not synthetic interpolated landmarks.
Primary/alternative observations and decision hashes remain distinct. This first
stage is a proposal, not an imported body or a default change.

## Verified implementation and next acceptance

The first pure screen is complete: all129 samples /1,032 segment slots; only132
is eligible and accepted. `artifacts/pose-mode-recovery-screen-01/manifest.json`
SHA `4498ed3023ab13de29e3eeb8d5ec1f7309938692702180207335a3bf36670cee`.
Elapsed0.093867s /41,484,288B peak RSS; no media decode, inference, native work
or API calls. The frozen criteria were not adjusted after inspecting results.
Subsequent selector edits only reject malformed/orphaned provenance and require
the existing encoded-frame audit for metadata/decoded-count differences.

`pose_mode_source.py` and `exercise-pose-recover-modes` now package both original
seals, exact raw observations, all decisions and implementation bytes. Ordinary
`exercise-body --evidence-run` recomputes the derivation and retains each selected
frame's actual IMAGE model metadata in evidence and spatial-body audits. Defaults
remain unchanged; no chaining with crop/identity repairs or bypassing body gates.

365 scoped cases pass in `.local/pose-mode-integrated-checks-01.xml` (10.51s).
The initial integration test placed the synthetic glitch next to an intentionally
missing sample; the selector correctly abstained. Three test failures are kept
in `pose-mode-source-checks-01.xml`; the fixture now uses a supported-neighbor
frame, with no relaxation of product rules. The unchanged legacy paths also pass.

Next: one frozen public-CLI package, one ordinary body construction and complete
unchanged floor preflight. Reuse saved original estimates; no inference, decoding,
dynamics, training or API calls. Each package/body stage has15s cooperative /20s
external wall bounds,128MiB artifact /512-file limits and1GiB observed RSS ceiling.
Floor geometry work remains separately bounded. Preserve source clocks, original
floor/scorer gates and prior direction regressions. No model/default/lab promotion
or old-body equipment transforms are authorized by a source-only improvement.

The actual mode package, body and382-row floor check are complete. All sampled
floor checks pass at the unchanged0.025 limit; max0.0173876520 versus the prior
0.0718740450. The normal body builder also re-estimates median lengths, body-frame
orientation and first-pose placement, so this is a newly source-derived body,
not a controlled test with every numerical body parameter held fixed.

Seed attempt01 stopped before output/native execution: the guard allowed zero
model compilations, but existing action-contract validation compiles one static
human mass model. Preserve `.local/pose-mode-kettlebell-seed-01`; there is no
successful seed artifact. A focused count confirms one compilation, no state
allocation/forward/steps. The receipt now explains that its historical
`physics_models:0` means executable simulation instances, not static mass checks.
Attempt02 may permit exactly that one call site, with all other native work still
forbidden. The intermediate test expected two mass calls and correctly failed;
its receipt is retained. This is a supervision/measurement correction, not a
change to physics, source evidence or task gates.

## Acceptance complete — November 1, 2026, 8:17 PM PST

**405 distinct current checks pass.** Summary
`.local/pose-mode-test-summary-01.json`, SHA
`4fcd07592ddaad627650b4781c0e87d975f2b0bd0fad548ee7f2d0bac7d60c78`,
deduplicates repeated checks and the renamed static-mass test. Earlier fixture/count
failures remain; root reviewed this work, not an unavailable peer worker.

| Artifact | Manifest SHA-256 | Completed process / peak RSS |
| --- | --- | --- |
| `artifacts/pose-mode-recovery-kettlebell-01` | `ad1e11d2d6a90edaea5988e80e18c7c65d0c8e94fe430ba239198f48d5f1938e` | 0.783795s /165,396,480B |
| `artifacts/pose-mode-kettlebell-body-01` | `6cfb168cf0a61387bf32907771904cd79017b1f56611f7a8a435847134f6907f` | 1.443186s /248,414,208B |
| `artifacts/pose-mode-kettlebell-floor-01` | `318a14fffbc98989ada7ba97c9cf588a461138ee92b91e9563fa6054519ffe08` | 0.362351s /141,672,448B |
| `artifacts/pose-mode-kettlebell-seed-02` | `7cd8de9df7e31715f6e48cbdecd967ff1002a7e175ea3e3d486fc3dffe8da793` | 1.373916s /195,723,264B |

Source package:28 files /14,255,422B, zero native calls. Body:56 files /21,704,274B,
one model/data/floor-forward, no dynamics. Floor:8 files /1,070,425B, one model,
two data allocations and382 forwards. Seed:55 files /30,290,358B, exactly one
static mass-model compilation, zero data/forward/steps. First failed seed:
1.083297s /182,779,904B, zero native calls/output; retained under
`.local/pose-mode-kettlebell-seed-01`. Runtime, stage plans and supervision receipts
are `.local/pose-mode-{package,body,floor,seed}-plan-*.json`,
`.local/pose-mode-runtime-01` and `.local/pose-mode-seed-runtime-{01,02}`.
Do not edit frozen runtimes or reuse output directories.

Saved audit `.local/pose-mode-acceptance-audit-01.json`, SHA
`af85c13b7df23c90e1e4343b7cf13604d2705a2212d07a064751c9777836cc74`,
independently checks161 covered members across screen/package/body/floor/seed,
all129 original rows,1,032 decisions,257 source slots,127 body references and382
floor rows. It verifies exact replacement versus alternative/original bytes,
sample clocks, source-to-body-to-seed proof joins, control brackets/interpolation,
contact arithmetic and unchanged0.025 threshold. It uses no native/media code;
0.112912s. Largest length delta is forearm−0.0021957371 model units; base-height
delta+0.0008109998 comes from the ordinary first-pose placement, not a tuned floor.

Reproduce the new stages with fresh output names:

```sh
.venv-lab/bin/python -m weft exercise-pose-recover-modes \
  --primary-run artifacts/full-video-single-person-kettlebell-01 \
  --alternative-run artifacts/selective-crop-videos-03/kettlebell-pushup/whole_image \
  --video data/exercise/kettlebell-pushup.webm \
  --out artifacts/my-mode-recovery-01
.venv-lab/bin/python -m weft exercise-body \
  --evidence-run artifacts/my-mode-recovery-01 --out artifacts/my-mode-body-01
```

The commands alone have cooperative bounds; acceptance workers additionally enforce
external deadlines and native-call limits. Body/seed inputs are not playable labs.
The current lab remains available via
`.venv-lab/bin/python scripts/play_sandbox.py --example auto-equipment --console`;
the user starts it. No server/browser was started here.

### Remaining work and handoff

- [ ] Resolve the separate raw-direction/identity errors, equipment geometry/contact,
  and full-task learned control. One sampled floor pass does not resolve these.
- [ ] Evaluate any new coupled world with fresh body-specific source transforms,
  retained uncertainty and original task/scorer gates. Do not transplant prior
  121-reference-body equipment coordinates or promote this127-reference body by default.
- [ ] Extend unfamiliar-video/independent evidence coverage and complete the full
  generalizable video→world→RL/interaction/recording goal. No physical truth claim.

All owned children are terminal/reaped; no paid API/key/ledger activity. The shared
$5 Meta budget remains unchanged. No new deployment or source/model acquisition.
Work stays on `main`; preserve independent avatar edits and their intervening
`3874809` checkpoint. Standing merge/push approval applies to the15 owned paths;
final exact remote/index/stash result is `.local/pose-mode-publication-01.json`.
The full goal stays active, not complete or blocked. Next step is a new justified
source/body/equipment hypothesis, not rerunning this unchanged successful floor test.
