# Frozen body versus published joints

## Verified — November 2, 2026, 5:19 AM PST

The previous turn published `89d3c29`: a different motion reaches shared physics,
interaction and exact recording. This turn measures its agreement with a
separate published reference. **The full goal remains active.** Passing the
existing simulator task did not establish accurate motion.

On the **same 479 available joint/frame pairs**, absolute joint-position RMSE
is 67.765 mm for triangulated image joints, 69.433 mm for the constructed body,
and **156.583 mm for the recorded assisted simulation**. No alignment, timing,
scale, identity, body, ground, force or success-threshold adjustment was made.
The reference's own accuracy and anatomical joint definitions remain unverified;
these are reference-agreement measurements, not a physical-accuracy certificate.

## Source and evidence boundary

The matching `04ac-0006.c3d` was already complete in the cached leading 1 MiB of
the publisher's training-reference archive. No new download or installation was
needed. Extract only that 51,856-byte member at tar data offset 54,784. Its SHA:
`a45639e3f80fd047b7f4a1d18caf3e97ce7be57369be38e8e85289430545192c`.
The cached prefix hash and original range receipt are retained; full-parent
archive checksum verification remains unavailable. Rights remain CC0-1.0.

Before reference parsing, freeze the previously published body and recording:

- Body run: `c4000ffcdd2aca83cbb1d1a7e2d6c677a6744b7f3cbbb88e78c4b9d46e8e1ef4`.
- Recording: `4623f2aecc2f1566bfc9be5b8d0b8093e16389810eb2fe99da65b515e1d6dcff`.
- Protocol: `.local/aspset-next-reference-01/plan.json`, SHA
  `a3847654bd58b47e2cad6f63458259a1fd8192b85dba8d81e640d7afb30cd149`.

The existing pinned `c3d==0.6.0` wheel is imported directly, not installed.
It parses 185 frames numbered 1–185, 17 named joints, 50 Hz and millimetres.
All 3,145 residual/camera-flag entries are zero; that metadata does **not** prove
observed optical markers. Original frame 1 maps video frame 0. The exact prior
conditional-ground rotation and mm→m conversion are reused, with no fitted
translation/rotation/scale/time offset. The floor remains an authored assumption.

All new reference data are **assessment-only, exposed development**. No C3D
coordinate enters construction, training or a controller. Later inspection does
not turn this unit into a fresh final holdout.

## Reusable comparison and verification

`camera_reference.compare_world_frame` compares twelve explicitly named world
joints. It preserves absolute error, separately reports hip-midpoint displacement
and root-relative pose error, and retains unavailable slots. Removing root
translation never replaces the absolute result. Nonfinite points remain missing;
malformed/boolean/oversized coordinates and renamed joints reject.

`scripts/studies/assess_aspset_body.py` connects this comparator to sealed camera
body previews, triangulation, a published reference package and one explicitly
selected recorded episode. It checks source identities, exact camera bytes,
coordinate placement and original timestamps. It reads saved coordinates only;
no native model, physics step, inference, optimizer or task re-evaluation runs.
The chosen second recorded episode is the reset plus all 184 programmed actions.
Missing kinematic poses are not interpolated into observations.

**42 distinct cases pass**, using the latest result per case:

- `.local/world-joint-reference-tests-01.xml` — 27 comparator/legacy cases.
- `.local/aspset-assessment-boundary-tests-02.xml` — 13 source/time/coverage,
  fixed-camera and immutable-path cases; the earlier seven-case receipt remains.
- `.local/world-joint-reference-input-delta-tests-02.xml` — 11 current world cases,
  including mixed boolean and integer-overflow input bounds; unchanged legacy
  projection results are reused.

An independent NumPy audit recomputes all **6,660 joint rows**, all reported
distance statistics and the 479-point matched subset directly from published
arrays, saved native endpoints and the original action journal. It checks 421
input members, without importing the new comparator. The input-binding-tightened
`assessment-02` keeps byte-identical comparison rows and identical statistics/
worst frames to `assessment-01`. Both originals remain intact.
The final path/camera/source/event guards also pass on the actual inputs;
`.local/aspset-next-assessment-02/current-input-guards.json` pins that check and
the current code. Later edits tighten admission and improve report labels, not
distance arithmetic; no extra native/scoring run was needed for those guards.

## Results and limits

| Stage | Available joint slots /2,220 | Absolute RMSE, all available | Absolute RMSE, matched 479 |
| --- | ---: | ---: | ---: |
| Triangulated image joints | 479 | 67.765 mm | 67.765 mm |
| Constructed body poses | 564 | 83.340 mm | 69.433 mm |
| Recorded simulated motion | 2,220 | 160.965 mm | 156.583 mm |

Use the matched column for comparisons between stages. Body poses exist at 47
source times; the recording covers all 185 frames. The body retains 1,656 missing
joint slots, and triangulation retains 1,741. Root-relative RMSE is reported
separately, not substituted for these absolute errors.

The largest constructed-body error is the left wrist at video frame 124:
542.785 mm. The recording's largest error is the left wrist at frame 9:
449.690 mm. Prior floor and task passes remain true only for their stated
simulator criteria; they are not promoted to reference accuracy.

### Tracking diagnostic, without reference-driven fitting

Comparing the recorded dynamics with its own constructed targets at 47 matching
times gives **exactly zero joint-position difference at reset**. This rules out
an initial coordinate/placement mismatch for this case. During motion, target
position RMSE reaches 245.188 mm at the left wrist and 240.766 mm at the right
ankle. Left-shoulder twist error is .577608 rad RMSE.

The source target requests peak left-shoulder twist speed 29.6924 rad/s.
Left-hip twist saturates its existing force bound on 69/184 actions;
left-shoulder twist on 59/184. These are observations, not proof of one cause.
A read-only equivalent-Euler-branch check finds only one locally shorter
interval, with a small reduction; do not claim a general chart-wrap fix is ready.
No alternate body/controller is installed or qualified by these diagnostics.

## Inspect or reproduce

Open the [plain-language report](../../artifacts/aspset-next-reference-report-01/index.html).
It separates matched comparisons from differing coverage, links every joint
slot and explains the reference/assistance limitations. HTML structure, values
and links were checked without a browser; no visual/browser QA is claimed.

For a new saved-data assessment, use a new output name:

```sh
.venv-lab/bin/python -B scripts/studies/assess_aspset_body.py \
  --run artifacts/aspset-next-joint-body-lab-01/run \
  --reference artifacts/aspset-next-reference-01 \
  --camera-inputs artifacts/aspset-next-camera-input-01 \
  --recording artifacts/aspset-next-shared-01/lab/sessions/session-bba962504d70 \
  --episode 1 --out artifacts/my-body-reference-assessment
```

The [existing lab launch command](aspset-next-clip.md#inspect-it) is unchanged.
These generated local artifacts are not included in a fresh Git clone.

| Package | Members | Manifest SHA-256 |
| --- | ---: | --- |
| `artifacts/aspset-next-reference-01` | 10 | `eaf88a4b1e53f20b4b94361298a3a459981ab6476351babbae0edca674aaaffc` |
| `artifacts/aspset-next-body-assessment-01` | 10 | `943ac4bf3387fef3a5318be15ea1b8cd3ac441b7aa4819d833ac4344009cb678` |
| `artifacts/aspset-next-body-assessment-02` | 10 | `7f4963edaef72e2b041f9c1941e83b5bd9ca6a14d8e8b6f439dc272e38dcd8f4` |
| `artifacts/aspset-next-reference-report-01` | 4 | `7647a769d514b11aa8328966891a12709c5b5f539ea15001f0bcd74e775fafe3` |
| `artifacts/aspset-next-tracking-diagnostic-01` | 5 | `9794d7875a60b935fdb89be470c68e87b09457fb83e25c4950ef327757cf2a04` |

## Ownership, bounds and next safe work

- Root owns this comparator, study, tests and docs. Preserve the separate avatar
  renderer/tests. Current base is `f54ed0f`, following `89d3c29`; refresh remote
  state before publication. Stashes were empty. Standing merge/push approval
  applies; publication receipt: `.local/aspset-next-assessment-publication-01.json`.
- Reference extraction/parse: `.local/aspset-next-reference-01`; 30 wall/20 CPU
  seconds, 256 MiB RSS, ≤8 MiB/20 files. Actual .131 s /77,496,320 B; complete
  output 151,832 B. Supervisor exit zero, terminal/reaped.
- Assessment controls: `.local/aspset-next-assessment-{01,02}`. Each has frozen
  script/runtime/plan/source seals and a 30 wall/20 CPU second, 384 MiB RSS,
  32 MiB/32-file limit. Latest actual .259486 s /66,699,264 B; 3,445,821 output B.
  Both terminal/reaped. Independent saved audit .247 s. No live root worker.
- Report rendering reuses the exact `-02` results, without scoring again. Later
  comparator changes only tighten malformed-input handling; valid numeric
  arithmetic and the archived scorer remain unchanged.
- Two exploratory diagnostics accidentally used system Python, which lacks
  NumPy; neither created a run. Use `.venv-lab` for all numerical work. No failed
  experiment or valid result was overwritten.
- No API/key/ledger access, new acquisition, paid call, server, browser or
  deployment. The existing shared $5 Meta budget remains unchanged.
- **Next:** prepare one bounded source-only motion-feasibility study using the
  frozen body, force/mass assumptions and original image constraints. Distinguish
  noisy/weakly constrained targets, actuator saturation and feedback lag before
  selecting a repair. Keep the published-reference assessment separate from
  fitting/selection, retain both failures and improvements, preserve all task
  gates and force bounds, and compare full intervals plus changed starts.
- Independent ground/equipment contact, unfamiliar subjects/cameras, full-body
  balance and the complete generalizable product remain open. Do not replace
  those requirements with this one reference comparison.
