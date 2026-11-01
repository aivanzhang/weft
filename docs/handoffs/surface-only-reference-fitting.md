# Surface-only reference fitting

## Ownership and current status

November 1, 2026, 2:29 AM PST. Implementation is frozen
on published base `046261b`; **31 distinct focused cases pass**. Root's one actual
public-CLI authored native acceptance also **passed**: the missing-image row
improved using its qualified interval and all twelve unchanged gates passed.
This lane prepared inputs/helper and read only the two terminal root receipts;
it did not launch native work or repeat the full saved-output audit.
This lane owns only `src/weft/whole_body_reference_refinement.py`, new
`tests/test_surface_only_reference_fitting.py`, and this note. Root owns config,
workflow/worker, report/CLI integration and any later public acceptance. No Git,
source, media, inference, native physics or paid work is authorized here.

This extends the existing construction optimizer; it does not repair a violation
of the old contract. That contract intentionally scheduled only usable-image
rows. Existing source failures, first-pose foot intersections and uncertain
contact remain unchanged.

## Agreed callable and output contract

The existing `propose_whole_body_reference_refinement` adds
`missing_image_policy='preserve'`. Accepted values are exactly `preserve` and
`qualified_surfaces`; the latter requires an existing surface objective
(`surface_interval_v1`, `camera_surface_interval_v1`, or
`camera_multiview_surface_interval_v1`). Invalid/inapplicable values reject
before geometry. No new method, profile, solver or target is introduced.

Default calls and explicit `preserve` retain their exact serialized fields and
numerical work. Only the nondefault option adds:

- `missing_image_policy: qualified_surfaces` to the candidate/overlay's
  `whole_body_reference_refinement` marker and `report.objective`.
- `surface_only_eligible` and `surface_only_attempted` booleans to every
  `report.references` row. Eligibility describes the baseline noninitial row:
  zero qualified image weights, at least one evaluated bound interval channel,
  and finite strictly positive interval loss. Attempted means a QP was scheduled
  for that missing-image row, flipped immediately before the existing solver
  call. A partial-budget interruption can precede its completed QP journal row.
- `changed_surface_only_proposal` for a changed missing-image row, or
  `unchanged_no_accepted_surface_step` for an eligible row without an accepted
  change, including a budget stop before any QP (`surface_only_attempted=false`).
  Other existing statuses remain; the first row remains frozen.

Current positive interval evidence is rechecked before every block, not just at
the beginning of a sweep. A missing, excluded, mismatched or already-satisfied
interval cannot schedule a missing-image correction. Source/model image weights
remain zero and image errors remain null; geometric predictions are not relabeled
observations. No interpolation target is added to the control grid.

## Invariants and bounds

Reuse the already-prepared exact source/body/entity/world/clock joins, interval
residual/Jacobian, existing QP and independent fresh-final geometry. Preserve all
source slots, original first pose/root/floor/shape, every angle/rate/depth/floor/
image/per-channel gate, and rejected/incomplete attempts. No midpoint, contact,
grasp, load, anatomy or physical-admission claim follows from a fit.

Unchanged production bounds: 30 seconds fit +8 finalization; 120,000 geometry
calls; 20,000 interval evaluations with complete final-audit reserve; 4 MiB trial
journal; existing input/output caps. These are cooperative per-call limits, not
a new isolation guarantee. Focused tests use analytical kinematics plus the
actual bounded QP, at most five seconds per case, with native/model/media/network/
subprocess operations denied. No real-source or native acceptance is authorized.

## Focused verification and release

Captured three default byte/call goldens from original fitter SHA
`1749db879d283a6e3767fdff4e1e9b8deda1c47e5a25b0a3818ec6362fafcbab`, retained at
`.local/surface-only-reference-goldens-01`. The digests exclude only the report's
elapsed seconds, not geometry/QP/projection counts, statuses or numerical values.
Both omitted policy and explicit `preserve` match those exact old bytes in all
three surface modes. The default missing-image row retains positive interval loss
without moving; new opt-in schedules its separately qualified interval.

All commands used `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 PYTHONPATH=src .venv-lab/bin/python -m pytest -q`,
`-o junit_family=legacy`, and the following focused selections/receipts:

1. Three `test_preserve_matches_published_bytes_and_calls` cases plus the camera
   `test_qualified_interval_moves_missing_image_row_without_fabricating_targets`,
   with `WEFT_SURFACE_ONLY_GOLDEN_OUT=.local/surface-only-reference-goldens-01`:
   `.local/surface-only-reference-red-01.xml`. Three baseline captures passed;
   the opt-in failed as expected because the old callable lacked the new keyword.
2. New `tests/test_surface_only_reference_fitting.py`, with
   `WEFT_SURFACE_ONLY_FIXTURE_OUT=.local/surface-only-reference-analytic-01`:
   `.local/surface-only-reference-checks-01.xml`, **28 passed / .380s**.
3. Four strengthened unavailable-interval cases, missing-first-pose preservation
   and current-interval recheck before another block/sweep:
   `.local/surface-only-reference-checks-02.xml`, **6 passed / .132s**.
4. `test_budget_before_first_qp_does_not_claim_an_attempt`:
   `.local/surface-only-reference-checks-03.xml`, **1 passed / .071s**.

Latest deduplicated outcomes: **31 passed**. No unchanged suite was repeated.
Cases cover all three surface modes; exact default preservation; null image errors;
full slots; unknown/missing/withheld/mismatched interval abstention before derivative
or QP work; already-inside and zero-gradient behavior; immutable first pose;
source-clock/rate preservation; nonlinear floor/camera-depth rejection; existing
floor failure; fresh-final interval failure; and complete final-audit/journal
reserves on budget exhaustion. Actual existing QP plus analytical kinematics was
used. Native/model/media/network/subprocess calls were denied; all cases finished
inside five seconds, no paid calls.

| Mode | Missing-row interval loss before, m² | After, m² | Geometry / interval calls |
| --- | ---: | ---: | ---: |
| Ordinary, 5 references | .001045986678004538 | 2.613660356115129e-10 | 80 / 31 |
| Single camera, 3 references | .001197619473125697 | 6.718985193317583e-10 | 34 / 14 |
| Two cameras, 3 references | .001197619473125697 | 6.6682392803755e-10 | 34 / 14 |

All original final gates passed in these descent cases. These are authored test
results, not measured reconstruction accuracy. The first pose, source clocks,
body dimensions, root and floor remain unchanged.

Saved actual analytical camera producer output:
`.local/surface-only-reference-analytic-01`, **12 members/189,641 bytes**;
manifest SHA `604fb12b9a10644e697df401f2b9c2e69fd1011255c66679c4f7d00a0bdb32f7`.
`report.json` SHA `b8b5146cedf2813e54bee6f9d35606d35f52bccd88332d45b897fb89014e2ce6`;
`angle-proposal.json` SHA `9c3eecc2fc39998b0701f073aaea32c6959977e50a3388abb9c9d679cf11f2e1`.
This package is for saved-schema composition, not an authenticated source ancestor
or a new native input truth.

| Released file / receipt | SHA256 |
| --- | --- |
| `src/weft/whole_body_reference_refinement.py` | `5766e6cd797671c54db20b70a8ca52a9cc8f142005ea4e642e78d1c65b228702` |
| `tests/test_surface_only_reference_fitting.py` | `cfa4db54752fd6b42bb3cfbaa535171d7a6305c26452353cad3ec143b53ae4db` |
| `.local/surface-only-reference-red-01.xml` | `eeea414211735765ddf98555c4bcb0103382cb58143d11809fa7c3d8f73baf93` |
| `.local/surface-only-reference-checks-01.xml` | `277f272b6a5a63f92fdabfb0a24a2eb7cc4266f3482311554e2d3e43e21a1a23` |
| `.local/surface-only-reference-checks-02.xml` | `e0c4a84b6e5f27be3fa6c03e35359678a25af01ab30a5268f7bf0a5002ca9dd1` |
| `.local/surface-only-reference-checks-03.xml` | `9c4563c96707786f0f41ac21d03107ebf63cd0af3c7341bf796d19558d75e3b3` |

No live handle or Git mutation. Product/test bytes are frozen for independent
review; root owns public routing/admission. The ordinary native acceptance uses
**five** references because one entirely unavailable image row in a three-row
sequence would violate the unchanged root estimator's minimum three confident
torso samples and 80% observed-root coverage. Four observed rows out of five
preserve both gates; the one existing short interpolated root stays explicitly
derived. The three-row camera fixtures instead retain already pinned original
camera roots. No source threshold or interpolation rule is relaxed.

## Frozen native inputs and public acceptance

Inputs-only package `.local/surface-only-reference-native-inputs-01` contains
**23 files/138,554 bytes**. Every declared parent member is materialized, including
the opaque authored `source.webm` bytes with their real byte hash. It is not a
playable recording or authenticated source ancestry. Existing reviewed decoder
source is retained only to exercise the metadata interface, never executed in
fixture preparation. Original numerical camera, body lengths/mobility, fixed
body→world translation `[.5,-.1,-1]`, authored height range `[.2,.6]`, pixel
annotation `[720,337.5]`, uncertainty 6 pixels and weight 2 are reused from the
prior authored `.local/surface-interval-native-inputs-01` input documents, not
from fitted outputs. New source identities and every dependent raw/canonical pin
are joined explicitly.

There are five ordered pose/reference records at original frames 0,2,4,6,8 and
all ten source slots. Frame 4 retains its exact authored coordinates but all
model confidence values are `.49`, below the unchanged `.5` threshold. Its
independently authored surface annotation is not derived from that confidence.
Four observed root samples give exactly 80% coverage. The existing pure root
adapter records frame 4 as `interpolated_reference_only`, originally
`low_confidence`, using frames 2 and 6; no new interpolation or observed position
is fabricated. The numerical root translations remain zero and the original
first pose/floor are unchanged. Canonical surface inputs retain five construction
slots and five unsampled slots.

Preparation command, executed once:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  .venv-lab/bin/python .local/prepare_surface_only_native_inputs_01.py
```

Preparation took .029394 seconds with one pure root-reference metadata call and
zero fitter, QP, model, physics, decoder, inference or network calls. Native,
model, media, network and subprocess imports were denied. The eight old input
pins were checked before writing the exclusive new directory. No unchanged
tests or studies were rerun.

| Prepared member | SHA256 |
| --- | --- |
| Fixture `manifest.json` | `baa63c3371b35847a10fb7ec18b4a5ff37639a56a7a54e64b658a41d7b90a2e8` |
| `fixture.json` | `d55d05c76e6afa0792ff0894f650e37ddc3e372ebaeb623560706eb3e2918cf0` |
| `body-source/manifest.json` | `88301dc1a9706b0d1ff7d9ab1d8acf3bea56688a5b7e2f24517e0486ef52cd92` |
| Preparation helper | `f481fbf7a7ce718e3a9e61c79cf82610ce694659955845a3646325eca9acf9f5` |
| `.local/surface-only-reference-native-plan-01.json` | `9850642b94b150a4c09d5b505cd5614dbc6971988a20870580a36be893c86d35` |
| `.local/run_surface_only_reference_native_01.py` | `fd577655b1b4e6e0d98370ce79c6f725cb948352007732fe8850e339603242a7` |
| Root's frozen runtime `runtime-manifest.json` | `196d0414c79ca4fab6b3f95cb099afdb389b5e88cc20786afc0c57caf8db2c72` |

Root froze `.local/surface-only-reference-native-runtime-01` from exact committed
`046261b6922f1b42318a6cfe48ff888d041d754b` plus only the reviewed core/config/worker/
report overrides: 291 files/6,299,504 bytes, unchanged scorer. The helper calls the
actual public `body-reference-propose --objective surface-intervals` command,
not a direct fitter shortcut. It uses the existing isolated worker, then only
saved-output checks. The declared fit is 5 seconds plus 8-second final audit,
workflow 30 seconds, parent 45 seconds and root supervisor 50 seconds. Existing
120,000 geometry/20,000 interval/4-MiB journal limits remain; workflow is bounded
to 64 MiB/96 files, worker output to 16 MiB, offline HTML to 2 MiB, and the separate
acceptance receipt to 128 KiB. RSS is observed, not a hard memory guarantee.

The falsifiable hypothesis requires frame 4 to be eligible, actually attempted
and changed, with strict interval-loss descent, zero image weights/null errors,
all twelve unchanged final gates passing, full source slots and unknown contact.
The original first reference, root, floor, shape, clocks and source evidence must
remain exact. Rejection, incomplete solve or failed hypothesis remains retained;
no tuning or unchanged retry is authorized.

The [prior blocked-call diagnosis](body-worker-denials.md) identified MuJoCo's
optional Darwin startup probe. No guard is relaxed: all nested processes remain
forbidden. This plan pins initializer SHA `24000c0b3c195f2bdb28a65420c2004b019a9bb6f6737e4caa1de798ffc064c5`
and the exact dependency versions. It accepts zero denial attempts or only that
exact predeclared six-frame blocked startup event with zero omitted diagnostics;
any unexpected denial fails. A blocked attempt is not a successful subprocess.
The earlier failed acceptance remains failed and its caller remains historically
unknown; the new plan does not relabel it.

Prepared invocation (executed once by root, not by this lane):

```sh
.venv-lab/bin/python -I -B .local/run_surface_only_reference_native_01.py \
  9850642b94b150a4c09d5b505cd5614dbc6971988a20870580a36be893c86d35
```

This lane parsed helper syntax only. Root independently reviewed and executed the
exact frozen command, owning its external supervisor/log/reap. Output is
`artifacts/surface-only-reference-native-01`. Product/test bytes remain at the
release pins above.

### Root's completed native result

The external process exited zero and was reaped in **.608321 seconds**. The
acceptance receipt took .556819 seconds; the guarded worker took .241893 seconds.
Together the timely caller and pending-caller receipt establish terminal pass.
The candidate remains **`unadmitted_construction_proposal`**, not an installed or
physically validated body.

- Frame 4: `surface_only_eligible=true`, `surface_only_attempted=true`,
  `changed_surface_only_proposal`. Its interval loss fell from **.0163805** to
  **.001038643912756187 m²**. Image weights remain zero and errors null.
- All **twelve** original gates passed; five references, ten source/channel
  slots and the single explicitly interpolated root were retained. First pose,
  root, floor, shape, source evidence and clocks remained unchanged.
- Sum of five interval losses: .0819025 → .03513191265111153 m². The maximum
  outside distance remains .181 m; this is not a claim every pose reached its
  envelope or that source geometry/contact became physically correct.
- **23 accepted updates, 69 interval evaluations, 204 geometry/forward calls,
  2 model compilations and 4 data allocations**, one fitter call. Physics steps,
  resets, media and inference calls are zero.
- Parent peak RSS **94,420,992 B**; worker peak RSS **78,774,272 B**. These are
  observed peaks, not hard memory ceilings. Output: **67 files/1,184,687 B**.
- The one exact predeclared MuJoCo startup probe was blocked. Its 672-byte,
  six-frame trace has zero omitted events. No parent denial or unexpected worker
  denial occurred; no permission exception or successful nested process was
  introduced. The previous unrelated failed acceptance remains unchanged.

| Terminal receipt / artifact | SHA256 |
| --- | --- |
| `.local/surface-only-reference-native-process-01.json` (488 B) | `5ec2075ab5c1f6c4ed270dce3334a6fe773de4d0dbd2f1bde7a7d0291c0999a2` |
| `.local/surface-only-reference-native-acceptance-01.json` (8,160 B) | `ce46a0e31aa730234ecafc74fdcbc8a0c822e5a49e47ed8848ae0e393fb0bae3` |
| Workflow seal | `b94613614a407b83452ef2336f9e1065bb73a27b46725309b80ca2b462ed7182` |
| Proposal seal | `e31e7a3b5d9369fd406549fd1c12124e2bece1c4ab300e50e0e756441bad4e71` |
| Inspection seal | `5addcf12c4bef07843264173ef86dc996bfcabfcf7268e3ad3339e11f4f31bde` |
| Saved report | `5f136298578dbdc88bf5bb6597605eeb06e4a937503a1c2eea294d1a31928a29` |
| Saved candidate | `6a12bd5993d8c02741783ada31ca867b99470c3562b684444fdfc02db68a98e9` |

This lane's terminal update read exactly the two receipts above (**8,648 B**),
not source data, the output tree, geometry or the fitter. UI owns independent
saved-report presentation checks. No live handle or Git mutation remains here.
The next safe action is root's scoped publication after its other lanes finish;
do not repeat this successful fit. Real unfamiliar-video surface qualification,
contact evidence and physical accuracy remain unresolved dependencies.
