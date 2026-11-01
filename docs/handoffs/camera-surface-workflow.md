# Camera-aware surface fitting

## Current state — October 31, 2026, 6:12 PM PDT

Implemented, independently reviewed, native-verified on an authored fixture and
independently audited from saved outputs. Root owns the CLI, input packaging, parent,
worker, final acceptance and common documentation. The fitter, metadata loader
and offline report lanes released their reviewed files. The parallel
`/root/exercise_reconstruction` lane owns only `exercise_surface_inputs.py`, its
new tests and `exercise-surface-inputs.md`; its thirty-check collection/compiler
milestone is now released and root-reviewed for the same checkpoint. It is a
callable API; common CLI wiring is next. Preserve unrelated avatar changes and
saved studies.

The combined generalized-video goal remains active. This is a construction
proposal path, not reliable unfamiliar-video reconstruction or contact proof.

## Product path

Use the existing `body-reference-propose --objective camera-surface-intervals`
with `--surface-config FILE --camera-image-scale VALUE`; see the
[local instructions](../local-lab.md#use-an-existing-camera-based-body).
The explicit scale is restricted to `[1e-6,1]` authored loss units per pixel.
It does not convert pixels to metres. The supplied camera and all original body
bytes stay unchanged. Output is `angle-proposal.json`, never a replacement
`candidate-human.json`; body/controller installation is unsupported.

The separate [camera metadata loader](camera-body-reference-inputs.md) verifies
selected exact source bytes and camera/body/root/ground joins. It does not decode
media, authenticate unavailable historical ancestors or construct a model.
The camera in the surface config must equal the source camera byte for byte
before output creation. Default and older objective routes remain unchanged.

The [fitter](camera-body-fit.md) uses forward perspective/radial projection and
absolute endpoint derivatives, including the pelvis contribution. It preserves
original source knots, complete slots, first pose, roots, floor and all existing
gates. Raw-pixel nonregression is separate from authored objective scaling.
Image eligibility uses construction-partition admission independently of missing
surface ranges; withheld evidence remains inaccessible. Source ranges contribute
only at original knots, never invented intermediate image/surface measurements.

The worker has an exact 29-module runtime closure, existing restricted execution
and resource bounds. The offline report validates copied inputs, saved original
slots, projection/error arithmetic, loss units, unknowns and gate results without
native work. It retains rejected attempts and cannot imply an installed body.

## Verification

**173 distinct focused test cases pass**: root workflow/worker/CLI 81, camera
fitter and image admission 40, source loader 21, report and legacy dependencies
31. Counts were deduplicated by XML case identity across targeted reruns. Three
test-only failures remain in their original receipts. This is not a whole-repo,
browser or physical-accuracy result. No unchanged passing suite was repeated.
Consolidated receipt `.local/camera-surface-workflow-checks-01.json`, SHA
`a1345c1be074845a5bc833780bb86be27564c8f6b5f13bc3d4b0dbbfda46a79b`.

- Root: `.local/camera-surface-workflow-tests-01.xml` through `04.xml`.
- Fitter: `.local/camera-body-fit-first-01.xml`, `rest-01.xml`,
  `boundaries-01.xml`, `boundaries-02.xml`; [details](camera-body-fit.md).
- Loader: `.local/camera-body-reference-inputs-checks-01.json`, SHA
  `1cb366b33f80a9298df109bb780e37a068960cbf366e1cea3f08a8d671874364`.
- Report: `.local/camera-body-reference-report-checks-01.json`, SHA
  `ba6c4f75599f5c2f89a0bb7c58d4dd558bbc6c929a52efe19b6a787226f151b3`.

The actual saved analytical fitter result passed the real metadata loader and
report using explicitly synthetic sidecars. No fit/native run was repeated.
Corrected saved-only receipt
`.local/camera-body-reference-report-producer-01/receipt-02.json`, SHA
`f7f4f1f8ab21290d4f527e29fb2154b83033318334790ca28c83c26c1b9f0f11`.
Its earlier checker looked for the wrong export filename and remains preserved.
Synthetic media IDs and opaque ancestor hashes are declarations, not recovered
source history. The 200-switch scripted report check is not browser performance.

## Native check and preserved failure

One new-mode native workflow was attempted from committed `61af0d6` plus eight
reviewed exact overrides, excluding avatar work. Runtime: 38 files/744,776 bytes,
manifest SHA `5f267171760efeeea7263c27a3ba0999c940b76432cb3075e62c940b5a3b602c`.
Frozen plan `.local/camera-surface-native-plan-01.json`, SHA
`e7218d652790732df8cf96a58ddd9990f09fcf056d02c9d1cdd4aff50010c9cd`.

`artifacts/camera-surface-fit-native-01` **failed before model compilation or
geometry**: the analytical fixture has an incomplete `body_frame`. One fitter
entry, zero models/data/forwards/steps; 0.185-second worker. The known dependency
startup probe was blocked with its exact recorded identity, as predeclared;
no permission exception was made. This failure is not a successful native check.

The new `.local/camera-surface-native-inputs-02` corrects only the explicit assumed
frame schema and camera travel-domain declaration, plus dependent pins. Real pure
body/source/surface validators passed under denied-native stubs. Original angles,
roots, floor, cameras, observations and clocks remain unchanged. Fixture receipt
SHA `aa48ff0ffab59af42f8ed54650a3f0d1c1a57a5e6c05e485d1817eee5102f184`;
this remains synthetic metadata with absent media and opaque historical ancestors.

Root froze a separate plan before execution:
`.local/camera-surface-native-plan-02.json`, SHA
`acc47adcb1bc39ddef3d489bab1f5439d0786b17dcd81101cb4e016313436496`.
It reuses the exact reviewed runtime above and predeclares the one known blocked
startup probe; any additional/different/omitted event fails. No blocked operation
is permitted, and the original failed check remains failed.

**Check 02 passed once** through the actual public Python workflow, including
source loader, worker and offline report. CLI routing is separately unit-tested.
Outer 0.371 seconds; workflow 0.302 seconds; worker 0.199 seconds, fitting 0.076
seconds. Two models, four data objects, 147 forward calls, one fitter call; zero
physics/reset/media/inference/training/API work. Observed worker peak RSS was
78,331,904 bytes, not a hard cap or sustained-use guarantee.

Ten updates change two of three references. All fourteen construction gates are
true. Mean image error is 4.213256→3.189308 pixels; conditional surface loss is
0.004888687→0.002466389 m². Worst range excess remains 0.057088744 m at the fixed
first pose. These are authored coordinates and targets, not measured human or
equipment accuracy. The result remains `unadmitted_construction_proposal` with
unknown contact, unverified physical validity and no installed body.

- [Inspect the native result](../../artifacts/camera-surface-fit-native-02/inspection/index.html).
- Workflow seal `6ee9e356e5ea96638c72478fd4dc4854a90957ab240ce16e56c038928d9a10ee`.
- Proposal seal `450c3b3b0918900ad3fab9173fadaac5e605bce53cf491034bd378fc5ed3faf9`.
- Report seal `e19b11a95ab192706e13a2f88cc63157a32d1a97a4e1fac6cec45ccd3586804d`.
- Outer receipt `.local/camera-surface-native-acceptance-02.json`, SHA
  `87dc4fc78f639242339b106e5270e09ec3775332710becfb121f529674d7dadc`.

Both native handles are terminal. No successful check or old failed fixture is
to be rerun. The independent saved audit passed:
`.local/camera-surface-native-saved-audit-02.json`, SHA
`b58da89d5fabd57a55444129a4efc5dc18a2cf1e0376895fafa7643b53d8ba63`.
All 81 artifact files/1,280,889 bytes, seals, copied inputs, source clocks, 36 pixel
residuals, interval arithmetic, saved gates, journals and offline links verify.
The audit reused unchanged guarded runtime hashes and made 112 cached reads/
1,471,658 bytes in 0.045 seconds; no native/report/fitter rerun. Its first receipt
made an incorrect assumption about a diagnosis hash field; that auditor-only
failure is preserved, and the corrected join uses the exact saved stack and the
separate frozen-plan initializer pin. Native01 is still failed.

## Next actions and limits

1. Check `.local/camera-surface-publication-01.json` for the actual scoped
   commit/remote outcome. An absent receipt means publication is unfinished.
   Pre-release base/remote match `61af0d6`; stash/index were empty. Root owns the
   twenty-two-path checkpoint; all 1,249 avatar status entries are excluded.
   The earlier nineteen-path pin list is superseded by
   `.local/camera-surface-publication-pins-02.json`. No whole-worktree clean claim.
2. Connect the released source-side collection/qualification API to the existing
   CLI. Its thirty focused cases and root review bring this checkpoint to
   **203 distinct scoped passes**, not a full-repository result. Final receipt,
   exact API and release pins live in [its handoff](exercise-surface-inputs.md).
3. Continue later explicit
   updated-controller/body admission. Actual unfamiliar exercise, ground and
   equipment contact accuracy still need evidence; an angle overlay alone does
   not complete the generalized product.

No paid call, new source inference, media decoding, training, server, browser or
deployment. The protected scorer remains
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
Existing camera-policy lab and recordings are outside this changed closure and
remain unchanged; do not repin or rerun them for this milestone.
