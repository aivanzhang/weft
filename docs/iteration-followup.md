# Orientation and appearance follow-up

Verified **October 28, 2026, 1:02 PM PDT**. Local development evidence; no paid call, deployment or submission.

## Final end-to-end check

`artifacts/explicit-modes-e2e-03/` completed five cases in **43.381 seconds**, with two bounded local workers. All five produced executable environments and recordings; **four passed all declared gates**.

| Case | Explicit inference | Result | Replay / servo error | Learned / random / coast success |
| --- | --- | --- | --- | --- |
| Single-leg squat | video | Gates passed | 0.210 rad | 100% / 18% / 0% |
| Squat | video | Gates passed | 0.236 rad | 100% / 10% / 0% |
| Kettlebell pushup | video | Gates passed | 0.218 rad | 100% / 82% / 0% |
| Jumping jacks/burpees | image | Executable; servo gate failed | 0.394 rad; limit 0.35 | 100% / 10% / 0% |
| Grayscale rolling | appearance regions | Gates passed | 3.12 px | 100% / 42% / 88% |

Rates use 50 matched evaluation episodes and one training seed. They measure authored target tasks, not whole-exercise imitation or physical transfer. The high no-control rolling rate shows that task is easy. Gates were not relaxed.

Each case recorded **12 transitions / 13 video frames**; actions and states match exactly, exports decode fully, and immutable run manifests remain unchanged. The study seal covers **487 files**, SHA-256 `29dd0646fc64d8f2c0f2adb58573d5054873afac6440e94b119f9fa8a122d47e`.

```sh
# Inspect and interact without rebuilding.
.venv-lab/bin/python -m weft lab --resume artifacts/explicit-modes-e2e-03/kettlebell-pushup
.venv-lab/bin/python -m weft lab --resume artifacts/explicit-modes-e2e-03/rolling-grayscale
```

## Preserved successes and failures

- [Body orientation](body-fidelity.md): actual physics now uses a fitted fixed torso orientation. Frozen visual development labels improve from 3/9 to 8/9 within 15°. Final pipeline outputs reproduce mean error **8.41°**, max **18.98°**. `artifacts/body-orientation-candidate-02/` preserves labels, predictions, grader, input manifest hashes and seven sealed files; manifest `42e9b24470e648630747a32e227b004a00ee8600d5f822460e35ac854ff7b28e`. These are assistant estimates, not expert biomechanical ground truth.
- [Pose inference](pose-identity.md): independent IMAGE inference removes jumping-jack extra hypotheses but creates false extras on single-leg squat. VIDEO remains default; IMAGE is explicit, with no silent fallback or dropped poses. The intermediate all-IMAGE suite is preserved at `artifacts/exercise-body-e2e-02/`: three executable cases, two gate passes, single-leg/lunge perception failures. Its 328-file seal is `e741dc28893c4a795e42ebf61c7d2e86df1e4c84a850c58729b2d2600884b56c`. Both modes miss a second person in one of five visual montage controls; person-count verification is not claimed.
- [Appearance tracking](general-tracking.md): the grayscale ball is the existing real source transformed, **not another independent trial**. All 157 frames are retained; only preselected first-pass frames 39–60 enter reconstruction. The second pass is excluded. Electrostatic and metallic-pendulum clips remain unsupported physical reconstructions because of background movement, missing tracks and unidentified forces.
- Root translation, floor/kettlebell contacts, balance, out-of-plane joints and muscle forces remain unsupported. Jumping-jack execution does not erase its failed servo gate; lunge remains rejected.

## Checks and Meta readiness

**280 distinct Python tests verified**, across the 278-test regression and affected final tests (two new tests). The sandbox denied loopback binding for 21 HTTP checks; all 21 passed with narrowly scoped local-socket permission. No assertion failed. Saved reports: `.local/iteration-body-appearance-tests.xml`, `.local/iteration-http-permitted-tests.xml`, `.local/explicit-modes-tests.xml`. Fourteen deterministic UI checks also passed. Real browser/device QA remains unperformed.

The final pushup prepared a construction-only Meta request at `artifacts/meta-pushup-oriented-dryrun-02/`, with zero requests sent. The approved **$5 total** application ledger remains configured with $0 committed; no project key is present. It is an estimated application-side limit, not a provider-enforced hard cap. [Key setup and live-call limits](meta-advisor.md).

No verification workers or temporary HTTP servers remain running. Preserve all failed runs and labels; use new output directories. Next work is targeted pose/ground-contact fidelity and independently labelled general-object motion, not rerunning unchanged passing cases.
