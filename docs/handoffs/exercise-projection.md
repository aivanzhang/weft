# Exercise projection consistency

Study terminal 2026-10-31 11:51:55 PDT. Owner: `/root/exercise_reconstruction`.

## Current state

Implemented, 19 focused tests passed, and the single saved-source study completed. **The fixed-camera hypothesis was rejected:** foot direction error 33.9294° exceeded the frozen 5° limit at 100% foot-axis coverage. No geometry improvement, physical-ground/contact validation or body admission resulted. All jobs are terminal; the three new product/test/handoff paths are frozen and released to root for review/publication.

Root approved the exact source-only hypothesis in `.local/reconstruction-next-hypothesis-01.md` (SHA-256 `c5dd5d3623f5b5388b710e75ddf3fe05e72e116d81ea0c21f016dd42b815f03d`). No model, focal length, threshold, source selection or geometry was retried or changed after the result.

Owned paths: new `src/weft/exercise_support_projection.py`, new `tests/test_exercise_support_projection.py`, this handoff, unique `.local/exercise_projection_01.py` helper and `artifacts/interval-foot-projection-consistency-01/` output. Published support/ground/equipment files, root guides, camera/RL/lab files and avatar work remain untouched. Root owns Git publication; this agent performs no Git mutation.

## Frozen hypothesis and protocol

Test whether a fixed authored pinhole camera accounts for model heel/toe image-direction discrepancies. Input is the original hip-relative model geometry, not an articulated simulator body. Camera `fx=fy=640 px`, principal point `(320,240)`, original axes, zero distortion. No focal/rotation search, landmark adjustment, floor objective, native construction/physics, inference, media decode, network or paid requests.

Use the saved Interval first-60% split: 772 pose rows; first 463 construction rows, exactly 126 previously selected usable rows, 337 construction rejections, 309 later exclusions. Reuse this selection; do not recompute supports. Metadata and whole-document integrity checks may visit all bytes; rejected/later landmark payloads must not enter fitting or evaluation. All decoded, metadata-only, unsampled, missing and rejected slots remain explicit.

One linear three-variable camera-translation solve per selected row from available core points `11–16,23–28` only, confidence ≥0.5; require at least six points, rank three, condition ≤1e8. Points `29–32` are held out from fitting. Reject nonfinite solutions and nonpositive projected depths. Report endpoint residuals and signed-vector direction differences for heel→toe axes plus ten fixed bilateral torso/limb core segments. Zero-length axes remain unavailable. No correction is applied.

Conditional support requires ≥90% evaluable foot axes and both median foot/core direction errors ≤5°. Adequately covered failure rejects only this fixed camera hypothesis; missing coverage/conditioning is inconclusive. These authored diagnostic cutoffs are not physical accuracy gates. No outcome admits a floor or verifies contact, gravity, biomechanics or generalization.

Work bounds: one worker, ≤900 pose rows, ≤126 solves / 252 foot axes in this study, 32 MiB total input, 10 seconds computation, outer 30-second deadline, 512 MiB RSS and 8 MiB generated output. Fixed single attempt, no retries. Provider spend $0; no API/key/ledger access.

## Frozen inputs

- `artifacts/interval-pushups-full-pose-03/pose/evidence.json`: `f6956ee7e02022c76c72cf8832a913b32ae14be2e486ea3c4d71c1b7917a0989` (13,048,442 bytes).
- `artifacts/interval-pushups-full-pose-03/ground-support-selected.json`: `3c42bbb500614ab51bb54e805cad48ba8ecacb784aed471e65a2eae6f4c6beb6` (2,197,909 bytes).
- Input seal: `bec762dd6747c01f540f617fc3d76b5f8b4a90cd506debc5e2f04954db3d7614`; original source `e93ea83a78d47c27972f245a61d114118793598cfd059c318e120c453fd40bbe`, 1,543 decoded / 1,544 declared frames, 640×480, 30 fps.
- Existing development evidence, not an unfamiliar/final holdout. No original media/model file needs reading.

## Prepared implementation and checks

- `src/weft/exercise_support_projection.py`: SHA-256 `577cb900b4b569bf42712206bc7e351f8e844eda77e8ee136c9caa2fa8c68550`.
- `tests/test_exercise_support_projection.py`: `53c805fb3dc49fe943512061ef187adbc37252ed8fbdf56aab7b63d9096b2704`.
- `.local/exercise_projection_01.py`: `99ea16b66de7222ce0af5b5cdf1f0c3e33cf1e23123313dd8a3f6063166da689`.
- Frozen `artifacts/interval-foot-projection-consistency-01/plan.json`: `94af48f4f3a0ccb8c5f298c6f639654bbe6ba646ebdbaee59449aedfc9d5256c`.
- Unchanged dependency `exercise_support_evidence.py`: `9bf26ba6b35102148fea6d4a6650fa42bccd702cec6401db4096fa3f7d67a77e`; package initializer `92c241eadbc977563d4a2d4081145adba883298a12336f0a5f1829e33212ab9d`. The study contains exact runtime/helper copies and references the existing sealed inputs without duplicating their 15.2 MiB into the 8 MiB new-output budget.

Focused check: `.venv-lab/bin/python -m pytest tests/test_exercise_support_projection.py -q --junitxml=.local/exercise-projection-tests-01.xml` — **19 passed**. XML SHA-256 `32ee850b5e71addda24ced7571b9fd5653f74174672c02a97b40b6784f7d259c`. Cases cover exact synthetic geometry, held-out foot error without translation contamination, excluded payload nonuse, missing confidence, rank/conditioning, points behind camera, zero-length directions, source/clock/canonical bindings and input/output/deadline guards.

Guarded complete import-only preflight passed at 2026-10-31 11:51:20 PDT in 0.0549 s, peak RSS 35,274,752 bytes. Python 3.14.7 / NumPy 2.5.4. Receipt `.local/exercise-projection-preflight-01.json`; zero source reads, diagnostic calls, linear solves, media/inference/physics/API calls. NumPy supplies only small CPU least-squares systems. Network/child processes and media/inference/native-physics imports are denied. RSS is enforced by a 20 ms process-peak watchdog, not a kernel allocation cap; between-sample overshoot is possible and measured peak will be reported.

Exact preflight/preparation commands used the following environment prefix with helper subcommands `preflight` and `prepare`; preparation succeeded without evaluating geometry:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv-lab/bin/python .local/exercise_projection_01.py run
```

This was the only source-run invocation. No existing public module/default/evaluator identity was changed. Root published previous work at `0e4b1475e9c87fbc4b4884d12cde6c850220fff3`; this new lane remains unstaged for root integration and separate avatar modifications are preserved.

## Terminal evidence and outcome

`artifacts/interval-foot-projection-consistency-01/` is sealed, 8 files / 2,303,192 bytes:

- Manifest SHA-256: `64d70703307d7ab7aed95a79a7d7f37d6b2d2a99440a64c9bc7a8e3d53a0eb3b`.
- `projection-consistency.json`: `2a2579c0e5234b0a0be43540da30c8dc50b3aec04fe70442e9a144657d86bbd5`.
- `report.json`: `455c221dd8f4e980d0b96f850dd112c98a6761e76bed24b684b08e52fce11108`.
- One diagnostic call, **126 linear solves**, 0.633852 s diagnostic / 0.700727 s total, peak RSS 108,806,144 bytes. All frozen compute/output bounds passed. Zero media decodes, inference, native body/physics, network/API calls or paid spend. Source/runtime hashes matched before and after; no live handle remains.

All **252/252 foot axes** were evaluable. Median projected foot direction error **33.9294198242°**, versus raw world-XY **20.6271344415°**. The fixed projection made this directional discrepancy larger, not smaller. Median projected core direction error **2.5045469228°** across **756/1,260** core-axis slots. The remaining core controls were unavailable, not scored as zero error.

Of 2,016 selected endpoint slots, 1,596 projected in front of the camera and **420 low-confidence slots remained unavailable**. The result also retains all **337 construction-rejected rows, 309 later-excluded rows, 771 unsampled decoded frames and one metadata-only frame**. There are **3,088 total source foot-axis slots**; the decision denominator is explicitly the frozen 252 selected construction axes.

The failure is adequately covered, not a timeout or missing-foot-data result. It rejects only the authored `f=640`, centered, identity-axis camera hypothesis for the raw model feet. Good fitted-core agreement plus poor held-out-foot directions does not distinguish unmeasured intrinsics, local model foot error and real foot articulation. Gravity, true sole/contact locations and prop support heights remain unknown. The saved 73.57° stationary foot-plane candidate must not be promoted to physical ground.

## Next safe action

Root's saved-only audit passed: `.local/exercise-projection-saved-audit-01.json`,
SHA `ba40e7b731dbaff318e535a98d813cf5b7ff55a6a40f381a485940cd68b9935c`.
All eight outputs and three exact input metadata files were read once:
17,553,328 bytes, 0.205 seconds active verification. It checks seals, partitions,
failure/coverage denominators and recomputed aggregate errors, without estimator,
test or native execution. Original media/model bytes and unrelated input-manifest
members were not reread; historical execution counts remain sealed report claims.

Root can review and publish the three new files without rerunning the study or unchanged tests. Existing ground/equipment files have an empty diff, all source outputs are immutable, and this agent made no Git mutations. Do not retry focal lengths, widen limits, or rebuild a body around this rejected explanation. The next reconstruction hypothesis needs distinct source-backed calibration/contact evidence or an explicitly qualified foot model; this diagnostic supplies no physical admission evidence by itself.
