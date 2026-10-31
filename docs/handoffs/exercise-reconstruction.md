# Exercise reconstruction workstream

## Ownership and current status

- Owner: `/root/exercise_reconstruction`, delegated by root after the user's parallel-agent request on October 31, 2026 (America/Los_Angeles).
- Active implementation: source-only temporal support candidates and an opt-in support-conditioned path in the existing ground-orientation estimator. No new simulator, camera fitting, task changes or automatic body admission.
- Owned files: `src/weft/exercise_support_evidence.py`, `src/weft/ground_orientation.py`, their focused tests, `docs/ground-orientation.md` and this handoff. Root owns CLI/camera-root/worker/sandbox/recording integration and Git publication. Separate avatar work is untouched.
- No live jobs or native/inference runs owned by this workstream. The source-only comparison will use new outputs and be recorded before execution.

## Fixed hypothesis and limits

The previous ground estimator treats all four inferred heel/toe points as possible ground points. Lifted or moving feet can invalidate that assumption. The new opt-in path first finds stationary **image-point candidates** inside the unchanged first 60% construction prefix, under an explicit authored fixed-camera prior. It then uses only candidate heel/toe points in the existing equal-frame plane estimator. Every missing, moving, rank-deficient and excluded slot remains counted; fewer than three non-collinear points cannot supply a plane.

Stationarity and hand/equipment co-motion are not proof of ground, contact, force, support or grasp. Heights, camera calibration and physical validation remain unknown. No ankle point is used as a ground sample. Later observations, simulator floor scores and task outcomes cannot select candidates. Existing estimator defaults and screening gates remain unchanged.

## Preserved evidence

- Original single-leg body: 8/118 source and 16/235 grid floor failures; maximum 0.08313043391431332 versus the unchanged 0.025 limit. Root's separate camera proposal now completes but remains unchanged; do not retry it or loosen gates.
- Earlier angle, depth, floor-priority and four-source ground hypotheses remain failed/exposed development evidence. Existing kettlebell full-reference trials still fail equipment stability; grasp is unknown.
- New licensed Interval Push-ups source is already preserved at `artifacts/interval-pushups-source-screen-02`: SHA `e93ea83a78d47c27972f245a61d114118793598cfd059c318e120c453fd40bbe`, 1,543 decoded frames / 51.4 s / 640×480; metadata declares 1,544. Root's sample inspection sees two handheld supports and partial occlusion. It is not equipment-free/static-ground proof. Do not redownload or call it a fresh final holdout.

## Next actions and completion gate

1. Implement bounded source-clock/identity/missing-slot support evidence and integrate the optional selector into `estimate_ground_orientation` without changing its default path.
2. Verify analytical moving/lifted-foot, missing/gapped observations, same-source hand/equipment coupling, held-out exclusion and numerical-rank abstention cases.
3. Freeze one source-only comparison against saved evidence in a new directory. No inference, physics, paid API or source-media decode is needed for that first comparison.
4. Report actual source results, including abstention and regressions. A conditional plane improvement is not reliable physical reconstruction. Further unfamiliar-video inference requires an explicit bounded plan coordinated with root; no duplicated jobs, paid calls or model downloads.

This milestone is complete only when the opt-in existing-estimator path, relevant checks, source comparison and exact resumable receipts are documented. The overall unfamiliar-video/contact reconstruction goal remains open.
