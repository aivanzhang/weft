# Conditional temporal equipment geometry

## Complete scoped implementation — November 2, 2026, 10:37 AM PST

Owner: `/root/ground_equipment_reconstruction`. Root owns integration, later
native admission/task comparisons, Git and combined handoff updates. No commit,
push, browser, server, deployment, source acquisition, paid call or key/ledger
access in this lane. Preserve unrelated root/body-motion/avatar work.

Owned paths: new `src/weft/equipment_temporal_silhouette.py`,
`tests/test_equipment_temporal_silhouette.py`,
`scripts/studies/qualify_temporal_equipment.py`, this note; approved narrow
extension to `silhouette_geometry.prepare_silhouette_evidence` for verified
cached binary masks. Existing PNG behavior and the single-frame proposer stay
unchanged. No scorer, gate, force limit, floor, body, task or source edits.

## Implemented hypothesis

The old single-frame solid-band method freezes a whole band whenever that frame
contains competing foreground. The new opt-in method uses an explicitly supplied
fixed-camera/static-rigid prior to let a different clear construction frame
constrain the band. Every selected baseline-reachable positive row hull remains
required, including occluded frames. A supporting frame must contain the whole
original projected band in its target box, have no empty reachable row, and have
no competing foreground on any reachable row. Missing competitors freeze the
view; unknown pixels never supply exterior evidence.

Fixed authored image screening: at least 0.2 seconds, maximum box-edge range
3 pixels and maximum full-span fitted edge drift 2 pixels. These checks do not
measure camera motion or rigid stationarity. Holes remain filled. Original
depth, object pose, floor, mass, materials and original centered-box inertia
remain fixed. Source fit and collision-proxy volume are reported separately.
Baseline foreground misses remain visible; no new miss is permitted. Every
rejection is retained; no native admission, contact, grasp or load is claimed.

**34 distinct dedicated cases pass**. The first 27 cases are recorded in
`.local/equipment-temporal-silhouette-01/tests-01.xml` and `tests-01.json`:
0.970855 seconds /123,060,224 bytes peak RSS. Authored fixtures,
zero physics steps, network and child processes denied. Includes positive
multi-frame occlusion repair, fully occluded abstention, missing-competitor
uncertainty, unchanged depth/inertia, filled holes, static contradiction,
withheld-coordinate exclusion, input/time/pixel bounds, immutability and exact
cached-binary custody. Seven additional preflight/rejection regressions pass in
`tests-02.xml` / `tests-02.json`: 0.476173 seconds /122,535,936 bytes. No unchanged
passing suite was repeated. Preflight preserves each original unresolved
identity and rejects before PNG decoding, source proof or native preparation;
its passing status means only that full prepared custody is still required.
This is not source-video or physical qualification.

## Frozen source study

One fixed proposal each for `sam-2` and `sam-3`, using the existing automatic
scene at `artifacts/equipment-frame-consensus-native-03/build/lab/run`.
Frames `[8,18,28,56,80,116,152]` are frozen before candidate results: include both
original construction observations (28/18), early initialization and spaced
construction coverage. No assessment/final evidence fits a candidate.

Reuse `artifacts/pose-equipment-silhouette-02` for 14 exact binary equipment
masks and `artifacts/kettlebell-shared-evidence-01` for seven cached human PNGs.
Existing complete source equivalence is reverified with hashes, not decoded.
Original source SHA256:
`4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`.

Plan/runtime/supervisor are saved in
`.local/equipment-temporal-silhouette-01/`; output uses a new
`artifacts/equipment-temporal-silhouette-01/`. Bounds: 60 wall /45 CPU seconds,
512 MiB observed RSS, four native model compilations, seven PNG decodes,
zero simulation data/steps/controls/video decodes/model inference/API calls,
64 MiB input /32 MiB output /128 files, one worker. External and recurring
cost: $0. Runtime derives from current committed `main` with only the two owned
module overlays and the new study script. Unrelated dirty files are excluded.

### Actual source rejection

The one source invocation ended before fitting: PID 9566 exited 1 and was reaped
after **0.836475 seconds /179,453,952 bytes** child high-water RSS. It compiled
one model for existing validation, decoded seven cached human PNGs and reused
14 equipment binaries. Zero simulation data, steps, controls, video decodes,
inference or paid calls. Preserve the original
`artifacts/equipment-temporal-silhouette-01/failure.json` and every 01 receipt.

`sam-2` has `unresolved_association` with the person at **all seven** selected
frames. At frame 152, both equipment identities belong to a changed unresolved
group. `sam-3` therefore also fails the frozen plan. No prior was relaxed,
identity reclassified or post-result frame subset selected. Source fit was not
attempted; the collision proxy remains unchanged. **No contact improvement.**

Runtime 01 contains 319 files from commit
`58e6ac96801bab15764e40586222661d8b3d0456`, plus the two owned module overlays
and study script. Plan SHA256:
`b0a209978dddbeb7abeaae0a8e4a9675c135981562c30584c864812cd8c3e120`.
Historical command (do not rerun into the occupied output):

```sh
.venv-lab/bin/python -B .local/equipment-temporal-silhouette-01/supervisor.py
```

### Saved-only rejection preservation

The failed worker did not persist its prepared context. It was not presented as
resumable and its completed decoding/proof preparation was not repeated.
Instead, exact mask metadata already pinned in the failure receipt was passed
to the new preflight for both original objects, using the unchanged plan.

`artifacts/equipment-temporal-eligibility-01/` retains the original failure,
plan, exact mask bytes, released implementation, reproducible helper and both
seven-slot rejection results. It has seven files /1,806,433 bytes; all six
sealed members verify. Seal:
`bf4fbf0d84474b625e351a0553a1f2a86640f525bfa976372140be07574ec802`.
Saved-only run: **0.209152 seconds /82,362,368 bytes**, zero PNG decodes, source
proofs, native calls, fits or physics steps. It is rejection preservation, not
another reconstruction study. Completed command:

```sh
.venv-lab/bin/python -B -P .local/equipment-temporal-eligibility-01/run.py
```

Exact release pins, tests, resource receipts and saved-artifact audit are in
`.local/equipment-temporal-eligibility-01/saved-audit.json`. No job is active;
all local processes have terminated. The five owned tracked/new paths remain
uncommitted for root integration; unrelated changes were not touched.

## Remaining dependency and next action

- [ ] Root: review/integrate the reusable method, binary-input extension and
  tests. Its source qualification remains rejected on this clip.
- [ ] Establish independent evidence resolving the equipment/person association
  across a declared construction interval, or use a new qualified stable track.
  Do not remove frame 152 or relabel current identities to obtain acceptance.
- [ ] Only after a separately justified source candidate qualifies: run normal
  native admission and the original frozen task comparison. No such candidate
  or physical-contact result exists in this lane.

This lane is done when the reusable opt-in geometry and conservative preflight
are integrated with the preserved rejection. The broader video/body/equipment/
RL product and physical accuracy remain unfinished.
