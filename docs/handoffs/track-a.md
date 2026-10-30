# Codex — Track A handoff

- Scope: video understanding and scene reconstruction for one generalizable Weft pipeline. [Parallel ownership](../parallel-workstreams.md).
- Revised assignment: [Track A goal text](track-a-goal.md). The user cleared the old runtime goal, installed this scoped objective through a pasted attachment and set it active. The attachment was read before continuing; its historical pause footer is superseded by that activation.
- Status: **active, Track A only**. agy completed B's spatial-lab milestone and is implementing capability/actor controls in the same checkout. Preserve `docs/handoffs/agy.md`, `src/weft/spatial_lab.py` and other B-owned work; use path-scoped commits, not blanket staging or stashing.
- Workspace: `/Users/ivan/kintsugi/weft`, `main`; goal thread `01a18139-c198-70d3-8cfe-e8e79b56fa2a`.
- Shared code/handoff baseline: `f819a93`, pushed and remote-verified; subsequent documentation receipts may advance `main`. The reviewed old stash was removed only after preservation in a verified local bundle. [Checkpoint details](../checkpoints/2026-10-29-collaboration.md). Verify actual Git state before resuming.
- Last research verification: **October 29, 2026, 5:28 PM PDT**. The conditional track-shape study and independent audit are terminal; no live A job remains. [Results and exact commands](../track-shape-diversity.md).
- Evidence: [conditional track-shape study](../track-shape-diversity.md), [feature-history study](../feature-track-diversity.md), [original appearance study](../surface-feature-diversity.md), [conditional camera/depth failures](../conditional-rigid-camera.md), [human multi-view findings](../freemocap-three-view.md).
- Local verification: `.local/track-shape-diversity-verification-01.json`; 40 new + 71 input files, five used source snapshots, all 684 frames / 1,090 object slots, 35,596 raw point slots and 56 controls checked. Independent projection difference ≤5.68e−13 px. The prior appearance-history results and fourteen conflicts remain unchanged in their linked guide.
- Paid use: no new request. Shared ledger last verified at eight settled calls / $0.186152 of $5, no unsettled reservation. Keep keys/ledger local and coordinate across checkouts.

## Completed shape-reconstruction checkpoint

- The preceding goal turn was **progress**: authorized prompt cleanup and scoped goal activation were pushed as `8ca9b60`. This turn adds a source-reconstruction method and verified evidence; it does not complete the broader goal.
- A-owned files: `src/weft/track_shape.py`, `tests/test_track_shape.py`, `scripts/studies/check_track_shape_diversity.py`, `scripts/studies/verify_track_shape_diversity.py`, `docs/track-shape-diversity.md`, this handoff and coordinator-owned `TODO.md` / architecture summary. B's runtime, CLI, UI, recording, capability and actor files are untouched.
- Twenty-five focused tests pass in `.local/track-shape-tests-01.xml` (0.05 s). Relative 3D reconstruction, noise, reflections, missing data, mismatches, insufficient depth motion, ambiguous camera constraints, coherent deformation and bounds are covered. No dependency/default/schema/scorer changes.
- `artifacts/track-shape-diversity-01`: terminal **40-file** seal `f95561b9eb4ba78abe96685a7045aab164a0c31243b29cd5e72691324aaed659`. The frozen method consumes fixed eight-frame windows from the prior journals without video/model inference. All 140 windows remain: 95 insufficient tracks, 21 insufficient depth motion, 15 camera-constraint conflicts, one affine conflict, eight short suffixes, **zero real shape proposals**. No geometry or physics promotion.
- Twenty-four known rigid controls reconstruct, worst normalized pairwise-distance RMS 0.002111. Twenty-four degeneracy/wrong-match controls reject. Eight coherently expanding objects still pass the image model: passing is not verified rigidity. All 56 saved controls are independently checked without refitting.
- Study 0.440 s; two workers; 30 s/case, 16 MiB/case, 96 MiB total bounds. Twenty maximum-size calls (32 frames × 128 tracks): median 2.459 / p95 12.111 / max 13.184 ms. Not UI latency. Zero paid calls; ledger rechecked at eight settled / $0.186152 of $5 / no unsettled reservation.
- Exact verification: `.venv-lab/bin/python scripts/studies/verify_track_shape_diversity.py --out .local/track-shape-diversity-verification-01.json` (exit 0, 0.389 s). Tests were reused unchanged. Side-box and exercise projection diagrams were inspected. Source/model/clock/identity gaps remain; these inputs are exposed development evidence.
- Git at checkpoint preparation: shared baseline `f7b5f2f` is B's completed spatial-lab implementation, following `8ca9b60`. Use only explicit A/coordinator paths for this checkpoint. Current B capability/actor work is unfinished and must not be staged, stashed or reset. The commit containing this checkpoint identifies its publication revision; recheck the remote before further integration.
- No live A process, server, lock or trainer remains. New experiments require fresh output directories. Do not rerun this unchanged study or repeat a material search before obtaining stronger source geometry.

## Resume without duplicating B

1. Read the latest `TODO.md`, B handoff and ownership contract; verify worktree/remote state and live jobs before editing.
2. Keep B's runtime, UI, recording, CLI and task implementations untouched. Propose interface changes instead of editing shared files concurrently.
3. Obtain longer or cross-view source correspondences and observable out-of-plane motion; short-window shape fitting currently yields no real candidate. Continue camera/shape/pose work without retuning the rejected windows. Adjacent links do not recover the original anchor for the side box at frames 28–32; tiny balls have no persistent feature tracks. Do not promote counts to real point accuracy.
4. Preserve all source slots, scene-edit uncertainty and the existing conditional-depth/human-contact regressions. Choose fresh independent evidence before claiming generalization.
5. Update this note with exact commands, settings, evidence identities, failures, live handles, Git state and the next safe action at every meaningful checkpoint.

Source/model inference and physics were not rerun for the feature-history study; it reuses saved descriptors. Reproduction command (new output directory only):

```sh
.venv-lab/bin/python scripts/studies/check_feature_track_diversity.py --out artifacts/track-a-feature-histories-new
```

This is not an instruction to run it again unchanged. Existing results are sufficient; move to the unresolved source camera/shape/pose and correspondence requirement while B integrates the lab.
