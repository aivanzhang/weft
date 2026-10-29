# Codex — Track A handoff

- Scope: video understanding and scene reconstruction for one generalizable Weft pipeline. [Parallel ownership](../parallel-workstreams.md).
- Revised assignment: [Track A goal text](track-a-goal.md). The runtime goal remains paused with its old full-project objective because the exposed goal controls cannot edit unfinished objective text; do not falsely mark it complete to replace it. User/UI replacement is still required for the runtime label.
- Status: autonomous development **paused by the user**. Handoff/Git preparation is separate authorized work; no new reconstruction run is required to hand off B.
- Workspace: `/Users/ivan/kintsugi/weft`, `main`; goal thread `01a18139-c198-70d3-8cfe-e8e79b56fa2a`.
- Last research verification: October 29, 2026, 4:14 PM PDT. `feature_tracks.py` and its twelve tests are complete; `feature-track-diversity-01` and independent verification are terminal. No live study job remains from this work.
- Evidence: [feature-history study](../feature-track-diversity.md), [original appearance study](../surface-feature-diversity.md), [conditional camera/depth failures](../conditional-rigid-camera.md), [human multi-view findings](../freemocap-three-view.md).
- Local verification: `.local/feature-track-diversity-verification-01.json`; 71 new sealed files plus 91 unchanged input files, 125 runtime hashes, 684 source/video frames, 1,090 object/frame slots, 35,596 current-point slots, 17,283 links and fourteen preserved conflicts. Twelve passing tests reused. Synthetic 75-frame errors independently recomputed for 9,173 point comparisons.
- Paid use: no new request. Shared ledger last verified at eight settled calls / $0.186152 of $5, no unsettled reservation. Keep keys/ledger local and coordinate across checkouts.

## Resume without duplicating B

1. Read the latest `TODO.md`, B handoff and ownership contract; verify worktree/remote state and live jobs before editing.
2. Keep B's runtime, UI, recording, CLI and task implementations untouched. Propose interface changes instead of editing shared files concurrently.
3. Continue source camera/shape/pose and complete correspondence work. Adjacent links do not recover the original anchor for the side box at frames 28–32; tiny balls have no persistent feature tracks. Do not promote counts to real point accuracy.
4. Preserve all source slots, scene-edit uncertainty and the existing conditional-depth/human-contact regressions. Choose fresh independent evidence before claiming generalization.
5. Update this note with exact commands, settings, evidence identities, failures, live handles, Git state and the next safe action at every meaningful checkpoint.

Source/model inference and physics were not rerun for the feature-history study; it reuses saved descriptors. Reproduction command (new output directory only):

```sh
.venv-lab/bin/python scripts/studies/check_feature_track_diversity.py --out artifacts/track-a-feature-histories-new
```

This is not an instruction to run it again unchanged. Existing results are sufficient; move to the unresolved requirement after the user resumes A.
