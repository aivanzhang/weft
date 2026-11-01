# Shared exercise lab package capacity

The complete lab package could exceed its 1,024-file cap even when each source, policy, evaluation and runtime input satisfied its own limits. Root first changed the aggregate file cap to **1,536**, then released producer buffers before the unchanged full output readback. The **192 MiB byte cap**, per-input limits, exact source/runtime checks and task semantics remain unchanged.

Root/UI's saved metadata estimate is **1,192 files**, with a mandatory lower bound of 1,151, and **168,773,639 fixed bytes**. This leaves 32,552,953 bytes under the existing byte cap for the remaining generated package content. That estimate comes from `.local/split-limb-feedback-play-01/packaging-preflight-02.json`; this test lane did not reread the real source or policy artifacts.

## Verification

Four targeted analytical tests pass in `tests/test_shared_exercise_lab_capacity.py`:

- A sealed authored source with 512 declared JSON members and a 512-entry runtime snapshot prepares and resumes through the public APIs. All source/runtime bytes and pins, the exact scene/task and the original source remain intact. The complete package has 1,035 files.
- The same independent input sizes, with 510 existing inspection-view aliases, require 1,545 files and are rejected before creating output.
- A reduced test-only aggregate byte ceiling rejects before creating output. It exercises the production byte-bound branch without allocating a large fixture or changing the production limit.
- A written noncritical source member is changed immediately before final readback. The original loader rejects its SHA256 mismatch, retains failure evidence and does not publish the ready lab configuration. The source and manifest's original pin remain unchanged.

The runtime fixture includes the real `runtime_files(scene)` bytes; final runtime readback and loader validation are not bypassed. Authored mass/floor/identity-rotation fixtures and empty reference frames replace geometry only. No simulation, source processing, rendering or learning is performed.

The first capacity test failed at the old aggregate bound; that red result is preserved. The file-cap change passed the initial three cases in 1.053600 seconds. After the memory-release change, those three affected cases plus the new tampering case passed: **4 passed, 0 failed**, 1.145076 seconds including guarded process startup. Both guards recorded zero native attempts and one denied subprocess attempt during import; no child process was dispatched.

## Memory overlap review

Root preserved the separate real-play RSS failures; the latest exceeded the 512 MiB observed ceiling. The producer previously retained complete source/policy/evaluation/runtime buffers while `load_shared_exercise_lab` reread the written package. It now captures the file count, releases those buffers, and removes attachment file maps before that same full readback. No file, validation or hash recheck is omitted.

Read-only review found no dangling default-path reference: every deleted variable is initialized on all successful paths and unused afterward. Camera policy identities and summaries are already retained separately; fitted-reference return identity survives the narrowed attachment. The returned count, scene/task and attachment identities keep their meaning. This review and the four tests establish validation continuity, **not measured RSS improvement**. Root owns a bounded comparison at the unchanged 512 MiB ceiling; this lane ran no real play.

### Attempt 04 and fresh-process ownership

Root reports that attempt 04 assembled a ready **1,195-file lab** with the buffer-release fix, then exceeded the 512 MiB observed RSS ceiling during `Lab` construction/initial summary in that same process. The ready lab and failed combined-process evidence remain preserved. Ready assembly does not establish successful live play or policy success.

Static review found no full-archive retention in the session or summary:

- `load_shared_exercise_lab` returns scene/task/contract, bounded policy or recipe bytes, the source-verification report and artifact name/size rows—not its full `files` mapping.
- `SharedExerciseSession` keeps those projections plus the environment/recording contract. `Lab.load_session` caches that one session; subsequent summaries do not reload the archive. Summary copies contain scene/task/state, not training, evaluation or source-file buffers.
- The first evidence-provenance summary parses up to five individually bounded metadata documents, then caches only its derived summary. Those document trees are temporary, but their allocator memory need not immediately leave process RSS.

Opening the ready lab in a fresh process removes the earlier build/readback allocator history and matches the normal build-then-open lifecycle. Static reasoning alone does **not** prove that admission, model construction or first summary stays below 512 MiB; the observed scope is recorded below.

### Observed fresh-process check: attempt 05

Root reports these completed stage checkpoints; this lane did not rerun or independently inspect the native outputs:

| Stage | Reported elapsed time | Observed peak RSS | Outcome |
|---|---:|---:|---|
| Fresh initial open | 4.019508 s | 395,706,368 B | Passed |
| 266 live actions, export and exact replay | 11.159369 s | 516,505,600 B | Passed: 532 total controls, 8,512 substeps, 268 frames |
| Second open in that same process | Started 11.5547 s; killed 12.1206 s | 623,230,976 B | Exceeded the unchanged 512 MiB ceiling |

The completed open/control/export stages stayed below **536,870,912 bytes (512 MiB)**. The later second-open failure remains preserved; it does not erase those stages or make the combined attempt successful. The normal server caches its existing session after export, so it does not perform that second admission on each summary. Root is separating a fresh, read-only resume verification to match user close/open behavior; controls and export must not be repeated for that check.

These observations establish only the measured stage outcomes, not a general memory optimization or a bound for every package. No speculative loader change is needed on this evidence alone.

A larger load-local overlap remains: `_verified_inputs` retains the entire package while three sequential `collect_shared_source_files` calls independently verify learning, evaluation and root source evidence. Each source check rereads pinned observations and full source-equivalence bundles, including their retained media bytes, then builds construction metadata. The three proof maps do not accumulate simultaneously, but each overlaps the complete package. Root estimates approximately 173 MB of package bytes and 67 MB for the final source-proof read; these are not new measurements by this review.

The smallest stronger conditional fix is to build the existing small return projection before the final root-source check, release the full package and its learning/evaluation/source/attachment submap aliases, then perform that same check with the exact manifest pin and deadline and attach `proof.report`. Earlier proof calls and every hash/semantic check stay intact. Camera admission contains only metadata/hash maps, not attachment files; fitted return requires its exact recipe bytes, not its envelope or selected proof corpus. Source-equivalence inspection reports contain no back-reference to raw bundle bytes. This ordering can therefore release the large maps without changing the return contract. The stage measurements must determine whether this final overlap is the actual peak.

A smaller duplicate also exists: `_validate_policy_package` keeps its parsed training/settings while the full verifier parses them again. Root's selected `training.json` is only 326,135 bytes (scene 481,780; task 1,009), so do not prioritize that change over the larger proof overlap without measurements. Both lifetime proposals remain conditional backlog, **not implemented**. Root owns the separate fresh, read-only resume verification. No source edit or native work occurred in this review lane; the existing four passing capacity checks and release 02 were reused without rerunning them.

Evidence:

- Red: `.local/shared-exercise-lab-capacity-01/red-01.xml`, SHA `426c30a552f8774d59511ef0a620c0c0ca1eb79a7345192c9dd094ba1d8e7d75`.
- Initial green/release: `.local/shared-exercise-lab-capacity-01/green-01.xml` and `release-01.json`, preserved unchanged.
- Current green: `.local/shared-exercise-lab-capacity-01/green-02.xml`, SHA `4f0180e3007ce18c844a1ec9dd2797b57abd2b5c5e9c9703fac0c5ef5297b1e6`.
- Current release: `.local/shared-exercise-lab-capacity-01/release-02.json`.
- Product: `src/weft/shared_exercise_lab.py`, SHA `382f2bd3dec89d87bbf2446261654d345fc88cee796d9d1bf4769b5cde6cdb1c`.
- Tests: SHA `37398c92f71bfeb8b18c63a40a7d3c43f6762f14da1670f85f1c6bfc781d4701`.

This closes the package-capacity regression, not the learned-policy success gap. Root owns the separate bounded play acceptance. Previous training results, saved policies and failed packages remain unchanged; no native retry or policy promotion occurred in this lane.

## Fresh-resume outcome

Root completed the separate read-only resume in **2.885010 seconds**, peak RSS
**399,474,688 bytes**, with 28 model constructions, one data instance, two forwards
and one reset; **zero new controls, substeps, replay or media decoding**. The full
run and all six recording members validate. This establishes the normal separate-
process lifecycle, not a fix for the retained same-process memory failure. The
conditional loader proposals above remain unimplemented. [Exact proof and launch
path](split-limb-feedback-play.md). No passing checks or completed physics were rerun.
