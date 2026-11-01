# Shared exercise loader buffer lifetime

Updated **2026-11-01 10:50 PST**. Owner: `/root/contact_reconstruction`; root owns native measurement and integration. **Implementation, 24 focused analytical checks and root's bounded real-load measurement are complete. The candidate finishes two retained, read-only opens under 512 MiB; the baseline is killed during its second open. This does not validate the previous full play/export/reload workload.**

## Change and boundary

The saved lab has 1,192 run files totaling 173,860,491 bytes. Previously, `load_shared_exercise_lab` retained the complete package and its learning/evaluation/attachment aliases while sequential source proofs reread their own sealed evidence. This duplicates large byte buffers during admission.

`src/weft/shared_exercise_lab.py` now builds the same small return projection, queues the original learning/evaluation/root source proofs, releases the full archive and all aliases, then runs those proofs sequentially. Each proof keeps its exact root, manifest pin and original deadline argument. Only the root proof report remains in the return value, as before. The proof object's raw-file map is not retained between calls.

All package/source hashes, source-equivalence checks, runtime/policy identities, limits, task definitions and validators remain. No cache substitutes for validation. **Failure order across validator categories changes:** evaluation metadata, policy status, source-video and configuration checks now precede the deferred learning/evaluation source reads. Within the deferred proof queue, the order remains learning → evaluation → root. A failure in any proof prevents return; no partially usable session escapes.

Projection preparation has no live-session side effect. Existing policy metadata validators may construct mass models. The fitted-reference attachment prepares an inert recipe provider; camera admission contains metadata/hash maps, not retained raw files. Live environment construction and controller sampler admission happen only after the loader returns successfully inside `SharedExerciseSession`. No source or held-out partition is reclassified.

Product SHA256: `a721d4de760da115532ea4bb60c971318570a033d6903a6d9ec1275419e8260d`. Tests: `tests/test_shared_exercise_lab_retention.py`, SHA256 `b7812591d85dfd1f44ca6643481549863b0a0db73a2cb3454e0561b6e97d6fd4`.

## Focused verification

**24 distinct cases pass**, with no native execution:

- Eleven new authored/mock cases: weak references prove archive/alias/prior-proof release before every read; five loader modes preserve exact return projections; corrupt learning, evaluation and root proofs expose no session; source tampering, invalid configuration and final timeout still reject.
- Four existing package-capacity cases pass, including complete manifest readback and corruption rejection.
- Nine existing analytical public prepare/resume cases pass across feedback, fitted-reference, camera-policy and coupled-feet paths. Their existing geometry fixtures replace physical calculations; these are not real native acceptance runs.

The first batch preserved `checks-01.xml/json`: 15 passed, two failures and seven setup errors. The two failures were native quaternion calls stopped by the analytical guard; the seven errors arose because that guard's blocked `MjModel` lacked the attributes legacy fixtures patch. No native operation ran. A narrowly adapted guard preserved those attributes and used the existing authored identity-rotation/mass substitutions. Only those nine previously unsuccessful cases were rerun; `checks-02.xml/json` records nine passed, zero native attempts, 1.034758 seconds. Both import guards denied MuJoCo's Darwin subprocess probe without dispatching a child. The original failed receipt remains unchanged.

Commands and exact evidence live under `.local/lab-loader-retention-01/`:

```sh
# First batch: 15 pass; preserve the guard/legacy-fixture incompatibilities.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable .venv-lab/bin/python -B \
  .local/split-limb-feedback-actor-01/run_pure.py \
  .local/lab-loader-retention-01/checks-01.xml \
  .local/lab-loader-retention-01/checks-01.json \
  tests/test_shared_exercise_lab_retention.py \
  tests/test_shared_exercise_lab_capacity.py \
  tests/test_shared_exercise_feedback_lab.py::test_lab_prepare_and_resume_preserve_complete_proof_with_original_deadlines \
  tests/test_camera_reference_lab.py::test_new_lab_and_resume_keep_reference_separate_from_policy \
  tests/test_camera_shared_lab.py::test_camera_lab_preserves_source_and_distinct_task \
  tests/test_shared_exercise_feet_lab.py::test_feet_lab_seals_exact_task_source_and_new_runtime_closure \
  -o junit_family=legacy

# Only the nine affected legacy cases; terminal success.
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable \
  .venv-lab/bin/python -B -P .local/lab-loader-retention-01/run_remaining_checks.py
```

JUnit SHA256s: first `be23c0a6f0b8e503f10b169a70152bfdaabfdd6eb2f42498bf9bc7f6864fbeae`; second `c9f4ebf2f5c2eddd2903ba9571ac58c59482bdae81e7febb2aac7eb0a04f93a1`. Existing passing checks were not rerun. Scoped whitespace checks passed. No test or runtime policy/scorer change was made to obtain these results.

## Frozen native measurement — executed once by root

Root reviewed the complete loader/worker/supervisor/plans, then executed each plan once, sequentially. The plans use the exact existing lab `artifacts/split-limb-feedback-play-05/lab`. Its run manifest remains `36fefcb67333f51e212860555e1af95f4e594a52b3bf17688bcf11f324a372af`; configuration remains `1dcff82c8cd3bb4226a72ddfc80c4f4dce746dc19d1fffe95e41e73d2bb3421a`. No new lab export, controls, replay, source acquisition, media decode or paid call occurred.

| Variant | Frozen runtime | Runtime manifest SHA256 | New output |
|---|---|---|---|
| Baseline | `.local/split-limb-feedback-play-04/runtime` | `6acc810d058c7bf422110e7f411738265d259ff4c6d11728705670318fa034f5` | `artifacts/lab-loader-retention-baseline-01` |
| Candidate | `.local/lab-loader-retention-01/runtime-candidate` | `4d23784db749d9b2b2fbb55e2e442865623bb7b16028252819cda8dcc7df3fe1` | `artifacts/lab-loader-retention-candidate-01` |

The candidate differs from runtime04 only in `load_shared_exercise_lab`; all surrounding bytes of that module and all other 290 runtime files remain exact. The baseline loader is `382f2bd3dec89d87bbf2446261654d345fc88cee796d9d1bf4769b5cde6cdb1c`. All 25 policy-closure hashes remain exact, including task/scorer `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb` and selected policy `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`. This worker must not import the independently changing checkout.

Both variants use the same worker and supervisor:

- Worker `.local/lab-loader-retention-01/worker.py`, SHA256 `52b77b3bb0230c31a38e01581d845f456bd90be835f2b7e9d89a71d360d66832`.
- Supervisor `.local/lab-loader-retention-01/supervisor.py`, SHA256 `d71b5e72c5233701696a68da263b58b273899c234089ea42d887c94fb40d24cf`.
- Existing descriptor-aware guard `.local/split-limb-feedback-native-01/file_guard.py`, unchanged SHA256 `1a37cde2e6f3508e2b136d3b7a6275239b14bbca0e519c4c38462dca4a454673`.
- Baseline plan SHA256 `418636e6f30a66416bdb39f1f887ddb1d6bb47e2dc9491ff1a74c4001c41a710`; candidate plan SHA256 `bafd25f46c1d07247f4ae2e6c82cec6c53c0b7e5296a84ec38afb9d68a228f3c`.

Each worker creates two distinct `Lab`/session/environment objects and calls the ordinary `Lab.summary()` once on each. The first Lab stays explicitly alive through the second open. Both summaries must preserve the original nominal state, scene, task, policy and action contract. Fifteen declared public fields must agree within each process; their hashes allow the same comparison across variants. Both full summaries are saved for review.

Expected totals per process are 56 model constructions, two data objects, four forwards and two resets; zero controls, substeps, replay or decoding. Guarded constructors enforce each load's share. Native steps, rendering, recording/export and training paths are blocked. Flush-on-stage checkpoints preserve elapsed time, RSS high-water and counters even if the supervisor kills the process. First-object retention and two distinct sessions are explicit acceptance assertions.

Each variant has a **512 MiB observed RSS ceiling and 60-second external bound**, including a five-second cleanup reserve; 50-second cooperative work and 55-second CPU bounds. Run sequentially, never concurrently. One child, no retry, a sanitized environment, no credentials/network, 4 MiB/16 result files, and two logs capped at 1 MiB each. The supervisor samples only its child and reaps its process group. This is sampled RSS plus final OS high-water validation, not hard memory/disk isolation.

Each new worker hashes its 291-file runtime once, checks six pinned metadata inputs totaling less than 2 MiB, and performs stat-only run/recording inventories before/after. It adds **no full 173 MB package scan**: both ordinary loaders retain all required package and source-proof reads. It does not rehash or decode saved recording payloads. Preparation copied/verified 6,312,858 runtime bytes once, read 163,441 small metadata/source bytes for freezing, and read zero saved-lab payload bytes. Earlier inspection read 219,960 bytes of small manifests/config only. `preparation-01.json` and `static-review-01.json` preserve exact bounds and identities. Static checks confirm that the matched plans differ only in variant, runtime, runtime search path, output and plan argument. No measurement worker or supervisor has been executed by this lane.

Exact commands root executed **sequentially**; do not rerun these occupied output paths:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/lab-loader-retention-01/supervisor.py \
  /Users/ivan/kintsugi/weft/.local/lab-loader-retention-01/baseline-plan-frozen-01.json \
  418636e6f30a66416bdb39f1f887ddb1d6bb47e2dc9491ff1a74c4001c41a710

/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/lab-loader-retention-01/supervisor.py \
  /Users/ivan/kintsugi/weft/.local/lab-loader-retention-01/candidate-plan-frozen-01.json \
  bafd25f46c1d07247f4ae2e6c82cec6c53c0b7e5296a84ec38afb9d68a228f3c
```

The exact argv arrays also appear in `preparation-01.json`. Logs/process receipts are in the lane's `baseline/` and `candidate/` directories. Both handles are terminal and both processes were reaped. Preserve the baseline's partial output/failure receipt. Do not rerun the same plan/output or expand bounds.

### Observed result and partial-baseline limit

| Variant | Outcome | Elapsed time | OS high-water RSS | Preserved output |
|---|---|---|---|---|
| Baseline | Killed during second summary; exit −9, reaped | 5.733592 s until failure | 546,373,632 B | 2 files / 828,624 B: plan and first summary |
| Candidate | Both opens complete; exit 0, reaped | 6.245397 s including supervisor cleanup before receipt | 499,630,080 B | 5 files / 1,663,137 B, complete seal |

The unchanged ceiling is 536,870,912 bytes (512 MiB). Baseline sampled peak was 546,357,248 bytes; the final OS high-water check also failed. Candidate sampled peak equals its OS high-water, and its final resource check passed. Candidate totals are 56 models, two data objects, four forwards, two resets, and **zero controls, substeps, replay, decode or forbidden attempts**. The first Lab/session/environment remained alive through the second open.

Root verified every candidate manifest member and exact equality of all 15 declared fields across the **baseline first summary and candidate first/second summaries**. Their scoped hash is `732dae3421eb83c5fc311537868502d5de451a220e29edb2a009fd305956f83c`. Both candidate summary files are byte-identical, SHA256 `7190a538a640dfc19caa842bc373579d760d1cd1062d9f080e8eaaa5e87578c7`.

The original symmetric requirement that both processes complete **is not met**: the baseline has no second summary or success seal. Do not report a full-baseline time comparison, speedup, a completed-baseline memory saving percentage, or a passing baseline. The measured result is narrower: the candidate completes the repeated-open workflow under the same cap that stops the baseline, with checked public semantics preserved for every available summary. The previous full play/export/reload failure remains unchanged.

Evidence:

- `.local/lab-loader-retention-01/comparison-01.json`, SHA256 `cc2eb13f345bafcab123352d9911b00aef123da6c9d5ee2ad8b2ab35d45db6c3`, root's verified comparison.
- `baseline/process.json`, SHA256 `9a99f8f1d14d1d36eebf5f9c92acd6346fd0815087abb1c63d8a081e3f241791`; PID 21568, terminal/reaped.
- `candidate/process.json`, SHA256 `1fab02178c35aad1f337d303f8fdd3b28d5aa87c9732dd0ada45bf315fcd6efd`; PID 21634, terminal/reaped.
- Candidate `result.json`, SHA256 `1e4ba96f4e7a42b123ff621d903da14c86ade57fa6b9e67f0c04b1ace2b380b8`; candidate seal `manifest.json`, SHA256 `b01092602805306a065768edf9e477ff9e39f0b9a2043834f679b8e5d001b8fc`.

This lane read only the 5,960 bytes of comparison/process receipts after root's validation. It did not repeat the native execution, source reads or summary/member hash verification. `release-01.json` records the final handoff identities and exact result scope.

## Interpretation and next action

The old full play/export/second-open attempt remains failed; its completed export remains preserved. The fresh one-open resume passed around 399 MB as documented in [the prior capacity handoff](shared-exercise-lab-capacity.md). The new comparison measures **repeated read-only opens**, not that earlier full build/play/export allocator history. A pass is not evidence of physical accuracy, policy success, arbitrary-video reconstruction or all-workload memory safety.

This lane is complete. Root owns the local-lab launcher update to the reviewed candidate runtime, root TODO/combined documentation, and integration/checkpoint/push. No live process handle remains here. This lane made no Git mutation, server/browser/API/native execution, source acquisition, avatar edit, previous conditional-contact file edit, or root TODO/handoff edit. Root's native executions are recorded above. Preserve every failed run and all independent dirty work, including external avatar commit `66f8a9e` above `79c2b6a`.
