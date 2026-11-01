# Reuse saved conditional equipment geometry

## Current result

Implemented an opt-in `shared-exercise-build --geometry-run` route. It reuses an existing sealed, source-qualified compound alternative and its exact saved task, then calls ordinary lab admission. It does not infer a new shape from a fresh video, refit masks, carve handles, change depth, rerun clearance or train a controller. Default build/config routes are unchanged.

**Qualified on one existing saved source:** the public build produced a ready lab, and a separate inspector-only recovery produced the [offline report](../../artifacts/exercise-geometry-resume-inspect-01/inspection/index.html). Root completed the report recovery on **November 1, 2026, at 2:06:29 p.m. PST**. Both earlier harness failures remain preserved below.

Focused guarded verification covers **68 distinct cases:63 new and5 affected legacy**, all latest outcomes passing. The ready build performed normal source proof/native admission and121 kinematic reference snapshots, with **zero physics steps or controls**. The recovered report performed no native calls. This proves saved-source integration, not a completed exercise, automatic fresh-video reconstruction or physical accuracy.

## Why this path

The existing silhouette producer already retains the box alternative and proposes conservative solid bands under an explicit exterior prior. The qualified saved study improves conditional image fit for `sam-3`; `sam-2` remains unchanged under occlusion. Available masks do not establish empty handle space, recovered depth or correct collision/contact geometry. Repeating a source screen or authoring another shape would not resolve those unknowns.

The missing integration was reuse: ordinary exercise assembly always rebuilt box proposals. This route carries the existing alternative through the same lab without replacing its body, floor, source clocks, original gates or task duration. In particular, the selected study retains its original two-second/60-control task, not a later full-reference task.

## Public contract

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --geometry-run artifacts/silhouette-shared-world-03 \
  --geometry-manifest-sha256 63184495388e6fd1e865ff83dbaaaceaf198ef1070fea03af83c79dd7775be70 \
  --out NEW_BUILD --max-seconds 60
.venv-lab/bin/python -m weft shared-exercise-inspect --run NEW_BUILD --out NEW_REPORT
```

These are interface examples; the exact executed acceptance commands and artifacts are below. The manifest pin is optional; a supplied pin must match exactly. `--geometry-run` is mutually exclusive with `--run`, `--human-run` and `--video`. Construction, body, mapping, root, task-interval and training overrides are rejected, including explicitly supplied default-valued flags such as `--body-scale 1`, `--pose-mode video` and `--training-seed 1729`. Output and wall-time bounds remain available; inspection uses its separate ordinary report command.

Accepted input is a completed/finished `conditional-silhouette-world-study-v1` root with the literal `geometry/` candidate, ready recorded admission, complete recorded clearance and unverified physical status. Selection is not an arbitrary report-provided path. Source-fit and clearance values remain **retained study claims**, not newly independently measured results.

## Files and custody

- `src/weft/exercise_geometry_inputs.py`: pure `read_geometry_resume_inputs` and `recheck_geometry_resume_inputs`; verifies selected metadata pins and configuration/body/source/camera/clock joins, existing geometry invariants, exact original-box provenance, and task-only scene-hash rebind. It does not call `validate_shared_world` or compile a model.
- `src/weft/exercise_scene_pipeline.py`: opt-in `_prepare_geometry_scene`, leaving the original route intact. Copies the selected snapshot to `geometry_source/`, then invokes `prepare_shared_exercise_lab(candidate_run, new_lab, max_seconds=...)` without policy, task or root overrides. Exact returned identities, actual scene/task bytes and copied source manifest must match before the lab is marked ready.
- `src/weft/cli.py`: explicit saved-input route and supplied-override rejection before work.
- `src/weft/exercise_build_report.py`: actual geometry-source/lab stages; links original box, selected compound, original/selected tasks, config and retained reports. Runtime admission remains `not_checked` by the inspector. A failed/not-ready pre-snapshot interruption remains reportable; missing snapshots cannot qualify ready builds.
- `tests/test_exercise_geometry_inputs.py` and `tests/test_exercise_geometry_resume.py`: authored metadata packets and mocked lab boundary only. They are not qualified source or native geometry evidence.
- Root separately owns the mixed human/equipment scope-label correction in `shared_exercise_lab.py`, SHA256 `e242f616ab7a43033b41866c2b43028023c825f92bc47e5ffffa8a2b4339cace`. No task or runtime behavior changes accompany that label.

The snapshot renames source seals to `study-manifest.json`, `candidate-manifest.json` and `baseline-manifest.json`; it is explicitly not a complete source package. The normal lab still performs complete source proof and native admission on the original selected package. Source signatures are rechecked around callbacks and after lab creation. A later callback/sealing failure retains an already verified ready inner lab; failure-only seals do not expose unsealed partial outputs.

## Bounds and saved source

Metadata admission reads at most20 files/16 MiB in5 cooperative seconds, at most8 MiB/member and1 MiB/manifest, including cached manifest reuse. Its receipt is at most32 KiB so a bounded parent failure receipt remains possible. Existing build bounds remain384 MiB/1600 files; the separately requested inspector retains its128 MiB/128-copied-file bound. No API, media decode, model download or simulation is performed by this metadata adapter/report.

Root independently confirmed the intended saved source identities:

| Artifact | SHA256 |
| --- | --- |
| Study manifest,263 members | `63184495388e6fd1e865ff83dbaaaceaf198ef1070fea03af83c79dd7775be70` |
| Selected candidate manifest,134 members | `a8c85573c4896392b603280ea77be0933df42dad24d6489b7f4b57256e2ab8ef` |
| Selected scene bytes | `4329cd5bcbb4ea45edcf7b79249dec16b6739b58d83aa67ca54bdf1ab681c37a` |
| Selected task bytes | `f4fb5e44c3e541b35415d6c1465302d205d9b7adb87dea06b7b7db98779ac32d` |
| Original study report | `474f33942913356ac028c8388cefc9dca2ba482557dcce4f5b9be7f620e37048` |
| Original study settings | `c2a934921865a599c9b3d0125f8a5646d83b27fc71ddcb32c96f3656b6004c92` |

No existing artifact is rewritten. The adapter/report adds no native call beyond normal lab preparation. Observed complete preparation, including its mandatory post-save reload, used **7 model creations,2 data objects,123 forwards,122 resets and zero steps/copies**. The initial five-model estimate omitted two human-mass compilations in the final scene/task validation; that harness-only error is preserved below.

## Verification and review

Receipts/XML are in `.local/exercise-geometry-resume-01/`. The unchanged guard is `.local/split-limb-feedback-actor-01/run_pure.py` (`c8d10113ef3344428cebf51157adf7f6cb3964917a1b7de96569761bb5c7d08d`):10 CPU seconds,4 MiB/file, no network/children and blocked native entry points. Each run records the known denied MuJoCo import-time subprocess probe; native attempts are empty.

- `check-01`:67 cases,63 passed and4 failed. The real defect was a missing new snapshot directory before the existing package writer. The source fix is one `mkdir` after the initial progress callback. Three boundary tests were also tightened to prove the mocked lab was reached.
- `check-02`:all7 affected assembly/report cases passed after that fix.
- `check-03`:the new cached-read byte-cap regression and accepted metadata path passed. A one-line length check preserves the tighter manifest cap when the bytes were already cached as a member.
- Latest outcomes across those receipts cover68 distinct cases, not the sum of rerun counts. Coverage includes wrong/resealed identities, body/floor/mass/camera/task/clock mutation, original-box retention, incomplete/unsafe inputs, byte/time bounds, source mutation, normal admission rejection, exact lab bytes, early/late progress failure, report links/claims and explicit default-valued CLI overrides.

Independent reviewer closed two report findings: missing pre-snapshot failure reporting and misleading empty-equipment wording. The final cached-limit delta is also reviewed and closed; its reverse delta reproduces the prior reviewed helper exactly. The reviewer parsed existing receipts without rerunning tests or native work.

## Public acceptance and preserved failures

All executions were performed by root. They reused runtime seal `c1588d7acd85d2c1f16d82fbeebabbcebe39ee1704b051efaee03eb3e4ef5bf2`: pinned Git `8b1e5e3662242e9fb24c7f0b620d8e24cfdab3ec` plus exactly the four released feature modules and root's scope-label file,296 files/6,433,480B. No independent avatar changes, new controller, task/scorer changes or source refit entered the runtime. Later attempts reused it without copying or modifying it.

| Execution | Terminal outcome | Elapsed / observed peak RSS | Preserved output |
| --- | --- | --- | --- |
| Native01 | Harness stopped at model cap5 during mandatory final lab reload; inspect never started | 1.842272s /294,387,712B | 494 files/80,920,945B |
| Native02 | **Build ready** in1.847018s; later inspector rejected an existing-ancestor mkdir before creating report files | 2.500625s /319,766,528B | 500 files/80,939,179B |
| Inspector recovery | **Completed**, one public inspect call; zero native/proof/build calls | 0.656014s /118,292,480B | 30 files/3,197,076B; report24 files/3,187,189B |

All three child processes were reaped. No API, media decode/generation, training, fitting, clearance rerun or physics control occurred; external and recurring cost were$0. There is no running worker/server in this lane.

**Native01 correction:** `prepare_shared_exercise_lab` calls `load_shared_exercise_lab` after writing its sealed files. The reload validates the scene, then validates the task through another scene validation: two additional human-mass model compilations, no data/reset/forward/reference calls. Full static tail review also confirmed that inspection adds no native calls. Native02 changed only fresh paths and the exact model cap/final expectation5→7; all other limits, source and runtime stayed exact. Proof: `.local/exercise-geometry-resume-native-02/native-count-proof.json`, SHA256 `8431bb2f667f93e0a7770ce12aa085d5bfc07af6105e9ef51375d88fa5103019`.

**Native02 inspector correction:** the inspector's safe directory opener tries `mkdir` on existing absolute ancestors and expects `FileExistsError`. The descriptor guard instead denied the first outside-output no-op before its syscall. Inspector-only recovery added a harness shim that returns `FileExistsError` only for already-existing real ancestors after the unchanged descriptor/traversal/no-symlink checks. It permits **no outside syscall or mutation**. Three focused guard cases passed; the frozen product, runtime and original file guard were unchanged. No rebuild or source-proof repeat was used for recovery.

Failure receipts remain at `artifacts/exercise-geometry-resume-native-01/failure.json` (`b2fca25daf766d2c85c34f2471b15dc68b7d95e40ea19527dffff787a553188b`) and `artifacts/exercise-geometry-resume-native-02/failure.json` (`8537e832bd0fc759c0053d0585a4f1d2f2bd28ea14955cdecd9659bc53f845cb`). Their matching `.local/` directories retain plans, scripts, process receipts and logs. The second failure describes the overall harness, **not a failed build or lab**.

### Ready artifacts and independent audits

| Artifact | SHA256 |
| --- | --- |
| `artifacts/exercise-geometry-resume-native-02/build/manifest.json` | `49196c57417d75cb1499173c685831c1ef4655f797e18048aec885aea582fa9f` |
| `lab/run/manifest.json` under that build | `94da20ea68f3fe8fdccf29fe4a8938ec0caa39f836aed824bd19ba52c5d6a32c` |
| `artifacts/exercise-geometry-resume-inspect-01/manifest.json` | `fb10f5a5d7261de8bb71b5811199e73eced1f36f5fb80cf8b02b6ceef09f59a6` |
| `inspection/manifest.json` under the recovery | `64ec6320f639b3cb9c99c4c9eda7daf3f2f1325046371824ae5efc471f4c5afb` |
| `inspection/report.json.txt` | `3139d81ffa6a947fc43e3299e7e079cf9faa8e0f9f68552f3349236903e426eb` |

Root independently audited the ready build once:487 sealed members/80,850,991B,465 inner-lab members,17 copied source metadata files,15 protected scene fields and object provenance, exact task,121 references, five runtime overlays and the correct human/equipment label. Receipt: `.local/exercise-geometry-resume-native-02/saved-build-audit-01.json`, SHA256 `05369f551321d86a171842854e4442e36ab342004780296ef612d13ef3f326d4`.

Root independently audited the recovered report once:29 sealed members/3,192,416B,23 inner-report members,21 copied artifacts matching the ready build,30 local HTML links and no required external links. Stages, retained-claim wording and zero native/proof/rebuild/API work were checked. Receipt: `.local/exercise-geometry-resume-inspect-01/saved-audit-01.json`, SHA256 `04633a7ffdd8da81fb40dc2540f5ab588955a446b152a43f3844f7462f06506b`. This lane did not duplicate either full audit.

### Executed commands and limits

Historical commands are preserved for provenance; **do not rerun them into occupied outputs**:

```sh
.venv-lab/bin/python -B .local/exercise-geometry-resume-native-01/supervisor-02.py 4852e87e9faaae4af49c0b07421a46cc9b6074c428092ba1d02a5202a4a3c185
.venv-lab/bin/python -B .local/exercise-geometry-resume-native-02/supervisor.py efd93a61b83d7d1dd9f0468d107a8304bf40b42fd7797d411cccd46434ebeab4
.venv-lab/bin/python -B .local/exercise-geometry-resume-inspect-01/supervisor.py 2d1f06e9d4f80dfa99f8e8d2a6a8a1c8c40ed9773f1232533e0ad2df62a356e3
```

Native attempts each had60 CPU/cooperative seconds,90 external seconds,768 MiB observed RSS,384 MiB/1600 build files +128 MiB/132 report files +4 MiB/16 worker files. Recovery used30 CPU/cooperative seconds,45 external seconds,512 MiB RSS and only the132 MiB/148 report-plus-worker allowance. Each separately reserved two1 MiB logs and a64 KiB observer receipt. The supervisor's narrow escalation allowed `/bin/ps` sampling of its owned child; it did not grant the child network, credential or outside-output write access. Observed RSS is not hard memory isolation.

The ready build completed2 ordinary source proofs/15,093,810 reported bytes. Its task, body, floor, coordinates, clocks and gates remain exact. The report retains `runtime_admission=not_checked` and does not rerun or claim the build's native admission. No60-control task episode was performed; no grasp/contact, recovered cavity/depth, learned policy or physical validity was established.

## Ownership and next safe action

Implementation lane: only the four product files above, two new tests, this handoff and lane scratch. Root owns the label file, TODO/combined, native acceptance and publication. No Git mutation or native/media/API work occurred in this lane; no background process is owned.

Production/test pins and pure receipts remain frozen in `.local/exercise-geometry-resume-01/release-01.json` (`8c3df4fa56b151a79b65133c84f296efe5dcb14716e5bcb3aea85164bc3c94d7`); its earlier handoff pin describes the pre-acceptance revision, not this terminal update. Source and tests have not changed. Root owns guide/TODO/combined updates and publication. The safe next action is to use the ready lab/report or begin separately scoped follow-up work—not rerun this acceptance, source fitting or audits. The remaining reconstruction gap is new evidence for ambiguous shape/depth/contact, not more confidence in this saved conditional alternative.
