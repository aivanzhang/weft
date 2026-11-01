# Current coupled learning — attempt 01 failed before controls

## Terminal attempt 01; correction 02 pending

Root's one reviewed invocation through `supervisor.py` (SHA prefix `b0911c43…`) is **terminal failed and reaped**, after 1.144 seconds. Exact launcher selection remains in `.local/fullspan-feedback-current-01/root-execution.json`; execution-plan pin `ee228f1dc63be736be2aa7da6deafcfa0119b5b979d83ec3d8c443aee5585ceb`. The failed output `artifacts/fullspan-feedback-current-01` retains 14 files / 178,008 bytes, seal `50af1a9f644a67c5b98fe39389be33dc5a84fa5d9be660d2b8d7418ad4bfbbd4`.

- **No live job:** former root-owned handle `73650` is closed. Two counted model compilations; **zero data allocations, forwards, resets, controls or physics substeps**. Selection/evaluation never started; there is no learned-control result.
- The terminal worker receipt SHA is `33e873a222c1ba3fade81ef92b206da316fee88e3c1c3a32285f546e136a8f41`; its 57-byte stderr (`3eb5634ede3fc6b26d1e66cae18c0cb0ec77c67cc890a070446e9be59b847d86`) says `Package destination is missing, occupied or unsafe`.
- Recorded denials are the known, still-blocked MuJoCo startup probe and a filesystem mutation outside the output guard. Static package code creates nested directories with `os.mkdir(part, dir_fd=fd)` and files with descriptor-relative `os.open`; the harness incorrectly interpreted those relative names against its working directory. The first guard did not save the exact relative filename. This is a packaging-guard failure, not a scored task/body/floor failure.
- **Next preparation:** a new `.local/fullspan-feedback-current-02` helper/plan will resolve real directory descriptors and bound exact targets to a new output; `open` auditing also needs the wrapper's descriptor context, and writable `fdopen` must not accept arbitrary descriptors. No blanket relative-path exception, changed source/runtime/seeds/gates, or permission to execute a native rerun follows from this diagnosis.
- Attempt 01, all frozen helpers and failures remain immutable. Root owns any reviewed 02 supervisor/execution. The historical preparation below is retained, not a success claim.

## Next bounded check

Use the existing **v4 feedback-residual policy on the original shared-world-v2 scene**. This is not a new scene, policy family, scorer, or training pipeline. No training, simulation, model, source decoding, or paid call has run in this preparation.

The previous full-span coupled study completed but learned/authored each scored **0/3 success**, failing `sam-2` stability. Its human/world/scorer implementation pins differ from the current runtime. Preserve that result as historical evidence; do not repin its policy or call it a matched current baseline. Current person-only and camera learners each match authored **3/3**, with modest reward changes and mixed error tradeoffs—not a task-success advantage or useful whole-body learning demonstration.

The cheapest unresolved check is the existing six-parameter learner on the full coupled task. It learns three joint groups × position/velocity feedback coefficients, bounded ±.25, around unchanged authored feedback. Root assistance remains explicit and unchanged. No equipment-state features or equipment forces are added.

## Frozen inputs and runtime

- Plan: `.local/fullspan-feedback-current-01/plan.json`, SHA `358832885a36ed436edd70af81781ec764c0a7023037801fbfeb69935377cb28`.
- Runtime: exact committed `046261b6922f1b42318a6cfe48ff888d041d754b`, **no overrides**; 291 files / 6,292,000 bytes. Runtime manifest SHA `e3806cc1ee2fa9ec8ae38b2b81b544202323f6e6207672978d765708dc2ea4c6`. Archive and extracted files are under the preparation directory.
- Original source: `artifacts/fullspan-exercise-source-01`, seal `f300b46cfb71697ed2ec4cdf4746c96ceda7bf0995a77412f0835d274b6d055f`; 192 declared members / 67,977,626 bytes. Only its manifest/scene/task were read; other named members were statted. Full payload/proof rechecks remain with production.
- Exact scene bytes: `1d16db133a81bcc55b7093d961446a686f2867e4e565a4b44335c4b7a72eb209`; task bytes: `ca9f9b6e22f00119fac65c508a6691eca154e523525ac0843e0742cccb30045f`. Canonical scene/task pins are in the plan.
- Protected scorer: `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`; gates `063a93773406db92b583daf947855098727e34009bf645e82e0a5a2e044e0002`.

The task retains 257 source slots, 121 original body references, the entire supported frame-2→256 reference interval, **254 controls at 30 Hz / 16 substeps**, two passive objects, and no contact requirements. Original floor, geometry, masses, source clocks, root trajectory, body angles and gates remain untouched. The six-frame equipment initialization gap remains an authored static extrapolation, not observed contact.

### Metadata checks and deferred admission

The unchanged pure human/task/interval/reset validators accept the exact metadata without human/task canonical changes. No metadata migration is indicated. Observation remains `shared-exercise-observation-v1`, size 190; actions remain 19 joints + three explicit roots. The task excludes the engine's six object-force channels.

Complete world admission is **not** a pure metadata operation: rigid corners call `mju_quat2Mat`, then action limits compile the human mass model. The first preflight's inert MuJoCo guard blocked that quaternion lookup; `metadata-preflight-failure.json` preserves it. No native call executed and no replacement quaternion/mass math was used. The changed-scope receipt `metadata-preflight-02.json` checks only the exact human/task/interval metadata; it does not claim native floor/collision/reset or source-proof admission.

Production order remains `_verified_inputs` → `validate_shared_world` → exact task validation → actual source-equivalence collection → unchanged `SharedExerciseTaskEnv`/`SharedWorldEnv` → resets/controls. Stop and retain any failure; do not migrate or weaken the scene to make the run pass.

## Frozen study and commands

One sequential public journey, not yet launched:

```text
python -m weft shared-exercise-train --run artifacts/fullspan-exercise-source-01 --out artifacts/fullspan-feedback-current-01/training --policy-family feedback-residual --seed 2026110101 --generations 2 --population 8 --max-seconds 60
python -m weft shared-exercise-select --learning artifacts/fullspan-feedback-current-01/training --out artifacts/fullspan-feedback-current-01/selected --max-seconds 15
python -m weft shared-exercise-policy-evaluate --learning artifacts/fullspan-feedback-current-01/selected --out artifacts/fullspan-feedback-current-01/evaluation --seed 2026110102 --trials 3 --jitter-scale 0.10 --max-seconds 60
```

Commands must run through a separately reviewed frozen-only launcher, never the mutable checkout. Omit `--task`/`--root-assistance`: the exact sealed task already supplies both authority and assistance.

- Train resets: seeds `1263521311`, `137031775`, existing human jitter 0/.05. Two generations × eight candidates × two resets reserve 32 trials / 8,128 controls. Existing zero/prior-best caching permits at most 28 new episodes; failed starts stay recorded.
- Evaluation: **four controllers**—`coast`, `random`, `authored`, `learned`—on three matched human-jitter-.10 starts: `1841308058`, `1870335751`, `203828300`. Equipment reset state stays unchanged. The 12 episodes reserve 3,048 controls. Seeds differ from training and known prior saved-study seeds.
- Selection preserves the original training-only feasible/least-violating choice; it performs no new controls. Evaluation is frozen afterward and never feeds back into search.
- Maximum reserved work: **11,176 controls / 178,816 substeps**; one process at a time, at most three CLI calls, no replay/render/export or extra baseline run.
- Cooperative limits 60/15/60 seconds; hard process walls 70/20/70 plus at most three seconds cleanup each; 180 seconds overall. Hard CPU limits 65/18/65 seconds per process; no child launches. Peak RSS acceptance 512 MiB is observed, not hard memory isolation.
- Existing output caps stay 128 MiB / 512 files per stage. Aggregate acceptance is 400 MiB / 1,900 files including frozen runtime/receipts, not a hard live disk quota. Streams ≤1 MiB per CLI; diagnostic 24 MiB aggregate, 8 MiB/file, 8 KiB/line; metadata 4 MiB/file.
- External cost **$0**. Existing sealed media may be hashed/copied for custody; no acquisition, decoding, inference, fitting, browser/server, installation, or API access.

## Interpretation and next action

Retain all failed resets/episodes and original denominators. Compare current matched success counts and body/equipment errors, not just reward. Verify nonzero selected coefficients and realized action changes from saved diagnostics. An unchanged authored selection, equal success counts, or a higher reward with worse errors does not establish the remaining whole-body learning goal. Changed starts are development evaluation, not new-video or physical holdout evidence.

**Next owner: root.** Review the exact plan/runtime and original native-admission path, then prepare/review the bounded launcher before authorizing one run. No live handle exists. The old failure, initial metadata-check failure, source inputs and preparation scripts remain immutable. No product files, scorer, Git state, or historical artifact was changed by this lane.

## Execution-helper preparation (still unexecuted)

Root requested a guarded worker after the input-plan review. `worker.py` is frozen at `fbc9c2924e9bff826fc0c962ece6ce670a234563125f5fe3a973a9164a0f117b`; `execution-plan.json` at `ee228f1dc63be736be2aa7da6deafcfa0119b5b979d83ec3d8c443aee5585ceb`, both under the preparation directory. Root separately owns `supervisor.py` and its review/receipt. The unused `supervise.py` draft is not an approved launcher; its optional pin in this execution-plan draft does not bind root's separate supervisor.

The worker invokes the original public CLI once per declared stage. Frozen-only imports, exact installed versions, source-seal guards, unchanged actor/scorer bytes, zero object-force channels, and counted task/native calls are enforced. The worker compares the original manifest before/after; unchanged production code hashes each stage's complete inputs before/after its work. The preparation's named source-stat inventory is separate: root's supervisor must explicitly implement that comparison before claiming it. Per-stage caps additionally allow at most 256 model compilations / 128 data allocations; selection permits zero task resets or physics steps. These are guard ceilings, not predictions. `MUJOCO_GL=disable` avoids rendering. All process/network attempts are denied. Only the exact diagnosed, pinned MuJoCo initializer's blocked Darwin `sysctl` attempt may be recorded separately; it never executes.

Root approved a necessary stream addendum, SHA `47b639c74ca468fe286afd15fc3ef7cdfc315ff17afb3be9953d51787069e7b4`: **5 MiB combined stdout/stderr per CLI**, replacing the original plan's 1 MiB. Existing saved evaluation metadata is 3,480,011–4,127,305 bytes and the CLI prints the full document. Valid logs must remain complete. Producer metadata/output caps are unchanged; the entire new output, including logs and receipts, must remain ≤400 MiB. The runtime is an external frozen input, not a repeated output copy.

Only syntax parsing and source/code inspection verified these local helpers. No process, solver, native model, training stage, or new admission ran. Root must close its exact supervisor/worker review before execution; preserve any later failure instead of rerunning unchanged.
