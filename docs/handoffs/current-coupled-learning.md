# Current coupled learning — no feasible policy

## Terminal result

The three public production stages completed once on **2026-11-01 (PST)** using the frozen **046261b** runtime. Root-owned handle `89403` is terminal, exit 0 and reaped; **no job is live**. The existing **v4 feedback policy on a shared-world-v2 scene** is executable, but this run does **not** demonstrate useful coupled learning: all four controllers score **0/3 task success** on matched changed starts.

`artifacts/fullspan-feedback-current-02` retains 1,046 files / 120,573,956 bytes; root seal `228a4c757046716874f22f6318b9a728c9890fd42c656390a8d1d028e723211c`. Caller time was 61.578710 seconds; supervisor time 61.496514 seconds. Actual work was **9,525 controls / 152,394 physics substeps / 40 resets / 34 model compilations / 2 data allocations**. Peak observed RSS was 284,377,088 bytes. External cost was **$0**; no source decoding, new inference, fitting, replay or rendering ran.

| Evaluation controller | Complete intervals | Task success | Mean score |
| --- | ---: | ---: | ---: |
| Coast | 3/3 | 0/3 | 0.369632 |
| Random | 0/3 | 0/3 | -0.942994 |
| Authored | 3/3 | 0/3 | 0.747771 |
| Learned | 3/3 | 0/3 | 0.733232 |

All starts and failures remain recorded. Random stops after 82, 4 and 41 controls with `human_velocity_envelope_exceeded`; actual substeps are 1,311, 61 and 654, preserving partial-control failure accounting. Coast completes but fails angular/root mean and peak error gates. Its equipment stays stable. Completion alone is not task success.

### Exact remaining gate failure

Authored and learned controls pass **every body and floor gate** on all three evaluation starts. Both fail only `equipment_stability:sam-2`; all four constituent equipment limits fail on every start. `sam-3` stays within every limit. Values below are ranges of each trial's accumulated maxima, not measured physical motion.

| `sam-2` metric | Fixed limit | Authored range | Learned range |
| --- | ---: | ---: | ---: |
| Displacement, model units | 0.03 | 0.4645–0.4999 | 0.3888–0.4563 |
| Orientation change, radians | 0.15 | 0.7938–1.1331 | 0.9244–1.4258 |
| Speed, model units/s | 0.15 | 1.1835–1.3570 | 1.2331–1.2802 |
| Angular speed, radians/s | 0.75 | 5.1677–5.7713 | 6.7562–7.6054 |

Learning lowers displacement, but worsens rotation and spin. Mean angular RMSE worsens **0.150644→0.159059 radians**; root RMSE **0.019607→0.020751 model units**; mean squared joint effort **0.148154→0.257590**. Peak body errors improve slightly while remaining below their gates. These mixed tradeoffs and lower reward are **not a control-quality improvement**.

### Why the lower-reward policy was selected

Training evaluated **28 episodes / 7,112 controls**, with two cache hits and **14 unique candidates, none feasible**. Generation 1 candidate 4 is the exact least-violating choice under the unchanged training-only rule: first minimize the largest positive normalized gate excess, then its mean, then maximize original reward.

The worst training term is `sam-2` displacement on seed `137031775`: authored **0.515633** versus selected **0.422929**, against **0.03**. Consequently, the largest normalized excess falls **16.187768→13.097629**. Mean normalized excess worsens **2.817281→2.985644**, and training reward falls **0.747648→0.734517**. Each candidate still violates eight of 26 training gate terms. The primary rank criterion wins; this is the declared selection rule, not a hidden reward improvement or a selector defect.

The separate selection stage reports `retained_training_infeasible_selection`, performs **zero new controls**, and does not use evaluation. The policy remains unqualified; no promotion or attachment was performed.

### Real learning, without success

The selected six coefficients are nonzero. Saved live-actor instrumentation confirms **all 762 learned evaluation actions** changed at least one joint command relative to authored control evaluated at that same state. Root and bias rules were unchanged; **zero learned root corrections** were observed. Across training and evaluation, 7,366 live samples changed joint commands.

All 762 paired saved trajectory joint actions also differ. Later root actions differ because the trajectories reach different states under the same root-feedback rule; that is not learned root control. Trajectory action differences must not be confused with the ±0.25 same-state residual bound.

Exact policy bytes remain unchanged through selection/evaluation: `e62722f8b575e3be26c9225f0c481dff8cf0d51ddfcba497f607c056043b2b62`.

## Evidence and saved-only diagnosis

- Training manifest: `06751632c3df28c713ad614aaac960a551b8980cd2174ffb45f5fd268e7ab57d`.
- Root terminal custody audit: `.local/fullspan-feedback-current-02/terminal-custody-audit.json` (SHA prefix `33ccc2d7…`). It joins all three stage seals, same policy, reaped zero-exit workers and exact runtime; no duplicate custody pass was needed.
- Independent interpretation: `.local/fullspan-feedback-current-02/saved-interpretation-01.json`, SHA `acbc6ff8c162fb69733657b9c244f83a13e0777f0f43a58945485197907bc76b`. This checks saved gate/rank arithmetic, seed/reset/state joins, policy/runtime/source identities and realized-action evidence; it is not another experiment or replay.
- Selected metadata: **15 unique files / 4,556,455 bytes**. Cached summaries omit raw diagnostic streams and initial/final state arrays; no original source, full 1,046-file readback, or 18,897,401-byte diagnostics scan occurred. Cache SHA `56a7f40170ca22cd368693f479f42893c42d4756215fca57684243490aed37e1`.
- The first saved checker expected `result.json`, but this supervisor uses `supervision.json`. Its missing-path failure is preserved in `saved-checker-failure-01.json`; only the root manifest was read before that failure. Conservative extra charge: 1 MiB, yielding **5,605,031 charged saved-artifact bytes**, below the 12 MiB allowance. Corrected arithmetic reused the cache; no study rerun occurred.

## Immutable inputs and runtime

- Runtime: exact committed `046261b6922f1b42318a6cfe48ff888d041d754b`, **no product overrides**; 291 files / 6,292,000 bytes under `.local/fullspan-feedback-current-01/runtime`. Manifest SHA `e3806cc1ee2fa9ec8ae38b2b81b544202323f6e6207672978d765708dc2ea4c6`. Do not repin this evidence to later UI or avatar commits.
- Original source: `artifacts/fullspan-exercise-source-01`, seal `f300b46cfb71697ed2ec4cdf4746c96ceda7bf0995a77412f0835d274b6d055f`; 192 declared members / 67,977,626 bytes. Production retained its before/after hash checks, and root compared named source stat identities before/after.
- Scene raw/canonical: `1d16db133a81bcc55b7093d961446a686f2867e4e565a4b44335c4b7a72eb209` / `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d`.
- Task raw/canonical: `ca9f9b6e22f00119fac65c508a6691eca154e523525ac0843e0742cccb30045f` / `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da`.
- Protected scorer: `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`; gates `063a93773406db92b583daf947855098727e34009bf645e82e0a5a2e044e0002`.

The task retains **257 original source slots, 121 body references, the full frame-2→256 interval, 254 controls at 30 Hz / 16 substeps**, two passive objects and no contact requirements. Observation remains size 190; actions remain 19 joints plus three explicit roots. Object-force channels are forbidden. Source clocks, angles, trajectory, geometry, masses, floor and gates are untouched. Equipment starts from a six-frame authored static extrapolation; neither grasp nor source contact is verified. Physical accuracy remains unverified.

Training seeds are `1263521311`, `137031775`, with the original human jitter 0/.05. Evaluation seeds are `1841308058`, `1870335751`, `203828300`, with jitter .10 and unchanged equipment reset state. All four controllers share each exact initial-state hash; these three states and seeds differ from training. This is independent changed-start **development evaluation**, not unfamiliar-video or physical holdout evidence.

## Exact public journey and bounds

Attempt 02 input plan: `adaf954c3d118a3ce4811eee42f4619675ef0923af26b65934c7229722a3ecf9`; execution plan: `7d24d179b66445b6e48bcb397ed1f988507a3525e9a45d5b0d7c1ada0316f084`; worker: `4d1d57718adb2cffdcdff84603e750ac920140d0a99d9ed9905140d8d938a563`. All are under `.local/fullspan-feedback-current-02`; `root-execution.json` pins root's selected supervisor and reviews. Release receipt: `abc265de806da194e56b61d15b61b81ca4ee222520489888484b8f1855024982`.

These commands already ran through that frozen-only launcher; **do not repeat them unchanged**:

```text
python -m weft shared-exercise-train --run artifacts/fullspan-exercise-source-01 --out artifacts/fullspan-feedback-current-02/training --policy-family feedback-residual --seed 2026110101 --generations 2 --population 8 --max-seconds 60
python -m weft shared-exercise-select --learning artifacts/fullspan-feedback-current-02/training --out artifacts/fullspan-feedback-current-02/selected --max-seconds 15
python -m weft shared-exercise-policy-evaluate --learning artifacts/fullspan-feedback-current-02/selected --out artifacts/fullspan-feedback-current-02/evaluation --seed 2026110102 --trials 3 --jitter-scale 0.10 --max-seconds 60
```

Reserved limits were 11,176 controls / 178,816 substeps; one CLI process at a time; hard walls 70/20/70 seconds and 180 overall; CPU 65/18/65 seconds per process; observed RSS acceptance 512 MiB. Each stage retained its 128 MiB / 512-file cap; outer acceptance was 400 MiB / 1,900 files, not a hard live disk quota. The reviewed stream addendum `47b639c74ca468fe286afd15fc3ef7cdfc315ff17afb3be9953d51787069e7b4` allowed 5 MiB combined streams per CLI. Diagnostic and metadata caps remained 24 MiB aggregate / 8 MiB per diagnostic file / 8 KiB per line / 4 MiB per metadata file. Selection was capped at zero task controls. The blocked, pinned MuJoCo startup `sysctl` attempt was recorded once per CLI; no permission was added and no child process executed.

## Preserved attempt 01 and preparation limits

Attempt 01 failed and was reaped after 1.144 seconds: `artifacts/fullspan-feedback-current-01`, 14 files / 178,008 bytes, seal `50af1a9f644a67c5b98fe39389be33dc5a84fa5d9be660d2b8d7418ad4bfbbd4`. It compiled two counted models but allocated no data and performed **zero forwards, resets, controls or substeps**. Its 57-byte error was `Package destination is missing, occupied or unsafe`; receipt `33e873a222c1ba3fade81ef92b206da316fee88e3c1c3a32285f546e136a8f41`, stderr `3eb5634ede3fc6b26d1e66cae18c0cb0ec77c67cc890a070446e9be59b847d86`.

The harness misinterpreted descriptor-relative package writes against its working directory. No exact leaf was recorded, so no filename is inferred. The corrected guard `1a37cde2e6f3508e2b136d3b7a6275239b14bbca0e519c4c38462dca4a454673` resolves actual Darwin descriptors with `F_GETPATH`/`fstat`, confines writes to the new output, rejects symlink/outside/descriptor-reuse paths, and preserves syscall arguments/errors. Eight tiny authored filesystem checks passed in .003819 seconds, receipt `9648be197ca960866ba79d8334b0219d960a94f0b032589b016cb2f87930f046`. No scientific runtime, task, seed or gate changed. Attempt 01 and all original helpers remain immutable.

Initial metadata preparation also deliberately stopped at the native quaternion/mass boundary; `metadata-preflight-failure.json` and corrected pure-scope `metadata-preflight-02.json` remain under attempt 01 preparation. Neither was a native admission claim. The actual production admission path ran in attempt 02. Historical full-span policies used different runtime pins and are not matched current baselines.

## Next distinct hypothesis — proposed only

**Do not rerun unchanged CEM with more seeds or budget.** The current actor ties shoulders and hips into one proximal group, and elbows/knees into one distal group (`shared_exercise_learning._groups`; `shared_exercise_feedback_residual.action_sample`). That forces arm and leg feedback corrections to change together. The saved tradeoff is consistent with a capacity limitation, but aggregate metrics do not identify the responsible limb/contact or prove that the task is feasible.

The smallest proposed algorithm change is an **explicit opt-in arm/leg split**: torso, arm-proximal, arm-distal, leg-proximal, leg-distal, each retaining the two existing normalized position/velocity features—**10 coefficients instead of six**. Reuse the current trainer, feasibility rule and evaluator; version/pin the new actor contract. Keep saved v4 artifacts unchanged and preserve default v4 numerical/action behavior; new implementation identities must remain truthful. Do not tie groups to `sam-2`, hardcode this clip, add equipment forces or new privileged features, retime references, alter reset distributions, change root/bias control, increase residual bounds, or modify scorer/reward/gates.

- [ ] Root coordinates the narrow implementation and analytical group-routing/zero-anchor/old-profile regressions; no implementation or new native run belongs to this completed slice.
- [ ] Only after review, freeze a distinct bounded study with training-only selection and new independent changed starts under the same task. Retain all candidates and failures. Require real action changes and report every original gate, not only the rank or reward.
- [ ] Useful coupled-learning evidence still requires passing body **and** equipment gates and a matched-baseline benefit without hidden tradeoffs; source/physical generalization remains a separate unverified goal.

**Next owner: root.** Checkpoint this terminal diagnosis, preserve both attempts and current policy, then decide the distinct opt-in algorithm test. No current result is promoted, and no live handle or further source/native work remains in this lane.
