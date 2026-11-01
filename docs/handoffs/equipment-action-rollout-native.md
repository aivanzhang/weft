# Full nominal equipment projection rollout

Updated **2026-11-01 (PST)**. Preparation/diagnosis owner: `/root/contact_reconstruction`; execution owner: `/root`. **Rollout02 completed as an experiment, accepted 49 controls, then abstained at control 50. The actual task remains incomplete; no full-interval controller success is established. Saved-only diagnosis rejects a simple secant-model refinement as the next implementation.** Rollout01 remains stopped at control 3 on its overly conservative provisional-mean rule; neither result is relabeled. No native worker was executed by this lane. Do not edit completed one-decision proofs (`b615e79`), frozen execution inputs, or occupied outputs. No live handle remains in this lane; root reported both runs terminal/reaped.

## Rollout01 scope (historical)

Scratch: `.local/equipment-action-rollout-native-01/`; intended new output: `artifacts/equipment-action-rollout-native-01`. Reuse the exact Native02 293-file runtime read-only; do not copy it or import the current checkout. Projector `04510c0a46ca13ce397e0d73ff6a5edcebd325d3df3ccd6092e2643b1d90acab`, predictor `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd`, runtime manifest `ab0b266943aef665403ef5e2b13939529363a3d7abd25145366e32e6e0820ad7`. All original policy/source/task/scorer pins and the five projector settings remain unchanged.

Start at the same nominal seed 29, without jitter. At each of at most **254 controls**, sample the original v5 callable on the **current actual state**, then create one explicit projector. Original-recording action/root comparisons are not valid after a correction. Preserve same-state base root bytes, dtype, action/trust bounds, and exact predicted-vs-actual full endpoint/contact identity. The original task metrics, not the helper's endpoint checks, determine the actual outcome.

**Rollout01-only stop rule:** stop without fallback on `unavailable`/`no_verified_candidate`, endpoint mismatch, any actual `failed_gates` after an accepted step, early engine stop, or resource/output admission refusal. That treatment of prefix mean gates was overly conservative and is corrected only in the separate Rollout02 harness below. `completed_interval=false` alone never stops a trial. Success always requires all 254 controls **and** the unchanged evaluator's `completed_interval` and `success`; a completed experiment can still be an incomplete/failed task.

## Fixed aggregate budget (unchanged for Rollout02)

- At most **6,096 predictions/copies**, **97,536 scratch + 4,064 live = 101,600 substeps**, **6,352 forwards**, **255 data objects over the trial**, **11 models**, and **one reset**.
- One helper lifetime per control; release it and its scratch data before the next decision. Global counters never reset; per-decision limits also remain 24 predictions/copies, one scratch object, 384 hypothetical substeps plus one 16-substep actual step.
- **40-second CPU / 50-second cooperative / 60-second external / 512 MiB observed RSS** bounds; no paid calls. Before admitting another projector, reserve its maximum native/call work and **seven seconds CPU/wall** (five-second decision plus two-second cleanup/sealing reserve). Guard work at 38 CPU/48 wall seconds and finalization at 40 CPU/50 wall seconds; the supervisor retains its five-second kill/reap reserve.
- At most **16 MiB / 12 output files**, plus two supervisor logs capped at 1 MiB each. Before the next decision, reserve **2 MiB diagnostic + 256 KiB proof/metrics + 512 KiB result/manifest space**. Enforce actual encoded totals. Flush completed rows to `decisions.jsonl`; do not discard rejected candidates or past decisions to fit a budget.

Each streamed row retains base/selected actions, root/trust/dtype evidence, complete diagnostic/evaluations/rejected values, prediction trace, actual before/after task metrics, endpoint/contact comparison, and attempted/completed native deltas. After actual comparison, replace the full `verified_endpoint` with its hash. An exceptional pre-comparison failure retains the bounded endpoint and available current row/action; unavailable diagnostic data remains null. If bounded failure-record writing itself fails, the result explicitly marks `incomplete_decision_record`. Flushed prior rows remain; failed finalization must not imply a success seal.

The independent reviewer measured one Native02 compact row at **26,287 bytes** (6,676,898 bytes if repeated 254 times) and 24-prediction active work at **0.0858286 seconds** (~21.8005 seconds ×254). These are single-case illustrations, not maximum sizes or measured full-trial latency. The draft additionally preserves the predictor trace; actual encoded-budget checks are authoritative. Estimated external and recurring cost: **$0**.

## Rollout01 execution identity (completed)

- Preserved `plan-draft-01.json`: `9737d6fb320d1e60710a212722acaebda57d4a38dcb5f2fb3d35e7b56ddd0fc3`.
- `plan-frozen-01.json`: `29ec2ae3b2f1ff6f039bdf2a7ab75bfe21574f9579ec393952a474d3330b91ae`.
- `worker.py`: `98e7d0bae53efedea5a9ec4ae7eea05b89764346b7cd8e71985eb49b12e27923`.
- `supervisor.py`: `a9090a11f9265f7f4956d5895be5670d801489fb7c78a2afad606835a4790455`.

`freeze-review-01.json` records both reviews, the exact launch and unchanged reused runtime. Syntax was parsed without executing either script. No new production module, infrastructure, broad test suite, runtime copy, saved payload scan, Git mutation, avatar edit, native run, source acquisition, media operation or API call occurred in this lane. Root owns one native execution with this exact command:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-rollout-native-01/supervisor.py \
  29ec2ae3b2f1ff6f039bdf2a7ab75bfe21574f9579ec393952a474d3330b91ae
```

Preserve all older native outputs and the completed [one-decision handoff](equipment-action-controller-native.md). The Rollout01 command above has now completed; do not repeat it or infer task success from process completion.

## Rollout01 observed result and stop-rule limitation

Root completed/reaped Rollout01 in **0.870669 seconds**, peak RSS **135,233,536 bytes**, four files / 58,022 bytes. It stopped at control 3 on `angular_rmse_rad = 0.3551332751277711`, above the unchanged `.35` final mean gate. All three actions were unchanged; every predicted endpoint/contact report matched actual execution. No correction or contact failure occurred. Actual counts: three predictions/controls/copies, 96 total substeps, eight forwards, four data objects, 11 models and one reset.

The original task divides squared tracking error by **valid elapsed duration**, so prefix mean error can recover. Root independently compared all three actions and complete after-step task metrics to original recording events 14–16: exact equality. The original control 3 already has this mean; at original control 48 it is **0.24993595389976453**. This was a conservative harness interpretation error, not a new controller regression. This lane did not replay or reread that prefix to rediscover root's comparison. The existing stopped result is not reclassified as a completed interval or successful trial.

- `result.json`: `f211776f4eed2566e8d6ace952b1d4f246ebe1c1abda018b2eb5ae47f1988358`.
- `decisions.jsonl`: `6de7d7a2e61400d387c3b6e736eb84d404bd82bf7282ca43f4c820255e95e19c`.
- `manifest.json`: `550fbef0d3dd457eb09638676afc10cecfe27594cf802f59d87bd3bf0df6913c`.

## Rollout02 corrected-stop trial — completed

New scratch/output: `.local/equipment-action-rollout-native-02/` and `artifacts/equipment-action-rollout-native-02`. Runtime, helper, source, scorer, task, policy, projector configuration and every resource limit remain identical to Rollout01. Only the worker's stop interpretation and unique paths change:

- Keep raw actual metrics and `failed_gates` unchanged. Record finite prefix crossings of `angular_rmse_rad` / `root_rmse_model_units` as `provisional_mean_gate_crossings`, with the original limit beside each value.
- Stop immediately for metric failure, unavailable/nonfinite failed means, reported non-mean protected gates, parity failure, abstention, engine faults, or the existing resource stops. At completed interval, failed mean gates also stop. All 254 controls plus the original `metrics.success` / `completed_interval` remain necessary for success.
- The fixed task has **no contact requirements**; the worker asserts that existing fact. This narrow change does not introduce generic partial-contact-window semantics, relax thresholds, change a scorer, or infer a best-case future trajectory.

Twelve small pure cases pass in **0.009539 seconds** using the saved control-3 final metrics, unchanged gate literals, and clearly authored classification variants. They cover provisional angular/root means, unchanged final rejection, null/metric failures, peak/floor/equipment/contact failure reports and root's later recovered mean. Inputs are not mutated. Only the AST-extracted pure function executes; no Weft import, native simulation or prefix replay occurs. Receipt: `stop-rule-checks-01.json`, SHA `1bc979ec36fea82740e774cfd97278906e27ceddcf9b0a245a6e530c7df03fb5`.

Execution pins (original draft preserved):

- `plan-draft-01.json`: `edcf270ca96712ccd8d5a4da297098ffd46c6f8c758aacaf65306405f129167c`.
- `plan-frozen-01.json`: `afb8c8d7e2ae0923615504631635555723bad195449b9b55d0d3387b68ca9000`.
- Worker: `865628952a286063cfa3555e122c2a4015ff472435663ff4e16713d91fb37eeb`.
- Supervisor: `2d0e92c3f3882c35da416bdbfc415b1fa5f090d712b06efe20d63b8005725724`.

Root reviewed the corrected semantics and supervisor; at freeze, the plan corrected stale parent-provenance wording to distinguish the completed Native02 one-step proof from the then-unexecuted full rollout. `freeze-review-01.json` records that review. Root executed the following changed-input trial once; **do not rerun this command**. No budget expansion or broader implementation is included.

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-rollout-native-02/supervisor.py \
  afb8c8d7e2ae0923615504631635555723bad195449b9b55d0d3387b68ca9000
```

### Rollout02 observed result

Root completed/reaped the process with exit 0 in **1.561130 seconds**, peak RSS **135,135,232 bytes**, four files / **718,926 bytes**. Fifty projector decisions made **95 predictions** and accepted **49 actual controls**: 48 unchanged actions, the previously verified correction at 49, then `no_verified_candidate` at 50. The control-50 reason is `nonlinear_candidates_fail_measured_constraints`; no fallback or actual step occurred. Current actual `failed_gates=[]`, but `completed_interval=false`. Process completion is not task completion.

Actual completed native counts: **11 models, 51 data objects, 146 forwards, one reset, 2,304 substeps, 95 copies**. There were no forbidden calls, media decodes or replays. These counts include 95 × 16 detached and 49 × 16 actual substeps; no additional simulation was needed for the diagnosis below.

Root verified every output member and the global saved-only audit. That audit verifies all 49 predicted/actual endpoint and contact matches, unchanged root bytes, action dtype/trust bounds, contiguous task metrics, and no live/task advance at abstention. Provisional mean crossings occur only at controls 3–4. This lane did not duplicate that global audit.

- `result.json`: `8f27289511838a1402a15a26592d696052a8343666935ef5f70d9674c974619e`.
- `decisions.jsonl`: `ae66fd6282fb7c1107beecd1b152294660ce70135687061e6cbf4517d0168a69` (694,950 bytes; 50 rows).
- `manifest.json`: `040698301ae85427b79e059cc34c31e9332a51b92ccdcbc5b95aa05817ea04aa`.
- Root audit `.local/equipment-action-rollout-native-02/saved-audit-01.json`: `b80067fa61f80c1765629beee584f14b0a2e094c1d7077defbaf8283996a05a2`.

### Saved control-49 comparison and control-50 diagnosis

Control 49 exactly matches Native02 across all seven compared fields: base action, selected action, actual/verified endpoint hashes, complete actual after-step task metrics, maximum endpoint errors and contact equality. Endpoint SHA: `ca2d641777574ec914d8d430db7d1bd3db11c7beb168c18bdb124b437c0a99a5`. Comparison source: Native02 `result.json`, SHA `9dac5e6d9321982fa62e5e13ddda6314832da45c9b719686306ef3a8c798af1e`. This was saved-data comparison, not replay.

At control 50, the linear solver **converged in two sweeps**, with maximum normalized linear violation `4.440892098500626e-16`. It is not the earlier iteration-limit failure. All **23 predictions** returned normally: base + 19 one-sided differences + three candidate scales. The final repeat was unused because no action passed. Active projection time was **0.073554083 seconds**; this was not a time or prediction-budget stop.

Only `sam-2` equipment speed and angular speed fail; all other ten measured constraints pass for every evaluated action. The full candidate's largest joint correction is **0.14527519**, within the unchanged `.25` bound. Its linear prediction is feasible, but its actual detached response is not:

| Saved probe | Speed (limit .15 model units/s) | Angular speed (limit .75 rad/s) |
| --- | ---: | ---: |
| Base action, actual prediction | .47983709 | 2.86758861 |
| Full candidate, linear estimate | .14999985 | .65735226 |
| Full candidate, actual prediction | .25955757 | 1.87978962 |
| Half candidate, actual prediction | .30479411 | 1.59253549 |
| Quarter candidate, actual prediction | .40011877 | 2.36676090 |

The best tested individual difference is joint 0 / −.1, but it also fails both gates (.31237108 / 1.62537726). No tested action passes; this is **not** proof that no feasible action exists within `.25`. Poor local response prediction is measured. A changed contact regime is only a hypothesis: rejected probes retained complete contact-cache hashes, not the signed velocities or contact-pair data needed to establish that cause.

The bounded diagnosis is `.local/equipment-action-rollout-native-02/diagnosis-01.json`, SHA `65c112041b8bc6fa98cae225e19b70354525a01ab186b93b2964192331006c83`. Its two-row derived cache is `diagnosis-input-01.json`, SHA `b8cecc8b66bb49af9775c8e4611e93af15af542918db10ae4c3a201cafd0d9c0` (97,547 bytes), pinned to the original decision stream. Reuse this cache for further saved-only analysis instead of rescanning the stream.

### One saved-data secant hypothesis — not supported

At root's request, `check_saved_secant.py` evaluated **one fixed rank-one update** to the existing response matrix, using the measured full candidate to fit it. It then compared predictions against the already saved half/quarter responses. No solver, optimizer, parameter sweep, Weft import, native call, new action or source change occurred. The full candidate fits exactly by construction and is not validation; the smaller-scale checks are development diagnostics, not an independent holdout.

| Saved check | Original maximum limit-normalized error | Secant maximum limit-normalized error |
| --- | ---: | ---: |
| Half correction | .22657992 | 1.04153816 |
| Quarter correction | .06897518 | .33850394 |

Both checks worsen. The simple full-response secant does not explain these observations, so it does **not** justify replacing the fixed scale probes. Receipt: `.local/equipment-action-rollout-native-02/secant-check-01.json`, SHA `ba0a6e4a82762b6494f8695382445b9c1cb88f82a61c321891123c168e0b1605`. The script is a one-shot writer and refuses to overwrite this receipt.

**Next proposed hypothesis, not implemented or executed:** model signed equipment velocity components before applying the unchanged speed norms, instead of fitting only nonnegative scalar speeds. The rejected-probe record does not contain those vectors, so current saved data cannot validate that proposal. Any follow-up must retain the already computed signed endpoint responses, preserve the default strategy, root bytes, actual gates, `.25` trust radius, 24 predictions / five-second helper budget and aggregate solver/work limits, and still verify a selected action with an actual nonlinear prediction and repeat. This is not evidence that a signed model will find a feasible action. Root owns the decision and any new changed-input execution; no further sweep or replay is queued.

Git/checkpoint ownership remains root's. This diagnosis changed only this handoff and new scratch analysis files; it did not alter runtime/source/helper/policy/task/scorer, either sealed rollout, the one-decision handoff, or unrelated avatar work. External spend: **$0**.
