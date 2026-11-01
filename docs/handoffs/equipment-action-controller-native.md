# Single-decision equipment projection pilot

Updated **2026-11-01 11:10 PST**. Preparation owner: `/root/contact_reconstruction`; execution owner: `/root`. **Native02 selects one corrected action whose full predicted endpoint matches actual execution exactly; all protected gates still pass at that boundary. The full interval remains incomplete.** Native01's earlier abstention is preserved. Both root processes are terminal and reaped. No native execution was started by this preparation lane. This completed handoff is frozen for checkpoint; future rollout preparation belongs in a separate handoff. Do not edit frozen inputs, rerun occupied plans, expand bounds or promote a controller from this one-decision check.

## Frozen scope

- Use the original recorded scene/task/v5 policy and nominal seed 29. Execute exactly 48 original base controls and compare every action/full task snapshot to the saved recording. The original base control 49 must also match before projection.
- Call `prepare_action_projector(env, config=ProjectionConfig(...))`, then `project(base_action)` exactly once. Config: 24 predictions, 5 seconds active, difference `.1`, trust radius `.25`, 64 linear-solver sweeps. The helper limits its result to 2 MiB.
- For `selected`/`unchanged`, require original dtype, exact root bytes, normalized action bounds, and actual `abs(selected[:19] - base[:19]) <= .25` without a tolerance; take exactly one live step and compare every predicted endpoint field, time and complete contact report exactly to actual state. Save unmodified actual protected task metrics and root/trust evidence.
- For `unavailable`/`no_verified_candidate`, preserve the None action and reason; do not substitute an action or take another live step. Experiment completion is explicitly **not controller success**.
- No full rollout, training, media, recording/export replay helper, retries, policy promotion or physical-accuracy claim. The required 48-control prefix only prepares the already exposed development state.

Native caps are **11 models, two data objects (live + one reusable scratch), 75 forwards, one reset, 24 copies and 1,168 substeps**: 768 prefix + at most 384 hypothetical + optional 16 selected. Guards enforce each phase separately and record actual attempted/completed counts. Every prediction and the complete projection call must leave live state/task metrics unchanged. The wrapped live environment `step` rejects calls during projection.

Keep the reviewed **60-second external / 512 MiB observed RSS** limits, 40-second worker CPU/cooperative bound, one child, sanitized environment, descriptor-aware output boundary and no network/subprocess inside the worker. Outputs total at most 4 MiB/eight files; each supervisor log is capped at 1 MiB. The separate helper agent's unexecuted 20-second proposal does not replace this plan.

## Exact inputs and execution

Preparation: `.local/equipment-action-controller-native-01/`. Output: `artifacts/equipment-action-controller-native-01`. The new runtime contains **293 files / 6,346,464 bytes**: all 292 qualified predictor-runtime files unchanged, plus only the reviewed new projector. All 25 original policy-runtime hashes remain exact; no task/scorer/source change occurs.

| Item | SHA256 |
|---|---|
| `plan-frozen-01.json` | `2df8d16338ce531aa88ac3b46a2ad63122f1705335296e2fc9b2044b04edcf48` |
| `runtime/manifest.json` | `58f5cd8b2a4ec97ef8616ee7f7a8ae81c857144debb68a4cccbd958d38861095` |
| `worker.py` | `e211e19be80a75788bef0a13789627c56c7764494132cf84e5b9e5b60ccd243f` |
| `supervisor.py` | `5f410301abb3aa10e43e76121322daa780edf387b10fe4da2343539d79381955` |
| Projector | `ad8d2a537fe12563ed938e1d6a67ae5d7671367530bd88a58d0c92d633169e50` |
| Qualified predictor | `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd` |
| Original policy | `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015` |
| Original task/scorer file | `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb` |
| Saved recording manifest | `94ee100c5093ad51880578972aa123cf01b11f75a791362379ea23a053db91cd` |

Root's exact Native01 launch (completed; do not repeat):

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-controller-native-01/supervisor.py \
  2df8d16338ce531aa88ac3b46a2ad63122f1705335296e2fc9b2044b04edcf48
```

Root approved the final worker delta: an exact selected joint trust-bound assertion, root-byte/dtype/trust evidence, and explicit no-controller-success flags. The helper agent reports **42 mocked tests passed**, source/test pins frozen; this lane did not rerun them. Preparation parsed source syntax without importing Weft, verified/copied the 6,325,520-byte parent runtime once, and read no saved recording/lab payload. The first metadata preflight assumed an absent `file_count`; it failed before any copy/runtime creation. `preparation-failure-01.json` preserves that failure. The corrected freeze derives the count from exact pinned members and policy closure from the exact runtime04 parent, checking all inherited members. `preparation-01.json`, SHA256 `13352787276a4e4a6cc69b66bab9cb837c4efe9fb350e7a532efb55699ded57f`, records every preparation read and pin.

## Native01 outcome

Root completed/reaped the pilot on **2026-11-01 at 11:05:52 PST**: **1.106289 seconds**, **139,640,832 B peak RSS**, five files / 101,445 bytes. Root verified all output members. The outcome is `no_verified_candidate`: the linear solver reached its 64-sweep limit with residual approximately `1.0240089e-6`. No nonlinear candidate was tested and no selected live step occurred. This is a solver-budget abstention, not proof that the physical task is infeasible.

Actual counts: **20 predictions/copies, 48 live controls, 1,088 substeps, 70 forwards, two data objects, 11 models and one reset**. All 48 original actions/snapshots match. Every prediction and the projection call preserve live/task state. No replay helper, media, paid call, retry or promotion occurred.

- Process receipt `.local/equipment-action-controller-native-01/process.json`: `694684f08e81e247c5fdb59683ca71db72dba43df09f01fc3a4090a670e514d9`.
- Output `manifest.json`: `596c1614a831a23e2f17352c0793411389a94afcad93d453cf4135571855fbd1`.
- `result.json`: `61faf5ccd8424621c563e250ba62b9cdf774ab558b0593097023dca97153f29b`.
- `projection.json`: `cc0082369fc96d7f0bc156655c558d79b66c68a38c5cc08a4e222c7956cbad0f`.

## Native02 — completed changed-input follow-up

Root approved a changed-input follow-up: allow the bounded last linear iterate to receive the already reserved nonlinear checks even when the 64-sweep convergence threshold is not reached. The same 24-prediction, 64-sweep, task/gate, time, memory and output bounds remain. The equipment agent owns the solver code/tests; root independently checked the one-block delta and final pins before freeze. Five affected/new/saved-vector mocked cases pass; the previously passed 42-case suite was not rerun. This is not an unchanged-input retry of Native01.

Workers under `.local/equipment-action-controller-native-02/` differ from Native01 by exactly one unique output-path substitution per file, with no other code change:

- Worker: `1da20c25c4a64c1850a76ac0c9fb9f5a7aa659edc5a3a34a1998b2519dcb5647`.
- Supervisor: `6c4a664f6eb6389c726bbb28a76da05765837247cf97b0caed5f5d8b0193ecb4`.
- `preparation-draft-01.json`: `15da72c2cd75c0c9f6f6fe77a7d0aa5a9297fd6adccf4382ab457a3846471693`.

The frozen runtime retains 293 files / 6,346,589 bytes. Every inherited member was compared to Native01: only `shared_exercise_action_projection.py` changes, to `04510c0a46ca13ce397e0d73ff6a5edcebd325d3df3ccd6092e2643b1d90acab`. Test pin: `4991c6a0ef1c78132636923735e4002442c0bd51b2d41a7890327124f2d75c57`. All other files, all 25 policy hashes, the plan's limits, phase limits, full config and acceptance remain exact. No saved recording/source payload was reread during preparation.

- Plan: `8a007cfef0b44545b4bb385a2f449676525b2815657b3b671a70b15288994a34`.
- Runtime manifest: `ab0b266943aef665403ef5e2b13939529363a3d7abd25145366e32e6e0820ad7`.
- `preparation-01.json`: `511dad1fac86a70d373fc21649a7d958978eb1433cd492e3547c7529edc22146`.

Exact root-owned launch, completed once into `artifacts/equipment-action-controller-native-02`; do not repeat:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-controller-native-02/supervisor.py \
  8a007cfef0b44545b4bb385a2f449676525b2815657b3b671a70b15288994a34
```

Root completed/reaped Native02 on **2026-11-01 at 11:10:50 PST**, PID 25407: **1.074171 seconds**, **139,935,744 B peak RSS**, five files / 169,863 bytes. Root verified every output member. The projector selects the full-scale nonlinear candidate (`correction_scale=1`), then actual control 49 matches all predicted state fields, time and endpoint contacts exactly. Original root bytes are unchanged and the selected action respects the frozen trust/action bounds. Every protected gate still passes at that boundary, but `completed_interval` is false. This is a successful one-decision prediction/verification check, **not a successful full-interval controller**.

Native01 remains an abstention, with no selected step. Native02 is a changed-input follow-up, not a reclassification or overwrite of that result. No further one-decision repeat is needed; a separately reviewed full nominal rollout is the next distinct question.

- Process receipt `.local/equipment-action-controller-native-02/process.json`: `ac51e9c9377dcae55e7a62dfeb1eb4154923c1afcd0a134a10307a2bdece2126`.
- Output `manifest.json`: `71afa27c64072d7fbb142331b42c2e6b0b2750802e1af4a14a9c93bf514cec58`.
- `result.json`: `9dac5e6d9321982fa62e5e13ddda6314832da45c9b719686306ef3a8c798af1e`.
- `projection.json`: `26902044ed86759550b1ba21d2c11ca0033dfaabab2e2d3097f4c795b36f4696`.

No paid calls, Git mutation, source/policy/scorer edits, avatar edits, browser/server or native runs occurred in this preparation lane. Root owns verification, execution and the next decision. The parent reports main at external checkpoint `4ade45a`, with predictor checkpoint `fdab07b` as an ancestor; this lane did not inspect or change Git. Preserve independent dirty work and every earlier result. A selected local action or exact prediction is not evidence of completing the exercise interval or arbitrary-video reconstruction.
