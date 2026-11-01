# Split-limb feedback — improved tradeoffs, no task success

## Terminal result

Root's one reviewed v5 invocation is **completed, exit 0 and reaped**, after 61.261301 seconds. Former root-owned session `41540` is closed. `artifacts/split-limb-feedback-native-01` retains **1,046 files / 120,782,513 bytes**, seal `cb6249e97ec4208507ab5d11685113cb925a41a7b0b8e822c44e1bbcc0cb4e1e`.

The arm/leg split changes real actions and improves several metrics, but **all four controllers remain 0/3 task success**. The policy remains infeasible and was not promoted. **At native-comparison completion, it was not yet attached to a lab; root owns a separate play acceptance.** That later play is inspection, not successful-policy promotion. This completes the bounded computation, not the useful whole-body learning goal.

| Controller | Complete intervals | Success | Mean score |
| --- | ---: | ---: | ---: |
| Coast | 3/3 | 0/3 | 0.369632 |
| Random | 0/3 | 0/3 | -0.942994 |
| Authored | 3/3 | 0/3 | 0.747771 |
| Learned v5 | 3/3 | 0/3 | 0.745110 |

All twelve evaluation starts remain recorded; none failed reset. Coast fails angular/root mean and peak tracking gates. Random terminates on `human_velocity_envelope_exceeded` after **82, 4 and 41 controls**, retaining partial-substep accounting. Authored and learned complete every 254-step interval and pass every body/floor gate, but fail only **`equipment_stability:sam-2`**. `sam-3` remains stable for every controller.

## Remaining equipment failure

Each row is the learned trial's accumulated maximum. Distances are **model units**, not calibrated metres; values are simulator results, not measured contact or physical motion.

| Evaluation seed | `sam-2` displacement | Orientation (rad) | Speed (model units/s) | Spin (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Fixed maximum | 0.030000 | 0.150000 | 0.150000 | 0.750000 |
| 1841308058 | 0.087322 | 0.658110 | 1.578679 | 10.201635 |
| 1870335751 | 0.389792 | 0.798891 | 1.637387 | 6.218104 |
| 203828300 | 0.093408 | 0.633226 | 0.718625 | 5.612274 |

**All four equipment limits fail on all three starts.** The second start exposes much larger displacement than the other two. `sam-3` has the same values throughout: displacement `0.000096060526`, orientation `0.000000089407` radians, speed `0.0204375`, and spin `0.000023168943` radians/s—all below its unchanged limits. No contact-duration requirement exists, and no image/support evidence was converted into task credit.

### Tradeoffs against authored and v4

The v4 comparison uses only the prior saved interpretation; its evaluation payload was not reread. Coast, random and authored **scores, body/equipment summary values, per-seed failure lists and episode lengths match exactly**. The actor/runtime changed explicitly, but protected task, dynamics, source and reset streams did not.

| Mean across three starts | Authored | Learned v4 | Learned v5 |
| --- | ---: | ---: | ---: |
| Score | 0.747771 | 0.733232 | 0.745110 |
| Angular RMSE (rad) | 0.150644 | 0.159059 | 0.149348 |
| Root RMSE (model units) | 0.019607 | 0.020751 | 0.018871 |
| Squared joint effort | 0.148154 | 0.257590 | 0.115502 |
| `sam-2` maximum displacement | 0.476517 | 0.424527 | 0.190174 |
| `sam-2` maximum orientation (rad) | 1.004236 | 1.206709 | 0.696743 |
| `sam-2` maximum speed | 1.264847 | 1.255934 | 1.311564 |
| `sam-2` maximum spin (rad/s) | 5.468985 | 7.320816 | 7.344004 |

V5 gains `0.011878` score over the prior searched policy but remains `0.002661` below authored. Mean tracking error, effort, floor penetration and equipment displacement/rotation improve. Peak angular error worsens **0.479803→0.492239 radians**, peak root error **0.107606→0.108044**, and equipment speed/spin worsen on average. Body peaks still pass their gates. The remaining equipment failures prevent a success or broad control-quality claim.

These seeds were already exposed in the v4 diagnosis and informed the new hypothesis. This is a **matched development comparison**, not a fresh final holdout or unfamiliar-video generalization result.

## Training selection is consistent, not feasible

Training evaluated **28 episodes / 7,112 controls**, with two cache hits and **14 unique infeasible candidates**. Generation 0 candidate 6 is selected by the original training-only feasibility rule. Selection performs **zero new controls** and reports `retained_training_infeasible_selection`; evaluation does not feed back into the optimizer.

| Training rank | Initial authored zero anchor | Selected g0c6 |
| --- | ---: | ---: |
| Largest positive normalized gate excess | 16.187768 | 6.499293 |
| Mean normalized excess | 2.817281 | 1.214913 |
| Violated terms / total | 8/26 | 8/26 |
| Original mean reward | 0.747648 | 0.745010 |

The worst initial term is `sam-2` displacement: `0.515633` against `0.03`, on training seed `137031775`. The selected worst term becomes spin: `5.624470` against `0.75`, on seed `1263521311`. Both maximum and mean violation improve, so the unchanged lexicographic rule selects g0c6 despite its `0.002638` lower reward. All four `sam-2` limits still fail on both training starts; no feasible alternative was hidden or promoted.

All **ten coefficients are nonzero** and remain within ±0.25. Root's terminal-worker audit independently confirms **762/762 learned evaluation actions differ from authored joint control computed at that same state, with zero root-action changes**. This stronger live comparison was not rerun by the saved interpreter. All 762 saved learned joint-action rows also differ from the separate authored trajectory. Trajectory root actions can diverge after the reached states diverge; that is not a learned root correction. No root coefficient or equipment force was learned. The actor uses only the existing normalized position/velocity feedback, with separate torso/arm-proximal/arm-distal/leg-proximal/leg-distal routing.

## Exact identities and scope

- New policy: `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`, profile `shared-exercise-policy-v5`. Exact bytes are unchanged between training and evaluation.
- Training seal: `eb5db3c9bec8f962a375f2bb95304af9c6a0876cc43e80b184151abfa4a0cd92`.
- Original source seal: `f300b46cfb71697ed2ec4cdf4746c96ceda7bf0995a77412f0835d274b6d055f`.
- Scene/task canonical pins: `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d` / `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da`.
- Protected scorer/gates: `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb` / `063a93773406db92b583daf947855098727e34009bf645e82e0a5a2e044e0002`.
- Frozen runtime: committed `52a2559fb2944deb7b274f8882481b8d6648f1a4` plus seven exact reviewed overrides; manifest `e1dcc006c9c17b4926d0a01e3c3de1e93e8dd0abade5ab32e4f7716a91daf79e`. Actor `3aefc6ef0e5be09fc3e710ffd277dc200c44bbc0b6051f852a13795f6a89268c`. Old v4 policy/runtime artifacts remain immutable, not silently repinned.
- Input/execution plans: `e1734213f1ba222c968459d9ffd90505f942adcc25fa0041bf9b9f011a871e20` / `478fa5aa7252226023df82dd3928991484bc9a318af804fe69a09c30027a1ce5`, under `.local/split-limb-feedback-native-01`.

The **257 declared scene/source slots**, 121 body references, frame-2→256 interval and **254 controls at 30 Hz / 16 substeps** remain unchanged. The scene denominator is not a claim that the parent video's native metadata count is 257; upstream evidence separately retains the 259-metadata/257-decoded boundary. Training seeds `1263521311` and `137031775` use original jitter 0/.05; evaluation uses `.10` and seeds shown above. Training/evaluation streams are `2026110101`/`2026110102`. Evaluation states differ from training and are matched exactly across all four controllers. Equipment reset state is unchanged; source/body/floor/clock/assistance/gates were not tuned.

The six-frame equipment initialization gap remains authored extrapolation. Contact/grasp, recovered anatomy, unassisted balance, physical accuracy and unfamiliar-video performance remain unverified. External cost is **$0**. No source inference, image decode or extra baseline/replay run belongs to this attempt.

## Independent saved-only interpretation

Receipt: `.local/split-limb-feedback-native-01/saved-interpretation-01.json`, SHA **`fd0756b7f0b80406ff8fdd212e68263d9f0af96b8566701e6531e4146526ec1a`**.

The audit read **11 selected new metadata files once / 4,548,236 bytes**, checked their raw pins against the completed outer seal and stage manifests, then reused a compact cache. One **24,791-byte** previous interpretation was read for comparison: `acbc6ff8c162fb69733657b9c244f83a13e0777f0f43a58945485197907bc76b`. Total selected input: **12 files / 4,573,027 bytes**, within the 15-file/8-MiB bound.

The receipt retains every controller/seed's body and equipment metrics, recomputes all training ranks from unchanged gates, confirms g0c6 selection, and checks source/task/scorer/runtime/seed/initial-state joins. Root separately owns process/seal custody. No whole-output inventory, raw diagnostics, original media/source, runtime reread, report regeneration, native call or repeated training occurred. Cache: `saved-selected-cache-01.json`, SHA `6ab0aba484c8e48e1142e364981ff569ac1ea919ec4111014dfd2325bef250a0`.

## Next safe action

Preserve this attempt and the failed/infeasible predecessors. **Do not repeat unchanged search with more budget or relabel the policy successful.** The split reduces some conflicts but leaves large equipment speed/spin and seed-sensitive displacement. Aggregate metrics do not identify the responsible contact sequence or prove task feasibility.

Root may scope a distinct next control/data hypothesis, using a bounded inspection of existing selected diagnostics if necessary before choosing more parameters. That inspection, any implementation and any new experiment are **not part of this audit**. Task/scorer/reset/source/contact gates remain fixed. Useful coupled learning still requires body and equipment success together with an honest matched-baseline benefit.
