# Equipment signed-velocity diagnostic

Updated **2026-11-01 (PST)**. Preparation/diagnosis owner: `/root/contact_reconstruction`; execution owner: `/root`. **Root completed/reaped the diagnostic. All required saved states and predictions match; no actual control 50 occurred. The signed affine model improves two saved checks but worsens the third, so it is not supported as the next solver change.** No native worker was executed by this lane. Do not modify frozen inputs or reuse its output directory.

## Question and narrow scope

[Rollout02](equipment-action-rollout-native.md) accepted 49 controls, then rejected all 23 tested actions at control 50. Its linear solver converged, but the full candidate's predicted equipment speed norms differed substantially from the actual detached response. A single saved-data secant update made both smaller-scale checks worse. The current trace retains only norms and hashes for rejected endpoints, so it cannot distinguish vector-response error from the additional nonlinearity of taking a norm.

This diagnostic retains **already computed signed equipment velocities** from one unchanged projector call at the exact saved pre-50 state. The required 49 directly applied saved actions reconstruct that input; they are **not another task qualification**. There are no prefix projector calls, new candidates, solver changes, fallback, optimization, or actual control-50 step—even if an unexpected selection occurs.

Each prediction saves its action, original endpoint/contact hashes, original scalar measurements and:

- Raw six-component free-joint `qvel`, object ID and `qpos`/`qvel` addresses.
- Signed linear components divided by engine scale, with model-units/second labels.
- Signed angular components in the **raw free-joint convention**, not mislabeled world-space coordinates.
- Endpoint orientation quaternion and scale, so later analysis can respect coordinate frames.
- Exact equality between norms reconstructed from these vectors and the unchanged projector's measured norms.

No additional native velocity query or prediction is needed. The prefix checks saved before/after states, snapshots and metrics; pre-50 state/metrics must equal saved control 49's endpoint. All 23 existing control-50 actions, endpoint/contact hashes, measurements and outcome must match the sealed trace. Any mismatch fails the diagnostic and preserves available evidence without stepping control 50. This does not establish feasibility within `.25`, model improvement, full-interval success or physical accuracy.

## Frozen inputs and limits

Scratch: `.local/equipment-action-velocity-probe-native-01/`. New output: `artifacts/equipment-action-velocity-probe-native-01/`.

- Plan: `bfee45649292b7eb60e2610f48c4ac424266afee3ae272d4d5d585e08f5e8b69`.
- Preserved draft: `9241536f32b3bc23c3ad5922011a265126a068841f82cfdebf15239da9d91261`.
- Worker: `14636e3b568cb000c95868566ef64eea1fb056c314d2dba20c85ae8c69bfbf8f`.
- Supervisor: `6a2dbb6fbaa42760a33de64f47ad267e2273d5a0eaaeaf8f2ce96ad681fb11dd`.
- Reused Native02 runtime: 293 files / 6,346,589 bytes; manifest `ab0b266943aef665403ef5e2b13939529363a3d7abd25145366e32e6e0820ad7`. No copy or source change.
- Projector: `04510c0a46ca13ce397e0d73ff6a5edcebd325d3df3ccd6092e2643b1d90acab`; predictor: `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd`. All 25 original policy files and source/task/scorer pins remain unchanged in the plan.
- Rollout02 seal: `040698301ae85427b79e059cc34c31e9332a51b92ccdcbc5b95aa05817ea04aa`; trace: `ae66fd6282fb7c1107beecd1b152294660ce70135687061e6cbf4517d0168a69`.
- Required pre-50 full state: `ca2d641777574ec914d8d430db7d1bd3db11c7beb168c18bdb124b437c0a99a5`; snapshot: `883635f2fb2101c73be47fcc01986f7ad7c61762bb1a5749dfaeb110418f25e7`.

Same Native02 one-decision maximums: **49 actual controls + at most 24 predictions = 1,168 substeps**, 75 forwards, 11 models, two data objects, one reset and 24 copies. Expected unchanged 23-prediction outcome uses 1,152 substeps / 74 forwards. Same 24-prediction, five-second, `.1` difference, `.25` trust-radius, 64-sweep helper settings. **40-second CPU/cooperative, 60-second external, 512 MiB observed RSS, 2 MiB/result, 4 MiB total / eight files**; no media, replay helper, retries or paid calls. Estimated external and recurring cost: **$0**.

## Preparation verification and completed launch

Root reviewed the worker, supervisor and draft plan. Only freeze metadata changed afterward. Both scripts parse; five small authored AST-only capture checks pass (signed values/scaling, second-object address mapping, angular norm, no input mutation and nonfinite rejection). No Weft/native imports or worker execution occurred. Receipt `capture-checks-01.json`: `9a90a1d94af7aad306330b5d137c24f743f2f8ffc63cb39d7d46b5330de0efb0`. This checks capture arithmetic, not simulator behavior. `freeze-review-01.json` records the approved pins and launch.

Root executed the following once, separately from the depth-prior lane. **Do not repeat it.**

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-velocity-probe-native-01/supervisor.py \
  bfee45649292b7eb60e2610f48c4ac424266afee3ae272d4d5d585e08f5e8b69
```

Output/log/receipt paths were absent at freeze. Root reported the process terminal/reaped with exit 0 in **1.496888 seconds**, peak RSS **139,018,240 bytes**, five files / **195,118 bytes**. Actual counts: **49 prefix controls, one projector, 23 predictions, 1,152 substeps, 74 forwards, 11 models, two data objects, one reset, 23 copies**. All 49 required saved prefix states and 23 original predictions/norms matched exactly; live state/task metrics remained unchanged during projection. No actual control 50, extra candidate, tuning, fallback or physics qualification occurred.

Root verified every output member:

- `result.json`: `e5ec246c00a1976df1a3c824759bd3f9365e572eb8578fb5a0068aa92dd16124`.
- `projection.json`: `cacd1915d3dd6001279f4bbc880becb508642cdc497e48b75bdbdb9ff8faebe3`.
- `manifest.json`: `9d8f81fe188b54dca439f0bba2ceccc11446cf8ecb8bc0235541afd56f2217f6`.

## One saved signed-affine hypothesis — mixed evidence, not adopted

`check_signed_affine.py` fits the six signed velocity components using **only the base and 19 existing one-sided differences**, then predicts the three already saved candidates. It compares the resulting speed/spin norms with the original scalar linear fit and the actual detached norms. No alternate model, parameter search, optimizer, source import or native call occurs. Candidates are held out only from this local fit, not from prior development or an independent evaluation set.

The predeclared primary check compares each candidate's **maximum limit-normalized norm error** across both speed metrics for both equipment objects. All component errors remain in the receipt:

| Candidate scale | Original scalar fit error | Signed affine fit error | Improves |
| --- | ---: | ---: | --- |
| Full | 1.62991649 | .49130921 | Yes |
| Half | .22657992 | .14367352 | Yes |
| Quarter | .06897518 | .19869632 | No |

For the failing equipment object (`sam-2`), the underlying predictions are:

| Candidate / metric | Old scalar estimate | Signed-model norm | Actual detached norm |
| --- | ---: | ---: | ---: |
| Full / speed | .14999985 | .18586119 | .25955757 |
| Full / angular speed | .65735226 | 1.63346397 | 1.87978962 |
| Half / speed | .31491847 | .31655050 | .30479411 |
| Half / angular speed | 1.76247043 | 1.70029063 | 1.59253549 |
| Quarter / speed | .39737778 | .39645489 | .40011877 |
| Quarter / angular speed | 2.31502952 | 2.21773866 | 2.36676090 |

Linear speed uses model units/second; angular speed uses radians/second. Unchanged actual limits remain `.15` and `.75`. All saved candidates still fail. The full-scale angular **vector residual norm is 1.24953275 rad/s**, despite a closer scalar norm; modeling signed components does not remove the underlying response mismatch.

The `sam-2` quaternion changes from the base endpoint are **.04906272 / .04009681 / .01869439 radians** for full/half/quarter candidates; the maximum difference-probe change is **.04090555 radians**. `sam-3` has zero measured relative change. These provide coordinate-frame context, not proof of contact-mode changes or a cause of the errors. The model uses raw free-joint coordinates without a world-frame label or rotation conversion.

The signed model fails the all-three improvement check and is **not adopted**. This does not prove that signed models can never help, that no feasible action exists within `.25`, or that a geometry/contact change caused the mismatch. It supports neither loosening gates nor adding solver sweeps. No production solver/helper/config change or new trial is queued.

- Analysis script `check_signed_affine.py`: `0429304b5695bb5b1c274903b0e54d8db219d3721150a54f000dc4b60d0a2345`.
- Receipt `signed-affine-check-01.json`: `29e0088f85632221bc39cf5e920005f46f6de5576bba66382cc8b95e667d13d9` (12,955 bytes). It pins the source artifacts and retains every object/metric comparison.

Next action belongs to root: preserve both rejected modeling hypotheses and decide on a separately justified change. Do not rerun unchanged inputs. No live handle is owned here. Git/checkpoint ownership remains root's; this work changes only its new scratch directory and this handoff, preserving all prior runs, production code and unrelated avatar work. External spend remains **$0**.
