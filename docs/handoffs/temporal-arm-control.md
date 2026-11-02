# Temporal arm control

## Completed — November 1, 2026, 5:40 PM PST

**All eight candidates complete the full interval; 0/8 succeed.** Every candidate
passes the five body/floor gates and four `sam-3` terms, but fails all four
`sam-2` stability terms. The exact saved authored baseline was reused, not rerun.

| First-window joint correction | Maximum normalized gate excess | First equipment crossing |
| --- | ---: | ---: |
| Saved authored baseline | 15.391766 | 57 |
| Shoulder twist +.25 | 16.202942 | 57 |
| Shoulder twist −.25 | 14.914896 | 57 |
| Shoulder lateral +.25 | 16.682251 | 57 |
| Shoulder lateral −.25 | 18.015056 | 57 |
| Shoulder flex +.25 | 18.892874 | 57 |
| Shoulder flex −.25 | 16.213458 | 49 |
| Elbow +.25 | 16.735524 | 57 |
| Elbow −.25 | 15.499077 | 48 |

Zero normalized excess means within the original limit; lower is better.
Only negative shoulder twist ranks before the baseline on maximum excess. Its
mean excess worsens **2.893448→3.185778** and mean reward falls
**0.747612→0.743716**. This weak one-start ordering is not successful learning
or a reason to promote it. No first crossing is delayed. Preserve all eight
failures; do not repeat these coefficients or blindly expand a gain sweep.

Actual run: **6.530473 seconds /215,842,816 B** child high-water RSS;
11 files/7,498,429 B; PID 4945 exited zero and was reaped. One data allocation,
14 model compilations, nine native resets, 2,050 forwards, exactly 2,032 controls
and 32,512 substeps. Every action uses its same-current-state authored base;
all 2,032 root-byte checks and saved residual/observation joins pass. No root
action mutation does **not** mean the resulting root trajectory is unchanged.
No extra rollout, media, source inference, API, trained policy or default change.

Native workload timing: 2,032 actor samples total 0.070675 seconds (mean 34.78 µs,
max 119.08 µs); physics steps total 4.808776 seconds, max 6.271 ms per control.
These are bounded backend measurements, not graphical or sustained-use latency.
Independent root audit recomputes **117 original scalar gate comparisons** from
the frozen evaluator and checks identities, flags, counters and seal joins using
467,736 B of cached metadata. Worker already verified all lossless journals;
root did not repeat journal/physics/media checks. Pure checks remain 30 passing.

Result SHA `a8cf16a778aa03e53cab94d165165e0d3a062d9bd62a0e8140c7283c404f5b63`;
seal `261881d6dab50026388f74f2dcc05b8ecb45fa0224563b8c0b23711ec39216dc`;
settings `e5b02c32e4e228bc5e897e41df60aec3e50d16765e8da173a63584c3ffd29dba`;
process `23b0b82de443f3573dccfed6f553c4b915d73210d93e4a8e2b4d8341a1a103a4`;
audit `cc8078c3ca2cb4dcf4f16a892ae9854841457f5824d1c88a64c9efe512ba415f`.
Pure receipt `c2309e7960d447ecd83cb1e860439fd356738417c39962afcbdea3675c8e9721`;
test source `17dc9a86ee0b3ad334c12d8a1e79a8edf4027187c1d0f67e3f041b28fbdcbed3`.

Next: return to source-qualified body/equipment geometry and contact evidence.
This result does not prove every bounded controller infeasible, and does not
authorize changing the task or inventing hidden geometry. The actor stays an
opt-in experimental helper, without a policy package, learner, lab attachment
or default. Keep the full generalized goal active and the verified lab usable.

## Reproduce and inspect

Saved results: `artifacts/temporal-arm-study-01/result.json`; each candidate has
its own lossless `.jsonl` journal. Harness, frozen plan, overlay and exact anchor:
`.local/temporal-arm-study-01/`. These artifacts are local, not bundled into Git.
Do not rerun the single-use supervisor or overwrite these directories. A justified
new experiment requires a new output path, explicit changed hypothesis and limits.
Actor checks were run once, before native execution, with the existing pure guard:

```sh
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  TMPDIR="$PWD/.local/shared-temporal-residual-01/tmp" \
  .venv-lab/bin/python -B -P .local/split-limb-feedback-actor-01/run_pure.py \
  .local/shared-temporal-residual-01/checks-01.xml \
  .local/shared-temporal-residual-01/checks-01.json \
  tests/test_shared_exercise_temporal_residual.py
```

The receipt path is exclusive-create; unchanged passing checks need no rerun.

## Original preparation — November 1, 2026, 5:37 PM PST

Root owned this bounded study, now completed above. It is not a trained policy.
The automatic-frame recording first exceeds equipment speed/spin at control 57,
with a coincident right-forearm boundary contact. Earlier old-world constant,
feedback and root interventions remain failed evidence, not a reason to retune
those same experiments. Test a smooth earlier arm intervention on the new world.

New `src/weft/shared_exercise_temporal_residual.py` preserves the same-state
authored base's three root action bytes and the existing ±.25 joint correction
bound. Three compact raised-cosine windows use elapsed task fraction, with zero
endpoint value/slope. This differs from the existing broad two-piece linear
phase residual; it does not claim phase-conditioned control is new. Sampling is
pure; metadata admission currently supports existing 22-channel v1/v2 coupled
tasks with 1–12 objects and no required contact. The caller must authenticate
the base/observation pair. No physics, reference, task, scorer or default changes.

**30 analytical checks pass**, `.local/shared-temporal-residual-01/checks-01.json`:
0.577401 seconds, zero native attempts. The blocked import-time subprocess probe
remains recorded. Coverage includes smooth convex basis, temporal rather than
source phase, immutable coefficients, finite input/metadata admission, float32/64
legal clipping and byte-exact unchanged roots. These are not native/learning tests.

Fixed study: four right-arm channels × two signs (±.25), first window only,
seed 19/no jitter, all 254 controls: at most 2,032 controls/32,512 substeps. Reuse
the saved complete authored baseline and require its exact initial snapshot.
Keep every candidate and all original 13 full-interval gate terms. This is one
nominal start, not the two-start training rank, a holdout or learned selection.
Lossless before/base/action/after journals receive saved arithmetic verification
without another simulator rollout. No media, source inference, API or promotion.

Inputs: canonical ready lab
`artifacts/equipment-frame-consensus-native-03/build/lab`; exact 299-file e5 runtime
from `.local/auto-equipment-console-01/runtime-manifest.json`; one frozen temporal
overlay. Authored anchor comes from the exact 265-event console journal, reset 10,
254 post-reset actions; source journal SHA
`3698a10c5b08f9688edf5a01fbc8587e43d0e73e6ed4987d62431ea2ecdafd3f`.
Anchor SHA `90ce186301a14a02e69daf949d04d4793df04ac17da749c0948444c5579e5895`;
initial snapshot `62290caff19ea4f249bbf576f83f3212d2c3ebf9ad6e3c97b9808ff453138e14`;
overlay `753b900e39fc660b2db64173cb76aaa19f947e04bf3dde834fd7085ef75f80fd`.

Scratch/output: `.local/temporal-arm-study-01` and
`artifacts/temporal-arm-study-01`. Frozen plan
`c9a16e327cf4283aeebf8e1051b9224cee0d72730fa6e2dc7de810979df0db7b`;
worker `0a1add3a6e4a10bb06d27bc445c361df62d90b4fa79331b985e0ba7499ee28d7`;
supervisor `1ceeab32bf1feb7a6864c079d40402e2ebe9f29b0516cf1212ef02eec3ad9202`.
Limits: 50 seconds cooperative/40 CPU/60 external, 512 MiB observed RSS,
32 MiB/16 output files, 1 MiB per log, one native environment/9 resets,
48 model compilations/2,064 forwards. Expected output about 12 MiB.
Cost: $0 external/$0 recurring; retain failures, no automatic cleanup.
Root reviewed the frozen harness and terminal flags; peers remain at usage limits,
so no peer review is claimed. Supervisor observes only its owned child and reaps it.

The preparation required an audit of exact unchanged gates and preservation of
failures; both are complete above. A future promising candidate still needs a
versioned learning/admission path and separate changed-start evaluation before
any default/generalization claim. Preserve independent avatar work; exact-path
checkpoint/push remains authorized by standing approval.
