# Reconstructed camera-body learning

## Verified checkpoint — November 2, 2026, PST

Root owns the integration of the exact reconstructed ASPset body with the
existing bounded learner, changed-start comparison, playable lab and policy
recording. Both reconstruction and simulation tracks remain active. Standing
user approval covers verified checkpoint publication; preserve avatar edits.

The new native regression reproduced policy attachment failure after successful
training/evaluation: the policy dependency inventory excluded eight body-fit
validators. The repair selects the complete inventory from coupling v2, retains
v1 requirements, and rejects missing or changed reconstruction validators.
Physics, task, optimizer, scoring and thresholds are unchanged. The focused
115-case receipt passes at `.local/reconstructed-camera-learning-checks-02.xml`;
the original two failures remain in `reconstructed-camera-learning-red-01.xml`.

One frozen development study completed on
`artifacts/reconstructed-camera-source-01/lab/run`, manifest
`ff3e37a021e3d5fc187a0435b978887057511c7872e680e49f19918e77736f75`.
Keep all 184 controls /3.68 seconds, the exact reconstructed body, original
gates, 19 joint channels and three explicitly enabled authored root forces.
Training: seed 298817, two generations × eight candidates × two starts.
Evaluation: seed 713684215, three fresh starts at jitter .10, matched across
coast/random/authored/selected policy, with no feedback to training. Settings and
runtime were frozen before execution. Source has 188 slots; the unchanged task
ends at its last sampled pose, 3.68 seconds, not the video's final unsampled slots.

Controls/output: `.local/reconstructed-camera-learning-01/` and
`artifacts/reconstructed-camera-learning-01/`, new exclusive directories.
Budget: one local worker, 180 seconds external /160 CPU seconds, 1 GiB sampled
RSS, 70,000 physics substeps, 128 model compilations, 64 data allocations,
8,000 forwards, 2,500 files /384 MiB output. Actual work: **24.118 seconds,
610,254,848 B peak RSS, 62,094 physics substeps, 41 compilations, six data
allocations and 7,955 forwards**. The supervisor reports 1,285 output files /
363,169,766 B and a reaped exit-zero worker. No source decode, inference, network,
paid call or key/ledger access in the learning job. Do not repeat this study or
expose its evaluation starts to tuning and later call them fresh.

## Learning, comparison and playable result

Training evaluates 28 episodes /5,152 actions, reusing two duplicate candidates.
It selects generation 0, candidate 1 with 12 nonzero coefficients; 366/368
selected training actions change, maximum joint correction .100481. Training
score improves .005298. Training reset seeds are `1955158460` and `542326383`,
as recorded in `training/settings.json` and the saved audit.

| Controller | Success | Completed intervals |
| --- | --- | --- |
| Learned | 3/3 | 3/3 |
| Programmed reference | 3/3 | 3/3 |
| Coast | 0/3 | 3/3 |
| Random | 0/3 | 2/3; one velocity-envelope failure at step 176 |

Frozen evaluation reset seeds: `7421163`, `2034436424`, `199742131`, distinct
from training. Mean reward rises **.805921→.811256**, but success does not improve.
Mean joint error falls .002166 rad; mean root error rises .000203 model units,
joint effort rises .001021, and maximum floor penetration rises .0000392.
Retain these mixed metrics and all failures. This is joint-feedback learning
with privileged simulator/reference inputs and unchanged authored root forces,
not learned locomotion, unassisted balance or physical transfer.

The attached lab runs all 184 selected-policy steps and passes the original task.
Its recording includes six manual steps, one reset and 184 policy steps:
**190 actions /192 events and exported video frames**, all replayed exactly.
Every learned decision and its unchanged root channels is verified. A complete
saved export receipt is available. All 913 immutable run members and ready
lifecycle files remain unchanged after close. Thirty in-process policy/action/
state/JSON calls have median 1.473 ms /p95 1.504 ms; not browser, HTTP or
sustained-use QA. The real console resume/select-learned/step/quit check exits
zero and creates no extra recording.

```sh
PYTHONPATH="$PWD/.local/reconstructed-camera-learning-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/reconstructed-camera-learning-01/lab
```

Try `controls`, `mode learned`, `record start`, `step 6`, `reset 29`,
`record stop`, then `record evidence`. Manual, coast and reference modes remain
available. Replace `lab-console` with `lab` to start the graphical lab yourself;
no browser or server was launched for verification. Retain the frozen runtime;
do not rewrite old runs to match future code.

## Evidence and integrity

| Artifact under `artifacts/reconstructed-camera-learning-01/` | Manifest SHA-256 |
| --- | --- |
| `training` (170 members) | `d730e3b14fdcebe612959ee28f449c950db7e8b679352bcb6034ec349604fcdd` |
| `evaluation` (187 members) | `b8928e1d532c95cc6b5a69a8fe40bc861fe5893e6a33629d972a2f5222d1621f` |
| `lab/run` (913 members) | `4b186782bc0babd220de7cfd4f1a2f82ad4358607388dff372df80eb6d428fa5` |
| `lab/sessions/session-e638ce673e24` (six members) | `ad1a47ba5b944915856b92c5d4c7be33d0d0fe2643da8578dc982fe56de34ab5` |
| Recording `evidence/` (one member) | `2eddbc03ccb1590345d8714de7479b78689d37ab1b8e3ba91f71323b5f0131b0` |

Policy SHA: `03f4c9c4b759cc288dd9e0d14bab509c55f98225ad6eed4a827ee271b9d1c26b`.
The control directory retains the plan, worker, supervisor, source snapshot,
per-stage results, process receipt, latency samples, console log and independent
saved-only audit. `saved-audit.json` verifies six seals /1,671 members, exact
source/scene/task/gates, deterministic saved training/evaluation and all recorded
root-action identities without native execution. It takes 1.423 s /242.5 MB.
The existing 394-member source run remains unchanged.

Changed code: `camera_shared_receipt.py` owns the pure body-dependency list;
`camera_shared_contract.py` reexports it; policy receipt/admission select the
complete v2 inventory. Tests add real training/evaluation/attachment/replay with
roots off/on and reject missing/altered entries. Scorer/actor/optimizer sources
are unchanged. The 315-file runtime is the previous trusted source snapshot
plus exactly these four source changes. Publication receipt will be
`.local/reconstructed-camera-learning-publication-01.json`; preserve the separate
avatar renderer/test edits and inspect the current index before staging.

## Next unfamiliar clip

Independent source work has identified the next archive-order development clip,
ASPset `04ac-0006`, without viewing it or reading reference coordinates.
`.local/aspset-next-unit-header-01.json` preserves that selection. One exact
64 MiB +64 KiB range GET extends the existing archive prefix, checks its ETag
and 64 KiB overlap, and completes in 1.579 seconds. No full 23.7 GB archive,
test-partition payload, provider account or paid service is used. Complete parent
archive integrity remains unverified; individual retained members are hashed.

`data/aspset/04ac-0006-01` now holds three complete original videos, matching
publisher cameras and CC0 attribution: 13 members /47,550,377 B, source seal
`adad325e9fa9c187c2448512e518b69dbe7982671483ae0c60971d2c0369a680`.
Two bounded workers verify native and decoder clocks for all three views:
**185 frames each /555 source slots, 50 Hz, zero native/decoder offset**. The
inspection seal is `bbe3ce977d0b30967fa50a06704f19d1adb3170ffa5ee2cc484ecf7a652b0357`
at `.local/aspset-next-media-inspection-01/manifest.json` (24 members).
Supervisor `.local/aspset-next-media-process-01/result.json` reports 12.179 s,
all three workers reaped; individual worker peak RSS ≤285,573,120 B. Read-only
source audit: `.local/aspset-next-unit-source-audit-01.json`.

This is the same subject/camera setup but a different, previously unprocessed
motion; not a fresh final holdout, independent physical truth or a new-subject
test. No pose inference, reference-coordinate reading, body construction or
physics has run on it. Camera synchronization remains publisher-supplied.

Next safe action: apply the frozen pose/body method to the new source with
explicit person-region/calibration/ground provenance and all missing slots;
compare construction and task failures without tuning protected thresholds.
Keep the original source's image regressions, weakly constrained poses and
iteration caps. Broader contact/equipment, independent changed-action evidence,
longer/diverse source coverage and the full fourteen-dimension goal remain open.
All root-owned jobs are terminal; no shared budget was spent. This checkpoint
completes its local learning/recording integration, not the overall goal.
