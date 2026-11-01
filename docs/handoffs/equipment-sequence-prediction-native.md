# Two-action native prediction qualification

## Verified result — November 1, 2026, 1:23 PM PST

The new [fixed-action sequence predictor](equipment-sequence-prediction.md)
matches both actual simulator endpoints and every substep contact in one bounded
two-action trial. Before accepting control 49, it forecasts the equipment speed
and spin violation at control 50. This is prediction qualification, **not a
successful anticipatory controller, completed exercise or physical validation**.

Reconstructed the same 48 saved Rollout02 controls, then forecast the fixed pair
`[saved selected49, saved base50]` on one copied simulator state. Both forecasts
match the previously saved single-step predictions (global calls 72/73), compared
by action/state/contact identity rather than counter labels. Applying those same
two actions produces exact full-state and substep-contact matches. Forecasting
leaves live physics, task metrics and supplied actions unchanged.

| Measurement | Control 49 | Control 50 | Original limit |
| --- | ---: | ---: | ---: |
| Forecast endpoint equipment speed | Passing | 0.4798370892 | 0.15 model units/s |
| Forecast endpoint equipment spin | Passing | 2.867588606 | 0.75 rad/s |
| Actual accumulated equipment gate | Passing | Failed: `sam-2` | Unchanged |

Actual accumulated control-50 peaks are **0.5976082792 model units/s** and
**3.516482254 rad/s**, higher than endpoint values. The helper does not forecast
the task accumulator, reward or success. Exact endpoint/contact parity cannot be
substituted for all accumulated substep gates. The 254-control task remains
incomplete and success is false. No optimizer or alternative action ran.

## Bounds and measurements

- One local child, $0 external/recurring cost, zero paid calls/retries/media/
  replay/training/server work. Seed 29, original source/scene/task/scorer/policy.
- 48 prefix controls + two scratch controls + two actual comparison controls:
  **832 native substeps, 54 forwards, 11 models, two data objects, one reset/copy**.
  Attempted and returned native counts match exactly.
- Forecast alone: 32 substeps, two boundary forwards, one scratch allocation/copy;
  0.005479 seconds observed. No closed-loop policy callback or new public mode.
- External elapsed: **0.880425 seconds**. Child high-water RSS:
  **137,707,520 bytes**. Five saved files /512,180 bytes; stdout 2,436 bytes,
  stderr empty. PID 51031 was reaped with exit 0.
- Enforced envelope: 40 CPU/cooperative seconds, 60 external seconds including
  cleanup, 512 MiB sampled/high-water RSS, 4 MiB/eight output files, 2 MiB/result
  and two 1 MiB logs. RSS observation is not hard memory isolation.
- Native execution used the reviewed narrow escalation for the existing child
  `/bin/ps` observer. No network or credentials were supplied to the worker.

## Exact evidence and retained preparation correction

Runtime `.local/equipment-sequence-prediction-native-01/runtime` adds only the
new predictor to the prior 293-file frozen runtime: 294 files /6,371,923 bytes.
Every original runtime/policy/scorer file is unchanged. Runtime manifest:
`a71c264d135b0dce3af67301fad87151413c99c749c5e79722d8ad32818cf5e1`.

Frozen plan02:
`fa9a96faf625ffc91b771c6602e212298330f649b4a27186b61c1536108511eb`.
Worker02:
`23ccd849c2b2136f7513063fecd928ad9aee9e21dc47631f1761174018679723`.
Supervisor02:
`68047f84d17976e905f3222bdf5046c8e0acaca83c4f173d65adb5ae080a263d`.

Plan01/worker01 remain unexecuted. Review found that the new contact observer
ran before returned-native counters were incremented. Plan02 moves those counters
immediately after the native return, preserving successful work even if later
observation raises. One isolated fake-operation check passes; it performs no
native work. Receipt:
`5bf61c9796493454b1c4e437d38c67edbc82fed2c569db5ab5682cf26372c7a6`.
Helper source, runtime, inputs and numerical bounds did not change.

Output: `artifacts/equipment-sequence-prediction-native-01`.

- Seal: `60a9abdff611d2b1728c7ff3d12a3c48446e0dd4d07e4ab4406692765eb3fd53`.
- Result: `227480a6b631c6b4940db9982233d68d27a541cb5370d8953c0381a75f13a94f`.
- Forecast: `6d39762614151ca7cf8ef99d1394f3d1f4247e16e97c2110ea2ada8104eb7e7d`.
- Prefix: `82aa7913128303b665d1c7182c39596909f866e6dc1027bf6a0ed77fd68771b9`.
- Root saved-only audit: `.local/equipment-sequence-prediction-native-01/saved-audit-01.json`,
  `956eac8fde3e6d95f2066581d8fcafc6ead258c85915efc3d9af1b230596024a`.

The audit independently hashes all four sealed members and compares exact
forecasts, actual states/contacts, counts and task failures without another run.
All evidence is exposed development data, not a fresh final holdout.

Completed command; do not rerun unchanged evidence:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles PYTHONDONTWRITEBYTECODE=1 \
  .venv-lab/bin/python -B -P \
  .local/equipment-sequence-prediction-native-01/supervisor-02.py \
  fa9a96faf625ffc91b771c6602e212298330f649b4a27186b61c1536108511eb
```

## Next safe action

Develop a bounded anticipatory action proposal that uses both predicted
boundaries, keeps original root/action/trust/gate limits and independently checks
actual accumulated metrics. A fixed future action is not a recomputed policy
rollout. Do not restart the unchanged control-50 derivative-scale experiments.
Root owns native execution/Git; no live process or lock remains from this trial.
