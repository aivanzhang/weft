# Check an executable environment

`weft qualify` applies one bounded software-check protocol to existing sealed environments. It does not reconstruct another world, train a policy, alter a task, or certify physical accuracy.

Use the existing [frozen sandbox launcher](local-lab.md) to play the current saved environment. Qualification writes a new directory and never records into the user's live lab. The first three-environment check passed; results and limits are below.

## One environment

```sh
.venv-lab/bin/python -m weft qualify \
  --run /path/to/ready-lab-or-sealed-run \
  --out artifacts/runtime-check-01
```

The default baseline turns motors off. `--controller authored` selects an existing authored reference controller only where the original environment supports it. This is **not** evaluation of a learned policy. A selected policy's package is still validated and identified; no stale runtime pins are replaced.

Add `--manifest-sha256 <expected-hash>` to require a previously frozen input identity. For multiple cases, put `manifest_sha256` in each case; single-run settings cannot override a case file.

`--probe-channel` selects a declared continuous-action motor; inspect `weft capabilities --run /path/to/run` for its meaning. Discrete profiles use their explicit action adapter. The changed-action probe starts from the same seed and replaces one command while preserving the other saved commands.

## Several environments

Use `--cases /path/to/cases.json` instead of `--run`. Input paths are relative to the case file. For example:

```json
[
  {"id": "tabletop", "input": "./tabletop-run", "seed": 1729, "controller": "coast", "probe_steps": 6, "probe_channel": 0},
  {"id": "exercise", "input": "./exercise-lab", "seed": 1729, "controller": "authored", "probe_steps": 6, "probe_channel": 0}
]
```

Cases must be distinct sealed inputs. The protocol uses explicit adapters for rolling, planar, spatial-rigid, supported human bodies, shared worlds and human-only shared exercise. It does not guess a human action from a rigid object's left/right commands. It tests only the selected control target; other actors are not automatically qualified.

## Read the result

| Result | Meaning |
| --- | --- |
| Admission | Source seals, runtime identity, scene/task and required policy package validate before execution. |
| Complete interval | Execution reaches the original declared horizon. Early termination remains visible, not a fabricated full-interval pass. |
| Same-seed reset / exact action replay | A fresh environment reproduces the recorded initial state, actions, rewards, terminal flags and states exactly. |
| Changed-action response | A matched simulated command changes reported physical coordinates. This is not evidence of a real intervention. |
| Task result | Original authoritative success/failure metrics, unchanged by this runner. A usable simulator can still fail its task. |
| Physical validation | **Unverified.** Software checks cannot establish unseen geometry, material parameters or real-world transfer. |

`status: completed` means the bounded job finished without an execution failure. A case can still be `not_qualified`, including valid early task success that does not cover the complete interval. The command exits nonzero unless every case is operationally qualified; task success remains separate. Final-step engine failures cannot masquerade as complete intervals: authoritative completion and actual engine-clock coverage must agree. No camera, collision shape, source interval, success threshold, assistance or policy input is changed to obtain a pass.

## Evidence and bounds

The new directory retains a frozen `plan.json`, implementation/dependency hashes, per-case contracts, physical action journals, exact transition hashes, changed-action results, failures and a final manifest. It performs no video decoding, model inference, encoding, browser/server startup or paid calls.

Defaults are 60 seconds, 12,000 total environment steps **including replay**, and 64 MiB of new evidence. Hard ceilings are five cases, 120 seconds and 128 MiB. Requests that cannot cover the declared horizon inside the step budget do not silently shorten it. Native physics calls use cooperative time checks; a stuck native call requires an external process deadline. Source evidence is validated on admission; consumed runtime documents, outer lab metadata and manifest identities are rechecked around execution. An execution failure in one admitted case does not skip independent cases; global resource or integrity failures stop the suite.

Previously exposed videos/seeds remain development evidence. Reuse an equivalent existing receipt where its contract matches; do not rerun an unchanged passing study merely to attach this protocol's name.

## Verified development result

`artifacts/runtime-qualification-01` contains the single frozen three-case study at seed **62031**:

| Environment / controller | Complete interval | Exact replay and action response | Original task |
| --- | --- | --- | --- |
| `tabletop-lab-01` / coast | 300 steps /10 s | Pass | Parking failed |
| `authored-3d-trained-lab` / coast | 360 steps /3 s | Pass | Parking failed |
| `whole-body-visual-lab-01` / authored feedback | 60 steps /2 s | Pass | Tracking passed |

Each changed-action check contains six commands from an identical seeded start. Total **1,458 steps** includes 720 baseline, 720 fresh replay and 18 changed actions. Runtime **0.6815 s**, complete process **0.8893 s**, **2,750,943 bytes /24 files**, no paid calls. The external process deadline was 65 seconds around the 60-second cooperative bound. These are one-run measurements, not sustained UI performance or broad generalization.

Seal: `ddd771ce4d4ff846532cbb584f7955fc4090af408e88ae174a29f1004c86e8e7`. Root independently verified all 23 covered files, 720 transition hashes, 18 saved matched-action comparisons and unchanged consumed inputs without another physics run: `.local/runtime-qualification-root-verification-01.json`. **61 distinct scoped Python cases** pass across the runner and CLI; `.local/runtime-qualification-tests-summary-01.json` lists the receipts.

The current 254-action shared-exercise sandbox was **not rerun or relabeled**. Its separate exact-replay evidence remains valid within its frozen runtime, and equipment-stability task success remains **0/3**. None of these checks validates physical anatomy, materials, unseen geometry or real-world transfer.
