# Authored moving-object tasks

This path adds a supplied placement goal to an existing source-qualified moving world. It keeps the original exploration unchanged. The same body, equipment, floor, clock and simulator are used; task scoring is separate from the old parking diagnostic.

**Current verification:** the real kettlebell clip passes task creation, portal loading, six recorded actions and exact state/reward replay. A bounded training search completed but did not solve the task: unchanged authored feedback remained best and failed peak body-error and final placement-hold checks. This is not recovered exercise intent, gripping, validated biomechanics or physical accuracy.

## Try the verified local project

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-kettlebell-flip-01/portal
```

Open **Authored kettlebell hover-place.mp4** (`9c67eaa0f37b42f2ad6b59506e331e2a`), then its sandbox. The task card and controls are ready; recording `session-c8522314ab40` contains the verified six-action example. Its short interval is incomplete, not successful. The original unscored exploration remains a separate project.

Learning evidence is retained at `.local/moving-placement-source-acceptance-01/learning/`. No learned controller is installed in the sandbox. The original object-force limit supplies at most 6 world-units/s² upward acceleration against 9.81 gravity; unsupported hovering cannot be achieved with direct object forces alone. Human or other support would be needed. This is a conditional force-budget observation, not proof that all contact-assisted policies are impossible.

## Create a task child

Choose an admitted exploration lab and pin its `run/manifest.json` SHA-256. Supply a task JSON file with these fields:

```json
{
  "target_object_id": "raw-0",
  "target_position_world": [0.5, 0.5, 0.5],
  "position_tolerance_world": 0.03,
  "speed_tolerance_world_s": 0.1,
  "hold_duration_s": 0.5
}
```

The coordinates above illustrate the format, not a measured or recommended target. Choose coordinates in the selected world's declared units and bounds. A video does not uniquely determine a task.

```sh
.venv-lab/bin/python -m weft moving-equipment-task \
  --lab PATH_TO_EXPLORATION_LAB \
  --parent-manifest-sha256 EXACT_PARENT_MANIFEST_SHA256 \
  --task placement.json --out artifacts/placement-task-01
```

The output must be new. It retains the exact parent evidence, authored target, fixed scorer and runtime identities. No source video inference or provider request runs.

Passing requires the complete original interval, unchanged body/floor limits, and the specified final consecutive position-and-speed hold. An early stop is incomplete, not a task pass. Manual controls include explicit joint, root and equipment forces; virtual object guidance is not a recovered grasp.

## Open and record

With the portal stopped, import the new child:

```sh
.venv-lab/bin/python -m weft portal-import \
  --lab artifacts/placement-task-01 --root .local/portal
.venv-lab/bin/python -m weft portal --root .local/portal
```

Open the imported project and its sandbox. The task card shows the target, tolerances, elapsed interval, hold time and body/floor limits. The original exploration remains a separate project.

Record from reset. While recording is prepared in the background, controls pause; they resume only after the recording is ready. Export checks full action/state/reward replay against the frozen task. A trusted export receipt retained during the open lab session can be checked again without repeating physics. After reopening, a saved receipt without that trusted pin is only a self-consistent claim, not newly verified replay. Source authenticity and physical accuracy are separate from replay consistency.

To re-execute a saved task recording, retain the recording-contract hash from its trusted export, then run:

```sh
.venv-lab/bin/python -m weft moving-equipment-replay PATH_TO_SESSION \
  --contract-sha256 TRUSTED_RECORDING_CONTRACT_SHA256 --max-seconds 60
```

This checks exact state and task reward again; an incomplete recording still cannot pass the full task.

## Train a bounded feedback policy

```sh
.venv-lab/bin/python -m weft moving-equipment-train \
  --lab artifacts/placement-task-01 --out artifacts/placement-learning-01 \
  --generations 1 --population 4 --seed 1729 --max-seconds 90
```

The learner searches explicit feedback gains and compares with zero-force, random and authored controls. It retains failed trials, freezes the selected policy, and checks its nominal replay. It does not change the task, source world or open sandbox, or automatically enable a learned controller there.

This first task version resets to the exact original start. Different random seeds do not create different physical starts. Its comparisons are development results, **not independent changed-start validation**; an unchanged zero-force or authored winner is not presented as newly improved behavior.

Limits: 20,000 aggregate control steps, at most 60 seconds of training, at most 90 seconds including source admission, and 64 MiB of learning output. The child lab retains its 256 MiB cap; portal storage remains 2,000,000,000 bytes. There are no provider calls or recurring cloud resources in these commands.
