# Authored moving-object tasks

This path adds a supplied placement goal to an existing source-qualified moving world. It keeps the original exploration unchanged. The same body, equipment, floor, clock and simulator are used; task scoring is separate from the old parking diagnostic.

**Current verification:** the real kettlebell clip passes task creation, portal loading, six recorded actions and exact state/reward replay. A bounded training search did not solve the task; its unchanged authored policy also failed both tested changed starts. This is not recovered exercise intent, gripping, validated biomechanics or physical accuracy.

## Try the verified local project

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-kettlebell-flip-01/portal
```

Open **Authored kettlebell hover-place.mp4** (`9c67eaa0f37b42f2ad6b59506e331e2a`), then its sandbox. The task card and controls are ready; recording `session-c8522314ab40` contains the verified six-action example. Its short interval is incomplete, not successful. The original unscored exploration remains a separate project.

The task card's **Before you try** section now explains support requirements without disabling manual controls. It is advice about the saved task, not a feasibility certificate. To inspect the same check from the CLI without training or changing the world:

```sh
.venv-lab/bin/python -m weft moving-equipment-preflight \
  --lab .local/moving-placement-source-acceptance-01/task-child
```

For a simpler object-control example, open **Grayscale ball — authored 3D depth.avi** (`460d1f5ebb964c18821c99b0be8b3fda`) in the same portal. It supports manual and existing learned actions; recording `session-6f9c676e1807` contains three manual and six learned actions with exact replay. This saved world models only 21 of 157 source frames, with authored depth and zero gravity—not recovered tabletop physics. Its visible source note preserves those limits. No new training or reconstruction was performed by importing it; Muse review is unavailable for this imported rigid-world profile.

For the authored kettlebell task:

1. **Video** shows the original clip. **Play captured movement** in the 3D world shows estimated body poses without running physics; equipment there stays at its initial position.
2. In **Sandbox**, select an explicit motor or equipment-force channel. Equipment channels ending in `fx`, `fy` or `fz` push along that world axis. All unselected commands remain zero; there is no automatic balance controller.
3. Click **Start over**, then start recording. Controls pause during preparation; playback starts when recording is ready. Pause to inspect or select a channel, then use **Fine controls → Advance one step** for a controlled trial.
4. **Stop recording** saves and exports the trial. Open its evidence to see replay checks and whether the full task interval was completed. A short experiment is useful even when it does not pass the task.

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

## Check a frozen policy from different starts

The separate evaluator changes the simulated object's starting position and velocity, while keeping the original human start, source reference, task goal and scoring limits. It does not train, change the saved world, or install a controller in the sandbox.

```sh
.venv-lab/bin/python -m weft moving-equipment-evaluate \
  --lab artifacts/placement-task-01 \
  --policy artifacts/placement-learning-01/policy.json \
  --evaluation-seed 4101 --evaluation-seed 4102 \
  --position-jitter 0.025 --velocity-jitter 0.1 \
  --out artifacts/placement-evaluation-01 --max-seconds 90
```

Choose evaluation seeds different from the policy's training seed. The two spread values are explicit authored choices in the world's declared units, not measured physical variation. The command freezes its protocol and policy before exposing any starts, checks actual position/velocity states, and compares zero force, random, authored feedback and the frozen policy from identical starts. Duplicate, nominal, clearance-refused or fallback starts remain reported and are not counted as successful changed-start trials.

Inspect `result.json` for success counts and admitted/refused denominators, `start-*.json` for actual states and complete-interval metrics, and `protocol.json` for exact settings. To repeat the same protocol deliberately, replace the seed/spread flags with `--protocol PATH_TO_PROTOCOL_JSON` and use a new output directory. Repeating it does not create fresh evidence.

This is simulated development evaluation, not a fresh final holdout, physical validation or recording replay. Saved evaluation results include initial states and final metrics, not an action-by-action journal. The existing manual task still resets to its exact original start. Evaluation is bounded to 20,000 aggregate controls, 60 seconds of execution and 90 seconds including the single source-admission check, with no API calls.

The existing example's evaluation is already saved at `.local/moving-support-evaluation-source-01/evaluation/`: two distinct starts were admitted without fallback or exclusion, and every controller scored **0/2**. Reuse those results rather than repeating the same failed experiment. A separate six-hypothesis support screen found no solution within its restricted arm-only search; this does not prove that every possible contact-assisted controller must fail.
