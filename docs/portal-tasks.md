# Create and train a task in the portal

Moving-body-and-object projects now expose two separate actions. Neither sends media to an AI provider or changes the original world.

1. Open a ready moving exploration and choose **Configure task**. The saved world is checked in the background before its object list and coordinate bounds appear.
2. Choose the object and enter a target position, distance/speed tolerances and final hold duration. These are your simulated task assumptions, not intent recovered from the video. Confirm them and create the task.
3. Open the new task's sandbox to interact and record. The original exploration remains separate; manual forces are not recovered grasp or muscle activation.
4. On the task project, open its training settings. Choose bounded search settings and, optionally, tests from different starting positions and velocities. Starting training creates a separate report project; it does not replace sandbox controls.
5. Inspect the report and downloadable artifacts. Zero input, authored feedback, random input and the selected frozen policy are all shown, including failures. Changed-start results show admitted and refused starts, not just successful trials.

“Complete” means the requested search and report finished. It does **not** mean the task succeeded. A failed search also does not prove that no controller can solve the task. These are development comparisons in an assumed simulation, not final-holdout or physical validation.

## Local launch

Open the verified local task and its training report:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-task-source-01/portal
```

Choose **Authored task** to open its manual sandbox, or **Training and evaluation** to inspect the completed report. This separately authored floor-placement task remains unsuccessful: every controller failed both tested changed starts. The original hover task is unchanged. The new task/report workflow passed in 45.760 seconds without API calls; this is workflow verification, not learned task success.

For new uploads, use a separate root. The user starts and manages the server:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-new-videos
```

Task controls require an admitted moving exploration, not merely an uploaded video or body preview. Existing playable projects and their limitations are described in [Moving-object tasks](moving-placement.md). The older kettlebell portal root is near its storage limit; use a new root instead of deleting evidence or increasing limits to fit another job.

## Limits and retained evidence

- One existing supervised worker; local task jobs receive no provider keys. Creation and training are explicit separate requests.
- Each task job reserves up to 650 MB within the portal's 2 GB limit. Cancellation and restart preserve partial results without automatic retry.
- Training: 1–4 generations, 4–12 candidates, at most 60 seconds. Optional evaluation: 2–16 unique seeds distinct from the training seed, at most 60 seconds. Combined requested work must fit 20,000 controls.
- The task retains the complete native interval, original body/floor/force limits, immutable scorer and exact source/world identities. The evaluation protocol is saved before training.
- Candidate-generation reports, frozen policy, nominal comparisons, actual changed-start states, refusals and manifests remain inspectable. A policy download is not installation or proof of improvement.

The [implementation handoff](handoffs/portal-tasks.md) records current verification and remaining product work.
