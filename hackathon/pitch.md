# Weft — hackathon story and copy

**One demonstration. A world of practice.**

Copy prepared October 27, 2026 for the Meta Global AI Developer Hackathon. The vision and prototype copy are separated so future capabilities are not presented as shipped results. Nothing in this document has been submitted or published.

[Product goal](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/goal.md) · [First-version scope](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/v1.md) · [Current evidence](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/TODO.md) · [Submission requirements](analysis/requirements.md)

---

## The headline

### Turn a video into a world an agent can learn in.

Weft is building a path from a physical demonstration to an interactive training environment—with objects that have state, relationships grounded in physics, and results that can be tested.

**Watch once. Test what happens next.**

## The one-line pitch

Weft is a video-to-environment project: turn one physical demonstration into an executable world where AI agents can act, make mistakes, and learn.

## The problem

A video shows what happened once. Learning a task requires trying what could happen next.

Today, making that leap often means building a simulator: assembling objects, defining interactions, configuring physics, writing task rules, and checking whether the result behaves as intended. That work stands between a useful demonstration and a place where an agent can practice. [Related research: Video2Policy](https://arxiv.org/html/2502.09886v1)

Weft's ambition is to make the demonstration the starting point for that environment.

## The product vision

Show Weft a physical task. Its target is to identify the objects that matter, map how they interact, and construct a shared world with explicit state, controls, and constraints.

Then move beyond the recording. Change an action. Try another starting position. Train an agent. Inspect the assumptions that determine the outcome.

The intended result is more than a visual reconstruction: a working environment, accompanied by evidence of what it captures and what remains uncertain.

### Understand the interaction

Identify what moves, what supports it, what constrains it, and where the evidence is incomplete.

### Build a world that responds

Give the scene persistent state and meaningful controls, so an agent can explore more than the demonstrated sequence.

### Earn confidence through testing

Compare recorded and simulated behavior. Challenge the physics. Reject broken repairs. Measure whether an agent actually learns.

## Why it matters

For robotics developers and researchers, the potential benefit is a shorter path from observing a task to experimenting with it. Less environment authoring. More repeatable trials. A clearer account of where a simulator can—and cannot—be trusted.

We will measure that benefit through manual effort saved, time and cost per usable environment, and independently evaluated task performance. Those are targets to demonstrate, not savings already established by the prototype.

## What makes the approach worth pursuing

**Physics is part of the description.** The goal is to explain interactions through support, contact, forces, materials, and constraints—not just label objects in a frame.

**Uncertainty stays visible.** A video does not reveal every mass, hidden surface, or possible action. Weft separates observations from estimates and assumptions.

**The evaluation has its own authority.** A repair cannot rewrite the success conditions. A beautiful replay cannot cancel a failed physical check.

**Learning is the test of usefulness.** The intended endpoint is an agent practicing a meaningful task, not a clip playing back.

These are Weft's design commitments, not claims of being the first project to explore video-to-simulation.

## Short spoken pitch — vision

> A video shows an action once. An AI agent needs somewhere to try it again—and try it differently.
>
> We're building Weft to turn a physical demonstration into that training environment. It will map the objects and their physical relationships, construct an interactive simulator, and define a task an agent can learn.
>
> The critical question is whether the world works beyond the recording. We test its behavior, expose its assumptions, and keep the evaluator outside the repair loop.
>
> Our first prototype connects those stages on a restricted rolling-ball example. The next milestone is a shared world with several interacting objects.
>
> Weft: one demonstration, a world of practice.

## Prototype description — accurate today

Weft is an early physical-video-to-RL prototype. Its current local pipeline follows an isolated yellow ball in a real video, builds a restricted horizontal-motion simulator, tests bounded parameter repairs, and trains an agent on an authored parking task.

Each run preserves the source, measurements, world versions, evaluation records, and learning results for inspection. The prototype establishes a reproducible foundation for the broader goal: generating multi-object training environments from video.

It does not yet recover arbitrary 3D scenes or task intent, and its added controls are not inferred from the recording. Real-world policy transfer and live Meta-model use have not been verified.

## Meta's intended role

**Planned copy, to be updated from real runs before submission:**

Meta's multimodal and reasoning models are intended to turn visual evidence into inspectable object relationships, physical hypotheses, task proposals, and bounded repairs. Their contribution should be visible in the world that gets built and the changes that pass independent tests.

The current local pipeline does not establish that integration. Replace this section with the actual model IDs, decisions, traces, and measured outcomes once live use has been verified. Development-tool usage alone is not the proposed in-product contribution.

## Three-minute demo story

This is the target storyboard. Show only capabilities backed by a preserved run; label replay and time compression.

| Time | What the audience sees | What it establishes |
| --- | --- | --- |
| 0:00–0:20 | A short physical demonstration and the task it represents | A concrete problem and intended user |
| 0:20–0:50 | The objects, relationships, and physical assumptions extracted from the video | What the system understood and what remains uncertain |
| 0:50–1:20 | The generated world replaying the interaction, then responding to a changed action | An executable environment, not prerecorded animation |
| 1:20–1:55 | A discrepancy, proposed correction, and an independent accept/reject decision | A visible model contribution and trustworthy checks |
| 1:55–2:30 | Agent training and baseline comparisons with counts and conditions | Whether the environment supports useful learning |
| 2:30–3:00 | Evidence, limits, new hackathon work, and practical benefit | A result the audience can understand and inspect |

## Fit with the hackathon

Intended track: **Indie Developer**. Intended theme: **Push the Frontier**.

| Official criterion | Weight | The story Weft must substantiate |
| --- | --- | --- |
| Impact | 30% | Less work between a real demonstration and a useful training environment |
| Meta AI tooling | 25% | Essential model-driven interpretation or repairs, backed by actual execution |
| Demo | 25% | A clear transformation from video to interaction to learning |
| Technical execution | 20% | Physics-grounded state, protected evaluation, bounded repair, and reproducibility |

Weights and theme come from the [official participant guide](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details). The proposed alignment is our interpretation, not an organizer assessment.

## Claim discipline before submission

Use this as an editorial check, not public-facing copy:

- Do not say “any video,” “learns all physics,” “no human input,” or “real-world transfer” without matching evidence.
- Name the supported scene family and disclose authored controls, task goals, and calibration.
- Use measured benefit and model-use claims only after the relevant runs exist; do not invent savings or benchmark scores.
- Keep existing Loom infrastructure separate from newly built hackathon work.
- “WorldBench-informed evaluation” describes a design influence, not an official benchmark result or endorsement.
- Recheck current submission requirements before any future authorized submission. This draft does not authorize publishing, submitting, or changing repository visibility.

---

**A demonstration is a beginning. Weft's goal is to make it a place to learn.**
