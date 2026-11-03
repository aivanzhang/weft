# Weft — hackathon story and copy

Updated November 3, 2026 (PST). This is a local preparation document, not a submitted entry.
[Demo notes and submission checklist](analysis/demo-plan.md) · [Working walkthrough](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/hackathon-demo.md)

## Positioning

**From a video to a world you can test.**

Weft helps simulation and reinforcement-learning developers turn video evidence into conditional, interactive environments. Inspect the assumptions, change an action, compare the result and record a reproducible trial.

Reinforcement learning means learning actions through rewards inside an environment. Weft exposes actions, state, rewards and resets; a rendered pose replay alone is not an RL environment.

## The problem and intended user

A video records one outcome. A developer needs an executable environment to test other actions. Constructing that environment also means choosing geometry, contacts, controls and task rules that a video may not uniquely reveal.

Weft brings source evidence, explicit assumptions, simulation and tests into one workflow. The intended benefit is less manual environment setup and clearer debugging. No measured labor-saving or physical-transfer claim is made without a matched study.

## What the demo actually shows

- A local upload portal with source-bound processing artifacts and explicit failure states.
- Supported body or object reconstruction paths, with observed, inferred and authored inputs kept distinct.
- Interactive simulation, reset, control and action/state recording.
- Compatible avatar appearance driven by simulated state, not an unrelated animation.
- Real Meta construction reviews and separately bounded controller-choice experiments.
- Reproducible evidence, including rejected proposals and failed task checks.

Use the current [demo notes](analysis/demo-plan.md) for actual acceptance results. A processing success is not automatically task success, and neither proves accurate physical reconstruction.

## Meta's role

Meta SAM 3.1 supplies object tracking in the verified rigid-object studies. Muse Spark 1.3 reviews source and world assumptions. A separate controller-choice request can select bounded authored tests or abstain; the independent scorer determines whether a tested choice improves the simulation. The model cannot rewrite the scorer or install arbitrary code.

OpenAI scene routing remains separately disclosed. Avatar artwork and Muse model inference are different features. Do not imply that all reconstruction or avatar creation is performed by Meta.

## Spoken opening

“A video shows what happened once. An agent needs an environment where it can try something different. Weft makes the video the starting point: inspect the evidence, build a conditional simulation, change an action and test what happens. When the evidence is insufficient, that uncertainty stays visible.”

## Theme and eligibility boundary

Intended theme: **Push the Frontier**. Intended track: **Indie Developer**, subject to entrant eligibility. Weft is a developer tool for simulation/RL, not an exercise coach. Exercise footage is a motion-reconstruction example, not medical advice, exercise scoring or validated biomechanics.

The captured [official guide](sources/participant-guide.md) lists sports analyzers/coaches among anti-project examples. This distinction must be true of the product, not merely its wording.

## Claims to avoid

- “Any video,” recovered anatomy, measured muscles, universal contacts or real-world transfer.
- Calling authored reference controls a learned policy, or animation playback interactive physics.
- Calling a local controller-score improvement a measured physical correction.
- Presenting exposed development videos as untouched holdouts.
- Describing prior Loom infrastructure or third-party models as newly built Weft work.
- Implying a private local build is already a judge-accessible hosted product.

Current rubric: impact 30%, Meta tooling 25%, demo 25%, technical execution 20%, per the October 26 official guide capture. Prioritize a visible working result and an inspectable model contribution over feature count.
