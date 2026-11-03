<img src="assets/brand/exports/weft-logo.png" alt="Weft" width="144">

# From video to a world you can test

Weft helps developers turn video observations into interactive simulation and
reinforcement-learning environments. Inspect the evidence, explore a generated
world, test actions, and record the results.

**Observe → reconstruct → interact → evaluate.**

## What you can do

- **Inspect the process.** Review source observations, assumptions, artifacts and failures at each stage.
- **Explore a simulation.** Control supported rigid-object and articulated-body environments, reset them, and compare outcomes.
- **Change the appearance.** Use built-in avatars or import a compatible rig without changing the simulated physics.
- **Keep the evidence.** Export recordings and replay actions; use Meta models for tracking, review and bounded controller tests.

Weft is a research prototype. Reconstruction depends on the input and explicit
assumptions; a playable environment is not a verified physical replica.

## Get started

### Explore the landing page

Requires Node.js 22 or later. No dependency installation is needed.

```sh
npm run dev
```

Open **http://127.0.0.1:3000/**. This static preview runs independently of the sandbox.

### Run the sandbox

Complete the [Python and API setup](sandbox/README.md), then run:

```sh
npm run sandbox
```

Open the authenticated URL printed in your terminal. Upload a video, inspect the
reconstruction, and enter a supported environment. Use **Record & start → Stop & save**
to capture a session. A licensed [rolling-ball sample](data/README.md) is included.

## Documentation

| Guide | Contents |
| --- | --- |
| [Project overview](PROJECT.md) | Architecture, capabilities, limitations and roadmap |
| [Sandbox](sandbox/README.md) | Installation, API configuration and local storage |
| [Avatar Studio](tools/avatar-assets/README.md) | Character authoring, previews and export |
| [Website](website/README.md) | Landing-page development and assets |
| [Data provenance](data/README.md) | Sample attribution and evaluation boundaries |

## Current focus

Reliable reconstruction of unfamiliar videos, equipment contact, compatible
avatar deformation, and measurable improvements under independent evaluation.
See the [capability overview](PROJECT.md#capabilities-and-limitations) before
choosing an input or interpreting a result.
