<img src="assets/brand/exports/weft-logo.png" alt="Weft" width="144">

# From video to a world you can test

[![Licence: AGPL v3](https://img.shields.io/badge/Licence-AGPL%20v3-blue.svg)](LICENSE)

**Licensed under the GNU Affero General Public License v3.0.** If you run a
modified Weft as a network service, AGPL section 13 requires you to offer your
users the source of that modified version. See [LICENSE](LICENSE), and
[NOTICE](NOTICE) for third-party material the AGPL does **not** cover — reference
datasets, demo clips, fonts and provider outputs keep their own licences.

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

### Run the complete local demo

After the [Python setup](sandbox/README.md#requirements), start both the landing
page and interactive sandbox in one terminal:

```sh
npm run demo
```

Open `/demo` to watch finished examples; use the header’s Access popover to build your own. The landing page is at
`http://127.0.0.1:3000/`; the walkthrough stays at `http://127.0.0.1:3000/demo`. **Ctrl+C** stops
both servers and preserves saved runs. Stop individually running servers first.
Page edits refresh automatically. Python and example-catalog edits restart the
sandbox when idle; active jobs and loaded live worlds are preserved. See the
[hot refresh notes](sandbox/README.md#hot-refresh) for restart limits and opt-out.

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

## Licence

Weft is licensed under the **GNU Affero General Public License, version 3 or
later** (`AGPL-3.0-or-later`). The full text is in [LICENSE](LICENSE).

The AGPL is a copyleft licence with a network clause. Section 13 means that if you
modify Weft and let others use it over a network, you must offer those users the
source of your modified version — not only when you distribute binaries.

[NOTICE](NOTICE) lists what the AGPL does **not** cover: reference datasets under
`data/`, the demo clips, the Inter font, the MediaPipe pose model and saved
provider outputs all keep their own licences and attribution. Check it before
redistributing anything from this repository; some of that material is not
redistributable at all.
