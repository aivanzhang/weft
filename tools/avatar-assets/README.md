# Avatar Studio

Create and inspect rigged 3D characters, motion previews, and portraits for Weft.
Studio handles asset creation; the [Weft sandbox](../../sandbox/README.md) handles
simulation, interaction, and recording.

## Getting started

The tested environment is **Python 3.14 on macOS arm64**. Use a separate Python
environment because the authoring dependencies differ from the simulator:

```sh
cd tools/avatar-assets
python3.14 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Reuse an existing environment if it is already configured. Native authoring
requires Blender 4.5.13, available at `.tools/Blender.app` or through an explicit
compatible executable. The read-only gallery does not require Blender.

From the repository root, start the interface you need:

```sh
./tools/avatar-assets/run studio --port 8766
./tools/avatar-assets/run gallery --port 8767
```

Open the printed address. Studio manages creation jobs; the gallery previews
saved assets without provider calls. These ports leave the Weft sandbox's
port 8765 available. Selected sample assets are included in the repository;
complete development renders are not required for a fresh installation.

## Create an asset

**Save request → Prepare plan → Review estimate → Start creation → Inspect results**

Describe the character, its appearance, and the actions it should perform.
For independently controllable characters, choose **Two separate assets** and
provide a request for each. A combined mesh is not two independent characters.

Saving a request is local. Planning and generation use server-side API credentials
and require configured spending limits. Review both the generated media and
validation results: a successful import does not establish correct animation.

### Prepare from the command line

This example saves a request and estimate without making provider calls:

```sh
./tools/avatar-assets/run generate \
  --prompt 'A friendly emerald dragon with small wings' \
  --action 'Flap both wings while standing still.' \
  --preset preview --max-credits 35 --prepare-only \
  --out outputs/my-dragon-001
```

Use a new output directory for each request. `--plan-only` makes AI calls;
`--execute` can generate, author, and render. Reusing a model avoids new geometry
generation, but planning and motion authoring can still incur AI charges.
Run `./tools/avatar-assets/run generate --help` for available options.

## Capabilities and limitations

| Area | Current support |
| --- | --- |
| Asset creation | Saved-source workflows and selected authored examples produce rigged models and preview media |
| Multiple characters | Separate creation jobs, progress, and downloads; no automatic separation of fused models |
| Simulation integration | Package conversion and compatible-rig import; each asset requires RL-driven motion checks |
| General motion | In development: recent owl and rabbit models generated, but their animations failed rig or action checks |

Arbitrary prompts, seamless loops, synchronized character interactions, and
universal rig compatibility are not yet established. Native hair and browser fur
use different representations. Appearance previews are not physical simulations.
See the [roadmap](TODO.md) for current priorities.

## Data and costs

API keys stay server-side. OpenAI and geometry-provider usage have separate cost
controls; reusing geometry does not eliminate all charges. Keep usage ledgers and
unsettled reservations when resuming work so retries remain within configured limits.

Retain source inputs, generated assets, and validation reports in their original
output directories. Respect source licensing and retain attribution with outputs. Reference images
and videos are visual inputs, not verified 3D rigs.

## Development

From `tools/avatar-assets`, run:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Component tests complement visual review of native and exported motion. Every
candidate must also be checked in Weft before it is described as sandbox-ready.
