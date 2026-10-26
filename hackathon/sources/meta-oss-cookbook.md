# Source snapshot: meta-oss-cookbook

- Source: https://github.com/meta-models/meta-oss-cookbook
- Retrieved: 2026-10-26 03:00:24 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `b3a02d482c32ca71d511a58b180f18e4d6881ce2ecb33544ce4da1d1d3e019dc`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

1.  [meta-models](https://github.com/meta-models)
2.  [meta-oss-cookbook](https://github.com/meta-models/meta-oss-cookbook)

## Repository navigation

- 
- [Code](https://github.com/meta-models/meta-oss-cookbook)
- [Issues1 (1)](https://github.com/meta-models/meta-oss-cookbook/issues)
- [Pull requests11 (11)](https://github.com/meta-models/meta-oss-cookbook/pulls)
- [Actions](https://github.com/meta-models/meta-oss-cookbook/actions)
- [Projects](https://github.com/meta-models/meta-oss-cookbook/projects)
- [Security and quality](https://github.com/meta-models/meta-oss-cookbook/security)
- [Insights](https://github.com/meta-models/meta-oss-cookbook/pulse)

More items

[](https://github.com/meta-models/meta-oss-cookbook)

meta-models

**[meta-oss-cookbook](https://github.com/meta-models/meta-oss-cookbook)**

Public

- [Notifications](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-oss-cookbook)You must be signed in to change notification settings
- [Fork29 (29)](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-oss-cookbook)
- [Star110 (110)](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-oss-cookbook)You must be signed in to star a repository

## About

All recipes for oss models from Meta Inc.

### Resources

[Readme](https://github.com/meta-models/meta-oss-cookbook#readme-ov-file)

[Apache-2.0 license](https://github.com/meta-models/meta-oss-cookbook#Apache-2.0-1-ov-file)

### Code of conduct

[Code of conduct](https://github.com/meta-models/meta-oss-cookbook#coc-ov-file)

### Contributing

[Contributing](https://github.com/meta-models/meta-oss-cookbook#contributing-ov-file)

[Activity](https://github.com/meta-models/meta-oss-cookbook/activity)

[Custom properties](https://github.com/meta-models/meta-oss-cookbook/custom-properties)

### Stars

**110** stars

### Watchers

**3** watching

### Forks

[**29** forks](https://github.com/meta-models/meta-oss-cookbook/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fmeta-models%2Fmeta-oss-cookbook&report=meta-models+%28user%29)

main

[Branches](https://github.com/meta-models/meta-oss-cookbook/branches)[Tags](https://github.com/meta-models/meta-oss-cookbook/tags)

[](https://github.com/meta-models/meta-oss-cookbook/branches)[](https://github.com/meta-models/meta-oss-cookbook/tags)

Go to file

Code

Open more actions menu

## Latest commit

 

## History

[123 Commits](https://github.com/meta-models/meta-oss-cookbook/commits/main/)

[](https://github.com/meta-models/meta-oss-cookbook/commits/main/)123 Commits

## Folders and files

[TABLE]

## Repository files navigation

- 
- [README](https://github.com/meta-models/meta-oss-cookbook#)
- [Code of conduct](https://github.com/meta-models/meta-oss-cookbook#)
- [Contributing](https://github.com/meta-models/meta-oss-cookbook#)
- [Apache-2.0 license](https://github.com/meta-models/meta-oss-cookbook#)

More items

# Meta OSS Cookbook

[](https://github.com/meta-models/meta-oss-cookbook#meta-oss-cookbook)

**From your first prompt to a working agent, then make the model your own.**

Practical recipes for building with Meta's open-weight **Muse Spark Open** and **Muse Glimmer** models. Run inference, serve an endpoint, build tool-using agents, and adapt a model with supervised fine-tuning or reinforcement learning. Choose a local setup, your own GPU infrastructure, or a hosted provider.

[Choose your model](https://github.com/meta-models/meta-oss-cookbook#choose-your-model) · [Pick your path](https://github.com/meta-models/meta-oss-cookbook#pick-your-path) · [Explore recipes](https://github.com/meta-models/meta-oss-cookbook#recipes-worth-exploring) · [Contribute](https://github.com/meta-models/meta-oss-cookbook#contributing)

## Choose your model

[](https://github.com/meta-models/meta-oss-cookbook#choose-your-model)

| **Muse Spark Open** | **Muse Glimmer** |
|----|----|
| **Agentic and coding workloads at scale.** A sparse mixture-of-experts model with 816B total parameters and 42B active per token. | **Local-first agents on your own hardware.** A 30B model for tool use, multi-step tasks, and multimodal applications. |
| Text, image, audio, and video input; text output. | Text and image input; text output. Image support depends on the runtime. |
| Self-host on multi-GPU infrastructure, explore the Mac Studio recipe, or use a hosted API. | Start with quantized single-GPU inference or an Apple-silicon runtime; use larger hardware for BF16 and post-training. |
| [**Explore Muse Spark Open →**](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark) | [**Explore Muse Glimmer →**](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer) |
| [Get the model](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/quickstart) · [Hosted providers](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/hosted) | [Quickstart](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/quickstart) · [Hosted providers](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/hosted) |

> **Choose the setup, not just the model.** Hardware needs depend on checkpoint precision, runtime, context length, and whether you are serving or training. Check the linked guide for requirements and verification status before you run.

## Pick your path

[](https://github.com/meta-models/meta-oss-cookbook#pick-your-path)

Both cookbooks take you from setup to an application. Jump straight to the part you need:

| I want to… | Muse Spark Open | Muse Glimmer |
|----|----|----|
| **Get the model running** | [Download and setup](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/quickstart) | [Quickstart](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/quickstart) |
| **Prompt it and call tools** | [Prompting, reasoning, and multimodal input](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/prompting) | [Chat templates, tool calls, and the agent loop](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/agentic-fundamentals) |
| **Serve on my own hardware** | [Inference runtimes and settings](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/inference-server) | [Inference runtimes](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/inference-server) · [Hardware guides](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/platform) |
| **Use an API without provisioning GPUs** | [Hosted providers](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/hosted) | [Hosted providers](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/hosted) |
| **Build an end-to-end application** | [Deployment recipes](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/recipes) | [Agent recipes](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes) |
| **Fine-tune or train with rewards** | [SFT and RL](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/post-training) | [SFT and RL](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/post-training) |

## Recipes worth exploring

[](https://github.com/meta-models/meta-oss-cookbook#recipes-worth-exploring)

### Run beyond a single machine

[](https://github.com/meta-models/meta-oss-cookbook#run-beyond-a-single-machine)

- **[Muse Spark Open on Mac Studio](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/recipes/muse-spark-open-mac):** Serve the NVFP4 checkpoint across Thunderbolt-linked Mac Studio computers with MLX.
- **[Muse Spark Open post-training](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/post-training):** Explore TorchTitan SFT and RL, Miles, TRL, and hosted training options.

### Build a local agent

[](https://github.com/meta-models/meta-oss-cookbook#build-a-local-agent)

- **[Computer use](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/computer-use-web):** Let Muse Glimmer navigate a website through screenshots and mouse actions.
- **[Voice on macOS](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/voice-agent-macos):** Build a local conversation stack with speech recognition, Muse Glimmer generation, and speech synthesis using ExecuTorch on Apple silicon.
- **[Alert triage](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/alert-triage-copilot):** Turn alerts into a three-stage structured pipeline that grades its own output.
- **[Sandboxed tool use](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/sandboxed-agent):** Explore an agent sandbox that blocks attempted credential exfiltration.

### Make the model your own

[](https://github.com/meta-models/meta-oss-cookbook#make-the-model-your-own)

- **[Structured output](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/structured-output)** and **[reasoning control](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/recipes/reasoning-effort):** Enforce a response schema and explore the cost of different reasoning budgets.
- **[Supervised fine-tuning with Axolotl](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/post-training/sft/axolotl):** Adapt Muse Glimmer with QLoRA, LoRA, or full-parameter training.
- **[Data-analyst RL with TitanRL](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/post-training/rl/titanrl/data-analyst):** Train tool-using behavior with a machine-checkable reward.

## What to expect

[](https://github.com/meta-models/meta-oss-cookbook#what-to-expect)

- **Commands you can use, context you can understand.** Setup steps, examples, and explanations live together in the model-specific guides.
- **Hardware and runtime details where they matter.** Follow the recipe's checkpoint, precision, and server requirements. Measurements apply to the documented configuration, not every deployment.
- **A clear distinction between local and hosted.** Local inference can run offline once the required artifacts are cached. Hosted APIs require provider credentials; setup and training may need downloads or online tracking.
- **Room to adapt.** Use the examples as starting points for your own tools, tasks, datasets, and deployments.

The cookbook is growing. Some paths are verified, while others document a pending or unverified setup. Each model folder and runtime guide states its current status; a listing here is not a blanket verification claim.

## Contributing

[](https://github.com/meta-models/meta-oss-cookbook#contributing)

Add a recipe, improve a guide, or share a reproducible fix. See [CONTRIBUTING.md](https://github.com/meta-models/meta-oss-cookbook/blob/main/CONTRIBUTING.md) and the [Code of Conduct](https://github.com/meta-models/meta-oss-cookbook/blob/main/CODE_OF_CONDUCT.md). For Muse Glimmer recipes, start with the [shared recipe template](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/assets/RECIPE_TEMPLATE.md).

## License

[](https://github.com/meta-models/meta-oss-cookbook#license)

See the repository [LICENSE](https://github.com/meta-models/meta-oss-cookbook/blob/main/LICENSE). Model weights have their own terms: [Muse Spark Open's model license](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-spark/README.md#license) and [Muse Glimmer's model card](https://huggingface.co/meta-models/Muse-Glimmer-30B).

## About

All recipes for oss models from Meta Inc.

### Resources

[Readme](https://github.com/meta-models/meta-oss-cookbook#readme-ov-file)

[Apache-2.0 license](https://github.com/meta-models/meta-oss-cookbook#Apache-2.0-1-ov-file)

### Code of conduct

[Code of conduct](https://github.com/meta-models/meta-oss-cookbook#coc-ov-file)

### Contributing

[Contributing](https://github.com/meta-models/meta-oss-cookbook#contributing-ov-file)

[Activity](https://github.com/meta-models/meta-oss-cookbook/activity)

[Custom properties](https://github.com/meta-models/meta-oss-cookbook/custom-properties)

### Stars

**110** stars

### Watchers

**3** watching

### Forks

[**29** forks](https://github.com/meta-models/meta-oss-cookbook/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fmeta-models%2Fmeta-oss-cookbook&report=meta-models+%28user%29)

## Releases

## Packages

## Used by

## Contributors

## Languages
