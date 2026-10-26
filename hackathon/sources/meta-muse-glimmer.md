# Source snapshot: meta-muse-glimmer

- Source: https://dev.meta.ai/models/muse-glimmer
- Retrieved: 2026-10-26 03:00:22 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `eb11e511aa6a1392930e5466aa54fd37d3f141177a76ffeab1f4455a83429b45`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

[Meta](https://dev.meta.ai/)

[Documentation](https://dev.meta.ai/docs)

Products

Muse Code

[](https://dev.meta.ai/products/muse-code/)

Overview

Meta Model API

[](https://dev.meta.ai/products/meta-model-api/)

Overview⁣

[](https://dev.meta.ai/api/auth/login)

API Login

Meta Connectors

[](https://dev.meta.ai/products/connectors)

Meta AI Connectors

[](https://muse.ai/platform)

Muse Connectors ↗

Close popover

Models

Muse

[](https://dev.meta.ai/models/muse-wonder/)

Muse Wonder

[](https://dev.meta.ai/models/muse-spark/)

Muse Spark 1.3

[](https://dev.meta.ai/models/muse-spark-open/)

Muse Spark Open

[](https://dev.meta.ai/models/muse-glimmer/)

Muse Glimmer

[](https://dev.meta.ai/models/muse-image/)

Muse Image

[](https://dev.meta.ai/models/muse-voice-transcribe/)

Muse Voice Transcribe

Segment Anything (SAM)

[](https://dev.meta.ai/models/sam-3-1/)

SAM 3.1

Close popover

Resources

Documentation

[](https://dev.meta.ai/docs/)

Model API docs ↗

Learn

[](https://dev.meta.ai/docs/cookbook)

Cookbooks ↗

[](https://dev.meta.ai/resources/videos/)

Videos

[](https://dev.meta.ai/resources/blog/)

Blog

Close popover

Community

Github

[](https://github.com/meta-models)

Meta Models ↗

[](https://github.com/meta-llama/llama)

Llama ↗

Hugging Face

[](https://huggingface.co/meta-models)

Meta Models ↗⁣

Close popover

[Contact Sales](https://dev.meta.ai/contact-sales?referrer_path=%2Fmodels%2Fmuse-glimmer)

[Login](https://dev.meta.ai/api/auth/login)

# Muse Glimmer

An open model built for always-on local agents. 30B parameters, licensed under Apache 2.0 and runs on a single GPU — tuned for tool use, long tasks, and failure recovery.

[Download model](https://huggingface.co/meta-models/)

[Documentation](https://dev.meta.ai/docs/muse-glimmer/)

## Meet Muse Glimmer

Always-on agents

Built for agents that don't stop with reliable tool-calling, persistent state across restarts, and self-managed memory across hours-long sessions.

Optimized for local deployments

It's small enough to run on a single consumer GPU or Mac, enabling use cases that range from local agents to local coding.

## Benchmarks

Muse Glimmer is trained to deliver competitive agentic and coding performance, with multimodal perception built in.

Benchmark

Muse Glimmer-30BHigh reasoning

Gemma4-31BThinking mode

Qwen3.6-27BThinking mode

General agentic

MCP Atlas

75.5

54.2

62.5

DeepSearch QA

74.6

61.7

71.1

τ³-Banking

23.5

15.1

16.7

WildClawBench

47.6

37.6

43.2

GDPval-AA

953

811

1141

GAIA2

43.3

36.4

40.0

SkillsBenchWith skills

44.3

32.4

46.6

OSWorld-Verified

65.9

58.5

75.6

Agentic coding

SWE-Bench Pro

51.2

36.9

50.2

SWE-Bench Verified

76.0

66.6

77.2

TerminalBench 2.1

51.7

43.4

60.7

SciCode

43.6

43.4

39.8

Multimodal

Charxiv Reasoning

78.8

77.7

78.4

ScreenSpot Pro

75.4

75.9

76.1

OmniDocBench v1.5

75.8

72.5

77.8

MMMU Pro

74

73

75

Safety

CI Memories

Violation (↓): 26.4Coverage: 64.8

Violation (↓): 12.1Coverage: 53.0

Violation (↓): 53.4Coverage: 66.9

Siren AgentDojo

Attack Success Rate (↓): 28.4Utility: 94.2

Attack Success Rate (↓): 25.6Utility: 90.8

Attack Success Rate (↓): 40.3Utility: 92.7

General capabilities and reasoning

IFBench

77.0

76.0

70.8

AIME 2026

94.7

89.2

94.1

GPQA Diamond

83.5

85.7

84.2

Humanity's Last ExamText · No tools

22.0

23.6

23.1

AA-LCR

80.0

68.3

73.3

Beam 128K

65.1

58.2

63.0

For the full model card, please [see the Muse Glimmer page on Hugging Face](https://huggingface.co/meta-models/Muse-Glimmer-30B).

For more details about Muse Glimmer evaluations, [read the methodology report](https://research.meta.ai/static/muse-glimmer-methodology).

Meta is committed to promoting safe and fair use of its tools and features, including Muse Glimmer. [This Usage Policy (“Policy”)](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/USAGE_POLICY.md) applies to the access or use of Muse Glimmer.

## Where to run Muse Glimmer

[](https://huggingface.co/meta-models/Muse-Glimmer-30B)

huggingface logo image

HuggingFace

Run Muse Glimmer with transformers, llama.cpp, vLLM, Colab, and other popular platforms.

[](https://ollama.com/library/muse-glimmer)

ollama image

Ollama

Run Muse Glimmer on your own hardware in one simple, seamless CLI command.

[](https://unsloth.ai/docs/models/muse-glimmer)

unsloth image

Unsloth

Run, fine-tune, and deploy Muse Glimmer from one unified local UI interface with the open-source Unsloth Desktop app.

[](https://lmstudio.ai/models/muse-glimmer)

lm studio image

LM Studio

LM Studio Bionic is the easiest way to run Muse Glimmer for agentic tasks, locally on your computer.

[](https://fireworks.ai/models/fireworks/muse-glimmer-30b)

fireworks image

Fireworks

Fireworks offers production-grade Muse Glimmer deployments with precise reasoning controls and effortless scaling for unpredictable agent traffic.

[](https://www.together.ai/models/muse-glimmer)

together.ai image

together.AI

Run Muse Glimmer on Together Serverless Inference for long-running agentic workflows.

[](https://openrouter.ai/meta/muse-glimmer-30b)

openrouter image

OpenRouter

Run Muse Glimmer via a single unified API that routes across multiple inference providers for cost, speed, and availability.

## Build with Muse Glimmer

[Download model](https://huggingface.co/meta-models/)

[View documentation](https://dev.meta.ai/docs/muse-glimmer/)

## Cookbooks and quickstarts

[Read documentation](https://dev.meta.ai/docs/muse-glimmer)

[](https://dev.meta.ai/docs/muse-glimmer/prompting)

Prompting guideGet the most out of Muse Glimmer by using the correct chat template and following these prompting best practices.

Learn more

[](https://dev.meta.ai/docs/muse-glimmer/quantization)

QuantizationRun on smaller hardware by reducing the precision of model weights.

Learn more

[](https://dev.meta.ai/docs/muse-glimmer/spec-decode)

Speculative decodingSpeed up inference by using a smaller, faster draft model to propose candidate tokens that the full model verifies.

Learn more

[](https://dev.meta.ai/docs/muse-glimmer/vllm)

vLLMDeploy with vLLM for high-throughput, low-latency inference with an OpenAI-compatible API endpoint.

Learn more

[](https://dev.meta.ai/docs/muse-glimmer/llama-cpp)

llama.cppRun on your machine with llama.cpp, a C/C++ inference engine that supports CPU, mixed CPU/GPU and full GPU execution.

Learn more

[](https://dev.meta.ai/docs/muse-glimmer/executorch)

ExecuTorchRun on mobile phones, tablets, and edge devices with ExecuTorch, Meta's on-device inference framework.

Learn more

Meta blue bg

Stay up-to-date

#### Our latest updates delivered to your inbox

Subscribe to our newsletter to keep up with the latest AI updates, releases and more.

[Sign up](https://dev.meta.ai/subscribe)

1.  Models
2.  Muse
3.  Muse Glimmer

Meta

[](https://www.facebook.com/AIatMeta/)

[](https://twitter.com/aiatmeta/)

[](https://www.youtube.com/@aiatmeta)

[](https://www.linkedin.com/showcase/aiatmeta)

Products

[Muse Code](https://dev.meta.ai/products/muse-code/)[Meta Model API](https://dev.meta.ai/products/meta-model-api/)[Meta AI Connectors](https://dev.meta.ai/products/connectors/)[Muse Connectors](https://muse.ai/platform)

Models

[Muse Wonder](https://dev.meta.ai/models/muse-wonder/)[Muse Spark 1.3](https://dev.meta.ai/models/muse-spark/)[Muse Spark 1.2](https://dev.meta.ai/models/muse-spark-1-2/)[Muse Spark 1.1](https://dev.meta.ai/models/muse-spark-1-1/)[Muse Spark Open](https://dev.meta.ai/models/muse-spark-open/)[Muse Glimmer](https://dev.meta.ai/models/muse-glimmer/)[Muse Image](https://dev.meta.ai/models/muse-image/)[Muse Voice Transcribe](https://dev.meta.ai/models/muse-voice-transcribe/)[SAM 3.1](https://dev.meta.ai/models/sam-3-1/)

Documentation

[Meta Model API Docs](https://dev.meta.ai/docs/)[Muse Glimmer Docs](https://dev.meta.ai/docs/muse-glimmer/)[Llama Docs](https://dev.meta.ai/llama/)

Resources

[Cookbook](https://dev.meta.ai/docs/cookbook/)[Blog](https://dev.meta.ai/resources/blog/)[Videos](https://dev.meta.ai/resources/videos/)[Help Center](https://dev.meta.ai/help/)[Contact Support](https://dev.meta.ai/support/)

Community

[Meta-Models Github](https://github.com/meta-models)[Llama GitHub](https://github.com/meta-llama/llama)[Hugging Face](https://huggingface.co/meta-models)

Terms & policies

[Terms of Service](https://www.facebook.com/policies_center/)[Privacy Policy](https://www.facebook.com/privacy/policy/)[Cookie Policy](https://www.facebook.com/privacy/policies/cookies/)

Products \

[Muse Code](https://dev.meta.ai/products/muse-code/)

[Meta Model API](https://dev.meta.ai/products/meta-model-api/)

[Meta AI Connectors](https://dev.meta.ai/products/connectors/)

[Muse Connectors](https://muse.ai/platform)

Models \

[Muse Wonder](https://dev.meta.ai/models/muse-wonder/)

[Muse Spark 1.3](https://dev.meta.ai/models/muse-spark/)

[Muse Spark 1.2](https://dev.meta.ai/models/muse-spark-1-2/)

[Muse Spark 1.1](https://dev.meta.ai/models/muse-spark-1-1/)

[Muse Spark Open](https://dev.meta.ai/models/muse-spark-open/)

[Muse Glimmer](https://dev.meta.ai/models/muse-glimmer/)

[Muse Image](https://dev.meta.ai/models/muse-image/)

[Muse Voice Transcribe](https://dev.meta.ai/models/muse-voice-transcribe/)

[SAM 3.1](https://dev.meta.ai/models/sam-3-1/)

Documentation \

[Meta Model API Docs](https://dev.meta.ai/docs/)

[Muse Spark Open Docs](https://dev.meta.ai/docs/overview#muse-spark-open)

[Muse Glimmer Docs](https://dev.meta.ai/docs/muse-glimmer/)

[Llama Docs](https://dev.meta.ai/llama/)

Resources \

[Cookbook](https://dev.meta.ai/docs/cookbook/)

[Blog](https://dev.meta.ai/resources/blog/)

[Videos](https://dev.meta.ai/resources/videos/)

[Help Center](https://dev.meta.ai/help/)

[Contact Support](https://dev.meta.ai/support/)

Community \

[Meta-Models Github](https://github.com/meta-models)

[Llama GitHub](https://github.com/meta-llama/llama)

[Hugging Face](https://huggingface.co/meta-models)

Terms & policies \

[Terms of Service](https://www.facebook.com/policies_center/)

[Privacy Policy](https://www.facebook.com/privacy/policy/)

[Cookie Policy](https://www.facebook.com/privacy/policies/cookies/)

## Please confirm your identity

For added security, you need to confirm your identity. You will be taken to your single sign-on provider, where you can enter your password.

Cancel

Continue
