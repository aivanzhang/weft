# Source snapshot: meta-api-fundamentals

- Source: https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals
- Retrieved: 2026-10-26 03:00:24 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `737bcd05f98c730807ba66f60aa3728654bbcbbaf37a58ebb40f4a1e13020558`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

1.  [meta-models](https://github.com/meta-models)
2.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook)

## Repository navigation

- 
- [Code](https://github.com/meta-models/meta-model-cookbook)
- [Issues17 (17)](https://github.com/meta-models/meta-model-cookbook/issues)
- [Pull requests26 (26)](https://github.com/meta-models/meta-model-cookbook/pulls)
- [Actions](https://github.com/meta-models/meta-model-cookbook/actions)
- [Projects](https://github.com/meta-models/meta-model-cookbook/projects)
- [Security and quality](https://github.com/meta-models/meta-model-cookbook/security)
- [Insights](https://github.com/meta-models/meta-model-cookbook/pulse)

More items

[](https://github.com/meta-models/meta-model-cookbook)

## FilesExpand file tree

main

## Breadcrumbs

1.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook/tree/main)

/

# 01_api_fundamentals

/

Copy path

## Directory actions

## More options

More options

## Directory actions

## More options

More options

## Latest commit

 

## History

[History](https://github.com/meta-models/meta-model-cookbook/commits/main/01_api_fundamentals)

[](https://github.com/meta-models/meta-model-cookbook/commits/main/01_api_fundamentals)History

main

## Breadcrumbs

1.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook/tree/main)

/

# 01_api_fundamentals

/

Copy path

Top

## Folders and files

[TABLE]

## [README.md](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#readme)

Outline

# API fundamentals

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#api-fundamentals)

Ten universal API-primitive recipes for the Meta Model API (Muse Spark). These are the entry points: each one proves a piece of the API works and gives you a copy-paste starting point. They use no CLI — just the OpenAI Python SDK pointed at the Model API.

Start with [01 — Chat completions](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/01_chat_completions.ipynb). It sets up the client, helpers, and structure that every other recipe reuses, so the concepts in 02–10 build on it.

## Canonical configuration

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#canonical-configuration)

Every recipe uses exactly this setup. Copy it verbatim; do not vary the base URL, model id, or key variable.

| Setting         | Value                                                |
|-----------------|------------------------------------------------------|
| SDK             | OpenAI Python SDK (`from openai import OpenAI`)      |
| Base URL        | `https://api.meta.ai/v1`                             |
| Model           | `muse-spark-1.3`                                     |
| API key env var | `MODEL_API_KEY` (format `LLM|{numeric_id}|{secret}`) |

import os\
\
from openai import OpenAI\
\
\# The OpenAI SDK does not auto-read MODEL_API_KEY, so pass it explicitly.\
client = OpenAI(\
    base_url="https://api.meta.ai/v1",\
    api_key=os.environ\["MODEL_API_KEY"\],\
)

Note

The official Model API client package is out of date — do not use it. Use the OpenAI SDK.

## The 10 recipes

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#the-10-recipes)

| \# | Recipe | API shape | Demonstrates | Grounded in |
|----|----|----|----|----|
| [01](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/01_chat_completions.ipynb) | Chat completions | Chat Completions | OpenAI code works by changing `base_url` + model | [features/chat-completion](https://dev.meta.ai/docs/features/chat-completion) |
| [02](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/02_streaming.ipynb) | Streaming responses | Chat Completions (`stream=True`) | render tokens as they generate; final usage chunk | [features/chat-completion](https://dev.meta.ai/docs/features/chat-completion) (Streaming) |
| [03](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/03_tool_calling.ipynb) | Tool / function calling | Chat Completions (`function` tools) | detect `tool_calls`, run the execute→feed-back loop | [features/tool-calling](https://dev.meta.ai/docs/features/tool-calling) |
| [04](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/04_structured_output.ipynb) | Structured output (JSON mode) | Chat Completions (`response_format`) | schema-guaranteed JSON; Pydantic parse | [features/structured-output](https://dev.meta.ai/docs/features/structured-output) |
| [05](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/05_prompt_caching.ipynb) | Prompt caching | Chat Completions | reuse a stable prefix; read `cached_tokens` | [features/prompt-caching](https://dev.meta.ai/docs/features/prompt-caching) |
| [06](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/06_reasoning_tokens.ipynb) | Reasoning / thinking tokens | Chat Completions (`reasoning_effort`) + Responses | control effort; read `reasoning_content`; replay | [features/chat-completion](https://dev.meta.ai/docs/features/chat-completion), [features/responses](https://dev.meta.ai/docs/features/responses) |
| [07](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/07_vision_input.ipynb) | Vision input | Chat Completions (image content) | send images (URL + base64); structured analysis | [features/image-understanding](https://dev.meta.ai/docs/features/image-understanding) |
| [08](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/08_long_context.ipynb) | Long context (1M) | Chat Completions | structure repo-level context; 1,048,576-token window | [getting-started/models](https://dev.meta.ai/docs/getting-started/models), [getting-started/overview](https://dev.meta.ai/docs/getting-started/overview) |
| [09](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/09_error_handling.ipynb) | Error handling & retry | Chat Completions | backoff with jitter; never retry `400` | [getting-started/error-handling](https://dev.meta.ai/docs/getting-started/error-handling), [getting-started/pricing-rate-limits](https://dev.meta.ai/docs/getting-started/pricing-rate-limits) |
| [10](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/10_search_grounding.ipynb) | Search grounding | Responses (`web_search` tool) | ground answers in live web results; map `url_citation` annotations to footnotes | [features/search-grounding](https://dev.meta.ai/docs/features/search-grounding) |

Work them in order: each recipe is self-contained, but the concepts build on recipe 01.

## Prerequisites

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#prerequisites)

- Python 3.10+.
- A Model API developer account and an API key from [dev.meta.ai](https://dev.meta.ai/).
- The dependencies in this repo's `pyproject.toml` (`openai`, `papermill`, `notebook`, `ipykernel`), installed with `uv sync`.

Set your key once per shell:

export MODEL_API_KEY="LLM\|...\|..."

In Colab, add `MODEL_API_KEY` to the secrets manager (the key icon in the left sidebar). The setup cell falls back to it automatically, so the same notebook runs headless and in Colab.

## Running the recipes

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#running-the-recipes)

These commands run from the repo's `private/` directory.

**Run a recipe headless (verifies every cell against the live API):**

MODEL_API_KEY="LLM\|...\|..." uv run papermill \\\
  api_fundamentals/01_chat_completions.ipynb /tmp/out.ipynb

A clean run exits 0 with no error outputs. The run stays well within the per-team free-tier limits (2,400 requests/min, 800,000 tokens/min).

**Open a recipe interactively:**

uv run jupyter notebook --no-browser --port 8888 --ip 0.0.0.0

**Lint and format (this tree uses ruff, not BLACK/arc):**

uv tool run ruff check api_fundamentals/\
uv tool run ruff format --check api_fundamentals/

## Pricing and rate limits

[](https://github.com/meta-models/meta-model-cookbook/tree/main/01_api_fundamentals#pricing-and-rate-limits)

These recipes show token **usage** (`response.usage`) rather than dollar costs. For current per-token pricing, see [getting-started/pricing-rate-limits](https://dev.meta.ai/docs/getting-started/pricing-rate-limits). Watch the `x-ratelimit-remaining-*` response headers to throttle before you hit a `429`.
