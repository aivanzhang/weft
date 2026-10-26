# Source snapshot: baseten-reasoning

- Source: https://docs.baseten.co/inference/model-apis/reasoning
- Retrieved: 2026-10-26 03:00:25 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `a65fa37997438c7ed84c0e0cf63de6a7d40f1f24eecea569f06fa90c37b4717d`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

Model APIs

# Reasoning

Copy pageCopy page

Control reasoning for reasoning-capable models

Copy pageCopy page

Baseten tunes each hosted model for inference performance, so its reasoning behavior can differ from the model provider’s default behavior. Some models generate reasoning tokens before producing a final answer. For models that support reasoning controls, use `reasoning_effort` to adjust how much they reason. Set `BASETEN_API_KEY` to your Baseten API key. For the Python and JavaScript examples, install the [OpenAI SDK](https://docs.baseten.co/inference/model-apis/overview#use-the-openai-sdk).

## 

[​](https://docs.baseten.co/inference/model-apis/reasoning#control-reasoning)

Control reasoning

Reasoning is on by default for most models. Set `reasoning_effort` to `none` to request a direct answer, though some models still reason at a lower depth with `none`; [Set reasoning depth](https://docs.baseten.co/inference/model-apis/reasoning#set-reasoning-depth) shows the depth each value reaches on each model.

| Model | Slug | Reasoning | Control |
|----|----|----|----|
| Beam | `reflection/Beam-1.0` | Always on | None |
| DeepSeek V4 Pro 0813 | `deepseek-ai/DeepSeek-V4-Pro-0813` | On by default | `reasoning_effort` |
| DeepSeek V4 Flash 0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` | On by default | `reasoning_effort` |
| DeepSeek V4.1 Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | On by default | `reasoning_effort` |
| DeepSeek V4.1 Flash 1017 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | On by default | `reasoning_effort` |
| DeepSeek V4.1 Flash 1017 Fast | `deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast` | On by default | `reasoning_effort` |
| DeepSeek V4.1 Flash Fast | `deepseek-ai/DeepSeek-V4.1-Flash-Fast` | On by default | `reasoning_effort` |
| GLM 5.2 | `zai-org/GLM-5.2` | Off by default | `chat_template_args` |
| GLM 5.2 Fast | `zai-org/GLM-5.2-Fast` | On by default | `reasoning_effort` |
| GLM 5.3 | `zai-org/GLM-5.3` | On by default | `reasoning_effort` |
| GLM 5.3 Fast | `zai-org/GLM-5.3-Fast` | On by default | `reasoning_effort` |
| GLM 5.3 Flash | `zai-org/GLM-5.3-Flash` | On by default | `reasoning_effort` |
| Kimi K3 | `moonshotai/Kimi-K3` | On by default | `reasoning_effort` |
| Muse Spark Open | `meta-models/muse-spark-open` | Always on | None |
| Nemotron Super 3.5 | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` | Off by default | `reasoning_effort` |
| Nemotron Ultra | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` | Off by default | `chat_template_args` |
| OpenAI GPT 120B | `openai/gpt-oss-120b` | On by default | `reasoning_effort` |

Models not listed here don’t support reasoning in Chat Completions.

Beam and Muse Spark Open always reason. Beam accepts `none` and `enable_thinking: false` but keeps reasoning on; use `minimal` for its lightest budget. To enable Nemotron Super 3.5, set `reasoning_effort` to `high`. GLM 5.3 accepts `chat_template_args.enable_thinking: false`, but responses can still include `reasoning_content`.

Enable reasoning on GLM 5.2:

- Python

- JavaScript

- cURL

Set `chat_template_args.enable_thinking` to `true` to enable reasoning:

enable_thinking.py

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="zai-org/GLM-5.2",
        messages=[{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        extra_body={"chat_template_args": {"enable_thinking": True}},
    )

    print(response.choices[0].message.content)

Set `chat_template_args.enable_thinking` to `true` to enable reasoning:

enable_thinking.mjs

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "zai-org/GLM-5.2",
        messages: [{ role: "user", content: "Is 91 a prime number? Answer in one sentence." }],
        chat_template_args: { enable_thinking: true },
    });

    console.log(response.choices[0].message.content);

Set `chat_template_args.enable_thinking` to `true` to enable reasoning:

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "zai-org/GLM-5.2",
        "messages": [{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        "chat_template_args": {"enable_thinking": true}
      }'

### 

[​](https://docs.baseten.co/inference/model-apis/reasoning#set-reasoning-depth)

Set reasoning depth

Reasoning depth describes how much reasoning a model performs before producing its final answer. A higher setting can use more reasoning tokens and take longer to respond. Set the top-level `reasoning_effort` parameter to request a depth. Models can map several values to the same level; the effective depth column shows how each listed value behaves. A `(default)` label identifies the effort level used when you omit `reasoning_effort`.

| Model | Supported values | Effective depth |
|----|----|----|
| Beam | `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | As sent |
| DeepSeek V4 Pro 0813 | `none`, `low`, `high`, `max` | As sent |
| DeepSeek V4 Flash 0731 | `none`, `minimal`, `low`, `medium`, `high` (default), `xhigh`, `max` | `minimal` runs at `low`, `medium` at `high`, and `xhigh` at `max`; other values as sent |
| DeepSeek V4.1 Flash | `none`, `low`, `high` (default), `max` | `low` runs at the default depth; other values as sent |
| DeepSeek V4.1 Flash Fast | `none`, `low`, `high` (default), `max` | `low` runs at the default depth; other values as sent |
| DeepSeek V4.1 Flash 1017 | `none`, `low`, `medium`, `high` (default), `max` | As sent |
| DeepSeek V4.1 Flash 1017 Fast | `none`, `low`, `medium`, `high` (default), `max` | As sent |
| GLM 5.2 | `none`, `low`, `high`, `max` | As sent |
| GLM 5.2 Fast | `none`, `low`, `high`, `max` | As sent |
| GLM 5.3 | `none`, `low`, `high` (default), `max` | `none` runs at `low` |
| GLM 5.3 Fast | `none`, `low`, `high` (default), `max` | `none` runs at `low` |
| GLM 5.3 Flash | `none`, `low`, `high` (default), `max` | `none` runs at `low` |
| Kimi K3 | `none`, `low`, `high`, `max` (default) | As sent |
| Muse Spark Open | `low`, `medium`, `high`, `xhigh` (default) | `none` and `minimal` run at `low`; other values as sent |
| Nemotron Super 3.5 | `none` (default), `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | `none` turns reasoning off; every other value runs at one depth, `high` |
| Nemotron Ultra | `none`, `high` | `none` turns reasoning off; `high` turns it on |
| OpenAI GPT 120B | `none`, `minimal`, `low`, `medium` (default), `high`, `xhigh`, `max` | `none` can still reason; other values as sent |

`As sent` means the model uses the effort level you request. `none` turns reasoning off on every model whose row lists it, unless the effective depth column says otherwise. On the GLM 5.3 models it runs at `low`. OpenAI GPT 120B can return `reasoning_content` with `none`. Beam and Muse Spark Open reason regardless.

- Python

- JavaScript

- cURL

Set `reasoning_effort` to `high` to request high reasoning effort:

reasoning_effort.py

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        messages=[{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        reasoning_effort="high",
    )

    print(response.choices[0].message.content)

Set `reasoning_effort` to `high` to request high reasoning effort:

reasoning_effort.mjs

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "deepseek-ai/DeepSeek-V4-Flash-0731",
        messages: [{ role: "user", content: "Is 91 a prime number? Answer in one sentence." }],
        reasoning_effort: "high",
    });

    console.log(response.choices[0].message.content);

Set `reasoning_effort` to `high` to request high reasoning effort:

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        "reasoning_effort": "high"
      }'

## 

[​](https://docs.baseten.co/inference/model-apis/reasoning#parse-the-response)

Parse the response

The message object returns reasoning in `reasoning_content` and the final answer in `content`.

- Python

- JavaScript

- cURL

Read `reasoning_content` for the reasoning and `content` for the final answer:

parse_reasoning.py

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="zai-org/GLM-5.2",
        messages=[{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        extra_body={"chat_template_args": {"enable_thinking": True}},
    )

    message = response.choices[0].message
    print("Reasoning:", message.reasoning_content)
    print("Answer:", message.content)

Read `reasoning_content` for the reasoning and `content` for the final answer:

parse_reasoning.mjs

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "zai-org/GLM-5.2",
        messages: [{ role: "user", content: "Is 91 a prime number? Answer in one sentence." }],
        chat_template_args: { enable_thinking: true },
    });

    const message = response.choices[0].message;
    console.log("Reasoning:", message.reasoning_content);
    console.log("Answer:", message.content);

Read `reasoning_content` for the reasoning and `content` for the final answer:

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "zai-org/GLM-5.2",
        "messages": [{"role": "user", "content": "Is 91 a prime number? Answer in one sentence."}],
        "chat_template_args": {"enable_thinking": true}
      }' | jq '.choices[0].message | {reasoning: .reasoning_content, answer: .content}'

The response body contains both fields on the assistant message:

Response

    {
      "choices": [
        {
          "message": {
            "role": "assistant",
            "reasoning_content": "The user is asking whether 91 is a prime number... 91 = 7 × 13, so it is not prime...",
            "content": "No, 91 is not a prime number because it can be factored as $7 \\times 13$."
          }
        }
      ],
      "usage": {
        "prompt_tokens": 21,
        "completion_tokens": 203,
        "total_tokens": 224
      }
    }

Baseten includes reasoning tokens in `completion_tokens` and counts them toward your total usage and billing.

## 

[​](https://docs.baseten.co/inference/model-apis/reasoning#model-specific-behavior)

Model-specific behavior

### 

[​](https://docs.baseten.co/inference/model-apis/reasoning#beam-conversation-history)

Beam conversation history

Beam requires `reasoning_content` on prior assistant messages. Preserve it alongside `content` when constructing a multi-turn request. Omitting it returns HTTP 400. With an [OpenAI SDK](https://docs.baseten.co/reference/inference-api/chat-completions) installed and `BASETEN_API_KEY` set, send the first response’s reasoning back in the next request.

- Python

- JavaScript

**To continue a Beam conversation**:

beam_history.py

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    messages = [{"role": "user", "content": "What is 6 times 7? Reply with the number only."}]
    response = client.chat.completions.create(
        model="reflection/Beam-1.0",
        messages=messages,
        reasoning_effort="minimal",
        max_tokens=1024,
    )

    assistant = response.choices[0].message
    messages.append({
        "role": "assistant",
        "content": assistant.content,
        "reasoning_content": assistant.reasoning_content,
    })
    messages.append({"role": "user", "content": "And 6 times 8? Reply with the number only."})

    response = client.chat.completions.create(
        model="reflection/Beam-1.0",
        messages=messages,
        reasoning_effort="minimal",
        max_tokens=1024,
    )
    print(response.choices[0].message.content)

Output

    48

**To continue a Beam conversation**:

beam_history.mjs

    import OpenAI from "openai";

    const client = new OpenAI({
      baseURL: "https://inference.baseten.co/v1",
      apiKey: process.env.BASETEN_API_KEY,
    });

    const messages = [{ role: "user", content: "What is 6 times 7? Reply with the number only." }];
    const first = await client.chat.completions.create({
      model: "reflection/Beam-1.0",
      messages,
      reasoning_effort: "minimal",
      max_tokens: 1024,
    });

    const assistant = first.choices[0].message;
    messages.push({
      role: "assistant",
      content: assistant.content,
      reasoning_content: assistant.reasoning_content,
    });
    messages.push({ role: "user", content: "And 6 times 8? Reply with the number only." });

    const second = await client.chat.completions.create({
      model: "reflection/Beam-1.0",
      messages,
      reasoning_effort: "minimal",
      max_tokens: 1024,
    });
    console.log(second.choices[0].message.content);

Output

    48

If your client can’t preserve reasoning, set `chat_template_kwargs: {"allow_missing_thinking": true}`. In the OpenAI Python SDK, pass it through `extra_body={"chat_template_kwargs": {"allow_missing_thinking": True}}`. This bypasses the missing-history check; preserving reasoning is the recommended path. Beam returns reasoning in `message.reasoning_content` and counts it in `usage.completion_tokens_details.reasoning_tokens`.

### 

[​](https://docs.baseten.co/inference/model-apis/reasoning#reasoning-token-budget)

Reasoning token budget

`max_tokens` caps the whole completion, reasoning tokens included. `reasoning.max_tokens` caps reasoning tokens alone: once the model reaches that many reasoning tokens, it stops reasoning and spends the rest of `max_tokens` on its final answer. Set `max_tokens` above the reasoning budget so the model has room to answer. Only DeepSeek V4 Flash 0731 enforces `reasoning.max_tokens`. It stops reasoning at 4,096 tokens by default, including at `reasoning_effort: max`. Raise the budget when a problem needs more reasoning room:

- Python

- JavaScript

- cURL

Pass `reasoning` through `extra_body` since it extends the standard OpenAI API:

reasoning_budget.py

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        messages=[{"role": "user", "content": "What is the sum of the first 100 prime numbers?"}],
        extra_body={"reasoning": {"max_tokens": 20000}},
        max_tokens=24000,
    )

    print(response.choices[0].message.content)

Include `reasoning` directly in the request options:

reasoning_budget.js

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "deepseek-ai/DeepSeek-V4-Flash-0731",
        messages: [{ role: "user", content: "What is the sum of the first 100 prime numbers?" }],
        reasoning: { max_tokens: 20000 },
        max_tokens: 24000,
    });

    console.log(response.choices[0].message.content);

Include `reasoning` in the JSON request body:

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": "What is the sum of the first 100 prime numbers?"}],
        "reasoning": {"max_tokens": 20000},
        "max_tokens": 24000
      }'

## 

[​](https://docs.baseten.co/inference/model-apis/reasoning#next-steps)

Next steps

For more information about custom API endpoints and additional request parameters, see the [OpenAI Python SDK documentation](https://github.com/openai/openai-python#advanced-usage) and [OpenAI JavaScript SDK documentation](https://github.com/openai/openai-node#advanced-usage).

## Model APIs overview

Review supported models, pricing, and features.

## Structured outputs

Constrain reasoning model output to a JSON schema.

Was this page helpful?

Yes

No

[Previous](https://docs.baseten.co/inference/model-apis/decision-models)[](https://docs.baseten.co/inference/model-apis/vision)

Vision

Send images and videos alongside text to vision-capable models

Next

⌘I

## On this page

Back to top

- [Control reasoning](https://docs.baseten.co/inference/model-apis/reasoning#control-reasoning)
  - [Set reasoning depth](https://docs.baseten.co/inference/model-apis/reasoning#set-reasoning-depth)
- [Parse the response](https://docs.baseten.co/inference/model-apis/reasoning#parse-the-response)
- [Model-specific behavior](https://docs.baseten.co/inference/model-apis/reasoning#model-specific-behavior)
  - [Beam conversation history](https://docs.baseten.co/inference/model-apis/reasoning#beam-conversation-history)
  - [Reasoning token budget](https://docs.baseten.co/inference/model-apis/reasoning#reasoning-token-budget)
- [Next steps](https://docs.baseten.co/inference/model-apis/reasoning#next-steps)
