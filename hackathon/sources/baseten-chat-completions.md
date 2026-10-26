# Source snapshot: baseten-chat-completions

- Source: https://docs.baseten.co/reference/inference-api/chat-completions
- Retrieved: 2026-10-26 03:00:25 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `628a9d0978e9a7a4d39b5d277fafb716d05d00b69fd4e20eefa7bd716e96aacd`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

Model APIs

# Chat Completions

Copy pageCopy page

Create chat completions using Baseten Model APIs, an OpenAI-compatible endpoint for managed LLMs.

Copy pageCopy page

POST

/

v1

/

chat

/

completions

Try it

Python

Python

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ.get("BASETEN_API_KEY"),
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(response.choices[0].message.content)

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Hello!" },
        ],
    });

    console.log(response.choices[0].message.content);

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ]
      }'

    <?php

    $curl = curl_init();

    curl_setopt_array($curl, [
      CURLOPT_URL => "https://inference.baseten.co/v1/chat/completions",
      CURLOPT_RETURNTRANSFER => true,
      CURLOPT_ENCODING => "",
      CURLOPT_MAXREDIRS => 10,
      CURLOPT_TIMEOUT => 30,
      CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
      CURLOPT_CUSTOMREQUEST => "POST",
      CURLOPT_POSTFIELDS => json_encode([
        'model' => 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
        'messages' => [
            [
                    'role' => 'system',
                    'content' => 'You are a helpful assistant.'
            ],
            [
                    'role' => 'user',
                    'content' => 'Hello!'
            ]
        ]
      ]),
      CURLOPT_HTTPHEADER => [
        "Authorization: Bearer <token>",
        "Content-Type: application/json"
      ],
    ]);

    $response = curl_exec($curl);
    $err = curl_error($curl);

    curl_close($curl);

    if ($err) {
      echo "cURL Error #:" . $err;
    } else {
      echo $response;
    }

    package main

    import (
        "fmt"
        "strings"
        "net/http"
        "io"
    )

    func main() {

        url := "https://inference.baseten.co/v1/chat/completions"

        payload := strings.NewReader("{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}")

        req, _ := http.NewRequest("POST", url, payload)

        req.Header.Add("Authorization", "Bearer <token>")
        req.Header.Add("Content-Type", "application/json")

        res, _ := http.DefaultClient.Do(req)

        defer res.Body.Close()
        body, _ := io.ReadAll(res.Body)

        fmt.Println(string(body))

    }

    HttpResponse<String> response = Unirest.post("https://inference.baseten.co/v1/chat/completions")
      .header("Authorization", "Bearer <token>")
      .header("Content-Type", "application/json")
      .body("{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}")
      .asString();

    require 'uri'
    require 'net/http'

    url = URI("https://inference.baseten.co/v1/chat/completions")

    http = Net::HTTP.new(url.host, url.port)
    http.use_ssl = true

    request = Net::HTTP::Post.new(url)
    request["Authorization"] = 'Bearer <token>'
    request["Content-Type"] = 'application/json'
    request.body = "{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}"

    response = http.request(request)
    puts response.read_body

200

    {
      "model": "<string>",
      "choices": [
        {
          "index": 123,
          "delta": {
            "role": null,
            "content": null,
            "tool_calls": [
              {
                "index": 123,
                "function": {
                  "arguments": "<string>",
                  "name": null
                },
                "id": "<string>",
                "type": "function"
              }
            ]
          },
          "logprobs": null,
          "finish_reason": null,
          "stop_reason": 123
        }
      ],
      "id": "<string>",
      "object": "chat.completion.chunk",
      "created": 123,
      "usage": null
    }

Download the [OpenAPI schema](https://docs.baseten.co/reference/inference-api/llm-openapi-spec.json) for code generation and client libraries.

[Model APIs](https://app.baseten.co/model-apis/create) provide instant access to high-performance open-source LLMs through an OpenAI-compatible endpoint.

## 

[​](https://docs.baseten.co/reference/inference-api/chat-completions#replace-openai-with-baseten)

Replace OpenAI with Baseten

Switching from OpenAI to Baseten takes two changes: the base URL and API key.

- Python

- JavaScript

- cURL

To switch to Baseten with the Python SDK, change `base_url` and `api_key` when initializing the client:

chat_completions.py

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages=[{"role": "user", "content": "Hello!"}],
    )

To switch to Baseten with the JavaScript SDK, change `baseURL` and `apiKey` when initializing the client:

chat_completions.js

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages: [{ role: "user", content: "Hello!" }],
    });

To call Baseten with cURL, send a POST request to `inference.baseten.co` with your API key:

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'

Deploy a [Model API](https://app.baseten.co/model-apis/create) to get started.

For detailed usage guides including structured outputs and tool calling, see [Using Model APIs](https://docs.baseten.co/inference/model-apis/overview). For web search and fetch tools that Baseten executes within the request, see [Server-side tool execution](https://docs.baseten.co/reference/inference-api/server-side-tool-execution).

OpenAI-compatible models you deploy yourself also support the chat completions format at their own base URL: `https://model-{model_id}.api.baseten.co/v1/chat/completions`. See [deployed model endpoints](https://docs.baseten.co/reference/inference-api/overview#deployed-model-endpoints) for URL formats.

#### Authorizations

[​](https://docs.baseten.co/reference/inference-api/chat-completions#authorization-authorization)

Authorization

string

header

required

Pass your Baseten API key. Clients automatically send `Authorization: Bearer <key>`. Direct callers can also use `Authorization: Api-Key <key>`; both schemes are accepted.

#### Body

application/json

Request body for creating a chat completion.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-messages)

messages

object\[\]

required

A list of messages representing the conversation history. Supports roles: `system`, `user`, `assistant`, and `tool`.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-model)

model

string

required

The model slug to use for completion, such as `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Find available models at [Model APIs](https://app.baseten.co/model-apis/create).

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-frequency-penalty)

frequency_penalty

number

default:0

Penalizes tokens based on how frequently they appear in the text so far. Positive values decrease repetition. Support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-logit-bias)

logit_bias

Logit Bias · object

A map of token IDs to bias values (-100 to 100). Use this to increase or decrease the likelihood of specific tokens appearing in the output.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-logprobs)

logprobs

boolean

default:false

If `true`, returns log probabilities of the output tokens. Log probability support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-top-logprobs)

top_logprobs

integer

default:0

Number of most likely tokens to return at each position (0-20). Requires `logprobs: true`. Log probability support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-max-tokens)

max_tokens

integer

default:4096

Maximum number of tokens to generate. If your request input plus `max_tokens` exceeds the model's context length, `max_tokens` is truncated. If your request exceeds the context length by more than 16k tokens or if `max_tokens` signals no preference, context reservation is throttled to 49512 tokens. Higher `max_tokens` values slightly deprioritize request scheduling.

Required range: `1 <= x <= 262144`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-n)

n

integer

default:1

Number of completions to generate. Only `1` is supported.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-presence-penalty)

presence_penalty

number

default:0

Penalizes tokens based on whether they have appeared in the text so far. Positive values encourage the model to discuss new topics. Support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-response-format)

response_format

ResponseFormatText · object

Plain text response format.

- ResponseFormatText

- ResponseFormatJson

- ResponseFormatJsonObject

- ResponseFormatGrammar

- ResponseFormatStructuralTag

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-seed)

seed

integer

Random seed for deterministic generation. Determinism is not guaranteed across different hardware or model versions.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-stop-one-of-0)

stop

stringstring\[\]stringstring\[\]

Up to 32 sequences where the API stops generating further tokens. Can be a string or array of strings.

Required string length: `1 - 1000`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-stream)

stream

boolean

default:false

If `true`, responses are streamed back as server-sent events (SSE) as they are generated.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-stream-options)

stream_options

StreamOptions · object

Options for streaming responses. Set `include_usage: true` to receive token usage statistics in the final chunk. Requires `stream: true` to be set to non-null.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-temperature)

temperature

number

Controls randomness in the output. Lower values like 0.2 produce more focused and deterministic responses. Higher values like 1.5 produce more creative and varied output.

Required range: `0 <= x <= 4`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-top-p)

top_p

number

default:1

Nucleus sampling: only consider tokens with cumulative probability up to this value. Lower values like 0.1 produce more focused output.

Required range: `x <= 1`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-tools)

tools

ChatCompletionToolsParam · object\[\]

A list of tools (functions) the model may call. Each tool should have a `type: "function"` and a `function` object with `name`, `description`, and `parameters`.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-tool-choice-one-of-0)

tool_choice

enum\<string\>ChatCompletionNamedToolChoiceParam · objectenum\<string\>ChatCompletionNamedToolChoiceParam · object

Controls which tool (if any) the model calls.

- `none`: Never call a tool.
- `auto`: Model decides whether to call a tool.
- `required`: Model must call at least one tool.
- `{"type": "function", "function": {"name": "..."}}`: Call a specific function.

Available options:

`none`,

`required`,

`auto`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-parallel-tool-calls)

parallel_tool_calls

boolean

default:true

If `true`, the model can call multiple tools in a single response.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-user)

user

string

A unique identifier for the end-user, useful for tracking and abuse detection.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-best-of)

best_of

integer

Number of candidate sequences to generate and return the best from. Only a value of 1 is supported.

Required range: `1 <= x <= 1`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-top-k)

top_k

integer

default:50

Limits token selection to the top K most probable tokens at each step. Lower values like 10 produce more focused output. Set to -1 to disable.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-top-p-min)

top_p_min

number

default:0

Minimum value for dynamic `top_p`. When set, `top_p` dynamically adjusts but does not go below this value.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-min-p)

min_p

number

default:0

Minimum probability threshold for token selection. Filters out tokens with probability below `min_p * max_probability`.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-reasoning-one-of-0)

reasoning

Reasoning · object \| null

Per-request reasoning controls. `max_tokens` caps reasoning tokens independently of the top-level `max_tokens` and overrides the model's default reasoning cap. Enforced on DeepSeek V4 Flash 0731, where reasoning is otherwise capped at 4096 tokens per request; other models ignore it. Applies when reasoning is enabled.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-reasoning-effort-one-of-0)

reasoning_effort

enum\<string\> \| null

How much reasoning a reasoning-capable model performs before answering. The gateway accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`; models without reasoning support ignore the parameter. See the Model APIs reasoning page for per-model values.

Available options:

`none`,

`minimal`,

`low`,

`medium`,

`high`,

`xhigh`,

`max`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-repetition-penalty)

repetition_penalty

number

default:1

Multiplicative penalty for repeated tokens. Values greater than 1.0 discourage repetition, values less than 1.0 encourage it.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-length-penalty)

length_penalty

number

default:1

Exponential penalty applied to sequence length during beam search. Values greater than 1.0 favor longer sequences.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-early-stopping)

early_stopping

boolean

default:false

If `true`, stops generation when at least `n` complete candidates are found.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-bad-one-of-0)

bad

stringstring\[\]stringstring\[\]

Words or phrases to avoid in the output. Support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-bad-token-ids)

bad_token_ids

integer\[\]

Token IDs to avoid in the output. Support varies by model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-stop-token-ids)

stop_token_ids

integer\[\]

List of token IDs that cause generation to stop when encountered.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-include-stop-str-in-output)

include_stop_str_in_output

boolean

default:false

If `true`, includes the matched stop string in the output.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-ignore-eos)

ignore_eos

boolean

default:false

If `true`, continues generating past the end-of-sequence token.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-min-tokens)

min_tokens

integer

default:0

Minimum number of tokens to generate before stopping. Useful for ensuring responses are not too short.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-skip-special-tokens)

skip_special_tokens

boolean

default:true

If `true`, removes special tokens from the generated output.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-spaces-between-special-tokens)

spaces_between_special_tokens

boolean

default:true

If `true`, adds spaces between special tokens in the output.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-truncate-prompt-tokens)

truncate_prompt_tokens

integer

If set, truncates the prompt to this many tokens. Useful for handling inputs that may exceed context limits.

Required range: `x >= 1`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-echo)

echo

boolean

default:false

If `true` and the last message role matches the generation role, prepends that message to the output.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-add-generation-prompt)

add_generation_prompt

boolean

default:true

If `true`, adds the generation prompt from the chat template, such as `<|assistant|>`. Set to `false` for completion-style generation.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-add-special-tokens)

add_special_tokens

boolean

default:false

If `true`, adds special tokens like BOS to the prompt beyond what the chat template adds. For most models, the chat template handles special tokens, so this should be `false`.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-documents)

documents

Documents · object\[\]

A list of documents for RAG (retrieval-augmented generation). Each document is a dict with string keys and values that the model can reference.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-chat-template)

chat_template

string

A custom Jinja template for formatting the conversation. If not provided, uses the model's default template.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-chat-template-args)

chat_template_args

Chat Template Args · object

Additional arguments to pass to the chat template renderer.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#body-disaggregated-params)

disaggregated_params

DisaggregatedParams · object

Advanced parameters for disaggregated serving. Used internally for distributed inference.

Show child attributes

#### Response

200

application/json

Successful response

A chat completion response returned by the model.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-model)

model

string

required

The model used for the completion.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-choices)

choices

ChatCompletionResponseStreamChoice · object\[\]

required

A list of chat completion choices.

Show child attributes

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-id)

id

string

A unique identifier for the chat completion.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-object)

object

string

default:chat.completion.chunk

The object type, always `chat.completion` or `chat.completion.chunk` for streaming.

Allowed value: `"chat.completion.chunk"`

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-created)

created

integer

The Unix timestamp (in seconds) of when the completion was created.

[​](https://docs.baseten.co/reference/inference-api/chat-completions#response-usage)

usage

UsageInfo · object

Token usage statistics for the request. Only present when streaming with `stream_options.include_usage: true`.

Show child attributes

Was this page helpful?

Yes

No

[Previous](https://docs.baseten.co/api-reference/regional/wake-a-regional-environment-of-a-model)[](https://docs.baseten.co/reference/inference-api/messages)

Messages

Create Anthropic Messages API requests against Baseten Model APIs.

Next

⌘I

Python

Python

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ.get("BASETEN_API_KEY"),
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(response.choices[0].message.content)

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "Hello!" },
        ],
    });

    console.log(response.choices[0].message.content);

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ]
      }'

    <?php

    $curl = curl_init();

    curl_setopt_array($curl, [
      CURLOPT_URL => "https://inference.baseten.co/v1/chat/completions",
      CURLOPT_RETURNTRANSFER => true,
      CURLOPT_ENCODING => "",
      CURLOPT_MAXREDIRS => 10,
      CURLOPT_TIMEOUT => 30,
      CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
      CURLOPT_CUSTOMREQUEST => "POST",
      CURLOPT_POSTFIELDS => json_encode([
        'model' => 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
        'messages' => [
            [
                    'role' => 'system',
                    'content' => 'You are a helpful assistant.'
            ],
            [
                    'role' => 'user',
                    'content' => 'Hello!'
            ]
        ]
      ]),
      CURLOPT_HTTPHEADER => [
        "Authorization: Bearer <token>",
        "Content-Type: application/json"
      ],
    ]);

    $response = curl_exec($curl);
    $err = curl_error($curl);

    curl_close($curl);

    if ($err) {
      echo "cURL Error #:" . $err;
    } else {
      echo $response;
    }

    package main

    import (
        "fmt"
        "strings"
        "net/http"
        "io"
    )

    func main() {

        url := "https://inference.baseten.co/v1/chat/completions"

        payload := strings.NewReader("{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}")

        req, _ := http.NewRequest("POST", url, payload)

        req.Header.Add("Authorization", "Bearer <token>")
        req.Header.Add("Content-Type", "application/json")

        res, _ := http.DefaultClient.Do(req)

        defer res.Body.Close()
        body, _ := io.ReadAll(res.Body)

        fmt.Println(string(body))

    }

    HttpResponse<String> response = Unirest.post("https://inference.baseten.co/v1/chat/completions")
      .header("Authorization", "Bearer <token>")
      .header("Content-Type", "application/json")
      .body("{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}")
      .asString();

    require 'uri'
    require 'net/http'

    url = URI("https://inference.baseten.co/v1/chat/completions")

    http = Net::HTTP.new(url.host, url.port)
    http.use_ssl = true

    request = Net::HTTP::Post.new(url)
    request["Authorization"] = 'Bearer <token>'
    request["Content-Type"] = 'application/json'
    request.body = "{\n  \"model\": \"deepseek-ai/DeepSeek-V4.1-Flash-1017\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Hello!\"\n    }\n  ]\n}"

    response = http.request(request)
    puts response.read_body

200

    {
      "model": "<string>",
      "choices": [
        {
          "index": 123,
          "delta": {
            "role": null,
            "content": null,
            "tool_calls": [
              {
                "index": 123,
                "function": {
                  "arguments": "<string>",
                  "name": null
                },
                "id": "<string>",
                "type": "function"
              }
            ]
          },
          "logprobs": null,
          "finish_reason": null,
          "stop_reason": 123
        }
      ],
      "id": "<string>",
      "object": "chat.completion.chunk",
      "created": 123,
      "usage": null
    }
