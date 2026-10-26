# Source snapshot: baseten-quickstart

- Source: https://docs.baseten.co/quickstart
- Retrieved: 2026-10-26 03:00:25 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `d14f167be3fdfc964ecdd8bc1deffd5043e27025507d7441f02a93ce58f59e9c`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

Get started

# Quickstart

Copy pageCopy page

Call a shared Model API endpoint without deploying or managing infrastructure.

Copy pageCopy page

Call a shared [Model API](https://docs.baseten.co/inference/model-apis/overview) endpoint. Baseten hosts the model and manages the infrastructure; you send requests without deploying a model. You pay for the input and output tokens your requests use. See [pricing and limits](https://docs.baseten.co/inference/model-apis/pricing-and-limits).

## 

[​](https://docs.baseten.co/quickstart#create-an-api-key)

Create an API key

Use a [personal API key](https://docs.baseten.co/organization/api-keys#api-key-types) to authenticate your requests. Reuse a saved key or create one:

1.  Open [API keys](https://app.baseten.co/settings/api_keys) and sign in. If you don’t have an account, [create one](https://app.baseten.co/signup) first.
2.  Select **Create API key** and enter a name such as `quickstart`.
3.  Select **Personal**, then **Create API key**.
4.  Copy the key. Baseten only shows it once.

In your terminal, replace `YOUR_API_KEY` with your key and run:

macOS/Linux

Windows

    export BASETEN_API_KEY="YOUR_API_KEY"

    $env:BASETEN_API_KEY = "YOUR_API_KEY"

Run the examples from this terminal so they can read `BASETEN_API_KEY`. Keep the key out of source code and version control.

## 

[​](https://docs.baseten.co/quickstart#send-a-request)

Send a request

Call the [Chat Completions API](https://docs.baseten.co/reference/inference-api/chat-completions) with Python, JavaScript, or cURL. The SDK examples use Baseten’s base URL and API key; you don’t need an OpenAI account. The `model` field selects the hosted model, GLM 5.3 in this example, and `messages` contains your input.

- Python

- JavaScript

- cURL

You’ll need [uv](https://docs.astral.sh/uv/getting-started/installation/) to run this example.Save this code as `chat.py`:

chat.py

    import os
    from openai import OpenAI

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="zai-org/GLM-5.3",
        messages=[
            {"role": "user", "content": "Explain inference in one sentence."}
        ],
    )

    print(response.choices[0].message.content)

Run it. `--with openai` installs the SDK for this run:

Terminal

Example output

    uv run --with openai chat.py

    Inference uses a trained model to generate outputs from new inputs.

You’ll need [Node.js](https://nodejs.org/en/download). Install the OpenAI SDK in your project directory:

Terminal

    npm install openai

Save this code as `chat.mjs`:

chat.mjs

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "zai-org/GLM-5.3",
        messages: [
            { role: "user", content: "Explain inference in one sentence." }
        ],
    });

    console.log(response.choices[0].message.content);

Run it:

Terminal

Example output

    node chat.mjs

    Inference uses a trained model to generate outputs from new inputs.

Run this command in a Bash-compatible terminal:

Terminal

Example response excerpt

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "zai-org/GLM-5.3",
        "messages": [
          {"role": "user", "content": "Explain inference in one sentence."}
        ]
      }'

    {
      "choices": [
        {
          "message": {
            "role": "assistant",
            "content": "Inference uses a trained model to generate outputs from new inputs."
          }
        }
      ]
    }

The reply is in `choices[0].message.content`.

The exact wording varies. Change the message and run the request again.

## 

[​](https://docs.baseten.co/quickstart#next-steps)

Next steps

- [Choose another Model API](https://docs.baseten.co/inference/model-apis/overview) and replace the `model` value with its slug.
- [Stream the response](https://docs.baseten.co/inference/streaming) to display text as the model generates it.
- [Deploy from Explore](https://docs.baseten.co/examples/create-a-model-with-rest#dashboard) if you need a model on dedicated GPUs with your own scaling settings.

Was this page helpful?

Yes

No

[Previous](https://docs.baseten.co/overview)[](https://docs.baseten.co/concepts/howbasetenworks)

How Baseten works

The moving parts behind training, deployment, request routing, autoscaling, and environment promotion on Baseten.

Next

⌘I

## On this page

Back to top

- [Create an API key](https://docs.baseten.co/quickstart#create-an-api-key)
- [Send a request](https://docs.baseten.co/quickstart#send-a-request)
- [Next steps](https://docs.baseten.co/quickstart#next-steps)
