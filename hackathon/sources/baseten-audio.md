# Source snapshot: baseten-audio

- Source: https://docs.baseten.co/inference/model-apis/audio
- Retrieved: 2026-10-26 03:00:26 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `5d138453fcabf82a95145d0ae0ddc2dec55e38fe96bb4981ecf989e829157102`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

Model APIs

# Audio

Copy pageCopy page

Send audio recordings alongside text prompts to audio-capable models on Model APIs

Copy pageCopy page

Model APIs support audio input on audio-capable models. These models accept audio alongside text in the same request, using the `audio_url` content type, and process both modalities together: they can transcribe speech, answer questions about audio content, or reason over a recording and a text prompt at once.

To deploy dedicated models for batch or real-time transcription, or for speech generation, see [Speech and audio](https://docs.baseten.co/inference/audio/overview).

Not all models support audio. Check the table below before sending audio inputs.

## 

[​](https://docs.baseten.co/inference/model-apis/audio#supported-models)

Supported models

| Model           | Slug                          |
|-----------------|-------------------------------|
| Muse Spark Open | `meta-models/muse-spark-open` |

## 

[​](https://docs.baseten.co/inference/model-apis/audio#send-an-audio-request)

Send an audio request

Use the `audio_url` content type to include audio in your messages. The request shape matches [vision requests](https://docs.baseten.co/inference/model-apis/vision) with `audio_url` in place of `image_url`. Each model has its own audio requirements:

| Model | Requirements |
|----|----|
| Muse Spark Open | WAV, MP3, FLAC, OGG (Vorbis or Opus), or M4A/AAC; 8–48 kHz; mono, stereo, and multichannel audio are downmixed; up to 4 files, 25 MB and 5 minutes each; 100 MiB request-body limit |

The combined size of images, videos, and audio must fit within the 200 MB total-media limit. Large combinations can also exceed the [encoded-media budget](https://docs.baseten.co/inference/model-apis/vision#image-and-video-limits). Baseten retrieves audio URLs **from the inference service**. Use an HTTP or HTTPS URL reachable from Baseten’s environment, such as your own object storage. Muse Spark Open follows redirects and rejects localhost, internal, and cloud-metadata IP addresses. Prefer stable, direct HTTPS links. Base64 `data:` URLs also work, but they count toward the request-body limit, so pass audio by URL when you can. Send a recording alongside a text prompt like this:

- Python

- JavaScript

- cURL

audio.py

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="meta-models/muse-spark-open",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Summarize what is said in this recording.",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
                        },
                    },
                ],
            }
        ],
    )

    print(response.choices[0].message.content)

audio.js

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "meta-models/muse-spark-open",
        messages: [
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Summarize what is said in this recording.",
                    },
                    {
                        type: "audio_url",
                        audio_url: {
                            url: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
                        },
                    },
                ],
            },
        ],
    });

    console.log(response.choices[0].message.content);

Request

    curl https://inference.baseten.co/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $BASETEN_API_KEY" \
      -d '{
        "model": "meta-models/muse-spark-open",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Summarize what is said in this recording."
              },
              {
                "type": "audio_url",
                "audio_url": {
                  "url": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
                }
              }
            ]
          }
        ]
      }'

## 

[​](https://docs.baseten.co/inference/model-apis/audio#pricing)

Pricing

There is no additional per-request fee for audio. Audio is converted to input tokens and priced at the model’s standard input rate. Longer recordings produce more tokens and cost more to process. See the [Model APIs pricing page](https://www.baseten.co/pricing) for per-model rates.

## 

[​](https://docs.baseten.co/inference/model-apis/audio#next-steps)

Next steps

## Vision

Send images and videos alongside text

## Chat Completions reference

Full request and response schema for the `audio_url` content type

Was this page helpful?

Yes

No

[Previous](https://docs.baseten.co/inference/model-apis/vision)[](https://docs.baseten.co/inference/model-apis/web-search)

Web search

Get grounded answers in Model APIs with server-side tool execution

Next

⌘I

## On this page

Back to top

- [Supported models](https://docs.baseten.co/inference/model-apis/audio#supported-models)
- [Send an audio request](https://docs.baseten.co/inference/model-apis/audio#send-an-audio-request)
- [Pricing](https://docs.baseten.co/inference/model-apis/audio#pricing)
- [Next steps](https://docs.baseten.co/inference/model-apis/audio#next-steps)
