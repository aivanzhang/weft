# Source snapshot: baseten-vision

- Source: https://docs.baseten.co/inference/model-apis/vision
- Retrieved: 2026-10-26 03:00:26 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `4648784a6863ec9fdf93acb62a52ab687288f58e03b97ce26548dd71baae2fed`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

Model APIs

# Vision

Copy pageCopy page

Send images and videos alongside text to vision-capable models

Copy pageCopy page

Model APIs support text and vision inputs, but multimodal capability depends on the underlying model. Vision-capable models accept images alongside text in the same request, using the OpenAI-compatible `image_url` content type. The model processes both modalities together, so it can answer questions about image content, compare multiple images, or extract structured data from screenshots. Not all models support vision. Check the table below before sending image inputs.

## 

[​](https://docs.baseten.co/inference/model-apis/vision#supported-models)

Supported models

| Model | Slug |
|----|----|
| DeepSeek V4.1 Flash | `deepseek-ai/DeepSeek-V4.1-Flash` |
| DeepSeek V4.1 Flash 1017 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| DeepSeek V4.1 Flash 1017 Fast | `deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast` |
| DeepSeek V4.1 Flash Fast | `deepseek-ai/DeepSeek-V4.1-Flash-Fast` |
| GLM 5.2 | `zai-org/GLM-5.2` |
| GLM 5.2 Fast | `zai-org/GLM-5.2-Fast` |
| GLM 5.3 | `zai-org/GLM-5.3` |
| GLM 5.3 Fast | `zai-org/GLM-5.3-Fast` |
| GLM 5.3 Flash | `zai-org/GLM-5.3-Flash` |
| Kimi K3 | `moonshotai/Kimi-K3` |
| Muse Spark Open | `meta-models/muse-spark-open` |
| Nemotron Super 3.5 | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` |

Nemotron Super 3.5 accepts text and image inputs. Audio and video inputs aren’t supported through its Model API.

## 

[​](https://docs.baseten.co/inference/model-apis/vision#send-a-vision-request)

Send a vision request

Use the `image_url` content type to include images in your messages. Baseten retrieves image URLs **from the inference service**, so the URL must be reachable over HTTPS from Baseten’s environment (for example your own object storage, Hugging Face artifact links, or other hosts that allow server-side fetches). Prefer stable, direct HTTPS links. Optional `image_url.detail` controls preprocessing resolution: `low`, `high`, `original`, or `auto` (OpenAI-compatible). When in doubt, use `auto`. Send an image alongside a text prompt like this:

- Python

- JavaScript

- cURL

vision.py

    from openai import OpenAI
    import os

    client = OpenAI(
        base_url="https://inference.baseten.co/v1",
        api_key=os.environ["BASETEN_API_KEY"],
    )

    response = client.chat.completions.create(
        model="moonshotai/Kimi-K3",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the natural environment in the image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ],
    )

    print(response.choices[0].message.content)

vision.js

    import OpenAI from "openai";

    const client = new OpenAI({
        baseURL: "https://inference.baseten.co/v1",
        apiKey: process.env.BASETEN_API_KEY,
    });

    const response = await client.chat.completions.create({
        model: "moonshotai/Kimi-K3",
        messages: [
            {
                role: "user",
                content: [
                    {
                        type: "text",
                        text: "Describe the natural environment in the image.",
                    },
                    {
                        type: "image_url",
                        image_url: {
                            url: "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
                            detail: "auto",
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
        "model": "moonshotai/Kimi-K3",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "Describe the natural environment in the image."
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
                  "detail": "auto"
                }
              }
            ]
          }
        ]
      }'

## 

[​](https://docs.baseten.co/inference/model-apis/vision#image-and-video-limits)

Image and video limits

Baseten’s media encoder enforces the following per-request limits:

| Limit | DeepSeek V4.1 Flash | DeepSeek V4.1 Flash Fast | Kimi K3 | Muse Spark Open | GLM 5.2 | GLM 5.2 Fast | GLM 5.3 | GLM 5.3 Fast | GLM 5.3 Flash |
|----|----|----|----|----|----|----|----|----|----|
| Max images per request | 64 | 16 | 96 | 8 | 8 | 8 | 8 | 8 | 64 |
| Max videos per request | Not supported | Not supported | 12 | 2 | Not supported | Not supported | Not supported | Not supported | Not supported |
| Max total media size per request (URL) | 200 MB | 64 MB | 320 MB | 200 MB | 100 MB | 100 MB | 100 MB | 100 MB | 100 MB |
| Max size per image (URL) | 200 MB | 32 MB | 80 MB | 10 MB | 80 MB | 80 MB | 80 MB | 80 MB | 80 MB |
| Max size per video (URL) | Not supported | Not supported | 150 MB | 200 MB | Not supported | Not supported | Not supported | Not supported | Not supported |
| Max video duration | Not supported | Not supported | 30 seconds | 300 seconds | Not supported | Not supported | Not supported | Not supported | Not supported |
| Max request body (base64) | 45 MB | 64 MB | 62 MB | 100 MiB | 62 MB | 62 MB | 62 MB | 62 MB | 100 MB |

Pass media by URL whenever you can. The vision encoder fetches each URL and enforces the per-file and total-media caps directly, while the request body stays small. Base64-encoded media travels inside the request body and counts toward the model’s request-body limit. Muse Spark Open accepts JPEG, PNG, WebP, GIF, BMP, and TIFF images. For animated GIFs, it uses only the first frame. It accepts MP4, MOV, MKV, and WebM videos with H.264, H.265, VP9, or AV1 encoding, up to 4K at 60 frames per second. It samples at most 96 frames per video and ignores the audio track. Send speech as a separate [`audio_url` part](https://docs.baseten.co/inference/model-apis/audio). Muse Spark Open enforces a 200 MB combined limit for images, videos, and audio, plus a 896 MiB encoded-media budget per request. A request can fit within the file-size limits and still exceed the encoded-media budget. Combining 8 large images, 2 five-minute videos, and 4 five-minute audio files can return HTTP 400. Split large combinations across requests. Send files larger than about 75 MB by URL. Base64 encoding increases their size by about a third, and the complete request must fit within the 100 MiB request-body limit. Muse Spark Open doesn’t accept PDF files. Extract their text before sending it.

## 

[​](https://docs.baseten.co/inference/model-apis/vision#pricing)

Pricing

There is no additional per-image fee. Images are converted to input tokens and priced at the model’s standard input rate. Higher resolution images produce more tokens and cost more to process. The exact conversion from pixels to tokens depends on the model. Kimi K3 divides each image into 28×28 pixel tiles where each tile becomes one input token. The cost table below uses Kimi K3’s uncached input rate (\$3.00 per million tokens); for other models, use the rates on the [Model APIs pricing page](https://www.baseten.co/pricing).

| Image resolution | Tiles | Input tokens | Cost at \$3.00/M |
|------------------|-------|--------------|------------------|
| 256×256          | 100   | 100          | \$0.0003         |
| 512×512          | 361   | 361          | \$0.0011         |
| 1024×1024        | 1,369 | 1,369        | \$0.0041         |
| 1920×1080        | 2,691 | 2,691        | \$0.0081         |

For videos, token count scales with both resolution and the number of sampled frames.

## 

[​](https://docs.baseten.co/inference/model-apis/vision#next-steps)

Next steps

## Model APIs overview

Supported models, pricing, and the feature support matrix

## Chat Completions reference

Full request and response schema for the `image_url` content type

Was this page helpful?

Yes

No

[Previous](https://docs.baseten.co/inference/model-apis/reasoning)[](https://docs.baseten.co/inference/model-apis/audio)

Audio

Send audio recordings alongside text prompts to audio-capable models on Model APIs

Next

⌘I

## On this page

Back to top

- [Supported models](https://docs.baseten.co/inference/model-apis/vision#supported-models)
- [Send a vision request](https://docs.baseten.co/inference/model-apis/vision#send-a-vision-request)
- [Image and video limits](https://docs.baseten.co/inference/model-apis/vision#image-and-video-limits)
- [Pricing](https://docs.baseten.co/inference/model-apis/vision#pricing)
- [Next steps](https://docs.baseten.co/inference/model-apis/vision#next-steps)
