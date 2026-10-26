# Source snapshot: modal-gpu

- Source: https://modal.com/docs/guide/gpu
- Retrieved: 2026-10-26 03:00:26 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `6257a9e06347b6cae9d00eec5422af5a2077a949b6d3f2677c874808f9e575b4`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

[Guide](https://modal.com/docs/guide/intro) [Introduction to Modal ](https://modal.com/docs/guide/intro)[Getting started ](https://modal.com/docs/guide/getting-started)[Functions](https://modal.com/docs/guide/functions) [Functions ](https://modal.com/docs/guide/functions)[Invocation methods ](https://modal.com/docs/guide/function-invocation-methods)[Function lookups ](https://modal.com/docs/guide/trigger-deployed-functions)[Container lifecycle management ](https://modal.com/docs/guide/lifecycle-functions)[Parametrized Functions ](https://modal.com/docs/guide/parametrized-functions)[Dynamic configuration ](https://modal.com/docs/guide/dynamic-function-config)[Sandboxes](https://modal.com/docs/guide/sandboxes) [Sandboxes ](https://modal.com/docs/guide/sandboxes)[Running commands ](https://modal.com/docs/guide/sandbox-spawn)[Networking and security ](https://modal.com/docs/guide/sandbox-networking)[File access ](https://modal.com/docs/guide/sandbox-files)[Snapshots ](https://modal.com/docs/guide/sandbox-snapshots)[Exit Snapshots Beta](https://modal.com/docs/guide/sandbox-exit-snapshots)[Pricing and resources ](https://modal.com/docs/guide/sandbox-resources)[Sidecars Beta](https://modal.com/docs/guide/sandbox-sidecars)[Secret injection Beta](https://modal.com/docs/guide/sandbox-secret-injection)[Endpoints](https://modal.com/docs/guide/endpoints) [Endpoints ](https://modal.com/docs/guide/endpoints)[Shared Endpoints ](https://modal.com/docs/guide/shared-endpoints)[Dedicated Endpoints ](https://modal.com/docs/guide/dedicated-endpoints)[Endpoint candidates ](https://modal.com/docs/guide/endpoint-candidates)[Endpoint integrations ](https://modal.com/docs/guide/endpoint-integrations)[Training](https://modal.com/docs/guide/training-quickstart) [Quickstart ](https://modal.com/docs/guide/training-quickstart)[Tinker-compatible API Beta](https://modal.com/docs/guide/spindle)[Multi-node training ](https://modal.com/docs/guide/multi-node-training)[Images](https://modal.com/docs/guide/images) [Defining Images ](https://modal.com/docs/guide/images)[Using existing container images ](https://modal.com/docs/guide/existing-images)[Named images ](https://modal.com/docs/guide/named-images)[Fast pull from registry ](https://modal.com/docs/guide/fast-pull-from-registry)[GPUs and other resources](https://modal.com/docs/guide/gpu) [GPU acceleration ](https://modal.com/docs/guide/gpu)[Using CUDA on Modal ](https://modal.com/docs/guide/cuda)[Configuring CPU, memory, and disk ](https://modal.com/docs/guide/resources)[Scaling out](https://modal.com/docs/guide/scale) [Scaling out ](https://modal.com/docs/guide/scale)[Input concurrency ](https://modal.com/docs/guide/concurrent-inputs)[Batch processing ](https://modal.com/docs/guide/batch-processing)[Job queues ](https://modal.com/docs/guide/job-queue)[Dynamic batching ](https://modal.com/docs/guide/dynamic-batching)[Multi-node Clusters ](https://modal.com/docs/guide/multi-node-clusters)[Deployment](https://modal.com/docs/guide/apps) [Apps, Functions, and entrypoints ](https://modal.com/docs/guide/apps)[Managing deployments ](https://modal.com/docs/guide/managing-deployments)[Continuous deployment ](https://modal.com/docs/guide/continuous-deployment)[Scheduled Functions ](https://modal.com/docs/guide/cron)[HTTP Applications](https://modal.com/docs/guide/servers) [Servers ](https://modal.com/docs/guide/servers)[Sticky Sessions ](https://modal.com/docs/guide/sticky-sessions)[Web Functions ](https://modal.com/docs/guide/webhooks)[Streaming endpoints ](https://modal.com/docs/guide/streaming-endpoints)[Web Function URLs ](https://modal.com/docs/guide/webhook-urls)[Request timeouts ](https://modal.com/docs/guide/webhook-timeouts)[Proxy Tokens ](https://modal.com/docs/guide/webhook-proxy-auth)[Networking](https://modal.com/docs/guide/tunnels) [Tunnels ](https://modal.com/docs/guide/tunnels)[Proxies Beta](https://modal.com/docs/guide/proxy-ips)[Cluster networking ](https://modal.com/docs/guide/private-networking)[Data sharing and storage](https://modal.com/docs/guide/local-data) [Passing local data ](https://modal.com/docs/guide/local-data)[Volumes ](https://modal.com/docs/guide/volumes)[Storing model weights ](https://modal.com/docs/guide/model-weights)[Cloud bucket mounts ](https://modal.com/docs/guide/cloud-bucket-mounts)[Dicts ](https://modal.com/docs/guide/dicts)[Queues ](https://modal.com/docs/guide/queues)[Dataset ingestion ](https://modal.com/docs/guide/dataset-ingestion)[Secrets and environment variables](https://modal.com/docs/guide/secrets) [Secrets ](https://modal.com/docs/guide/secrets)[Environment variables ](https://modal.com/docs/guide/environment_variables)[Performance](https://modal.com/docs/guide/cold-start) [Cold start performance ](https://modal.com/docs/guide/cold-start)[Memory Snapshots ](https://modal.com/docs/guide/memory-snapshots)[High-performance LLM inference ](https://modal.com/docs/guide/high-performance-llm-inference)[Reliability and robustness](https://modal.com/docs/guide/retries) [Failures and retries ](https://modal.com/docs/guide/retries)[Preemption ](https://modal.com/docs/guide/preemption)[Timeouts ](https://modal.com/docs/guide/timeouts)[GPU health ](https://modal.com/docs/guide/gpu-health)[Troubleshooting ](https://modal.com/docs/guide/troubleshooting)[Security and privacy](https://modal.com/docs/guide/security) [Security and privacy ](https://modal.com/docs/guide/security)[Data residency ](https://modal.com/docs/guide/data-residency)[Customer-supplied encryption keys Alpha](https://modal.com/docs/guide/customer-supplied-encryption-keys)[Audit logs ](https://modal.com/docs/guide/audit-logs)[Integrations](https://modal.com/docs/guide/oidc-integration) [Using OIDC to authenticate with external services ](https://modal.com/docs/guide/oidc-integration)[Connecting Modal to your Datadog account ](https://modal.com/docs/guide/datadog-integration)[Connecting Modal to your OpenTelemetry provider ](https://modal.com/docs/guide/otel-integration)[Okta SSO ](https://modal.com/docs/guide/okta-sso)[Microsoft Entra SSO ](https://modal.com/docs/guide/entra-sso)[Custom SAML SSO ](https://modal.com/docs/guide/saml-sso)[SCIM Integration ](https://modal.com/docs/guide/scim)[Slack notifications Beta](https://modal.com/docs/guide/slack-notifications)[Workspace & account settings](https://modal.com/docs/guide/workspaces) [Workspaces ](https://modal.com/docs/guide/workspaces)[Environments ](https://modal.com/docs/guide/environments)[Service users ](https://modal.com/docs/guide/service-users)[User groups ](https://modal.com/docs/guide/user-groups)[Role-Based Access Control (RBAC) ](https://modal.com/docs/guide/rbac)[Billing ](https://modal.com/docs/guide/billing)[Network egress billing ](https://modal.com/docs/guide/network-egress-billing)[Budgets ](https://modal.com/docs/guide/budgets)[Other topics](https://modal.com/docs/guide/feature-maturity) [Feature maturity ](https://modal.com/docs/guide/feature-maturity)[Restricted Functions ](https://modal.com/docs/guide/restricted-access)[JavaScript/Go SDKs Beta](https://modal.com/docs/guide/sdk-javascript-go)[Modal 1.0 migration guide ](https://modal.com/docs/guide/modal-1-0-migration)[File and project structure ](https://modal.com/docs/guide/project-structure)[Developing and debugging ](https://modal.com/docs/guide/developing-debugging)[Jupyter notebooks ](https://modal.com/docs/guide/jupyter-notebooks)[Asynchronous API usage ](https://modal.com/docs/guide/async)[Global variables ](https://modal.com/docs/guide/global-variables)[Region selection ](https://modal.com/docs/guide/region-selection)[GPU Metrics ](https://modal.com/docs/guide/gpu-metrics)[Search syntax ](https://modal.com/docs/guide/log-search-syntax)

------------------------------------------------------------------------

Copy page

# GPU acceleration

Modal makes it easy to run your code on [GPUs](https://modal.com/gpu-glossary/readme).

Using a GPU requires having a valid payment method on file.

## Quickstart [](https://modal.com/docs/guide/gpu#quickstart)

Here’s a simple example of a Function running on an A100 in Modal:

    import modal

    image = modal.Image.debian_slim().pip_install("torch", "numpy")
    app = modal.App(image=image)

    @app.function(gpu="A100")
    def run():
        import torch

        assert torch.cuda.is_available()

## Specifying GPU type [](https://modal.com/docs/guide/gpu#specifying-gpu-type)

You can pick a specific GPU type for your Function via the `gpu` argument. Modal supports the following values for this parameter:

- `T4`
- `L4`
- `A10`
- `L40S`
- `A100`
- `A100-40GB`
- `A100-80GB`
- `RTX-PRO-6000`
- `H100`/`H100!`
- `H200`
- `B200`/`B200+`
- `B300`

For instance, to use a B200, you can use `@app.function(gpu="B200")`.

Refer to our [pricing page](https://modal.com/pricing) for the latest pricing on each GPU type.

## Specifying GPU count [](https://modal.com/docs/guide/gpu#specifying-gpu-count)

You can specify more than 1 GPU per container by appending `:n` to the GPU argument. For instance, to run a Function with eight H100s:

    @app.function(gpu="H100:8")
    def run_llama_405b_fp8():
        ...

Currently B300, B200, H200, H100, A100, L4, T4 and L40S instances support up to 8 GPUs (up to 2,304 GB GPU RAM), and A10 instances support up to 4 GPUs (up to 96 GB GPU RAM). Note that requesting more than 2 GPUs per container will usually result in larger wait times. These GPUs are always attached to the same physical machine.

## Picking a GPU [](https://modal.com/docs/guide/gpu#picking-a-gpu)

For running, rather than training, neural networks, we recommend starting off with the [L40S](https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413), which offers an excellent trade-off of cost and performance and 48 GB of GPU RAM for storing model weights and activations.

For more on how to pick a GPU for use with neural networks like LLaMA or Stable Diffusion, and for tips on how to make that GPU go brrr, check out [Tim Dettemers’ blog post](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) or the [Full Stack Deep Learning page on Cloud GPUs](https://fullstackdeeplearning.com/cloud-gpus/).

## B300 GPUs [](https://modal.com/docs/guide/gpu#b300-gpus)

[B300s](https://www.nvidia.com/en-us/data-center/dgx-b300/) are NVIDIA Blackwell Ultra GPUs, based on the Blackwell [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).

To request a B300, set the `gpu` argument to `"B300"`:

    @app.function(gpu="B300:8")
    def run_inference():
        ...

B300 requires CUDA version 13.1+. Make sure your container Image and libraries are compatible with CUDA 13 before requesting a B300.

## B200 GPUs [](https://modal.com/docs/guide/gpu#b200-gpus)

B200s are [NVIDIA data center GPUs](https://www.nvidia.com/en-us/data-center/dgx-b200/) based on the Blackwell [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).

To request a B200, set the `gpu` argument to `"B200"`

    @app.function(gpu="B200:8")
    def run_deepseek():
        ...

Check out [this example](https://modal.com/docs/examples/llm_inference) to see how you can use B200s to max out vLLM serving performance for LLaMA 3.1-8B.

Before you jump for this powerful GPU, make sure you understand where the bottlenecks are in your computations. For example, running language models with small batch sizes (e.g. one prompt at a time) results in a [bottleneck on memory, not arithmetic](https://kipp.ly/transformer-inference-arithmetic/). Since arithmetic throughput has risen faster than memory throughput in recent hardware generations, speedups for memory-bound GPU jobs are not as extreme and may not be worth the extra cost.

### Opt-in upgrade to B300 [](https://modal.com/docs/guide/gpu#opt-in-upgrade-to-b300)

Use `gpu="B200+"` to allow Modal to run requests on either B200 or B300 GPUs. B200+ is billed as B200, regardless of which GPU is used. Use this option only if your code is compatible with both types of GPUs. B300 requires CUDA version 13.1+. Use this to have access to a greater capacity pool automatically.

## H200 and H100 GPUs [](https://modal.com/docs/guide/gpu#h200-and-h100-gpus)

[H200s](https://www.nvidia.com/en-us/data-center/h200/) and [H100s](https://www.nvidia.com/en-us/data-center/h100/) are the previous generation of top-of-the-line data center chips from NVIDIA, based on the Hopper [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture). These GPUs have better software support than do Blackwell GPUs (e.g. popular libraries include pre-compiled kernels for Hopper, but not Blackwell), and they often get the job done at a competitive cost, so they are a common choice of accelerator, on and off Modal.

All H100 and H200 GPUs on the Modal platform are of the SXM variant, as can be verified by examining the [power draw](https://modal.com/docs/guide/gpu-metrics) in the dashboard or with `nvidia-smi`.

### Automatic upgrades to H200s [](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s)

Modal may automatically upgrade a `gpu="H100"` request to run on an H200. This automatic upgrade does *not* change the cost of the GPU.

Kernels [compatible](https://modal.com/gpu-glossary/device-software/compute-capability) with H200s are also compatible with H100s, so your code will still run, just faster, so long as it doesn’t make strict assumptions about memory capacity. An H200’s [HBM3e memory](https://modal.com/gpu-glossary/device-hardware/gpu-ram) has a capacity of 141 GB and a bandwidth of 4.8TB/s, 1.75x larger and 1.4x faster than an NVIDIA H100 with HBM3.

In cases where an automatic upgrade to H200 would not be helpful (for instance, benchmarking) you can pass `gpu=H100!` to avoid it.

## A100 GPUs [](https://modal.com/docs/guide/gpu#a100-gpus)

[A100s](https://www.nvidia.com/en-us/data-center/a100/) are based on NVIDIA’s Ampere [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture). Modal offers two versions of the A100: one with 40 GB of RAM and another with 80 GB of RAM.

To request an A100 with 40 GB of [GPU memory](https://modal.com/gpu-glossary/device-hardware/gpu-ram), use `gpu="A100"`:

    @app.function(gpu="A100")
    def qwen_7b():
        ...

Modal may automatically upgrade a `gpu="A100"` request to run on an 80 GB A100. This automatic upgrade does *not* change the cost of the GPU.

You can specifically request a 40GB A100 with the string `A100-40GB`. To specifically request an 80 GB A100, use the string `A100-80GB`:

    @app.function(gpu="A100-80GB")
    def llama_70b_fp8():
        ...

## GPU fallbacks [](https://modal.com/docs/guide/gpu#gpu-fallbacks)

Modal allows specifying a list of possible GPU types, suitable for Functions that are compatible with multiple options. Modal respects the ordering of this list and will try to allocate the most preferred GPU type before falling back to less preferred ones.

    @app.function(gpu=["H100", "A100-40GB:2"])
    def run_on_80gb():
        ...

See [this example](https://modal.com/docs/examples/gpu_fallbacks) for more detail.

## Multi GPU training [](https://modal.com/docs/guide/gpu#multi-gpu-training)

Modal currently supports multi-GPU training on a single node, with multi-node training in private Beta (email us at <support@modal.com> for access). Depending on which framework you are using, you may need to use different techniques to train on multiple GPUs.

If the framework re-executes the entrypoint of the Python process (like [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html)) you need to either set the strategy to `ddp_spawn` or `ddp_notebook` if you wish to invoke the training directly. Another option is to run the training script as a subprocess instead.

    @app.function(gpu="A100:2")
    def run():
        import subprocess
        import sys
        subprocess.run(
            ["python", "train.py"],
            stdout=sys.stdout, stderr=sys.stderr,
            check=True,
        )

## Examples and more resources [](https://modal.com/docs/guide/gpu#examples-and-more-resources)

For more information about GPUs in general, check out our [GPU Glossary](https://modal.com/gpu-glossary/readme).

Or take a look some examples of Modal Apps using GPUs:

- [Fine-tune a character LoRA for your pet](https://modal.com/docs/examples/diffusers_lora_finetune)
- [Fast LLM inference on big GPUs](https://modal.com/docs/examples/llm_inference)
- [Stable Diffusion with a CLI, API, and web UI](https://modal.com/docs/examples/text_to_image)
- [Rendering Blender videos](https://modal.com/docs/examples/blender_video)

[GPU acceleration](https://modal.com/docs/guide/gpu#gpu-acceleration)[Quickstart](https://modal.com/docs/guide/gpu#quickstart)[Specifying GPU type](https://modal.com/docs/guide/gpu#specifying-gpu-type)[Specifying GPU count](https://modal.com/docs/guide/gpu#specifying-gpu-count)[Picking a GPU](https://modal.com/docs/guide/gpu#picking-a-gpu)[B300 GPUs](https://modal.com/docs/guide/gpu#b300-gpus)[B200 GPUs](https://modal.com/docs/guide/gpu#b200-gpus)[Opt-in upgrade to B300](https://modal.com/docs/guide/gpu#opt-in-upgrade-to-b300)[H200 and H100 GPUs](https://modal.com/docs/guide/gpu#h200-and-h100-gpus)[Automatic upgrades to H200s](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s)[A100 GPUs](https://modal.com/docs/guide/gpu#a100-gpus)[GPU fallbacks](https://modal.com/docs/guide/gpu#gpu-fallbacks)[Multi GPU training](https://modal.com/docs/guide/gpu#multi-gpu-training)[Examples and more resources](https://modal.com/docs/guide/gpu#examples-and-more-resources)

Learn more

[High-speed inference with vLLM](https://modal.com/docs/examples/llm_inference)

[Edit images with a prompt](https://modal.com/docs/examples/image_to_image)

[GPU metrics on Modal](https://modal.com/docs/guide/gpu-metrics)

[GPU health on Modal](https://modal.com/docs/guide/gpu-health)
