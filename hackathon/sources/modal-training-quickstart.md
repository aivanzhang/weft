# Source snapshot: modal-training-quickstart

- Source: https://modal.com/docs/guide/training-quickstart
- Retrieved: 2026-10-26 03:00:27 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `0e113e283ffa93296175458957446a7332aa9c5f5c93844944d74d34ede5e3b1`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

[Guide](https://modal.com/docs/guide/intro) [Introduction to Modal ](https://modal.com/docs/guide/intro)[Getting started ](https://modal.com/docs/guide/getting-started)[Functions](https://modal.com/docs/guide/functions) [Functions ](https://modal.com/docs/guide/functions)[Invocation methods ](https://modal.com/docs/guide/function-invocation-methods)[Function lookups ](https://modal.com/docs/guide/trigger-deployed-functions)[Container lifecycle management ](https://modal.com/docs/guide/lifecycle-functions)[Parametrized Functions ](https://modal.com/docs/guide/parametrized-functions)[Dynamic configuration ](https://modal.com/docs/guide/dynamic-function-config)[Sandboxes](https://modal.com/docs/guide/sandboxes) [Sandboxes ](https://modal.com/docs/guide/sandboxes)[Running commands ](https://modal.com/docs/guide/sandbox-spawn)[Networking and security ](https://modal.com/docs/guide/sandbox-networking)[File access ](https://modal.com/docs/guide/sandbox-files)[Snapshots ](https://modal.com/docs/guide/sandbox-snapshots)[Exit Snapshots Beta](https://modal.com/docs/guide/sandbox-exit-snapshots)[Pricing and resources ](https://modal.com/docs/guide/sandbox-resources)[Sidecars Beta](https://modal.com/docs/guide/sandbox-sidecars)[Secret injection Beta](https://modal.com/docs/guide/sandbox-secret-injection)[Endpoints](https://modal.com/docs/guide/endpoints) [Endpoints ](https://modal.com/docs/guide/endpoints)[Shared Endpoints ](https://modal.com/docs/guide/shared-endpoints)[Dedicated Endpoints ](https://modal.com/docs/guide/dedicated-endpoints)[Endpoint candidates ](https://modal.com/docs/guide/endpoint-candidates)[Endpoint integrations ](https://modal.com/docs/guide/endpoint-integrations)[Training](https://modal.com/docs/guide/training-quickstart) [Quickstart ](https://modal.com/docs/guide/training-quickstart)[Tinker-compatible API Beta](https://modal.com/docs/guide/spindle)[Multi-node training ](https://modal.com/docs/guide/multi-node-training)[Images](https://modal.com/docs/guide/images) [Defining Images ](https://modal.com/docs/guide/images)[Using existing container images ](https://modal.com/docs/guide/existing-images)[Named images ](https://modal.com/docs/guide/named-images)[Fast pull from registry ](https://modal.com/docs/guide/fast-pull-from-registry)[GPUs and other resources](https://modal.com/docs/guide/gpu) [GPU acceleration ](https://modal.com/docs/guide/gpu)[Using CUDA on Modal ](https://modal.com/docs/guide/cuda)[Configuring CPU, memory, and disk ](https://modal.com/docs/guide/resources)[Scaling out](https://modal.com/docs/guide/scale) [Scaling out ](https://modal.com/docs/guide/scale)[Input concurrency ](https://modal.com/docs/guide/concurrent-inputs)[Batch processing ](https://modal.com/docs/guide/batch-processing)[Job queues ](https://modal.com/docs/guide/job-queue)[Dynamic batching ](https://modal.com/docs/guide/dynamic-batching)[Multi-node Clusters ](https://modal.com/docs/guide/multi-node-clusters)[Deployment](https://modal.com/docs/guide/apps) [Apps, Functions, and entrypoints ](https://modal.com/docs/guide/apps)[Managing deployments ](https://modal.com/docs/guide/managing-deployments)[Continuous deployment ](https://modal.com/docs/guide/continuous-deployment)[Scheduled Functions ](https://modal.com/docs/guide/cron)[HTTP Applications](https://modal.com/docs/guide/servers) [Servers ](https://modal.com/docs/guide/servers)[Sticky Sessions ](https://modal.com/docs/guide/sticky-sessions)[Web Functions ](https://modal.com/docs/guide/webhooks)[Streaming endpoints ](https://modal.com/docs/guide/streaming-endpoints)[Web Function URLs ](https://modal.com/docs/guide/webhook-urls)[Request timeouts ](https://modal.com/docs/guide/webhook-timeouts)[Proxy Tokens ](https://modal.com/docs/guide/webhook-proxy-auth)[Networking](https://modal.com/docs/guide/tunnels) [Tunnels ](https://modal.com/docs/guide/tunnels)[Proxies Beta](https://modal.com/docs/guide/proxy-ips)[Cluster networking ](https://modal.com/docs/guide/private-networking)[Data sharing and storage](https://modal.com/docs/guide/local-data) [Passing local data ](https://modal.com/docs/guide/local-data)[Volumes ](https://modal.com/docs/guide/volumes)[Storing model weights ](https://modal.com/docs/guide/model-weights)[Cloud bucket mounts ](https://modal.com/docs/guide/cloud-bucket-mounts)[Dicts ](https://modal.com/docs/guide/dicts)[Queues ](https://modal.com/docs/guide/queues)[Dataset ingestion ](https://modal.com/docs/guide/dataset-ingestion)[Secrets and environment variables](https://modal.com/docs/guide/secrets) [Secrets ](https://modal.com/docs/guide/secrets)[Environment variables ](https://modal.com/docs/guide/environment_variables)[Performance](https://modal.com/docs/guide/cold-start) [Cold start performance ](https://modal.com/docs/guide/cold-start)[Memory Snapshots ](https://modal.com/docs/guide/memory-snapshots)[High-performance LLM inference ](https://modal.com/docs/guide/high-performance-llm-inference)[Reliability and robustness](https://modal.com/docs/guide/retries) [Failures and retries ](https://modal.com/docs/guide/retries)[Preemption ](https://modal.com/docs/guide/preemption)[Timeouts ](https://modal.com/docs/guide/timeouts)[GPU health ](https://modal.com/docs/guide/gpu-health)[Troubleshooting ](https://modal.com/docs/guide/troubleshooting)[Security and privacy](https://modal.com/docs/guide/security) [Security and privacy ](https://modal.com/docs/guide/security)[Data residency ](https://modal.com/docs/guide/data-residency)[Customer-supplied encryption keys Alpha](https://modal.com/docs/guide/customer-supplied-encryption-keys)[Audit logs ](https://modal.com/docs/guide/audit-logs)[Integrations](https://modal.com/docs/guide/oidc-integration) [Using OIDC to authenticate with external services ](https://modal.com/docs/guide/oidc-integration)[Connecting Modal to your Datadog account ](https://modal.com/docs/guide/datadog-integration)[Connecting Modal to your OpenTelemetry provider ](https://modal.com/docs/guide/otel-integration)[Okta SSO ](https://modal.com/docs/guide/okta-sso)[Microsoft Entra SSO ](https://modal.com/docs/guide/entra-sso)[Custom SAML SSO ](https://modal.com/docs/guide/saml-sso)[SCIM Integration ](https://modal.com/docs/guide/scim)[Slack notifications Beta](https://modal.com/docs/guide/slack-notifications)[Workspace & account settings](https://modal.com/docs/guide/workspaces) [Workspaces ](https://modal.com/docs/guide/workspaces)[Environments ](https://modal.com/docs/guide/environments)[Service users ](https://modal.com/docs/guide/service-users)[User groups ](https://modal.com/docs/guide/user-groups)[Role-Based Access Control (RBAC) ](https://modal.com/docs/guide/rbac)[Billing ](https://modal.com/docs/guide/billing)[Network egress billing ](https://modal.com/docs/guide/network-egress-billing)[Budgets ](https://modal.com/docs/guide/budgets)[Other topics](https://modal.com/docs/guide/feature-maturity) [Feature maturity ](https://modal.com/docs/guide/feature-maturity)[Restricted Functions ](https://modal.com/docs/guide/restricted-access)[JavaScript/Go SDKs Beta](https://modal.com/docs/guide/sdk-javascript-go)[Modal 1.0 migration guide ](https://modal.com/docs/guide/modal-1-0-migration)[File and project structure ](https://modal.com/docs/guide/project-structure)[Developing and debugging ](https://modal.com/docs/guide/developing-debugging)[Jupyter notebooks ](https://modal.com/docs/guide/jupyter-notebooks)[Asynchronous API usage ](https://modal.com/docs/guide/async)[Global variables ](https://modal.com/docs/guide/global-variables)[Region selection ](https://modal.com/docs/guide/region-selection)[GPU Metrics ](https://modal.com/docs/guide/gpu-metrics)[Search syntax ](https://modal.com/docs/guide/log-search-syntax)

------------------------------------------------------------------------

Copy page

# LLM training quickstart

Modal provides many ways to get started with training.

The easiest way is to use [Modal Dojo](https://dojo.modal.dev), an open-source library that gives you the building blocks, tuned defaults, and observability you need for easy, production-grade SFT and RL.

For a Tinker-compatible API, you can deploy a [Spindle](https://modal.com/docs/guide/spindle) server.

And when you need maximum control over your stack, you can bring your existing training code and infrastructure to a [Clustered Function](https://modal.com/docs/guide/multi-node-training).

Here, we walk through how to start training with Modal Dojo. See [here](https://modal.com/docs/examples/paint_flowers) for a more fully-fledged example.

## Run your first example [](https://modal.com/docs/guide/training-quickstart#run-your-first-example)

Install the package:

    uv add modal-dojo

Set up [Modal](https://modal.com/docs/cli/latest/setup) and the [dashboard](https://dojo.modal.dev/guides/dashboard):

    modal-dojo setup

And empower your agents with the [skill bundle](https://dojo.modal.dev/guides/agent):

    modal-dojo skills install

Then, it’s as easy as:

    import re

    from modal_dojo import (
        HuggingFaceDataset,
        Qwen3_5_4B,
        Qwen3_5_4B_Recipe,
        TrainConfig,
    )

    model = Qwen3_5_4B()

    async def gsm8k_rm(args, sample, **kwargs) -> float:
        text = model.parse_response(sample.response or "").content
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        pred = boxed[-1] if boxed else (re.findall(r"-?[\d,]+(?:\.\d+)?", text) or [""])[-1]
        try:
            return float(float(pred.replace(",", "")) == float(sample.label))
        except ValueError:
            return 0.0

    config = TrainConfig(
        model=model,
        dataset=HuggingFaceDataset(
            hf_repo="skrishna/gsm8k_only_answer",
            hf_split="train[:120]",
            input_column="text",
            output_column="label",
            input_format="text",
        ),
        recipe=Qwen3_5_4B_Recipe(
            custom_rm_function=gsm8k_rm,
        ),
    )

    if __name__ == "__main__":
        run = config.launch()
        print(run.training_run_id)

See the [documentation](https://dojo.modal.dev) for more information.

[LLM training quickstart](https://modal.com/docs/guide/training-quickstart#llm-training-quickstart)[Run your first example](https://modal.com/docs/guide/training-quickstart#run-your-first-example)
