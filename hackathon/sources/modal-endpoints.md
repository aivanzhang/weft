# Source snapshot: modal-endpoints

- Source: https://modal.com/docs/guide/endpoints
- Retrieved: 2026-10-26 03:00:26 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `32920e868476d18097a9135fc76a39f28c446e44481a699975655a0343383532`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

[Guide](https://modal.com/docs/guide/intro) [Introduction to Modal ](https://modal.com/docs/guide/intro)[Getting started ](https://modal.com/docs/guide/getting-started)[Functions](https://modal.com/docs/guide/functions) [Functions ](https://modal.com/docs/guide/functions)[Invocation methods ](https://modal.com/docs/guide/function-invocation-methods)[Function lookups ](https://modal.com/docs/guide/trigger-deployed-functions)[Container lifecycle management ](https://modal.com/docs/guide/lifecycle-functions)[Parametrized Functions ](https://modal.com/docs/guide/parametrized-functions)[Dynamic configuration ](https://modal.com/docs/guide/dynamic-function-config)[Sandboxes](https://modal.com/docs/guide/sandboxes) [Sandboxes ](https://modal.com/docs/guide/sandboxes)[Running commands ](https://modal.com/docs/guide/sandbox-spawn)[Networking and security ](https://modal.com/docs/guide/sandbox-networking)[File access ](https://modal.com/docs/guide/sandbox-files)[Snapshots ](https://modal.com/docs/guide/sandbox-snapshots)[Exit Snapshots Beta](https://modal.com/docs/guide/sandbox-exit-snapshots)[Pricing and resources ](https://modal.com/docs/guide/sandbox-resources)[Sidecars Beta](https://modal.com/docs/guide/sandbox-sidecars)[Secret injection Beta](https://modal.com/docs/guide/sandbox-secret-injection)[Endpoints](https://modal.com/docs/guide/endpoints) [Endpoints ](https://modal.com/docs/guide/endpoints)[Shared Endpoints ](https://modal.com/docs/guide/shared-endpoints)[Dedicated Endpoints ](https://modal.com/docs/guide/dedicated-endpoints)[Endpoint candidates ](https://modal.com/docs/guide/endpoint-candidates)[Endpoint integrations ](https://modal.com/docs/guide/endpoint-integrations)[Training](https://modal.com/docs/guide/training-quickstart) [Quickstart ](https://modal.com/docs/guide/training-quickstart)[Tinker-compatible API Beta](https://modal.com/docs/guide/spindle)[Multi-node training ](https://modal.com/docs/guide/multi-node-training)[Images](https://modal.com/docs/guide/images) [Defining Images ](https://modal.com/docs/guide/images)[Using existing container images ](https://modal.com/docs/guide/existing-images)[Named images ](https://modal.com/docs/guide/named-images)[Fast pull from registry ](https://modal.com/docs/guide/fast-pull-from-registry)[GPUs and other resources](https://modal.com/docs/guide/gpu) [GPU acceleration ](https://modal.com/docs/guide/gpu)[Using CUDA on Modal ](https://modal.com/docs/guide/cuda)[Configuring CPU, memory, and disk ](https://modal.com/docs/guide/resources)[Scaling out](https://modal.com/docs/guide/scale) [Scaling out ](https://modal.com/docs/guide/scale)[Input concurrency ](https://modal.com/docs/guide/concurrent-inputs)[Batch processing ](https://modal.com/docs/guide/batch-processing)[Job queues ](https://modal.com/docs/guide/job-queue)[Dynamic batching ](https://modal.com/docs/guide/dynamic-batching)[Multi-node Clusters ](https://modal.com/docs/guide/multi-node-clusters)[Deployment](https://modal.com/docs/guide/apps) [Apps, Functions, and entrypoints ](https://modal.com/docs/guide/apps)[Managing deployments ](https://modal.com/docs/guide/managing-deployments)[Continuous deployment ](https://modal.com/docs/guide/continuous-deployment)[Scheduled Functions ](https://modal.com/docs/guide/cron)[HTTP Applications](https://modal.com/docs/guide/servers) [Servers ](https://modal.com/docs/guide/servers)[Sticky Sessions ](https://modal.com/docs/guide/sticky-sessions)[Web Functions ](https://modal.com/docs/guide/webhooks)[Streaming endpoints ](https://modal.com/docs/guide/streaming-endpoints)[Web Function URLs ](https://modal.com/docs/guide/webhook-urls)[Request timeouts ](https://modal.com/docs/guide/webhook-timeouts)[Proxy Tokens ](https://modal.com/docs/guide/webhook-proxy-auth)[Networking](https://modal.com/docs/guide/tunnels) [Tunnels ](https://modal.com/docs/guide/tunnels)[Proxies Beta](https://modal.com/docs/guide/proxy-ips)[Cluster networking ](https://modal.com/docs/guide/private-networking)[Data sharing and storage](https://modal.com/docs/guide/local-data) [Passing local data ](https://modal.com/docs/guide/local-data)[Volumes ](https://modal.com/docs/guide/volumes)[Storing model weights ](https://modal.com/docs/guide/model-weights)[Cloud bucket mounts ](https://modal.com/docs/guide/cloud-bucket-mounts)[Dicts ](https://modal.com/docs/guide/dicts)[Queues ](https://modal.com/docs/guide/queues)[Dataset ingestion ](https://modal.com/docs/guide/dataset-ingestion)[Secrets and environment variables](https://modal.com/docs/guide/secrets) [Secrets ](https://modal.com/docs/guide/secrets)[Environment variables ](https://modal.com/docs/guide/environment_variables)[Performance](https://modal.com/docs/guide/cold-start) [Cold start performance ](https://modal.com/docs/guide/cold-start)[Memory Snapshots ](https://modal.com/docs/guide/memory-snapshots)[High-performance LLM inference ](https://modal.com/docs/guide/high-performance-llm-inference)[Reliability and robustness](https://modal.com/docs/guide/retries) [Failures and retries ](https://modal.com/docs/guide/retries)[Preemption ](https://modal.com/docs/guide/preemption)[Timeouts ](https://modal.com/docs/guide/timeouts)[GPU health ](https://modal.com/docs/guide/gpu-health)[Troubleshooting ](https://modal.com/docs/guide/troubleshooting)[Security and privacy](https://modal.com/docs/guide/security) [Security and privacy ](https://modal.com/docs/guide/security)[Data residency ](https://modal.com/docs/guide/data-residency)[Customer-supplied encryption keys Alpha](https://modal.com/docs/guide/customer-supplied-encryption-keys)[Audit logs ](https://modal.com/docs/guide/audit-logs)[Integrations](https://modal.com/docs/guide/oidc-integration) [Using OIDC to authenticate with external services ](https://modal.com/docs/guide/oidc-integration)[Connecting Modal to your Datadog account ](https://modal.com/docs/guide/datadog-integration)[Connecting Modal to your OpenTelemetry provider ](https://modal.com/docs/guide/otel-integration)[Okta SSO ](https://modal.com/docs/guide/okta-sso)[Microsoft Entra SSO ](https://modal.com/docs/guide/entra-sso)[Custom SAML SSO ](https://modal.com/docs/guide/saml-sso)[SCIM Integration ](https://modal.com/docs/guide/scim)[Slack notifications Beta](https://modal.com/docs/guide/slack-notifications)[Workspace & account settings](https://modal.com/docs/guide/workspaces) [Workspaces ](https://modal.com/docs/guide/workspaces)[Environments ](https://modal.com/docs/guide/environments)[Service users ](https://modal.com/docs/guide/service-users)[User groups ](https://modal.com/docs/guide/user-groups)[Role-Based Access Control (RBAC) ](https://modal.com/docs/guide/rbac)[Billing ](https://modal.com/docs/guide/billing)[Network egress billing ](https://modal.com/docs/guide/network-egress-billing)[Budgets ](https://modal.com/docs/guide/budgets)[Other topics](https://modal.com/docs/guide/feature-maturity) [Feature maturity ](https://modal.com/docs/guide/feature-maturity)[Restricted Functions ](https://modal.com/docs/guide/restricted-access)[JavaScript/Go SDKs Beta](https://modal.com/docs/guide/sdk-javascript-go)[Modal 1.0 migration guide ](https://modal.com/docs/guide/modal-1-0-migration)[File and project structure ](https://modal.com/docs/guide/project-structure)[Developing and debugging ](https://modal.com/docs/guide/developing-debugging)[Jupyter notebooks ](https://modal.com/docs/guide/jupyter-notebooks)[Asynchronous API usage ](https://modal.com/docs/guide/async)[Global variables ](https://modal.com/docs/guide/global-variables)[Region selection ](https://modal.com/docs/guide/region-selection)[GPU Metrics ](https://modal.com/docs/guide/gpu-metrics)[Search syntax ](https://modal.com/docs/guide/log-search-syntax)

------------------------------------------------------------------------

Copy page

# Endpoints

Modal Endpoints let you deploy models from the [Modal Library](https://modal.com/library)—or your own custom weights—as production-ready inference APIs.

- **Fast inference** — tuned open-source serving engines, with speculative decoding where supported, behind Modal’s low-latency request proxy.
- **Managed infrastructure** — Modal handles provisioning, routing, and capacity management.
- **Open and inspectable** — use familiar OpenAI- and Anthropic-compatible APIs; [inspect or adapt the generated source](https://modal.com/docs/guide/dedicated-endpoints#view-the-generated-source) behind a Dedicated Endpoint.

Choose between two serving modes:

|  | [Shared Endpoints](https://modal.com/docs/guide/shared-endpoints) | [Dedicated Endpoints](https://modal.com/docs/guide/dedicated-endpoints) |
|----|----|----|
| **Best for** | Fast, fully managed inference | Isolated capacity and custom models |
| **Models** | Selected models from the Modal Library | All Modal Library models, plus custom weights |
| **Billing** | Per token | Compute resources |
| **Capacity** | Managed by Modal | Configurable autoscaling, including scale-to-zero |

## Create an endpoint [](https://modal.com/docs/guide/endpoints#create-an-endpoint)

Browse the [Modal Library](https://modal.com/library) to choose a model and see which serving modes it supports. Then create an Endpoint from the [**Endpoints**](https://modal.com/endpoints) tab in the dashboard.

## Proxy tokens [](https://modal.com/docs/guide/endpoints#proxy-tokens)

Shared Endpoints always require a [Proxy Token](https://modal.com/docs/guide/webhook-proxy-auth). Dedicated Endpoints require one by default. Create one with the CLI:

    modal workspace proxy-tokens create

Join the token ID and secret with a period (`.`) and pass them as a bearer token:

    Authorization: Bearer wk-<id>.ws-<secret>

The combined value can be used as the API key in an OpenAI-compatible client. See [Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth) for environment scoping and other authentication options. Dedicated Endpoints can also be created with `--unauthenticated`.

## Call an endpoint [](https://modal.com/docs/guide/endpoints#call-an-endpoint)

Text-generation models on both Shared and Dedicated Endpoints can be called through the OpenAI-compatible Chat Completions and Responses APIs or the Anthropic-compatible Messages API. Embedding models can be called through the OpenAI-compatible Embeddings API.

### Model names [](https://modal.com/docs/guide/endpoints#model-names)

Set `model` to the value shown in the dashboard’s request examples. You can also retrieve it through the OpenAI-compatible Models API:

    curl "<your-endpoint-url>/v1/models" \
      -H "Authorization: Bearer $MODAL_PROXY_TOKEN_ID.$MODAL_PROXY_TOKEN_SECRET"

### Send a request [](https://modal.com/docs/guide/endpoints#send-a-request)

This example uses the Chat Completions API and a [proxy token](https://modal.com/docs/guide/endpoints#proxy-tokens):

    curl "<your-endpoint-url>/v1/chat/completions" \
      -H "Authorization: Bearer $MODAL_PROXY_TOKEN_ID.$MODAL_PROXY_TOKEN_SECRET" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "<model-name>",
        "messages": [{ "role": "user", "content": "Hello!" }]
      }'

See [Endpoint integrations](https://modal.com/docs/guide/endpoint-integrations) for connecting coding agents like OpenCode, Codex, and Claude Code to a Shared Endpoint.

## Affinity routing [](https://modal.com/docs/guide/endpoints#affinity-routing)

LLM serving engines cache the KV state of a prompt prefix, so a request that reuses the prefix of an earlier request is fastest when it lands on the same container. Multi-turn conversations and agent loops, where each request repeats the previous turns, benefit the most.

To keep the requests of one conversation together, send the same routing key in the `Modal-Routing-Affinity-Key` header on each of them. The routing key is an arbitrary string; use one per conversation, task, or agent run.

    curl "<your-endpoint-url>/v1/chat/completions" \
      -H "Authorization: Bearer $MODAL_PROXY_TOKEN_ID.$MODAL_PROXY_TOKEN_SECRET" \
      -H "Modal-Routing-Affinity-Key: $CUSTOM_CONVERSATION_ID" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "<model-name>",
        "messages": [{ "role": "user", "content": "Hello!" }]
      }'

Choose the routing key to match the shared prefix:

- **One key per conversation**, not per user or application. A single key shared by many unrelated conversations concentrates all of that traffic on one container.
- **Use a new key when the prefix changes**, for example after compacting or summarizing a long context. The cached prefix no longer applies, and a new key lets the request move to a container with free capacity.

Note that requests may be routed to a different container when the Endpoint scales or a container is replaced or overloaded.

[Endpoints](https://modal.com/docs/guide/endpoints#endpoints)[Create an endpoint](https://modal.com/docs/guide/endpoints#create-an-endpoint)[Proxy tokens](https://modal.com/docs/guide/endpoints#proxy-tokens)[Call an endpoint](https://modal.com/docs/guide/endpoints#call-an-endpoint)[Model names](https://modal.com/docs/guide/endpoints#model-names)[Send a request](https://modal.com/docs/guide/endpoints#send-a-request)[Affinity routing](https://modal.com/docs/guide/endpoints#affinity-routing)
