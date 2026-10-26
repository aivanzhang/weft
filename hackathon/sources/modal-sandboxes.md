# Source snapshot: modal-sandboxes

- Source: https://modal.com/docs/guide/sandboxes
- Retrieved: 2026-10-26 03:00:27 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `fa0ef54984694cfbde39a9a185f4b37e7db96171dc7903d2579302d53a692338`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

[Guide](https://modal.com/docs/guide/intro) [Introduction to Modal ](https://modal.com/docs/guide/intro)[Getting started ](https://modal.com/docs/guide/getting-started)[Functions](https://modal.com/docs/guide/functions) [Functions ](https://modal.com/docs/guide/functions)[Invocation methods ](https://modal.com/docs/guide/function-invocation-methods)[Function lookups ](https://modal.com/docs/guide/trigger-deployed-functions)[Container lifecycle management ](https://modal.com/docs/guide/lifecycle-functions)[Parametrized Functions ](https://modal.com/docs/guide/parametrized-functions)[Dynamic configuration ](https://modal.com/docs/guide/dynamic-function-config)[Sandboxes](https://modal.com/docs/guide/sandboxes) [Sandboxes ](https://modal.com/docs/guide/sandboxes)[Running commands ](https://modal.com/docs/guide/sandbox-spawn)[Networking and security ](https://modal.com/docs/guide/sandbox-networking)[File access ](https://modal.com/docs/guide/sandbox-files)[Snapshots ](https://modal.com/docs/guide/sandbox-snapshots)[Exit Snapshots Beta](https://modal.com/docs/guide/sandbox-exit-snapshots)[Pricing and resources ](https://modal.com/docs/guide/sandbox-resources)[Sidecars Beta](https://modal.com/docs/guide/sandbox-sidecars)[Secret injection Beta](https://modal.com/docs/guide/sandbox-secret-injection)[Endpoints](https://modal.com/docs/guide/endpoints) [Endpoints ](https://modal.com/docs/guide/endpoints)[Shared Endpoints ](https://modal.com/docs/guide/shared-endpoints)[Dedicated Endpoints ](https://modal.com/docs/guide/dedicated-endpoints)[Endpoint candidates ](https://modal.com/docs/guide/endpoint-candidates)[Endpoint integrations ](https://modal.com/docs/guide/endpoint-integrations)[Training](https://modal.com/docs/guide/training-quickstart) [Quickstart ](https://modal.com/docs/guide/training-quickstart)[Tinker-compatible API Beta](https://modal.com/docs/guide/spindle)[Multi-node training ](https://modal.com/docs/guide/multi-node-training)[Images](https://modal.com/docs/guide/images) [Defining Images ](https://modal.com/docs/guide/images)[Using existing container images ](https://modal.com/docs/guide/existing-images)[Named images ](https://modal.com/docs/guide/named-images)[Fast pull from registry ](https://modal.com/docs/guide/fast-pull-from-registry)[GPUs and other resources](https://modal.com/docs/guide/gpu) [GPU acceleration ](https://modal.com/docs/guide/gpu)[Using CUDA on Modal ](https://modal.com/docs/guide/cuda)[Configuring CPU, memory, and disk ](https://modal.com/docs/guide/resources)[Scaling out](https://modal.com/docs/guide/scale) [Scaling out ](https://modal.com/docs/guide/scale)[Input concurrency ](https://modal.com/docs/guide/concurrent-inputs)[Batch processing ](https://modal.com/docs/guide/batch-processing)[Job queues ](https://modal.com/docs/guide/job-queue)[Dynamic batching ](https://modal.com/docs/guide/dynamic-batching)[Multi-node Clusters ](https://modal.com/docs/guide/multi-node-clusters)[Deployment](https://modal.com/docs/guide/apps) [Apps, Functions, and entrypoints ](https://modal.com/docs/guide/apps)[Managing deployments ](https://modal.com/docs/guide/managing-deployments)[Continuous deployment ](https://modal.com/docs/guide/continuous-deployment)[Scheduled Functions ](https://modal.com/docs/guide/cron)[HTTP Applications](https://modal.com/docs/guide/servers) [Servers ](https://modal.com/docs/guide/servers)[Sticky Sessions ](https://modal.com/docs/guide/sticky-sessions)[Web Functions ](https://modal.com/docs/guide/webhooks)[Streaming endpoints ](https://modal.com/docs/guide/streaming-endpoints)[Web Function URLs ](https://modal.com/docs/guide/webhook-urls)[Request timeouts ](https://modal.com/docs/guide/webhook-timeouts)[Proxy Tokens ](https://modal.com/docs/guide/webhook-proxy-auth)[Networking](https://modal.com/docs/guide/tunnels) [Tunnels ](https://modal.com/docs/guide/tunnels)[Proxies Beta](https://modal.com/docs/guide/proxy-ips)[Cluster networking ](https://modal.com/docs/guide/private-networking)[Data sharing and storage](https://modal.com/docs/guide/local-data) [Passing local data ](https://modal.com/docs/guide/local-data)[Volumes ](https://modal.com/docs/guide/volumes)[Storing model weights ](https://modal.com/docs/guide/model-weights)[Cloud bucket mounts ](https://modal.com/docs/guide/cloud-bucket-mounts)[Dicts ](https://modal.com/docs/guide/dicts)[Queues ](https://modal.com/docs/guide/queues)[Dataset ingestion ](https://modal.com/docs/guide/dataset-ingestion)[Secrets and environment variables](https://modal.com/docs/guide/secrets) [Secrets ](https://modal.com/docs/guide/secrets)[Environment variables ](https://modal.com/docs/guide/environment_variables)[Performance](https://modal.com/docs/guide/cold-start) [Cold start performance ](https://modal.com/docs/guide/cold-start)[Memory Snapshots ](https://modal.com/docs/guide/memory-snapshots)[High-performance LLM inference ](https://modal.com/docs/guide/high-performance-llm-inference)[Reliability and robustness](https://modal.com/docs/guide/retries) [Failures and retries ](https://modal.com/docs/guide/retries)[Preemption ](https://modal.com/docs/guide/preemption)[Timeouts ](https://modal.com/docs/guide/timeouts)[GPU health ](https://modal.com/docs/guide/gpu-health)[Troubleshooting ](https://modal.com/docs/guide/troubleshooting)[Security and privacy](https://modal.com/docs/guide/security) [Security and privacy ](https://modal.com/docs/guide/security)[Data residency ](https://modal.com/docs/guide/data-residency)[Customer-supplied encryption keys Alpha](https://modal.com/docs/guide/customer-supplied-encryption-keys)[Audit logs ](https://modal.com/docs/guide/audit-logs)[Integrations](https://modal.com/docs/guide/oidc-integration) [Using OIDC to authenticate with external services ](https://modal.com/docs/guide/oidc-integration)[Connecting Modal to your Datadog account ](https://modal.com/docs/guide/datadog-integration)[Connecting Modal to your OpenTelemetry provider ](https://modal.com/docs/guide/otel-integration)[Okta SSO ](https://modal.com/docs/guide/okta-sso)[Microsoft Entra SSO ](https://modal.com/docs/guide/entra-sso)[Custom SAML SSO ](https://modal.com/docs/guide/saml-sso)[SCIM Integration ](https://modal.com/docs/guide/scim)[Slack notifications Beta](https://modal.com/docs/guide/slack-notifications)[Workspace & account settings](https://modal.com/docs/guide/workspaces) [Workspaces ](https://modal.com/docs/guide/workspaces)[Environments ](https://modal.com/docs/guide/environments)[Service users ](https://modal.com/docs/guide/service-users)[User groups ](https://modal.com/docs/guide/user-groups)[Role-Based Access Control (RBAC) ](https://modal.com/docs/guide/rbac)[Billing ](https://modal.com/docs/guide/billing)[Network egress billing ](https://modal.com/docs/guide/network-egress-billing)[Budgets ](https://modal.com/docs/guide/budgets)[Other topics](https://modal.com/docs/guide/feature-maturity) [Feature maturity ](https://modal.com/docs/guide/feature-maturity)[Restricted Functions ](https://modal.com/docs/guide/restricted-access)[JavaScript/Go SDKs Beta](https://modal.com/docs/guide/sdk-javascript-go)[Modal 1.0 migration guide ](https://modal.com/docs/guide/modal-1-0-migration)[File and project structure ](https://modal.com/docs/guide/project-structure)[Developing and debugging ](https://modal.com/docs/guide/developing-debugging)[Jupyter notebooks ](https://modal.com/docs/guide/jupyter-notebooks)[Asynchronous API usage ](https://modal.com/docs/guide/async)[Global variables ](https://modal.com/docs/guide/global-variables)[Region selection ](https://modal.com/docs/guide/region-selection)[GPU Metrics ](https://modal.com/docs/guide/gpu-metrics)[Search syntax ](https://modal.com/docs/guide/log-search-syntax)

------------------------------------------------------------------------

Copy page

# Sandboxes

This page is a high-level guide to Sandboxes, secure containers for executing untrusted code on Modal. Sandboxes are useful for tasks like executing code generated by an LLM or isolating user-generated code.

For full reference documentation on the `modal.Sandbox` interface, visit our SDK docs ([Python](https://modal.com/docs/sdk/py/latest/Sandbox), [JavaScript](https://modal.com/docs/sdk/js/latest/Sandbox), [Go](https://modal.com/docs/sdk/go/latest/Sandbox)).

Sandboxes can be created using the [`Sandbox.create`](https://modal.com/docs/sdk/py/latest/Sandbox#create) constructor:

Python

Python (async)

JavaScript

Go

    sb_app = modal.App.lookup("my-app", create_if_missing=True)
    sb = modal.Sandbox.create(app=sb_app)

    p = sb.exec("python", "-c", "print('hello')", timeout=3)
    print(p.stdout.read())

    p = sb.exec("bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done", timeout=5)
    for line in p.stdout:
        # Avoid double newlines by using end="".
        print(line, end="")

    sb.terminate()

    sb_app = await modal.App.lookup.aio("my-app", create_if_missing=True)
    sb = await modal.Sandbox.create.aio(app=sb_app)

    p = await sb.exec.aio("python", "-c", "print('hello')", timeout=3)
    print(await p.stdout.read.aio())

    p = await sb.exec.aio("bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done", timeout=5)
    async for line in p.stdout:
        # Avoid double newlines by using end="".
        print(line, end="")

    await sb.terminate.aio()

    import { ModalClient } from "modal";

    const modal = new ModalClient();
    const app = await modal.apps.fromName("my-app", {
      createIfMissing: true,
    });
    const image = modal.images.fromRegistry("python:3.13-slim");

    const sb = await modal.sandboxes.create(app, image);

    const p = await sb.exec(["python", "-c", "print('hello')"], {
      timeoutMs: 3 * 1000,
    });
    console.log(await p.stdout.readText());

    const p2 = await sb.exec(
      ["bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done"],
      { timeoutMs: 5 * 1000 },
    );
    for await (const line of p2.stdout) {
      process.stdout.write(line);
    }

    await sb.terminate();

    package main

    import (
        "context"
        "fmt"
        "io"
        "os"
        "time"

        modal "github.com/modal-labs/modal-client/go"
    )

    func main() {
        ctx := context.Background()
        mc, _ := modal.NewClient()

        app, _ := mc.Apps.FromName(ctx, "my-app", &modal.AppFromNameParams{
            CreateIfMissing: true,
        })
        image := mc.Images.FromRegistry("python:3.13-slim", nil)

        sb, _ := mc.Sandboxes.Create(ctx, app, image, nil)
        defer sb.Terminate(ctx, nil)

        p, _ := sb.Exec(ctx, []string{"python", "-c", "print('hello')"}, &modal.SandboxExecParams{
            Timeout: 3 * time.Second,
        })
        stdout, _ := io.ReadAll(p.Stdout)
        fmt.Println(string(stdout))

        p2, _ := sb.Exec(ctx, []string{"bash", "-c", "for i in {1..10}; do date +%T; sleep 0.5; done"}, &modal.SandboxExecParams{
            Timeout: 5 * time.Second,
        })
        io.Copy(os.Stdout, p2.Stdout)
    }

**Note:** you can run the above example as a script directly with `python my_script.py`. `modal run` is not needed here since there is no [entrypoint](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps).

Sandboxes require an [`App`](https://modal.com/docs/guide/apps) to be passed when spawned from outside of a Modal container. You may pass in a regular `App` object or look one up by name with [`App.lookup`](https://modal.com/docs/sdk/py/latest/App#lookup). The `create_if_missing` flag on `App.lookup` will create an `App` with the given name if it doesn’t exist.

## Configuration [](https://modal.com/docs/guide/sandboxes#configuration)

Sandboxes are configured through arguments to [`Sandbox.create`](https://modal.com/docs/sdk/py/latest/Sandbox#create). Refer to `Sandbox.create` for full documentation on Sandbox configs, or see the [JavaScript](https://modal.com/docs/sdk/js/latest/Sandbox) and [Go](https://modal.com/docs/sdk/go/latest/Sandbox) SDK references.

Sandboxes support nearly all configuration options found in regular Modal [Functions](https://modal.com/docs/sdk/py/latest/Function). For example, [named Images](https://modal.com/docs/guide/named-images) and [Volumes](https://modal.com/docs/guide/volumes) can be used just as with Functions:

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(
        # prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
        image=modal.Image.from_name("sandbox-pandas"),
        volumes={"/data": modal.Volume.from_name("data-volume", create_if_missing=True)},
        app=sb_app,
    )

    sb = await modal.Sandbox.create.aio(
        # prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
        image=modal.Image.from_name("sandbox-pandas"),
        volumes={"/data": modal.Volume.from_name("data-volume", create_if_missing=True)},
        app=sb_app,
    )

    // prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    const image = await modal.images.fromName("sandbox-pandas");
    const volume = modal.volumes.fromName("my-volume");
    const sb = await modal.sandboxes.create(app, image, {
      volumes: { "/data": volume },
      workdir: "/repo",
    });

    // prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    image, err := mc.Images.FromName(ctx, "sandbox-pandas", nil)
    volume := mc.Volumes.FromName("my-volume", nil)
    sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Volumes: map[string]*modal.Volume{"/data": volume},
      Workdir: "/repo",
    })

Explicit CPU and memory [resource configuration](https://modal.com/docs/guide/sandbox-resources) is not necessary for many tasks because Sandboxes can opportunistically [burst above their requested resources](https://modal.com/docs/guide/sandbox-resources#pay-for-what-you-use) when needed. Heavier jobs can be provisioned with additional resources to guarantee availability:

Python

JavaScript

Go

    import modal

    app = modal.App.lookup("my-app", create_if_missing=True)

    sb = modal.Sandbox.create(
        cpu=0.5,
        memory=512,
        app=app,
    )

    import { ModalClient } from "modal";

    const modal = new ModalClient();
    const app = await modal.apps.fromName("my-app", { createIfMissing: true });
    const image = modal.images.fromRegistry("python:3.13-slim");

    const sb = await modal.sandboxes.create(app, image, {
      cpu: 0.5,
      memoryMiB: 512,
    });

    package main

    import (
        "context"

        modal "github.com/modal-labs/modal-client/go"
    )

    func main() {
        ctx := context.Background()
        mc, _ := modal.NewClient()

        app, _ := mc.Apps.FromName(ctx, "my-app", &modal.AppFromNameParams{
            CreateIfMissing: true,
        })
        image := mc.Images.FromRegistry("python:3.13-slim", nil)

        sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
            CPU:       0.5,
            MemoryMiB: 512,
        })
    }

Sandboxes are billed at `max(request, actual)`. See [Sandbox pricing](https://modal.com/pricing#sandboxes) for rates.

## Lifecycle [](https://modal.com/docs/guide/sandboxes#lifecycle)

### Events [](https://modal.com/docs/guide/sandboxes#events)

Every Sandbox moves through a series of lifecycle events as it progresses from creation to completion. Understanding these events is useful for monitoring, debugging, and building automations that react to Sandbox state changes.

The lifecycle events, in order, are:

1.  **Created** — The Sandbox has been requested and registered with Modal. At this point the Sandbox object exists and has an ID, but no compute resources have been allocated yet. This is the initial state immediately after calling `Sandbox.create`.

2.  **Scheduled** — The Sandbox has been scheduled to a specific worker. The worker is now provisioning the resources the Sandbox needs (CPU, memory, GPU, volumes, etc.) and preparing the container environment. The Sandbox will transition to **Started** once the container is fully initialized.

3.  **Started** — The Sandbox’s container has been launched on a worker and the entrypoint process (if any) is running. At this point you can begin executing commands inside the Sandbox with `sandbox.exec(...)`. Network tunnels and volume mounts are active.

4.  **Ready** — If [readiness probes](https://modal.com/docs/guide/sandboxes#readiness-probes) are enabled for the Sandbox, this event fires once the probe succeeds, indicating that the service inside the Sandbox is fully initialized and ready to accept traffic. This is especially useful for Sandboxes running web servers or other services that need warm-up time before they can handle requests. If readiness probes are not configured, this event is skipped.

5.  **Finished** — The Sandbox has stopped running. This can happen for several reasons: the entrypoint process exited on its own, the Sandbox was explicitly terminated (via the dashboard or `sandbox.terminate()`), the timeout or idle timeout was reached, or an out-of-memory condition occurred. Once finished, no further commands can be executed inside the Sandbox. You can learn more about why a Sandbox stopped running in the dashboard or by examining the exit code returned from `sandbox.poll()`.

### Timeouts [](https://modal.com/docs/guide/sandboxes#timeouts)

Sandboxes have a default maximum lifetime of 5 minutes. You can change this by passing a `timeout` of up to 24 hours to the `Sandbox.create(...)` function.

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(app=sb_app, timeout=10*60)  # 10 minutes

    sb = await modal.Sandbox.create.aio(app=sb_app, timeout=10*60)  # 10 minutes

    const sb = await modal.sandboxes.create(app, image, {
      timeoutMs: 10 * 60 * 1000, // 10 minutes
    });

    sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
        Timeout: 10 * time.Minute,
    })

If you need a Sandbox to run for more than 24 hours, we recommend using [Filesystem Snapshots](https://modal.com/docs/guide/sandbox-snapshots) to preserve its state, and then restore from that snapshot with a subsequent Sandbox.

### Idle Timeouts [](https://modal.com/docs/guide/sandboxes#idle-timeouts)

Sandboxes can also be automatically terminated after a period of inactivity - you can do this by setting the `idle_timeout` parameter. A Sandbox is considered active if any of the following are true:

1.  It has an active [command](https://modal.com/docs/guide/sandbox-spawn) running (via [`sb.exec(...)`](https://modal.com/docs/sdk/py/latest/Sandbox#exec))
2.  Its stdin is being written to (via [`sb.stdin.write()`](https://modal.com/docs/sdk/py/latest/Sandbox#stdin))
3.  It has an open TCP connection over one of its [Tunnels](https://modal.com/docs/guide/tunnels)

### Readiness Probes [](https://modal.com/docs/guide/sandboxes#readiness-probes)

After a Sandbox starts, you often need to run custom initialization logic before it’s ready for use — pulling code with `git pull`, installing dependencies, starting a server, writing config files, or other setup that isn’t baked into the image. Readiness probes give you a way to track when that initialization is complete, so you don’t have to build the polling or signaling yourself. Modal also uses probe results to give you observability into how long this startup phase typically takes.

A readiness probe is a check that Modal runs automatically inside the Sandbox at a configurable interval. You can then call `wait_until_ready()` to block until the probe succeeds.

There are two types of readiness probes:

- **TCP probe** — Checks whether a TCP port inside the Sandbox is accepting connections. This is the most common choice when your startup logic includes launching a server.
- **Exec probe** — Runs an arbitrary command inside the Sandbox and succeeds when the command exits with status code 0. Use this for any other readiness condition: checking that a file exists, verifying a setup script has completed, confirming dependencies are installed, etc.

Both probe types accept an `interval_ms` parameter (default: 100ms) that controls how frequently the check is retried until it succeeds.

#### TCP readiness probe [](https://modal.com/docs/guide/sandboxes#tcp-readiness-probe)

Use a TCP probe when your Sandbox starts a server and you want to wait until it’s listening on a port:

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(
        "python3", "-m", "http.server", "8080",
        readiness_probe=modal.Probe.with_tcp(8080),
        app=sb_app,
    )

    # Blocks until port 8080 is accepting connections
    sb.wait_until_ready()

    # The server is now ready — interact with it via tunnels, exec, etc.
    sb.terminate()

    sb = await modal.Sandbox.create.aio(
        "python3", "-m", "http.server", "8080",
        readiness_probe=modal.Probe.with_tcp(8080),
        app=sb_app,
    )

    # Blocks until port 8080 is accepting connections
    await sb.wait_until_ready.aio()

    # The server is now ready — interact with it via tunnels, exec, etc.
    await sb.terminate.aio()

    import { ModalClient, Probe } from "modal";

    const modal = new ModalClient();
    const app = await modal.apps.fromName("my-app", { createIfMissing: true });
    const image = modal.images.fromRegistry("python:3.13-slim");

    const sb = await modal.sandboxes.create(app, image, {
      command: ["python3", "-m", "http.server", "8080"],
      readinessProbe: Probe.withTcp(8080),
    });

    // Blocks until port 8080 is accepting connections
    await sb.waitUntilReady();

    // The server is now ready — interact with it via tunnels, exec, etc.
    await sb.terminate();

    package main

    import (
        "context"
        "time"

        modal "github.com/modal-labs/modal-client/go"
    )

    func main() {
        ctx := context.Background()
        mc, _ := modal.NewClient()

        app, _ := mc.Apps.FromName(ctx, "my-app", &modal.AppFromNameParams{
            CreateIfMissing: true,
        })
        image := mc.Images.FromRegistry("python:3.13-slim", nil)
        probe, _ := modal.NewTCPProbe(8080, nil)

        sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
            Command:        []string{"python3", "-m", "http.server", "8080"},
            ReadinessProbe: probe,
        })

        // Blocks until port 8080 is accepting connections
        sb.WaitUntilReady(ctx, 5*time.Minute)

        // The server is now ready — interact with it via tunnels, exec, etc.
        sb.Terminate(ctx, nil)
    }

#### Exec readiness probe [](https://modal.com/docs/guide/sandboxes#exec-readiness-probe)

Use an exec probe when readiness depends on something other than a TCP port — for example, waiting for a file to be created or a setup script to complete:

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(
        "bash", "-c", "sleep 5 && touch /tmp/ready && sleep 3600",
        readiness_probe=modal.Probe.with_exec(
            "sh", "-c", "test -f /tmp/ready",
            interval_ms=250,
        ),
        app=sb_app,
    )

    # Blocks until "test -f /tmp/ready" exits with code 0
    sb.wait_until_ready()

    # The sandbox is now ready
    p = sb.exec("cat", "/tmp/ready")
    sb.terminate()

    sb = await modal.Sandbox.create.aio(
        "bash", "-c", "sleep 5 && touch /tmp/ready && sleep 3600",
        readiness_probe=modal.Probe.with_exec(
            "sh", "-c", "test -f /tmp/ready",
            interval_ms=250,
        ),
        app=sb_app,
    )

    # Blocks until "test -f /tmp/ready" exits with code 0
    await sb.wait_until_ready.aio()

    # The sandbox is now ready
    p = await sb.exec.aio("cat", "/tmp/ready")
    await sb.terminate.aio()

    import { ModalClient, Probe } from "modal";

    const modal = new ModalClient();
    const app = await modal.apps.fromName("my-app", { createIfMissing: true });
    const image = modal.images.fromRegistry("python:3.13-slim");

    const sb = await modal.sandboxes.create(app, image, {
      command: ["bash", "-c", "sleep 5 && touch /tmp/ready && sleep 3600"],
      readinessProbe: Probe.withExec(["sh", "-c", "test -f /tmp/ready"], {
        intervalMs: 250,
      }),
    });

    // Blocks until "test -f /tmp/ready" exits with code 0
    await sb.waitUntilReady();

    // The sandbox is now ready
    await sb.terminate();

    package main

    import (
        "context"
        "time"

        modal "github.com/modal-labs/modal-client/go"
    )

    func main() {
        ctx := context.Background()
        mc, _ := modal.NewClient()

        app, _ := mc.Apps.FromName(ctx, "my-app", &modal.AppFromNameParams{
            CreateIfMissing: true,
        })
        image := mc.Images.FromRegistry("python:3.13-slim", nil)
        probe, _ := modal.NewExecProbe(
            []string{"sh", "-c", "test -f /tmp/ready"},
            &modal.ExecProbeParams{Interval: 250 * time.Millisecond},
        )

        sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
            Command:        []string{"bash", "-c", "sleep 5 && touch /tmp/ready && sleep 3600"},
            ReadinessProbe: probe,
        })

        // Blocks until "test -f /tmp/ready" exits with code 0
        sb.WaitUntilReady(ctx, 5*time.Minute)

        // The sandbox is now ready
        sb.Terminate(ctx, nil)
    }

**Note:** Readiness probes will run for a maximum of 5 minutes. If the probe does not succeed within that window, `wait_until_ready()` will raise a `modal.exception.TimeoutError`. This is Modal’s own error class rather than the builtin `TimeoutError`, so a bare `except TimeoutError` will not catch it. The probe timeout does **not** automatically terminate the Sandbox — you may want to catch the error and explicitly terminate the Sandbox if readiness is never achieved:

Python

Python (async)

JavaScript

Go

    try:
        sb.wait_until_ready()
    except modal.exception.TimeoutError:
        print("Sandbox failed to become ready")
        sb.terminate()

    try:
        await sb.wait_until_ready.aio()
    except modal.exception.TimeoutError:
        print("Sandbox failed to become ready")
        await sb.terminate.aio()

    try {
      await sb.waitUntilReady();
    } catch (err) {
      console.log("Sandbox failed to become ready");
      await sb.terminate();
    }

    if err := sb.WaitUntilReady(ctx, 5*time.Minute); err != nil {
        fmt.Println("Sandbox failed to become ready")
        sb.Terminate(ctx, nil)
    }

If you call `wait_until_ready()` on a Sandbox that was not configured with a readiness probe, an error will be raised. Similarly, calling it after the Sandbox has been terminated will raise an error. However, calling `wait_until_ready()` after the Sandbox has already become ready returns immediately.

## Return Codes [](https://modal.com/docs/guide/sandboxes#return-codes)

[Unix-style exit codes](https://tldp.org/LDP/abs/html/exitcodes.html) are provided to help diagnose conditions such as success, manual termination, or out-of-memory.

They are available on both:

- Processes in the sandbox (via [`ContainerProcess.returncode`](https://modal.com/docs/sdk/py/latest/container_process#returncode) / [`ContainerProcess.poll()`](https://modal.com/docs/sdk/py/latest/container_process#poll))
- The Sandbox itself (via [`Sandbox.returncode`](https://modal.com/docs/sdk/py/latest/Sandbox#returncode) / [`Sandbox.poll()`](https://modal.com/docs/sdk/py/latest/Sandbox#poll))

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(app=sb_app)

    # Read returncode of individual process
    p = sb.exec("sh", "-c", "exit 42")
    p.wait()
    print(p.returncode) # 42

    # Read returncode of finished sandbox
    # Terminate sends a SIGKILL, code 137
    sb.terminate(wait=True)
    print(sb.returncode) # 137

    sb = await modal.Sandbox.create.aio(app=sb_app)

    # Read returncode of individual process
    p = await sb.exec.aio("sh", "-c", "exit 42")
    await p.wait.aio()
    print(p.returncode) # 42

    # Read returncode of finished sandbox
    # Terminate sends a SIGKILL, code 137
    await sb.terminate.aio(wait=True)
    print(sb.returncode) # 137

    const sb = await modal.sandboxes.create(app, image);

    // Read returncode of individual process
    const p = await sb.exec(["sh", "-c", "exit 42"]);
    const returnCode = await p.wait();
    console.log(returnCode); // 42

    // Read returncode of finished sandbox
    // Terminate sends a SIGKILL, code 137
    const returnCodeSb = await sb.terminate({ wait: true });
    console.log(returnCodeSb); // 137

    sb, _ := mc.Sandboxes.Create(ctx, app, image, nil)

    // Read returncode of individual process
    p, _ := sb.Exec(ctx, []string{"sh", "-c", "exit 42"}, nil)
    returnCode, _ := p.Wait(ctx)
    fmt.Println(returnCode) // 42

    // Read returncode of finished sandbox
    // Terminate sends a SIGKILL, code 137
    returnCodeSb, _ := sb.Terminate(ctx, &modal.SandboxTerminateParams{Wait: true})
    fmt.Println(returnCodeSb) // 137

## Runtimes [](https://modal.com/docs/guide/sandboxes#runtimes)

Sandboxes run on one of two runtimes:

- [**gVisor**](https://gvisor.dev/), a container runtime developed by Google, provides strong isolation and is suitable for most workloads.
- **VMs** run the Sandbox in a virtual machine with its own Linux kernel. Use the VM runtime for workloads that expect a full Linux environment, such as running Docker inside the Sandbox, mounting [FUSE](https://man7.org/linux/man-pages/man4/fuse.4.html) filesystems, or managing resources with nested [cgroups](https://man7.org/linux/man-pages/man7/cgroups.7.html).

Switch runtimes via the `runtime` parameter, or leave unset to let Modal choose for you.

Python

JavaScript

Go

    sb = modal.Sandbox.create(app=sb_app, runtime="vm")  # or: runtime="gvisor"
    p = sb.exec("uname", "-srn")
    print(p.stdout.read()) # Linux modal 7.2.6
    sb.terminate()

    const sb = await modal.sandboxes.create(app, image, { runtime: "vm" });

    const p = await sb.exec(["uname", "-srn"]);
    console.log(await p.stdout.readText()); // Linux modal 7.2.6

    sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
        Runtime: modal.SandboxRuntimeVM,
    })
    p, err := sb.Exec(ctx, []string{"uname", "-srn"}, nil)
    stdout, err := io.ReadAll(p.Stdout)
    fmt.Println(string(stdout)) // Linux modal 7.2.6

[GPU](https://modal.com/docs/guide/gpu) Sandboxes are only supported with `runtime="gvisor"`.

Nested virtualization is available on the [Enterprise plan](https://modal.com/pricing) by setting `experimental_options={"enable_nested_virt": True}` on a VM Sandbox.

### Running Docker in a Sandbox [](https://modal.com/docs/guide/sandboxes#running-docker-in-a-sandbox)

Use the VM runtime to run Docker in a Sandbox. Start `dockerd` as the Sandbox entrypoint and run containers with `sb.exec`. Containers on the Docker bridge can reach each other, and `/var/lib/docker` is included in [Filesystem Snapshots](https://modal.com/docs/guide/sandbox-snapshots#filesystem-snapshots).

Python

JavaScript

Go

    # The named Image has Docker installed. It was built and published ahead of time:
    #
    #     image = (
    #         modal.Image.from_registry("ubuntu:24.04")
    #         .env({"DEBIAN_FRONTEND": "noninteractive"})
    #         .apt_install("docker.io")
    #     )
    #     image.build(sb_app).publish("sandbox-docker")

    sb = modal.Sandbox.create(
        "dockerd",
        app=sb_app,
        # prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
        image=modal.Image.from_name("sandbox-docker"),
        runtime="vm",
        readiness_probe=modal.Probe.with_exec("docker", "info", interval_ms=500),
    )
    sb.wait_until_ready()

    p = sb.exec("docker", "run", "--rm", "alpine", "echo", "hello from Docker")
    p.wait()
    print(p.stdout.read())

    sb.terminate()

    import { Probe } from "modal";

    const dockerImage = modal.images
      .fromRegistry("ubuntu:24.04")
      .dockerfileCommands([
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y docker.io",
      ]);

    const sb = await modal.sandboxes.create(app, dockerImage, {
      command: ["dockerd"],
      runtime: "vm",
      readinessProbe: Probe.withExec(["docker", "info"], {
        intervalMs: 500,
      }),
    });
    await sb.waitUntilReady();

    const p = await sb.exec(
      ["docker", "run", "--rm", "alpine", "echo", "hello from Docker"],
      { stdout: "pipe" },
    );
    await p.wait();
    console.log(await p.stdout.readText());

    await sb.terminate();

    dockerImage := mc.Images.FromRegistry("ubuntu:24.04", nil).DockerfileCommands([]string{
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update && apt-get install -y docker.io",
    }, nil)
    probe, _ := modal.NewExecProbe(
        []string{"docker", "info"},
        &modal.ExecProbeParams{Interval: 500 * time.Millisecond},
    )
    sb, _ := mc.Sandboxes.Create(ctx, app, dockerImage, &modal.SandboxCreateParams{
        Command:        []string{"dockerd"},
        Runtime:        modal.SandboxRuntimeVM,
        ReadinessProbe: probe,
    })
    sb.WaitUntilReady(ctx, 5*time.Minute, nil)

    p, _ := sb.Exec(ctx, []string{
        "docker", "run", "--rm", "alpine", "echo", "hello from Docker",
    }, &modal.SandboxExecParams{Stdout: modal.Pipe})
    p.Wait(ctx, nil)
    stdout, _ := io.ReadAll(p.Stdout)
    fmt.Println(string(stdout))

    sb.Terminate(ctx, nil)

## Environments [](https://modal.com/docs/guide/sandboxes#environments)

### Environment variables [](https://modal.com/docs/guide/sandboxes#environment-variables)

You can set environment variables using inline secrets:

Python

Python (async)

JavaScript

Go

    secret = modal.Secret.from_dict({"MY_SECRET": "hello"})

    sb = modal.Sandbox.create(
        secrets=[secret],
        app=sb_app,
    )
    p = sb.exec("bash", "-c", "echo $MY_SECRET")
    print(p.stdout.read())

    secret = modal.Secret.from_dict({"MY_SECRET": "hello"})

    sb = await modal.Sandbox.create.aio(
        secrets=[secret],
        app=sb_app,
    )
    p = await sb.exec.aio("bash", "-c", "echo $MY_SECRET")
    print(await p.stdout.read.aio())

    const secret = modal.secrets.fromObject({ MY_SECRET: "hello" });
    const image = modal.images.fromRegistry("python:3.13-slim");

    const sb = await modal.sandboxes.create(app, image, {
      secrets: [secret],
    });
    const p = await sb.exec(["bash", "-c", "echo $MY_SECRET"]);
    console.log(await p.stdout.readText());

    secret, err := mc.Secrets.FromMap(ctx, map[string]string{"MY_SECRET": "hello"}, nil)
    image := mc.Images.FromRegistry("python:3.13-slim", nil)

    sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Secrets: []*modal.Secret{secret},
    })
    p, err := sb.Exec(ctx, []string{"bash", "-c", "echo $MY_SECRET"}, nil)
    stdout, err := io.ReadAll(p.Stdout)
    fmt.Println(string(stdout))

### OIDC identity tokens [](https://modal.com/docs/guide/sandboxes#oidc-identity-tokens)

Pass `include_oidc_identity_token=True` to `modal.Sandbox.create` to inject a `MODAL_IDENTITY_TOKEN` environment variable containing an OIDC token into the Sandbox. See the [OIDC integration guide](https://modal.com/docs/guide/oidc-integration) for details.

### Custom Images [](https://modal.com/docs/guide/sandboxes#custom-images)

Sandboxes support [custom images](https://modal.com/docs/guide/images) just as Functions do. These can be defined using [method chaining](https://modal.com/docs/guide/images) or by referencing an [existing Image in an external container registry](https://modal.com/docs/guide/existing-images).

#### Separating Image builds from Sandbox creation [](https://modal.com/docs/guide/sandboxes#separating-image-builds-from-sandbox-creation)

To avoid blocking creation of new Sandboxes on rebuilding an invalidated Image, it’s recommended to use Modal’s named Images with sandboxes, rather than using inline Image definitions.

Use [`Image.build`](https://modal.com/docs/sdk/py/latest/Image#build) to trigger Image builds as part of a deployment flow or at a regular interval (e.g., in a [scheduled job](https://modal.com/docs/guide/cron) or CI pipeline), then publish the result as a [named image](https://modal.com/docs/guide/named-images). Wrap the build in `modal.enable_output()` to stream its build logs.

Sandboxes can then use [`Image.from_name`](https://modal.com/docs/sdk/py/latest/Image#from_name) to reference the Image in a way that’s guaranteed to not block on rebuilds.

Python

Python (async)

JavaScript

Go

    # build_sandbox_image.py
    app = modal.App.lookup("sandbox-app", create_if_missing=True)

    # Method-chained image
    image = modal.Image.debian_slim().pip_install("pandas")

    # Or, for an external registry image with a fixed tag:
    # image = modal.Image.from_registry("ubuntu:24.04")

    with modal.enable_output():
        image.build(app=app).publish("sandbox-runtime")

    # sandbox_app.py
    app = modal.App.lookup("sandbox-app", create_if_missing=True)

    # prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    image = modal.Image.from_name("sandbox-runtime")
    modal.Sandbox.create(app=app, image=image)

    # build_sandbox_image.py
    app = await modal.App.lookup.aio("sandbox-app", create_if_missing=True)

    # Method-chained image
    image = modal.Image.debian_slim().pip_install("pandas")

    # Or, for an external registry image with a fixed tag:
    # image = modal.Image.from_registry("ubuntu:24.04")

    with modal.enable_output():
        await image.build.aio(app)
    await image.publish.aio("sandbox-runtime")

    # app.py
    app = await modal.App.lookup.aio("sandbox-app", create_if_missing=True)

    # prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    image = modal.Image.from_name("sandbox-runtime")
    await modal.Sandbox.create.aio(app=app, image=image)

    // build_sandbox_image.ts
    const app = await modal.apps.fromName("sandbox-app", {
      createIfMissing: true,
    });

    const image = modal.images
      .fromRegistry("python:3.13-slim")
      .dockerfileCommands(["RUN pip install pandas"]);

    await image.build(app);
    await image.publish("sandbox-runtime");

    // app.ts
    const app = await modal.apps.fromName("sandbox-app", {
      createIfMissing: true,
    });

    // prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    const image = await modal.images.fromName("sandbox-runtime");
    await modal.sandboxes.create(app, image);

    // build_sandbox_image.go
    app, err := mc.Apps.FromName(ctx, "sandbox-app", &modal.AppFromNameParams{
        CreateIfMissing: true,
    })

    image := mc.Images.FromRegistry("python:3.13-slim", nil).
        DockerfileCommands([]string{"RUN pip install pandas"}, nil)

    builtImage, err := image.Build(ctx, app, nil)
    err = builtImage.Publish(ctx, "sandbox-runtime", nil)

    // app.go
    app, err = mc.Apps.FromName(ctx, "sandbox-app", &modal.AppFromNameParams{
        CreateIfMissing: true,
    })

    // prefer named Images in Sandboxes to ensure they are never built during Sandbox launch
    image, err = mc.Images.FromName(ctx, "sandbox-runtime", nil)
    sb, err := mc.Sandboxes.Create(ctx, app, image, nil)

Note

- **Modal treats external Image tags as immutable once pulled.** For [external registry](https://modal.com/docs/guide/existing-images) images, `Image.build` always returns the cached version — Modal does not detect upstream changes to mutable tags like `:latest`.
- To pick up a new version of an external registry image, update the tag in your deploy script (for example, `ubuntu:24.04` → `ubuntu:24.04-20240523`).

## Running a Sandbox with an entrypoint [](https://modal.com/docs/guide/sandboxes#running-a-sandbox-with-an-entrypoint)

In most cases, Sandboxes are treated as a generic container that can run arbitrary commands. However, in some cases, you may want to run a single command or script as the entrypoint of the Sandbox. You can do this by passing command arguments to the Sandbox constructor:

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create("python", "-m", "http.server", "8080", app=sb_app, timeout=10)
    for line in sb.stdout:
        print(line, end="")

    sb = await modal.Sandbox.create.aio("python", "-m", "http.server", "8080", app=sb_app, timeout=10)
    async for line in sb.stdout:
        print(line, end="")

    const sb = await modal.sandboxes.create(app, image, {
      command: ["python", "-m", "http.server", "8080"],
      timeoutMs: 10 * 1000,
    });

    sb, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Command: []string{"python", "-m", "http.server", "8080"},
      Timeout: 10 * time.Second,
    })

This functionality is most useful for running long-lived services that you want to keep running in the background. See our [Jupyter notebook example](https://modal.com/docs/examples/jupyter_sandbox) for a more concrete example of this.

## Referencing Sandboxes from other code [](https://modal.com/docs/guide/sandboxes#referencing-sandboxes-from-other-code)

If you have a running Sandbox, you can retrieve it using the `from_id` method.

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(app=sb_app)
    sb_id = sb.object_id

    # ... later in the program ...

    sb2 = modal.Sandbox.from_id(sb_id)
    p = sb2.exec("echo", "hello")
    print(p.stdout.read())
    sb2.terminate()

    sb = await modal.Sandbox.create.aio(app=sb_app)
    sb_id = sb.object_id

    # ... later in the program ...

    sb2 = await modal.Sandbox.from_id.aio(sb_id)
    p = await sb2.exec.aio("echo", "hello")
    print(await p.stdout.read.aio())
    await sb2.terminate.aio()

    const sb = await modal.sandboxes.create(app, image);
    const sbId = sb.sandboxId;

    // ... later in the program ...

    const sb2 = await modal.sandboxes.fromId(sbId);
    const p = await sb2.exec(["echo", "hello"]);
    console.log(await p.stdout.readText());
    await sb2.terminate();

    sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
    sbId := sb.SandboxID

    // ... later in the program ...

    sb2, err := mc.Sandboxes.FromID(ctx, sbId, nil)
    defer sb2.Terminate(ctx, nil)
    p, err := sb2.Exec(ctx, []string{"echo", "hello"}, nil)
    stdout, err := io.ReadAll(p.Stdout)
    fmt.Println(string(stdout))

A common use case for this is keeping a pool of Sandboxes available for executing tasks as they come in. You can keep a list of `object_id`s of Sandboxes that are “open” and reuse them, closing over the `object_id` in whatever function is using them.

## Named Sandboxes [](https://modal.com/docs/guide/sandboxes#named-sandboxes)

You can assign a name to a Sandbox when creating it. Each name must be unique within an App - only one *running* Sandbox can use a given name at a time. Note that the associated App must be a deployed App. Once a Sandbox completely stops running, its name becomes available for reuse. Some applications find Sandbox Names to be useful for ensuring that no more than one Sandbox is running per resource or project. If a Sandbox with the given name is already running, `create()` will raise an error.

Python

Python (async)

JavaScript

Go

    sb1 = modal.Sandbox.create(app=sb_app, name="my-name")
    # This will raise a modal.exception.AlreadyExistsError.
    sb2 = modal.Sandbox.create(app=sb_app, name="my-name")

    sb1 = await modal.Sandbox.create.aio(app=sb_app, name="my-name")
    # This will raise a modal.exception.AlreadyExistsError.
    sb2 = await modal.Sandbox.create.aio(app=sb_app, name="my-name")

    const sb1 = await modal.sandboxes.create(app, image, { name: "my-name" });
    // this will raise an AlreadyExistsError
    const sb2 = await modal.sandboxes.create(app, image, { name: "my-name" });

    sb1, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Name: "my-name",
    })
    // this will return an error
    sb2, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Name: "my-name",
    })

A named Sandbox may be fetched from a deployed App using `from_name()` *but only if the Sandbox is currently running*. If no running Sandbox is found, `from_name()` will raise an error.

Python

Python (async)

JavaScript

Go

    sb_app = modal.App.lookup("my-app", create_if_missing=True)
    sb1 = modal.Sandbox.create(app=sb_app, name="my-name")
    # Returns the currently running Sandbox with the name "my-name" from the
    # deployed App named "my-app".
    sb2 = modal.Sandbox.from_name("my-app", "my-name")
    assert sb1.object_id == sb2.object_id # sb1 and sb2 refer to the same Sandbox

    sb_app = await modal.App.lookup.aio("my-app", create_if_missing=True)
    sb1 = await modal.Sandbox.create.aio(app=sb_app, name="my-name")
    # Returns the currently running Sandbox with the name "my-name" from the
    # deployed App named "my-app".
    sb2 = await modal.Sandbox.from_name.aio("my-app", "my-name")
    assert sb1.object_id == sb2.object_id # sb1 and sb2 refer to the same Sandbox

    const app = await modal.apps.fromName("my-app", { createIfMissing: true });
    const sb1 = await modal.sandboxes.create(app, image, { name: "my-name" });
    // returns the currently running Sandbox with the name "my-name" from the
    // deployed App named "my-app".
    const sb2 = await modal.sandboxes.fromName("my-app", "my-name");
    console.assert(sb1.sandboxId === sb2.sandboxId); // sb1 and sb2 refer to the same Sandbox

    app, err := mc.Apps.FromName(ctx, "my-app", &modal.AppFromNameParams{
      CreateIfMissing: true,
    })
    sb1, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Name: "my-name",
    })
    // returns the currently running Sandbox with the name "my-name" from the
    // deployed App named "my-app".
    sb2, err := mc.Sandboxes.FromName(ctx, "my-app", "my-name", nil)
    // sb1 and sb2 refer to the same Sandbox
    fmt.Println(sb1.SandboxID == sb2.SandboxID)

Sandbox Names may contain only alphanumeric characters, dashes, periods, and underscores, and must be shorter than 64 characters.

## Tagging [](https://modal.com/docs/guide/sandboxes#tagging)

Sandboxes can also be tagged with arbitrary key-value pairs. These tags can be used to filter results in `Sandbox.list`. A Sandbox can carry up to 10 tags.

Python

Python (async)

JavaScript

Go

    sandbox_v1_1 = modal.Sandbox.create("sleep", "10", app=sb_app)
    sandbox_v1_2 = modal.Sandbox.create("sleep", "20", app=sb_app)

    sandbox_v1_1.set_tags({"major_version": "1", "minor_version": "1"})
    sandbox_v1_2.set_tags({"major_version": "1", "minor_version": "2"})

    for sandbox in modal.Sandbox.list(app_id=sb_app.app_id):  # All sandboxes.
        print(sandbox.object_id)

    for sandbox in modal.Sandbox.list(
        app_id=sb_app.app_id,
        tags={"major_version": "1"},
    ):  # Also all sandboxes.
        print(sandbox.object_id)

    for sandbox in modal.Sandbox.list(
        app_id=sb_app.app_id,
        tags={"major_version": "1", "minor_version": "2"},
    ):  # Just the latest sandbox.
        print(sandbox.object_id)

    sandbox_v1_1 = await modal.Sandbox.create.aio("sleep", "10", app=sb_app)
    sandbox_v1_2 = await modal.Sandbox.create.aio("sleep", "20", app=sb_app)

    await sandbox_v1_1.set_tags.aio({"major_version": "1", "minor_version": "1"})
    await sandbox_v1_2.set_tags.aio({"major_version": "1", "minor_version": "2"})

    async for sandbox in modal.Sandbox.list.aio(app_id=sb_app.app_id):  # All sandboxes.
        print(sandbox.object_id)

    async for sandbox in modal.Sandbox.list.aio(
        app_id=sb_app.app_id,
        tags={"major_version": "1"},
    ):  # Also all sandboxes.
        print(sandbox.object_id)

    async for sandbox in modal.Sandbox.list.aio(
        app_id=sb_app.app_id,
        tags={"major_version": "1", "minor_version": "2"},
    ):  # Just the latest sandbox.
        print(sandbox.object_id)

    const sandboxV1_1 = await modal.sandboxes.create(app, image, {
      command: ["sleep", "10"],
    });
    const sandboxV1_2 = await modal.sandboxes.create(app, image, {
      command: ["sleep", "20"],
    });

    await sandboxV1_1.setTags({ major_version: "1", minor_version: "1" });
    await sandboxV1_2.setTags({ major_version: "1", minor_version: "2" });

    // All sandboxes.
    for await (const sandbox of modal.sandboxes.list({ appId: app.appId })) {
      console.log(sandbox.sandboxId);
    }

    // Also all sandboxes.
    for await (const sandbox of modal.sandboxes.list({
      appId: app.appId,
      tags: { major_version: "1" },
    })) {
      console.log(sandbox.sandboxId);
    }

    // Just the latest sandbox.
    for await (const sandbox of modal.sandboxes.list({
      appId: app.appId,
      tags: { major_version: "1", minor_version: "2" },
    })) {
      console.log(sandbox.sandboxId);
    }

    sandboxV1_1, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Command: []string{"sleep", "10"},
    })
    sandboxV1_2, err := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
      Command: []string{"sleep", "20"},
    })

    sandboxV1_1.SetTags(ctx, map[string]string{"major_version": "1", "minor_version": "1"})
    sandboxV1_2.SetTags(ctx, map[string]string{"major_version": "1", "minor_version": "2"})

    // All sandboxes.
    it, _ := mc.Sandboxes.List(ctx, &modal.SandboxListParams{
      AppID: app.AppID,
    })
    for sandbox := range it {
      fmt.Println(sandbox.SandboxID)
    }

    // Also all sandboxes.
    it, _ = mc.Sandboxes.List(ctx, &modal.SandboxListParams{
      AppID: app.AppID,
      Tags:  map[string]string{"major_version": "1"},
    })
    for sandbox := range it {
      fmt.Println(sandbox.SandboxID)
    }

    // Just the latest sandbox.
    it, _ = mc.Sandboxes.List(ctx, &modal.SandboxListParams{
      AppID: app.AppID,
      Tags:  map[string]string{"major_version": "1", "minor_version": "2"},
    })
    for sandbox := range it {
      fmt.Println(sandbox.SandboxID)
    }

## Cleaning up client-side resources [](https://modal.com/docs/guide/sandboxes#cleaning-up-client-side-resources)

When you use a Sandbox handle in the SDK to interact with a Sandbox (e.g., calling `exec()`), it opens a direct connection to the remote Sandbox and keeps it open for subsequent commands. This connection and related resources are cleaned up automatically once they have been idle for a while, so you normally don’t need to manage them yourself.

However, if your client manages a large number of Sandboxes, you can release these resources eagerly by calling `detach()` on the Sandbox. This permanently closes all connections from that handle and prevents it from opening new ones. Detaching doesn’t terminate or otherwise affect the remote Sandbox; it only cleans up client-side resources.

Python

Python (async)

JavaScript

Go

    sb = modal.Sandbox.create(app=sb_app)
    sb.detach()

    sb = await modal.Sandbox.create.aio(app=sb_app)
    await sb.detach.aio()

    const sb = await modal.sandboxes.create(app, image);
    sb.detach();

    sb, err := mc.Sandboxes.Create(ctx, app, image, nil)
    defer sb.Detach()

[Sandboxes](https://modal.com/docs/guide/sandboxes#sandboxes)[Configuration](https://modal.com/docs/guide/sandboxes#configuration)[Lifecycle](https://modal.com/docs/guide/sandboxes#lifecycle)[Events](https://modal.com/docs/guide/sandboxes#events)[Timeouts](https://modal.com/docs/guide/sandboxes#timeouts)[Idle Timeouts](https://modal.com/docs/guide/sandboxes#idle-timeouts)[Readiness Probes](https://modal.com/docs/guide/sandboxes#readiness-probes)[TCP readiness probe](https://modal.com/docs/guide/sandboxes#tcp-readiness-probe)[Exec readiness probe](https://modal.com/docs/guide/sandboxes#exec-readiness-probe)[Return Codes](https://modal.com/docs/guide/sandboxes#return-codes)[Runtimes](https://modal.com/docs/guide/sandboxes#runtimes)[Running Docker in a Sandbox](https://modal.com/docs/guide/sandboxes#running-docker-in-a-sandbox)[Environments](https://modal.com/docs/guide/sandboxes#environments)[Environment variables](https://modal.com/docs/guide/sandboxes#environment-variables)[OIDC identity tokens](https://modal.com/docs/guide/sandboxes#oidc-identity-tokens)[Custom Images](https://modal.com/docs/guide/sandboxes#custom-images)[Separating Image builds from Sandbox creation](https://modal.com/docs/guide/sandboxes#separating-image-builds-from-sandbox-creation)[Running a Sandbox with an entrypoint](https://modal.com/docs/guide/sandboxes#running-a-sandbox-with-an-entrypoint)[Referencing Sandboxes from other code](https://modal.com/docs/guide/sandboxes#referencing-sandboxes-from-other-code)[Named Sandboxes](https://modal.com/docs/guide/sandboxes#named-sandboxes)[Tagging](https://modal.com/docs/guide/sandboxes#tagging)[Cleaning up client-side resources](https://modal.com/docs/guide/sandboxes#cleaning-up-client-side-resources)

See it in action

[Building a coding agent with Sandboxes](https://modal.com/docs/examples/agent)

[Building a code interpreter](https://modal.com/docs/examples/simple_code_interpreter)

[Running a Jupyter notebook](https://modal.com/docs/examples/jupyter_sandbox)
