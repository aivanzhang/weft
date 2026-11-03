# Local sandbox environment

The interactive video-to-simulation portal runs separately from the landing page.
It uses Python in the existing `.venv-lab`, its own loopback port and private
project storage. `npm run dev` starts only the landing page on port 3000; it does
not start Python, access sandbox projects or enable video processing.

From the repository root, start the sandbox in a separate terminal:

```sh
npm run sandbox
```

Open the complete `http://127.0.0.1:8765/#token=…` link printed by Python. Keep
that token private. The process stays in the foreground; Ctrl+C stops it while
preserving projects and recordings. No browser is opened automatically.

The default project directory remains `.local/portal`. Existing experiments and
recordings are not moved or copied. To open the existing hackathon demo instead:

```sh
npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal
```

Saved demo artifacts are local and are not included in a fresh Git clone.
Choose another port if a sandbox already owns 8765; do not start two processes
against the same project directory:

```sh
npm run sandbox -- --root .local/another-portal --port 8766
```

The launcher always uses the repository as its working directory. The existing
Meta and OpenAI credentials and canonical spending ledgers remain unchanged;
another project directory does not create a new API allowance. No credentials
are copied into the landing page. Starting the launcher does not itself request
paid inference; explicit portal actions retain their existing budget checks.

## Python setup

If `.venv-lab` already works, reuse it without reinstalling dependencies. For a
fresh checkout, the pinned lab environment was verified on Python 3.14 / macOS
arm64; other platforms are not verified against this lockfile:

```sh
python3.14 -m venv .venv-lab
.venv-lab/bin/python -m pip install -r requirements-lab.lock
.venv-lab/bin/python -m pip install --no-deps -e .
```

This is dependency and process separation, not a container or an OS security
boundary. The simulator implementation stays in `src/weft` so the portal, CLI
and recorded-evidence tools continue to share one pipeline.

## Check without starting a server

```sh
node sandbox/run.mjs --check
node --test sandbox/run.test.mjs
```

`--check` verifies the Python executable and prints the launch command without
importing the pipeline, creating project directories or making API calls. Full
portal behavior and current reconstruction limits are documented in
[the portal guide](../docs/portal.md) and [the demo guide](../docs/hackathon-demo.md).
