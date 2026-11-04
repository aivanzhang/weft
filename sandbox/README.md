# Video-to-simulation sandbox

Upload a video, inspect reconstruction evidence, interact with a supported world,
and record or replay your actions. See [capabilities and limits](../PROJECT.md).

## Requirements

- **Node.js 22+** for the launcher.
- **Python 3.14** with the pinned lab dependencies. The lockfile is verified on
  macOS with Apple silicon; other platforms have not been qualified.

From the repository root, create the environment once:

```sh
python3.14 -m venv .venv-lab
.venv-lab/bin/python -m pip install -r requirements-lab.lock
.venv-lab/bin/python -m pip install --no-deps -e .
```

Reuse an existing working `.venv-lab` rather than reinstalling it.

## Run

```sh
npm run sandbox
```

Open the complete authenticated link printed for `http://127.0.0.1:8765/`.
Keep its access token private. Stop with **Ctrl+C**; projects and recordings remain
in `.local/portal`. For a different collection or port:

```sh
npm run sandbox -- --root .local/my-projects --port 8766
```

For the guided walkthrough, visit `/demo` on the sandbox. Opening a run brings
only that run into focus: switch between its live world, recording and evidence.
Only the active view loads media or an embedded simulator. The access link is
needed once per browser tab; demo pages retain it for navigation and hide it
from the address bar. The demo code is separate from this API access key.

Run only one process per project directory. The sandbox and landing page use
separate processes and dependencies, not isolated containers. Saved experiments
are local data and are not included in a fresh clone.

## API configuration

Set `OPENAI_API_KEY` in the server environment for AI-backed analysis. Configure
its cumulative allowance with `.venv-lab/bin/python -m weft openai-budget --total-usd <amount>`.
Optional Meta features use `META_API_KEY` and
`.venv-lab/bin/python -m weft meta-configure --budget <amount>`; add `--key` to
enter a key through a hidden prompt instead. Replace `<amount>` with your chosen
allowance. Keys stay server-side, and starting the server makes no paid request.

Allowances and prior charges persist across project collections. Keep local
credentials and spending ledgers when archiving or cleaning saved projects.

## Verify the launcher

```sh
node sandbox/run.mjs --check
node --test sandbox/run.test.mjs
```

These checks do not start the server or make API calls. Use the portal's evidence
and evaluation reports to assess a generated world's quality separately.
