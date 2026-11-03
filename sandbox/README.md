# Local sandbox

From the repository root, run `npm run sandbox` in its own terminal. Open the
complete authenticated link printed for `http://127.0.0.1:8765/`; keep its token
private. No browser opens automatically. Ctrl+C preserves projects and recordings.

The sandbox uses `.venv-lab` and `.local/portal`, separate from the Node landing
page on port 3000. This is process/dependency separation, not container isolation.
Both share the repository; no credentials or saved data are moved or copied.

## Saved demo and alternate port

```sh
npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal
npm run sandbox -- --root .local/another-portal --port 8766
```

Saved demo artifacts are not in fresh clones. Never run two sandbox processes
against the same project directory. The launcher uses the repository working
directory and canonical Meta/OpenAI ledgers; another root grants no new budget.
Startup itself requests no paid inference. Portal actions retain budget checks.

## Python setup

Reuse a working `.venv-lab`. For a fresh checkout, the lockfile is verified on
Python 3.14 / macOS arm64; other platforms are unverified:

```sh
python3.14 -m venv .venv-lab
.venv-lab/bin/python -m pip install -r requirements-lab.lock
.venv-lab/bin/python -m pip install --no-deps -e .
```

## Check without starting a server

`node sandbox/run.mjs --check` verifies the Python executable and prints the
command without importing the pipeline, creating projects or making API calls.
`node --test sandbox/run.test.mjs` checks argument handling.
See [PROJECT.md](../PROJECT.md) for the workflow, evidence and current limits.
