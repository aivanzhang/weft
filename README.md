# Weft

Turn video evidence into an inspectable, interactive reinforcement-learning (RL)
environment. Supported inputs produce conditional simulations—not verified
physical replicas. Weft is separate from the neighboring Loom ingestion project.

## Run locally

From the repository root, use separate terminals:

| Interface | Command | Address |
| --- | --- | --- |
| Landing page | `npm run dev` | `http://127.0.0.1:3000/` |
| Video upload and 3D sandbox | `npm run sandbox` | Open the authenticated link printed for port 8765 |

The landing page needs Node.js 22+ with no npm install. The sandbox uses
`.venv-lab`; [setup instructions](sandbox/README.md). Neither command deploys.
Saved local demos are not included in a fresh clone. To reopen the hackathon demo:

```sh
npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal
```

Upload → inspect evidence/assumptions → control the world → choose a compatible
avatar → record/replay. Muse can review evidence and select bounded tests.
Unfamiliar-video reconstruction and Studio avatar deformation remain unresolved.

## Documentation

- [PROJECT.md](PROJECT.md): current status, architecture, remaining work and evidence.
- [Demo and submission checklist](hackathon/analysis/demo-plan.md).
- [Avatar authoring](tools/avatar-assets/README.md) · [website maintenance](website/README.md).
- [Historical studies](docs/HISTORY.md) · [source provenance](data/README.md).

The repository is private and hosted site offline. Publication and submission
require separate authorization. Preserve `.local` projects, credentials and
spending ledgers; Git does not back them up.
