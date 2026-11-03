# Separate local landing page and sandbox

Recorded November 3, 2026 (PST).

## Completed and verified

- Root `npm run dev` launches only the buildless landing page on
  `http://127.0.0.1:3000/`. Node.js 22+; no runtime dependencies or Python needed.
  Refresh after edits. `-- --port 3001` selects another loopback port.
- Root `npm run sandbox` launches the existing Python portal on port 8765 using
  `.venv-lab` and `.local/portal`. Open its complete printed authenticated link.
  `-- --root .local/portal-unfamiliar-source-acceptance-01/portal` opens the saved
  demo collection. Existing recordings and API ledgers stay at their original
  paths. Never run two portal processes against the same root.
- `sandbox/` owns the launch wrapper and setup instructions. The implementation
  remains shared in `src/weft`; this is dependency/process separation, not a
  container or OS isolation boundary. No data or credentials were moved/copied.
- `website/public-assets.mjs` is the explicit allowlist shared by the local
  static server and hosted asset bundler. Private files are not served. The
  landing server does not proxy APIs, accept uploads or start simulation workers.
- `npm run test:local`: eight checks pass (three launcher checks and five HTTP
  groups), including all sixteen public assets, private-path rejection, disabled
  upload behavior, exact source-video byte ranges and invalid options.
  Initial sandboxed HTTP tests failed because loopback binding was denied;
  rerunning with permission passed. The temporary test listener closed.
- Syntax checks pass for the changed website client and hosted asset bundler.
  `node sandbox/run.mjs --check` verifies the existing Python executable without
  starting a server. No AI call, paid resource, browser test or deployment ran.

## Preserved inherited work

The initial tree included unrelated avatar edits and a research draft. They are
checkpointed separately from the environment change, not presented as newly
completed product functionality:

- Avatar source-repair helpers and Studio verification/polling changes: 58
  focused Python checks and eight Node checks pass. The existing
  `tools/avatar-assets/TODO.md` retains unsupported owl/rabbit results and the
  unintegrated generated-source repair. No new visual/animation qualification.
- `src/weft/moving_equipment_body_sweep.py`: syntax checked only; untested,
  unintegrated research draft. Its earlier unfinished status is recorded in
  `docs/handoffs/hackathon-demo.md`. No default or task-success claim changes.
- No stashes existed at the initial inspection. No saved work was deleted.

## Next action and limits

Start either command manually in separate terminals; do not restart an already
running sandbox. The landing page remains usable without the sandbox. Hosting
stays intentionally offline. Browser appearance and live sandbox execution were
not re-tested for this launch-only change. Existing reconstruction, avatar-skin
and task-success limitations remain unchanged.
