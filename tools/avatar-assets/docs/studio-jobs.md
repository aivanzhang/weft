# Durable studio jobs

`StudioJobs` owns local lifecycle metadata, not avatar quality decisions or paid
authorization. The HTTP layer owns its session capability, uploads and opaque file
URLs. Pipeline outputs remain separate, immutable stage artifacts.

## Callable contract

```python
jobs = StudioJobs(root=ROOT, state_dir=None, executor=None)
job = jobs.create(prompt, style="", custom_actions=(), preset="preview",
                  reference=None, reuse_model=None, idempotency_key=None)
jobs.plan(job["id"])
snapshot = jobs.get(job["id"])
jobs.start(job["id"], plan_id=snapshot["plan_id"])
jobs.cancel_cooperative(job["id"])
jobs.list(limit=20, offset=0)
jobs.resolve_job(job["id"])  # Trusted ID lookup for the HTTP artifact boundary.
```

IDs contain24 lowercase hexadecimal characters. `create` preserves an immutable
request and selected-file identity without decoding, hashing or copying a large
input on the request thread. The detached worker checks that identity before and
after preparation; `pipeline.prepare` then preserves exact input bytes and hashes.
Later resumes use those preserved bytes, not the original selected file.

Custom actions are0–8 exact text descriptions, each1–600 characters. Presets are
`draft`, `preview`, `muse`. References are single PNG/JPEG/WebP images up to8MiB;
selected reusable GLBs are up to64MiB. The worker performs actual format/decoding
validation. Accepted paths are confined to this tool's `outputs/`, `references/`,
or `state_dir/uploads/`; the browser supplies opaque reference IDs, never paths.

Repeated creation with the same idempotency key and immutable inputs returns the
same job. Reusing that key for a different request fails. Planning/execution
requests also avoid duplicate active or completed invocations.

## Planning, execution and cancellation

Planning invokes `Pipeline.run(execute=False, plan_only=True,
cancel_requested=callback)` in the detached child. It cannot submit fresh Meshy
generation. Execution uses `execute=True, plan_only=False`. Both keep the existing
pipeline/agent/native bounds and exact source pins.

New Studio jobs pin five AI calls, one local repair and240seconds per call
(expected planner/author/critic:three). The generic CLI default remains separate.
Completed historical ten-call jobs remain readable; unfinished old requests
are not rewritten. Exact typed request validation rejects altered ceilings.

The35-credit preparation ceiling is not a grant. Start requires the exact current
plan ID when supplied and reads the canonical Meshy SQLite ledger in read-only
mode. It checks existing base reservations, job-scoped grants and uncertain calls,
then rechecks in the worker immediately before execution. Only the existing
Meshy adapter reserves charges. This layer never configures budgets, adds grants,
resets usage, reads credentials, retries uncertain submissions or buys credits.
The canonical ledger remains under the Weft project; an alternate state directory
does not create another ledger or spending allowance.

Cancellation is cooperative at pipeline boundaries, not an immediate native/API
stop. Existing charges, task IDs, native handles and partial files remain. A
completed sealed delivery takes precedence over a late cancellation request.

## Durable ownership and bounds

- Private state is0700; request/owner/receipt files and SQLite are0600.
- One registry is bound per tool root in `.local/studio-registry.json`.
- One expensive worker, four queued operations,256 retained job records,
  eight invocations per job,128 lifecycle journal entries per job.
- SQLite and atomic private receipts survive HTTP/page restarts. Only explicitly
  queued, not-yet-owned work dispatches on restart. Running/uncertain work is never
  duplicated.
- Each invocation retains its exact worker-source snapshots, request digest,
  PID/start identity, token, log, bounded-run receipt and terminal result.
- A global worker lock plus per-job lock prevents overlapping ownership. macOS
  start identity uses the installed SDK's `proc_bsdinfo` PID/UID/start-time fields;
  Linux uses process start ticks. PID presence alone is never ownership proof.
- Heartbeats run every2s. A heartbeat older than15s displays `needs_inspection`;
  uncertainty keeps the active lease. No automatic PID termination or replay.
- Controller wall time5400s, sampled controller RSS1GiB, control output64MiB and
  existing runner log limit16MiB. Native/agent subprocesses retain their separate
  existing bounds; controller RSS is not aggregate descendant RSS.
- Stored studio job/output policy is8GiB. A launch reserves2GiB for the pipeline
  plus64MiB for its controller. Actual byte counts are scanned by the detached
  supervisor, not UI polls. Crossing the2GiB pipeline threshold requests a safe
  cancellation; an in-flight bounded stage can finish before it observes that
  request. This is not a kernel filesystem quota. No output is automatically
  deleted. A full or uncertain registry requires local inspection and explicit
  storage reconciliation before further launches, not a new registry to evade it.

An unsealed exit, hard controller failure or conflicting owner is
`needs_inspection`, with the lease retained. Inspect the private invocation's
`worker-job.json`, `supervisor-result.json`, logs and preserved pipeline stage
handles before any manual recovery. Never clear a lease merely because a PID is
absent: a native/provider operation may still need reconciliation.

## Snapshot contract

Snapshots expose real states/events and elapsed current/last invocation time,
not invented completion percentages. They contain `id`, `title`, `status`,
`request`, `operation`, `plan_id`, `stages`, `plan`, `reviews`, `previews`, `result`,
`capabilities`, `storage`, `error` and `provenance`.

Artifact descriptors are `{path, kind, id, label, bytes}` with tool-root-relative
paths. Only existing selected images, clips, GLBs and editable sources appear.
Inputs, agent context, raw logs, provider downloads and credential paths are not
downloads. HTTP translates only these descriptors into opaque URLs.
Completed media comes from the delivery's selected pack or gallery, including
`media-portable-<attempt>`; older global media folders remain readable. Unaccepted
attempt portraits are labeled work-in-progress previews. Source arrays, signed
URLs and raw provider responses never enter snapshots.

## Verification and limits

```sh
.venv/bin/python -m unittest tests.test_studio_jobs -v
```

The tests inject `FixtureExecutor` and `FixturePipeline` and label all outputs
`analytic_fixture`. They cover queue/restart behavior, exact idempotency, private
paths, stale plans, read-only allowance checks, retained uncertain owners, safe
cancellation, source tampering, capacity, sanitized artifacts and supervisor
receipts. Placeholder fixture bytes are not generated assets or media proof.

A local bounded metadata benchmark used256 fixture jobs and50 listed jobs with
24 events each: list median98.45ms/max100.72ms; single-job median2.02ms/max2.41ms.
It launched no worker and establishes metadata responsiveness only. The native
macOS own-process identity lookup was also verified locally. No model, provider,
native-renderer or HTTP-server run was performed for this module's verification;
actual end-to-end avatar/HTTP verification belongs to the integration workstream.
