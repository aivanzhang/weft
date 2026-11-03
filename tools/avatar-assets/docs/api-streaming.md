# Bounded API streaming and stopped-job status

The optional `sse-v1` transport retains early request identities during long AI
calls. It does not recover the prior timed-out panda answer, establish its cost,
or complete the hands-off avatar acceptance run. Existing jobs remain unchanged. A new GPT-5.2 Studio request now uses streaming;
its actual website planner completed in31.228 s with3,577 input/2,010 output
tokens and$0.034400 usage. Evidence: `outputs/gpt52-panda-website-prepare-v1`
and `outputs/studio-15105a1353404792a185d61d/agents/planner`. The first author then stopped on an unsupported event at31.468 s. Its original
trace lacks the event name; no rig/answer was returned. New traces record only
sanitized event names, not payloads. A fresh job uses standard responses while
this streaming compatibility remains unresolved.

## New-job configuration

Add `"transport": "sse-v1"` to the owner-only `.local/openai-backend.json` before
preparing a new request. Omission preserves the existing non-streaming format.
The setting and transport implementation hash become immutable job inputs.
Changing settings never resubmits an old call or makes a failed job resumable.

The configured timeout remains an absolute bound, not a sliding inactivity timer.
Values up to 600 seconds are already supported, and the new standard-response owner setting is600 seconds. The previous job used240 seconds. Seven 600-second calls could consume 4,200 seconds before
native work inside the worker's 5,400-second limit; increasing one timeout is not
proof a complete job will fit. Socket operations are bounded; OS DNS still needs
the outer worker deadline. A late DNS return cannot submit a late POST.

The model, token limits, service tier, no-tools policy, `store:false`, seven-call
and two-repair owner settings, and spending ledger remain unchanged. There is no
background generation, polling service, provider-side saved-response recovery or
automatic request retry. No new service or recurring charge is introduced.

## Preserved evidence

Before HTTP submission, `request.json` saves an opaque client request ID derived
from the immutable job, role and request digest. `transport.json` records only
that ID, a sanitized server request ID, an early response ID, status, counters
and elapsed time. It contains no key, prompt, reasoning or partial model answer.
The trace is owner-only, atomically replaced and synced at bounded milestones.

OpenAI documents the client ID as a support-correlation mechanism when a timeout
prevents receiving the server ID; it is not an idempotency key or billing proof.
[API overview](https://developers.openai.com/api/reference/overview).

Streaming emits incremental events. This implementation waits for one full,
consistent terminal response, then applies the same pinned-model, usage, schema
and no-tool checks as the original runner. Only validated usage settles a charge;
early IDs or partial output never do.
[Streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses),
[event reference](https://developers.openai.com/api/reference/resources/responses/streaming-events).

Limits are 16 MiB total stream bytes, 4 MiB per event and terminal response,
65,536 events, and a 4 KiB trace. Invalid ordering, changed response IDs, tools,
refusals, incomplete/error events and truncated streams stop without retry.
Every original reservation and failed-attempt artifact remains preserved.

## Website status

Stopped-job snapshots distinguish an unknown charge, a settled unusable response,
an explicit unsupported result, and a received but invalid plan. Missing or
changed evidence is labeled unverified, without a cost claim. Missing delivery
assets take priority over AI diagnostics. No status read alters a lease, grant,
charge or execution permission.

The helper reads bounded request/receipt files and at most twelve indexed call
rows; it does not scan model images, source trees or full provider responses.
Twenty real stopped-job reads measured a 0.611 ms median and 15.715 ms maximum.
These are local observations, not sustained production latency guarantees.

## Verification and limitation

- 16 mocked HTTP/parser checks cover fragmentation, malformed streams, bounds,
  deadlines, durable IDs, tampering and partial responses.
- 13 runner/Studio/custody checks cover settings pinning, full validation,
  accounting holds, cached reuse, exact release proof and two complete
  parser-to-runner HTTP fixtures.
- The existing 112-check regression batch had one missing-image fixture. After
  supplying its analytic guide images, that failed check passed; validators were
  not changed. Existing fixture database-close warnings remain recorded.
- Website status has 53 Python and three pure Node checks, plus a new delivery
  error-priority regression. No browser was opened for these changes.
- Near-4 MiB parser fixtures took 13–15 ms; 65,536 events took 336 ms, with
  cumulative process peak RSS about 122 MB. These are analytic transport fixtures,
  not generated avatars or live provider latency.

Exact receipts and commands are in `outputs/openai-stream-transport-tests-v1` / `v2`,
`outputs/stream-runner-integration-tests-v1` / `v2`, and
`outputs/studio-ai-status-tests-v1` / `v2`. The current handoff records the
unchanged uncertain charge and remaining live acceptance work.
