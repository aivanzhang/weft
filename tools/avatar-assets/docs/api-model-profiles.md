# Explicit API model profiles

New requests can select a pinned OpenAI model in the server-only backend configuration. Existing requests retain their exact model, settings, response and price history. No model fallback occurs during a call.

| Model snapshot | Input / output per million tokens | Context | Full reservation at 16,384 output tokens |
| --- | --- | --- | --- |
| `gpt-5.4-2026-03-05` (unchanged default) | $2.50 / $15; long-context $5 / $22.50 | 1,050,000 | $5.618640 |
| `gpt-5.2-2025-12-11` (explicit alternative) | $1.75 / $14 | 400,000 | $0.929376 |

Official profiles checked November 2, 2026 (PST): [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2). Both support image input, structured output and streaming. Provider account availability and avatar quality still require a real run; lower price is not quality evidence.

Reservations cover the entire model context plus the configured output limit, using uncached rates and the higher long-context price where applicable. Output limits include reasoning tokens. They are conservative exposure bounds, not estimates of actual charges. Standard service tier, no tools, no automatic transport retries, no subscription or recurring services.

New non-default calls receive an immutable price binding alongside their ledger row. Historical unbound rows retain the original GPT-5.4 interpretation. A new model cannot reinterpret an old uncertain reservation. Request fingerprints and source snapshots pin the selected model implementation; response model and usage must match before settlement.

## Resumed panda trial

The prior d80 author call has no answer or provider response ID. Its $5.618640 hold remains unchanged. At the latest pre-run check, settled cost plus that hold was $9.618295, leaving $5.381705 within the standing $15 cap.

A new GPT-5.2 request has an expected three-call cost around $0.35–$1.00 (50,000 input and 1,000–8,000 output tokens per call), not a guaranteed invoice. Up to seven calls/two repairs remain subject to the remaining dollars before each request; five entirely uncertain calls would fit, six would not. Reuse the paid source geometry: zero new Meshy credits. Preserve all rejected attempts and do not call a new run successful until native artifacts, visual review and website delivery pass.

Local lease release and closing unused grant capacity are separate operations. Neither operation settles an uncertain charge, changes the old call records, resumes the old job, or submits a new request. Closing the old grant fences further calls from it; the original grant remains recorded and its actual/unknown exposure still consumes the project cap. A fresh request requires its own recorded grant under the existing user authorization.

## Actual transport results

Job `15105a1353404792a185d61d`: streamed planner completed in31.228 s for$0.034400. First author stopped at31.468 s because the local parser rejected an unsupported event. The original trace did not capture its name; its$0.929376 hold remains unknown. New traces record a sanitized event name without accepting additional types.

Job `447a849b435b4d2e962b3763` used standard responses with a600-second deadline.
Its planner and three author calls all completed and settled: total$0.495499.
Native finishing rejected all three designs (tuft vertex bound, nose face-count
minimum, then dragon face-count maximum); this is not a transport failure or an
accepted avatar. All preparation folders and rejected attempts remain immutable.

Website continuation `d8795ec6c273489da37954ab` reuses the authenticated plan and
native failure history, with no new planner call. It is running with exact tuft
capacity and combined count-error feedback. Before this run, settled usage was
$4.529554 plus$6.548016 unknown holds: conservative exposure$11.077570, leaving
$3.922430. Expected two to six calls cost roughly$0.35–$1.20; this estimate is
not permission to exceed the$15 aggregate cap or discard uncertain charges.
