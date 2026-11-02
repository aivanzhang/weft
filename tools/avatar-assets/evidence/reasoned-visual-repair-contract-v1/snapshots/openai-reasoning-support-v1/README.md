# Bounded reasoning support

Implemented after the current job closed; `runtime.patch` preserves the prepared scratch proposal and must not be blindly applied again. `baseline/` and `proposed/` preserve local review sources; `final/` contains final owned runtime/test snapshots. The continuation collection region has separate ownership; only `_role` was changed here.

## Behavior

- Optional owner-only `reasoning_effort`: `none`, `low`, `medium`, `high`. Missing remains `none` without rewriting old configuration snapshots.
- Request, runner settings, cache and payload hashes bind the effort. New owner configuration may select `medium`; no global configuration or ledger was changed here.
- Bounded Responses parsing accepts reasoning metadata plus exactly one completed final assistant JSON message. Tools, commentary messages, multiple answers, refusals and incomplete results still fail closed.
- `usage.output_tokens` already includes reasoning. Subcounts are validated but never billed twice. Existing output bound, full-context reservation, prices, no-retry behavior and project budget remain unchanged.

## Evidence and limits

55 active focused tests passed; 12 baseline runner cases additionally passed on the identical proposed runner. Legacy b082 planner receipt was read-only authenticated. No actual provider/native calls, key reads or live accounting changes occurred. See `result.json` for exact commands, hashes and remaining limits.

Official sources: [GPT-5.4 model](https://developers.openai.com/api/docs/models/gpt-5.4), [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning). The pinned snapshot supports medium. Reasoning uses output tokens and shares the maximum-output budget. A 2,000–6,000 extra-token scenario costs $0.03–$0.09/call at the current standard output rate; this is not measured usage. For seven calls that scenario adds $0.21–$0.63 within, not on top of, the existing approved cap.

Next: finish all owners' source freeze, configure medium for a new hash-bound job under the existing budget, then perform one bounded run. Do not reset grants or edit saved receipts.
