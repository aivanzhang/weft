# One fresh Studio generation: pricing check

Checked 2026-11-02 10:25:16 AM PST. No provider/API, credential or AI-role access.

**35 Meshy credits; two paid submissions; no automatic paid retries.**

| Stage | Pinned settings | Credits |
| --- | --- | --- |
| Text preview | meshy-7.1; standard model; Ultra4K geometry; triangular remesh; target35000faces; GLB |20+5=25|
| Text refine | same model; successful previewID;4K basecolor; PBR maps; GLB |10|

The current[API price table](https://docs.meshy.ai/en/api/pricing) confirms these amounts. The[Text to3D API](https://docs.meshy.ai/en/api/text-to-3d) supports this explicit model and two-stage configuration. Provider rig/motion generation is disabled; rig, fur, requested actions and final rendering are local. Exact not-yet-submitted payload templates are in pricing-review.json.

The supplied ledger snapshot is202/210:8remain,27short of this35-credit job. This review neither reads norchanges that ledger. API prices are in prepaid credits; the user's dollar conversion is unknown, so no dollar estimate or purchase authorization is invented.

Current flow: normally3 AI-role calls (planner, author, critic), atmost5 distinct calls with one repair. Studio's separate generic runner cap remains10 and240seconds/call; account-specificCLI costs are unknown. This is code inspection, not a live run.

[Published processing times](https://docs.meshy.ai/en/api/task-lifecycle) are medians, not a guarantee. Poll/resume the existing task rather than duplicate generation after a local wait ends.

Local source snapshots/hash references are retained. No production changes, accounts, subscriptions, provider requests or model calls occurred.
