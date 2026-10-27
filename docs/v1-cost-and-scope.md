# V1 execution envelope

This implementation is a local, CPU-only demonstrator, not a general video world model.

- External compute/API spend: estimated **$0** for the default local pipeline; no paid calls or cloud resources.
- Recurring cost: **$0** with no resident process, deployment, or schedule.
- Downloads: public open-source Python dependencies and one small, openly licensed physical video. Cache the source once and verify its hash before reuse.
- Analysis: one video, at most 60 seconds / 1,800 frames; bounded frame dimensions and a single tracked rigid object.
- Repair: finite parameter proposals; the evaluator and observation labels are not writable by the proposer.
- RL: a small tabular learner, fixed training episode cap, distinct training/evaluation random seeds, and simulation-only claims.
- Artifacts: bounded local files; record source attribution, evidence IDs, parameters, versions, seeds, timings, all candidate decisions, and unresolved limits.

## Iteration setup

The two checked-in presets each run 15,000 training episodes and 300 evaluation episodes per policy (three policies). Each has at most 100 steps per episode, two local repair proposals, one video decode plus replay generation, and a 120-second training wall-time limit. They share one cached source and run in separate output directories. At most two local processes were used concurrently for setup verification; there are no cloud workers, GPUs, queues, schedules, or paid retries.

Projected external charge for this two-run local pair is **$0 one-time and $0 recurring**. This excludes electricity, existing hardware, and ordinary internet service. Source copies and run artifacts consume local storage; actual directory sizes are recorded in the verification note. No automatic pruning or recurring work is configured. Decode, render, and total runtime are not protected by a whole-process timeout; input/frame bounds are the current limit.

Optional Meta pricing in `meta.py` was checked October 26, 2026 and must be reverified before paid use. The estimated one-call envelope is not a provider-enforced spending cap. This setup makes no Meta request and creates no paid resource.

No Meta credential was exposed by the process environment during initial inspection. A configured adapter must not silently replace a failed Meta call with fabricated model output. The default pipeline uses explicitly labeled local computer vision and numerical fitting. No claim of meaningful Meta use or hackathon readiness follows from that fallback.

Before enabling paid inference, record the selected provider's current token/image pricing, expected input/output tokens, call and retry bounds, total projected spend, and an explicit run budget. No budget or account authorization is implied by a credential merely being present.

No deployment, public release, hackathon submission, agreement acceptance, or purchase is part of this work.
