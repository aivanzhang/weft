# Two independent animal preflights

Recorded November 2, 2026, 8:00 PM PST. No API, Meshy, native-generation or ledger-writing calls occurred. These are validated requests, **not generated animals or successful motion demonstrations**.

## Concrete requests

- `owl-flight-request-v2.json`: one independent barn owl; modeled/painted feather appearance; one four-second takeoff, repeated wing flapping and forward travel action. No landing/return sequence added.
- `rabbit-hop-request-v2.json`: one independent cream rabbit; modest short fur; one four-second repeated forward-hop action with leg articulation.

Each has no reference, reused model, companion, scene, shared mesh or continuation. Both require their own new GLB, rig, materials, motion and outputs. Prepared outputs are sibling directories `outputs/animal-owl-flight-preflight-v2` and `outputs/animal-rabbit-hop-preflight-v2`; their separate identities prevent mistaken fused-asset delivery.

The free production `pipeline.prepare` path passed in 0.028 s and 0.011 s. It validated input limits, saved the API backend snapshot, output specifications and source pins. It did not create an AI plan or geometry. Provider/native entry points were mocked to fail if called. `preflight-results-v2.json` records exact request hashes. V1 preparations remain unchanged because the root added explicit single-asset planner guidance after those source snapshots; V2 includes current guide SHA `5e10705b24cb349f99609ad5c2ba40a7fd47e9b3f1fc3dfb0a85ceb1a54b0aa4`.

## Cost and authorization

| Work | Per animal | Both |
|---|---:|---:|
| Meshy-7.1 Ultra4K geometry |25 credits|50 credits|
| Meshy4K texture refinement |10 credits|20 credits|
| Total Meshy |35 credits|70 credits|
| Planned OpenAI roles |3 calls|6 calls|
| Illustrative OpenAI estimate |$0.5985|$1.197|

The illustration assumes 50,000 input tokens including images/schema and 8,000 total output tokens including reasoning per call, at GPT-5.2 $1.75/M input and $14/M output. It is not measured demand or a completion guarantee. Proposed backend snapshot: `gpt-5.2-2025-12-11`, medium reasoning,16,384 max output tokens,600-second request deadline,3 calls,0 repairs. The owner configuration is unchanged; a guarded website run needs this explicitly selected matching server profile. No automatic paid retries or credit purchases.

Read-only Meshy ledger:245 authorized credits,237 reserved/reported,8 remaining. **70 credits require62 additional credits**. Account-specific dollars per credit are unknown.

Authoritative `budget-effective.json` applies the saved append-only OpenAI budget reduction: **$15 total ceiling, $13.375724 exposure, $1.624276 remaining** at the recorded time. Historical unknown charges remain held. The first raw `budget-readonly.json` observed the original $17 ledger row and is retained as a superseded observation; it is NOT the applicable ceiling.

Each OpenAI call must first reserve **$0.929376**; two concurrent calls would require $1.858752. Run sequentially and stop if the unchanged $15 ceiling cannot cover the next reservation. Six full reservations sum to $5.576256, not expected charges. Settled usage can release unused reservation, but the remaining allowance cannot guarantee all six calls or repairs. No new OpenAI allowance was requested or created here.

Official documentation opened November 2, 2026, 7:57–7:58 PM PST:

- [Meshy API pricing](https://docs.meshy.ai/en/api/pricing): Meshy7.1 text preview20 credits, Ultra geometry+5,2K/4K refine10.
- [Meshy text-to-3D](https://docs.meshy.ai/en/api/text-to-3d): preview and refinement are separate steps; current Meshy7.1 supports4K geometry. These match current adapter fields.
- [OpenAI GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2): pinned snapshot and standard input/output pricing used above.

## Supported route and limits

Existing route: text planner → Meshy geometry+texture → inspect generated topology → local AI-authored nonhumanoid skeleton, weights and action curves → native/export checks → independent review → preview/download if accepted. Provider rig is `none`; no paid humanoid rig/motion is incorrectly applied to owl or rabbit. Wings/legs must actually separate and deform in generated topology; a prompt is not proof.

Feathers have no dedicated strand system/material enum: classify as `other` with feather finish, using generated feather geometry/paint and articulated wings. Do not label mammal fur as feathers. Rabbit fur uses existing bounded native/portable routes; quality and attachment may reject it. No physics, ground-contact or aerodynamic validity is claimed. Unsupported topology, failed deformation, insufficient meaningful movement or budget limits produce explicit failure/checkpoint, not stock animation or success.

The single job delivers one actor. Panda and dragon as independent assets therefore require separate jobs and separate downloadable models. A combined Meshy input, shoulder attachment, mesh labels or a single skeleton do not meet that requirement. A two-actor scene/composition manifest is not implemented by these source-free requests; do not claim these owl/rabbit tests prove it.

## Next action and commands

Root can request the single62-credit incremental Meshy decision now, keeping the OpenAI ceiling at15 and existing holds unchanged. After approval, create two separate canonical website jobs from the v2 JSON subject/style/action values, using the proposed3-call/0-repair profile. Retain canonical exact-job grants and ownership; do not directly execute these preflight folders as if they were website-created jobs.

Commands already run from `tools/avatar-assets`:

```sh
PYTHONPATH=. .venv/bin/python outputs/different-animals-preflight-v1/prepare_requests.py
PYTHONPATH=. .venv/bin/python outputs/different-animals-preflight-v1/prepare-current-sources.py
```

They use immutable new output folders and must not be rerun in place. The first invocation without `PYTHONPATH=.` failed on import before any request/output creation. The corrected invocation passed. A read-only budget helper import typo also failed before opening the ledger; the corrected `budget_snapshot` call produced the authoritative result. No source code was changed by this workstream.
