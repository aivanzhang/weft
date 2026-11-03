# Typed review and rejected-review recovery

A malformed review is an **unsupported result**, not an accepted avatar. New
reviews use four required named checks so duplicate appearance entries cannot
replace a deformation assessment. The existing acceptance and evidence gates
remain unchanged.

## Typed response, unchanged internal review

The API's v2 response has `schema_version: 2` and a closed `checks` object with
exactly these keys:

- `identity`
- `appearance`
- `framing`
- `deformation`

Each contains `passed`, `reason` and `evidence_ids`. Native and exported
appearance must be assessed together in the one appearance check. Missing
evidence is not a pass. State and action-requirement reviews remain required by
their existing acceptance rules.

`typed_critic_schema()` binds allowed citations to supplied evidence;
`typed_critic_prompt()` explains the named checks. `decode_typed_review()` converts
that response to the existing canonical list of named checks. The coordinator
then runs the unchanged `validate_review()` and `_bind_review()` checks. Conversion
alone does not establish acceptance. New API calls do not fall back to the legacy
wire format when v2 data is malformed.

Raw `agents/reviewer-N/answer.json`, provider responses, prompts, schemas, image
snapshots, receipts and charge records remain unchanged. A successfully validated
review receipt contains the canonical format. Historical legacy responses remain
readable; typed historical responses are decoded only for canonical comparison,
while their original raw hashes remain authoritative.

## Continuing a rejected settled review

The guarded continuation path supports the specific terminal profile
`stage: reviewer`, `failure_kind: review_contract`. It is not a transport retry or
a way to promote a rejected answer.

Before saving a continuation, the collector verifies:

- All parent calls settled; exact request, source/reference and source snapshots.
- The paid answer against its saved schema, response, request and artifact hashes.
- Exact prompt, submitted images, authored recipe and model-bound final evidence.
- Completed native model, reload, media, action, deformation and attachment proofs.
- The same review-contract rejection under saved/current validator semantics.
  Syntax trees are compared; archived source code is never executed.

A saved valid review receipt or delivery cannot be admitted through this rejected
review route. The collector carries the raw rejected answer under
`findings.rejected_review.status: invalid_untrusted`, alongside its rejection,
provenance, authenticated images and native measurements. It does **not** create a
valid `review`, invent a missing check, reuse an acceptance verdict or fabricate a
`review-N.json` receipt. Prior authenticated source failures remain available.

The next job reuses the validated plan and latest authored design, then runs new
authoring and independent review through normal gates. No new planner or Meshy
call is needed for an unchanged valid plan. Spending still requires the existing
[source/reference-bound standing policy](standing-reference-retries.md), exact
child grant and remaining aggregate allowance. Unknown charges remain held.

## Images for independent re-evaluation

Rejected review votes do not decide which images the next author sees. The image
consumer independently verifies the parent's inventory, paid review/image hashes,
source/model binding and final evidence, then selects a bounded set by evidence
type, including exported views, isolated exported poses, native views and motion.
A fabricated path or altered image is rejected.

Three source guides and the optional reference remain primary inputs. Repair
images fill only the remaining capacity: at most eight images total, 8 MiB each,
32 MiB combined, deduplicated by hash. Invalid review text remains diagnostic
input, not instructions or a valid quality finding.

## Recorded result and verification

Job `d8795ec6c273489da37954ab` ended **unsupported**. Four calls settled for
**$0.644975**. Its final build and media reached review, but the raw response
contained `identity`, `appearance`, `appearance`, `framing` and omitted
`deformation`. The exact failure was `Duplicate review check`. That answer remains
invalid and unchanged; native gate success does not supply the missing visual
verdict.

Focused evidence:

| Scope | Result | Evidence |
| --- | --- | --- |
| Typed schema/decoder and legacy semantics | 15 tests passed | [Typed critic contract](../outputs/typed-critic-contract-tests-v1/result.json) |
| Coordinator integration and existing gates | 59 distinct tests passed | [Pipeline tests](../outputs/typed-review-pipeline-tests-v1/result.json) |
| Continuation custody and typed historical comparison | 18 distinct tests passed | [Continuation tests](../outputs/rejected-review-continuation-tests-v1/result.json) |
| Rejected-image selection and existing image guards | 18 distinct tests passed | [Image tests](../outputs/rejected-review-image-tests-v1/result.json) |

These are analytic fixtures, not generated-avatar acceptance. Earlier fixture
setup failures and targeted reruns are recorded in those receipts.

The actual d879 read-only collector verified 671 files, four settled calls and
seven submitted evidence images in 0.718 seconds. Its 35,881-byte context retained
the invalid review and required zero new planner calls; output and ledger hashes
were unchanged. See [the custody proof](../outputs/rejected-review-continuation-tests-v1/actual-read-only.json).
The subsequent [actual image-consumer proof](../outputs/rejected-review-image-consumer-proof-v1/)
selected eight total author images and constructed a bounded prompt without a paid
call or native build.

Actual website Retry→Prepare then passed for child `45757c9565b545119553baca`,
without another planner call. See [website preparation](../outputs/gpt52-panda-website-retry-v4/)
and [Start observation evidence](../outputs/gpt52-panda-studio-live-v4/).
No accepted delivery is established by these recovery checks. The
[current handoff](../TODO.md) owns live status and the next safe action; do not
infer an active process from this historical note.
