# Reference-bound standing retries

Standing policy **v2** lets the website continue an unsupported avatar job using
its exact paid source, optional reference image, validated plan and authenticated
failure evidence. It does not copy an old manual rig or authorize fresh Meshy
work. Native builds and independent quality review still decide acceptance.

## Policy contract

The owner-only file is `.local/openai-standing-authorization.json` (mode `0600`).
Only the owner configures it; browser requests cannot supply a key, budget,
provider URL, source path or policy replacement.

Version 2 retains every v1 field and adds one required field:

| Field | Meaning |
| --- | --- |
| `schema_version` | `2` |
| `allowed_reference_sha256` | Exact lowercase SHA256, or `null` to require no reference |
| `allowed_source_sha256` | Exact SHA256 of the saved, already-paid GLB |
| `lineage_root_job_id`, `lineage_root_request_sha256` | Immutable root job and request for this retry chain |
| `approval_receipt` | Project-relative receipt path and SHA256 |
| `max_total_microdollars` | Existing total project ceiling, not a fresh allowance |
| `model`, `max_calls`, `max_repairs`, `max_successors` | Finite model, per-job call/repair and lineage bounds |
| `status`, `policy_id` | Active/revoked state and distinct policy identity |

A reference snapshot must stay under the job's `inputs/` directory, use a
supported `reference.png`, `.jpg`, `.jpeg` or `.webp` name, and match its exact
recorded bytes and hash within **8 MiB**. The source is bounded at **64 MiB**.
Symlinks, escaped paths, altered bytes and mismatched selections are rejected.
Intent comparison uses source/reference **bytes and hashes**, not their original
absolute paths. Prompt, style, requested actions and output intent still match the
approved lineage. A different image requires a different explicit owner policy.

V1 policies remain valid and still reject references. V1 is not silently
upgraded, and a v2 policy with a null reference hash cannot acquire an image later.

## Website sequence

1. Select the safely finished unsupported parent and choose **Retry saved work**.
   The authenticated retry endpoint saves one idempotent child; it launches no work.
2. Choose **Prepare**. The worker snapshots the immutable child request, verifies
   parent process custody and continuation evidence, and grants that exact child
   under the owner policy and canonical ledger.
3. Prepare reuses the authenticated validated plan when it remains valid. A
   supported semantic replan, if required, is explicitly recorded and costs a call;
   it is not silently skipped.
4. Choose **Start** for the exact prepared plan. Agents receive the prior proposal
   and verified failures; native construction and quality gates run again.

Status reads do not grant, spend or dispatch. At action time the worker rechecks
policy hash, custody, source/reference, lineage, call limits and available money.
The parent must have settled calls before retirement into its exact successor.
Unknown charges elsewhere remain fully held. Closed unused grants do not erase
those charges. A live owner, uncertain parent, policy change or exhausted bound
blocks continuation; no automatic credit purchase or budget increase occurs.

## Current configured lineage — November 2, 2026, PST

The owner installed a new v2 policy for the accepted raw panda/shoulder-dragon
source and accepted portrait. This is a new lineage under the same **$15 total
OpenAI cap**, including prior spend and unknown holds; it is not $15 per retry.

- Root: `studio-447a849b435b4d2e962b3763`.
- Model: `gpt-5.2-2025-12-11`; at most **7 calls / 2 repairs per child** and
  **3 successors** in this lineage. Available funds can stop it earlier.
- No new Meshy generation or credits are authorized.
- Immutable policy, approval, activation and previous v1 policy:
  [`outputs/openai-standing-reference-policy-v2/`](../outputs/openai-standing-reference-policy-v2/).
- Installed policy SHA256:
  `f43d5ede22c8c6f0a216db021295b0e04f3bc8558fe751e68fb1416981d9d826`.

The root's planner and three author calls all settled. All three native attempts
failed; the continuation collector authenticates those proposals and failed packs,
retains the latest ready proposal plus distinct source failures, and requires
**zero new planner calls** for this validated plan. No completed native build is
required to continue a well-recorded failed build.

The actual website retry/Prepare passed with evidence in
[`outputs/gpt52-panda-website-retry-v3/`](../outputs/gpt52-panda-website-retry-v3/).
It reused the validated plan without a new planner call and closed its temporary
browser/server. Child `d8795ec6c273489da37954ab` is now running through the actual
website Start observer, session **17883**. Do not launch a duplicate. Its first
native build succeeded but the deformation gate rejected belly vertices influenced
by opposite arms; the existing bounded repair loop is correcting the weights.
This is not final avatar acceptance. Consult the [current handoff](../TODO.md)
and saved run result for subsequent status.

## Verification

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest \
  tests.test_standing_authorization.StandingAuthorizationTests \
  tests.test_standing_reference_authorization.StandingReferenceAuthorizationTests \
  tests.test_standing_reference_authorization.StandingOptionalReferenceTests -q
```

**28 tests passed** in 0.225 seconds: unchanged v1 behavior, v2 reference and
no-reference flows, idempotence, source/reference tampering, unsafe paths, size
bounds, policy changes, finite lineage, unknown-charge rejection and spending
limits. These use analytic input bytes and temporary ledgers, not generated
avatars or provider calls.

The actual root's hypothetical v2 binding passed read-only in 0.015 seconds
before installation. That snapshot recorded **$11.077570 conservative project
exposure** and **$3.922430 available successor allowance**, preserving unknown
holds. These are historical measurements, not a current balance or permission to
exceed the ledger. Active policy and ledger hashes were unchanged by that check.
Evidence: [`outputs/standing-reference-policy-tests-v1/`](../outputs/standing-reference-policy-tests-v1/).

Implementation: [`standing_authorization.py`](../avatar_assets/standing_authorization.py).
Related: [website backend](api-website-backend.md), [hands-off Studio](hands-off-studio.md).
