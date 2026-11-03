# Clearer-frame equipment initialization

## Verified checkpoint — November 2, 2026 (PST)

The ordinary portal now lets equipment analysis select a clearer supplied image
instead of forcing the first image. One real API run advances to two conditional
world proposals; both reject **degenerate projected torso scale**. No playable
world, selected candidate, physics rollout or trained policy is claimed.

## Implementation and compatibility

- New `equipment_analysis_contract.py` defines the equipment-only v2 format.
  Each of at most three objects names an exact original `source_frame` from the
  at-most-six submitted images. Unknown, unsampled, reordered, duplicated or
  invalid frame identities reject. Coordinates retain source dimensions and
  outward rounding; selected image identities remain inspectable.
- `portal_pipeline.py` keeps the original object/scene default, schema and prompt
  byte-for-byte. Equipment opts into v2 explicitly. Request, response, image,
  format and schema joins remain closed and source-bound.
- Future equipment quotes bind v2; legacy quotes without the new profile still
  bind v1. Worker execution cannot silently change the quoted method. The same
  project API keys, persistent budgets, one-request bounds and no-retry behavior
  apply. No ChatGPT login or coding-agent runtime is involved.
- The complete original video still reaches tracking. Person initialization
  remains frame zero. Earlier equipment slots retain their actual returned or
  missing status and pre-initialization phase; no prefix is trimmed or fabricated.
- Preflight validation admits current producer bytes or the exact reviewed
  historical producer SHA
  `75434812f8b7727e79ba64b82d33bfb2e31fb752e18b8d99208865dfaea1479d`.
  Original body, timing, anchor and source proofs are still rederived. The
  historical test snapshot is inert, never executed; altered producers reject.
- `portal_body_reuse.py` can create a new child from a pinned completed equipment
  refusal with no selectable world. Four small parent metadata documents are
  retained. Active, partial, selected, ready or selectable results still reject;
  original artifacts and deduplication remain unchanged. This is not a retry of
  the old paid request.

**179 distinct scoped Python checks pass**, recorded in
`.local/sampled-equipment-checkpoint-01/tests.json`. Independent reviews:
`.local/perception-sampled-frame-review-01/` and
`.local/portal-body-reuse-refusal-review-01/`. Author release notes:
`.local/equipment-sampled-frame-01/` and `.local/portal-body-reuse-terminal-01/`.

## Actual complete-video portal execution

The prior first-image attempt and its recovered refusal remain unchanged in
project `2661cef9df634f81a70cc9664db6829f`. New ordinary child:
**`cb3ea4169eae4804bd0aa6f99d3453cb`**, under
`.local/portal-kettlebell-flip-01/portal/`.

Source MP4 SHA256 remains
`c289fc1afe0b60a418cc6ed9e4554c8414a4d970d2dfcaea1f6f82839204dac3`.
It is the full attributed CC BY-SA 4.0 kettlebell-flip recording, not a new
holdout. The exact retained body has all 442 references over 458 source frames.

1. `.local/portal-sampled-init-preflight-01`: ordinary saved-body CLI and portal
   preflight finish in **17.905 s** with zero API or inference calls. The new v2
   quote is served normally; worker takes 12.196 s, sampled peak 1,192,296,448 B.
   All original parent/body files and captured runtime remain unchanged.
2. `.local/portal-sampled-init-paid-01`: one guarded OpenAI request and one Meta
   request complete preparation, publication and conditional assembly in
   **71.530 s**. Preparation worker: 53.707 s / 1,052,344,320 B. Equipment worker:
   17.061 s / 914,653,184 B. Existing process, storage and task limits remain.

Cost/source plans are `.local/portal-sampled-init-cost-plan-01.json` and
`.local/portal-sampled-init-paid-plan-01.json`. The quote reserves at most
$0.3135 OpenAI plus $0.3664 Meta; actual charges are **$0.002358** and **$0.0916**
respectively, **$0.093958 total**. All charges settled, no retry or unknown charge.
Lifetime totals: OpenAI **$0.007886/$5**, Meta **$0.466027/$5**. No new recurring
infrastructure or deployment. Test servers, queues and workers closed/reaped.
The saved preflight runtime is byte-identical to the paid run's recorded runtime.

## What improved and what did not

The API selects original frame **91**, not frame zero. Its exact saved image is
unchanged from the earlier sample set. The proposed source ROI is
`[276, 309, 84, 84]` in the original 640×480 image. Equipment is more visible in
this frame, but the region remains loose and includes leg/background pixels.
Review: `.local/sampled-init-localization-review-01/`.

Saved tracking contains all **458 × 2 = 916 slots**, all returned. All 91 prefix
equipment slots keep the pre-initialization label. Native timing matches the
retained 458-frame clock; decoder time remains separate. Equipment median box
area falls from 9,070 to 529 pixels², while its minimum reaches 2 pixels², so this
is not a blanket tracking-accuracy claim. The role screen distinguishes the
person more clearly, but an unresolved proximity group still spans all frames.
Review: `.local/sampled-init-tracking-review-01/` uses 5.42 MB of saved JSON only.

The unchanged constructor retains two competing assignments. Both report
`proposal_geometry_rejected: Torso scale is degenerate`, remain unselectable and
preserve original task/source gates. This is progress beyond the prior lack of
a rigid proxy, not successful reconstruction or contact.

## Diagnosed root-contract mismatch and active fix

Saved-only diagnosis completes in 0.174 s / 143 MB using one 28.88 MB observation
document. Exactly two of 442 references have a projected torso span below eight
pixels: frame 121 (6.5086 px) and frame 122 (7.2659 px). Original normalized-to-pixel
conversion and fixed-region mapping agree exactly. Source IDs/times agree;
model-estimated torso depth supports foreshortening, not a coordinate mistake or
proof of physical anatomy.

Crucially, the original body already marks both roots `interpolated_reference_only`,
using original frames 120→123 over 0.100 s within its existing two-sample/0.25 s
limits. It uses 440 observed references for scale (median 49.6605 px). Equipment
proposal incorrectly recomputes the median over all 442 (49.4667 px). Removing the
per-frame guard alone would still change scale by 0.3919% and mismatch 441 root
translations, up to 0.01985 model units. That is not an acceptable fix.

Exact receipts: `.local/sampled-init-torso-diagnosis-01/` and
`.local/sampled-init-scale-contract-review-01/`. No paid repeat is needed.

Active non-overlapping ownership:

- Contract agent: `equipment_proposal.py`, a new source-root binding helper and
  dedicated tests; reuse/rederive the original sealed scale subset and all roots.
- Runtime agent: matching versioned validation in `shared_world_reconstruction.py`
  and dedicated tests, after agreeing the exact binding interface. Legacy strict
  behavior remains unchanged; interpolated roots must never become observations.
- Portal agent: find the existing saved-evidence constructor/import path for a
  fresh result without rewriting the current rejected project or repeating APIs.
- Root: integrate, preserve source/task/floor/interpolation bounds, run one
  bounded saved-only acceptance after review, and publish only verified changes.

Do not trim references, change body/root values, remove the old guard, move the
floor or weaken task gates merely to admit this clip. The correction must match
the original scale, root recipe, interpolation support and evidence exactly.

Inspect both projects with:

```sh
.venv-lab/bin/python -m weft portal \
  --root .local/portal-kettlebell-flip-01/portal --port 8766
```

The user manages the persistent server. Existing playable default-portal examples
remain intact. General ground/equipment contact, useful full-task learning and
independent physical validation remain unfinished. Preserve separate avatar work.
