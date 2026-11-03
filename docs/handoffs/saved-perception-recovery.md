# Saved perception recovery and equipment initialization

**Continuation:** [Clearer-frame initialization](sampled-equipment-initialization.md) is now implemented and real-API verified. It reaches two rejected geometry proposals; saved torso-scale diagnosis is the next dependency, not another provider retry.

## Verified recovery — November 2, 2026 (PST)

The original paid preparation is recovered without repeating either provider or
body inference. The ordinary constructor still **abstains**: the saved tracks
do not distinguish the person from equipment well enough to build a candidate.
The full video-to-world goal remains active; no new playable world is claimed.

### Implementation

- `portal-recover-perception` requires a stopped portal, an exact project ID and
  the caller's bridge-manifest hash. It uses the ordinary full source publisher,
  existing capacity checks and an immutable additive equipment recovery receipt.
- Original job, source, perception status/failure and provider receipts remain
  byte-identical. The recovery queues the existing equipment request schema;
  it cannot queue provider work, pose inference, training or automatic selection.
- Startup resumes only a valid never-started recovery queue. A durable exclusive
  start marker precedes expensive validation; interrupted construction does not
  silently retry. Worker admission rederives the normal source proof.
- The measured 267.09 MiB source inventory now fits a scoped 320 MiB allowance.
  The 64 MiB/member, 1,536-file, 30-second qualification, 650,000,000-byte job and
  1.5 GiB worker limits remain unchanged. The publisher honors the smaller of
  its deadline and the recovery caller's remaining time.

Implementation/review receipts: `.local/portal-perception-recovery-01/`,
`.local/perception-publication-size-01/`, and
`.local/portal-perception-recovery-review-01/`. Forty-two new focused cases
(27 recovery, 15 bound/deadline) and 17 selected existing cases pass. These are
scoped checks, not complete repository or browser QA.

The portal now distinguishes the retained preparation failure, zero-cost saved
recovery and current assembly result. Known identity failures have a plain-language
explanation; zero candidates no longer invites an impossible selection. An exact
recovery receipt is downloadable through the existing authenticated artifact API.
Only bounded metadata is read on the request thread; selected projects retain
their original recovery history. Ten additional server cases and 18 DOM groups
pass. Actual HTTP inspection in `.local/recovery-portal-view-01` returns the
7,025-byte summary and exact 1,075-byte receipt, denies an unauthenticated download,
and preserves all 238 existing job files. No worker starts; server closes. This
is API/scripted DOM verification, not visual browser QA.

### Actual saved-only acceptance

Harness: `.local/portal-saved-recovery-01/{run.py,recover.py}`. It captures the
current Python runtime before execution, invokes the ordinary CLI once, starts
a temporary authenticated loopback portal, waits for normal equipment dispatch,
and downloads its source-validation and candidate artifacts by HTTP.

| Stage | Elapsed | Peak memory |
| --- | ---: | ---: |
| CLI publication and queue | 5.342 s | 769,572,864 bytes |
| Ordinary equipment worker | 8.235 s | 865,402,880 bytes |
| Complete test | 14.489 s | 103,481,344 bytes in the supervisor |

Memory is sampled process peak, not a kernel allocation guarantee. All original
job files, frozen runtime files and canonical provider ledgers are unchanged.
Zero provider/inference/training calls; temporary server, queue and worker closed.

Project: `2661cef9df634f81a70cc9664db6829f`, under
`.local/portal-kettlebell-flip-01/portal/`. Complete saved bridge:
`42007e8f1efee448df7b55cd67137b18e8faa76a79ab807d8b1fea3292f1d11d`.
Published source descriptor:
`bfcfc85f4b4770c5638de6cf35c4db260470fa763e3fa555a3922d0d0b9ac691`.
Candidate report:
`8666493b0ba4043a515e53b620601c4594cf35c0b09d1cec79eeebb18b29549b`.

The source proof retains all 458 frames, 442 body references, both raw tracking
IDs and unresolved associations. Assembly reports `insufficient_role_margin:sam-1`,
`no_supported_rigid_proxy_candidates`, and `unresolved_body_correspondence:sam-1`.
No candidate, selection, physics rollout or learned policy resulted. The earlier
publication failure remains visible beside this later completed assembly attempt.

The user can inspect the project with:

```sh
.venv-lab/bin/python -m weft portal \
  --root .local/portal-kettlebell-flip-01/portal --port 8766
```

The user owns that persistent server. Do not repeat the completed recovery or
provider request. Existing playable examples in the default portal remain intact.

## Active parallel continuation

1. **Equipment initialization:** allow an explicitly versioned equipment-only
   analysis to choose an original frame from the already submitted sample set.
   Preserve exact legacy requests/readers and the original frame-zero person
   anchor. Earlier slots retain their pre-initialization phase and actual
   returned/missing status; no prefix is fabricated or removed. This does not
   establish identity or contact. No provider call is authorized by a test.
2. **Portal explanation — complete:** original failure, saved-only recovery and
   current assembly outcome use small, hash-bound metadata, verified above.
3. **Independent review/integration:** verify legacy source compatibility and
   new sample identity, then perform one bounded representative acceptance only
   when implementation and cost/source checks are complete.

The first image's proposed equipment box overlaps the person. Source-coordinate
conversion is correct. Local inspection of already saved images 91 and 274 shows
more distinct equipment, supporting the clearer-frame hypothesis—not a manual
replacement box, verified tracking, or an approved repeated unchanged request.

Budget after recovery is unchanged: Meta $0.374427/$5, OpenAI $0.005528/$5;
zero unsettled calls. Last published main checkpoint is `d509a2c`. Preserve
separate avatar changes. Ground/contact, useful full-task learning and broader
independent physical validation remain unfinished.
