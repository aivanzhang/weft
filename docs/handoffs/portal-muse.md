# Muse review in the product portal

**Follow-up:** [local control comparisons](portal-checks.md) now run through the
portal with exact traces and unchanged scoring. Physical measurement requests and
automatic repairs remain unresolved; the original live review below is unchanged.

## Verified milestone — November 2, 2026 (PST)

Muse is now available through an authenticated project action in the existing
portal, not just a standalone CLI or avatar feature. OpenAI still plans the
initial scene; Muse critiques construction evidence and proposes tests. This is
not the completed automatic repair/test/acceptance loop.

### Implementation

- `portal_review.py` pins source video, evidence and optional human-model bytes
  to the saved construction manifest and original upload identity. Exercise uses
  `build/body_source`; objects use `planar/run`. The existing advisor selects at
  most twelve construction observations and three thumbnails, excluding scores,
  policies, audit episodes and development-validation rows.
- `POST /api/projects/<id>/muse` queues one review per completed project. Repeated
  requests return the existing review. There is no retry or restart requeue.
  Review and build jobs share one supervised processing slot; at most four
  reviews wait. Reviews have a 90-second wall bound and the existing sampled
  1.5 GiB RSS guard. Cancellation preserves unknown charges and existing advice.
- `muse/status.json`, `binding.json`, `advice/*`, worker log/receipt are separate
  from the completed simulation's status and immutable run. A successful review
  means schema-valid proposals, never accepted or applied physical claims.
- Portal UI shows separate Meta/OpenAI allowances, evidence references,
  alternatives, required tests, unanswered questions and downloadable receipts.
  Model text uses text nodes. Sandbox access remains available during review.
- Project keys only, no ChatGPT login. Meta uses the original shared ledger;
  OpenAI credentials are not forwarded to the review worker. No dependencies,
  browser automation, hosted deployment or persistent server were added.

### Live acceptance

Exact harness/runtime/receipt: `.local/portal-muse-01/`. This directory contains
the plan, `check_http.py`, pinned runtime manifest and `result.json`. Do not rerun
it: the project already has its one review and the acceptance refuses duplicates.

The real authenticated HTTP action reviewed the existing source-bound exercise
job `4e0f1fe859664bb5ba788214e129451d`, using the licensed source
`single-leg-squat.webm`, SHA-256
`ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
No pose inference, training or physics rollout was repeated.

- One real `muse-spark-1.3` Standard request, zero retries, three proposals.
- Complete HTTP check 8.615943 s; worker 6.895824 s / 105,922,560 B peak RSS.
- 22 HTTP requests: mean 56.54 ms, max 1.18728 s, including sandbox initialization.
  These are this development workload's observations, not sustained-load proof.
- Cost **$0.013384**. Afterwards the shared Meta ledger had ten reservations,
  **$0.249336 committed / $4.750664 remaining**, zero unsettled. OpenAI unchanged.
- Duplicate HTTP submission made no second call. Proposal/evidence/receipt
  downloads and mounted sandbox status were verified through the same server.
- All 1,325 run members reverified. Manifest remained
  `d8cfe07cc44e3d36efafcd6039c49435188e684c763e28fb9ce11e5cc8d1fe26`;
  the parent project's terminal status bytes also remained unchanged.
- Proposed gravity/camera alignment, unverified contact/support, and partial
  articulation need new calibration/contact evidence. Muse's semantic description
  is itself unverified; do not treat its activity labels as ground truth.
- The temporary server shut down, the worker exited 0 and was reaped. No live job
  remains from this acceptance. Original source/scored runs/recordings are intact.

Official pricing was fetched from `https://dev.meta.ai/docs/pricing-rate-limits`
into `.local/meta-pricing-portal-20261102.html`: Standard input $1.25/M,
output $4.25/M. The 32,768-input / 4,096-output estimate remains $0.058368/request. This
application estimate is not a provider hard cap. No recurring resource was added.

### Verification and remaining work

**72 distinct Python checks and four portal UI checks pass.** Evidence is indexed
in `.local/portal-muse-01/audit.json`; retained JUnit results include the corrected
fixture failure below. UI checks are deterministic DOM tests, not browser visual QA.
The final unexpected-model rejection was verified locally after live acceptance;
the actual live response already matched the pinned model, so no paid repeat was made.

Focused tests cover evidence tampering/symlinks, excluded evaluation data,
unknown-cost preservation, cancellation, restart, repeated requests, auth/origin,
artifact access and safe UI rendering. One HTTP fixture initially omitted its
advice directory; only the fixture was corrected, with original failure retained
in `.local/portal-muse-http-01.xml` and passing correction separately.

Next: connect typed, bounded Muse hypotheses to independent test executors and
expose accepted/rejected repair comparisons. Preserve current scorers and source
evidence. The current live proposals need additional real observations, not a
simulated self-check relabeled as measured contact. Continue source-qualified
equipment/missing-input assistance and unfamiliar-video reconstruction work;
neither this feature nor the prior portal checkpoint completes the full goal.

Use `.venv-lab/bin/python -m weft portal --port 0` to start the user-managed
portal and inspect the saved exercise review without another charge. No key needs
to be re-entered. Hosted site stays offline. Preserve the concurrent avatar
session's dirty paths; publish only the explicit portal-owned paths.
