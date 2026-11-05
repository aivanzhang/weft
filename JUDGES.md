# For judges

Thanks for reviewing Weft. Two ways in, depending on how deep you want to go.

## 1. Hosted demo — zero setup

**https://weft.ivanzhangofficial.workers.dev**

The landing page explains the product; the **Environments** section is live:

- Browse the gallery, then press **Try it with your own video**.
- Supported input for the hosted tier: an MP4 up to 10 MB / 15 seconds showing
  **one isolated ball moving horizontally under a fixed camera** (the sample on
  the page shows the idea). The pipeline reconstructs a playable environment
  and shows the initial physics fit against the repaired one.
- **Access code**: the submission form in our hackathon entry includes a judge
  access code. Paste it into the optional *Access code* field before
  generating — it moves you to a reserved lane so public traffic can never
  exhaust your capacity. Without it you share the public daily budget.
- Results are private to your browser session and expire after 24 hours.

This hosted tier is deliberately narrow: the container runs with no
credentials and no outbound network, and every limit it enforces is published
in `hosting/README.md`.

## 2. Full product — local, ~10 minutes

The complete portal (video → evidence → MuJoCo world → RL task → recording and
replay, with Meta SAM tracking, Muse review, and the Muse controller test)
runs locally. Verified on macOS / Apple silicon with Python 3.14 and Node 22+:

```sh
python3.14 -m venv .venv-lab
.venv-lab/bin/python -m pip install -r requirements-lab.lock
.venv-lab/bin/python -m pip install --no-deps -e .
.venv-lab/bin/weft fetch-pose-model        # pinned local pose model, no account
.venv-lab/bin/weft fetch-exercise-sample   # attributed CC-licensed sample clips
npm run sandbox
```

Open the full authenticated link the launcher prints for
`http://127.0.0.1:8765/` (keep its token private). Then:

1. **Upload** a sample exercise clip (single person, mostly fixed camera,
   ≤60 s). The local check runs with zero API calls; admitted clips offer the
   saved-body sandbox, refused clips show their reasons — and when the defect
   is localized, the pipeline proposes its own supported window as a one-click
   retry.
2. **Interact**: reset, drive controls, set a placement task, run bounded
   training, record a continuous session, swap the character mid-take
   (built-ins prepare with one button), export the video and exact replay.
3. **Muse** (optional, needs your own Meta API key configured per
   `sandbox/README.md`): evidence review, and the bounded controller-choice
   test where Muse's actual pick is scored under the unchanged task — accept,
   reject, or abstain, with receipts either way.

Everything the pipeline claims is backed by preserved evidence: budgets and
API charges live in append-only ledgers, failures keep their artifacts, and
`PROJECT.md` indexes the evaluation snapshots.

## Honest limits

Single person and mostly-fixed cameras on the exercise route; multi-person
clips are refused with reasons. Fast small objects can exhaust the tracker
ladder. Simulations are conditional models with disclosed assumptions — not
verified copies of the physical world — and the product says so wherever it
matters.
