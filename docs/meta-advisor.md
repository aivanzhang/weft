# Meta scene advisor

The scene advisor asks Meta's Muse Spark model about **competing physical explanations** for a video: gravity versus camera projection, contact and support, articulated movement, actuator assumptions, and what cannot be inferred about muscles. It produces proposals and suggested tests, not verified physics or automatic simulator changes.

## What is implemented

`weft.scene_advisor.advise_run(run, out, questions=None, execute=False, max_cost_usd=None)` accepts a completed local lab `run/` and writes a **new, separate output directory**. Preparation is offline. It includes at most 12 construction observations and three aspect-preserving thumbnails from the pinned source video. Development-validation rows, audit episodes, learned policies, scores, and evaluator code are excluded.

Each response hypothesis has a concept, claim, evidence/object references, model-reported confidence, alternatives, and at least one proposed falsification test. Unknown evidence/object references, extra mutation fields, malformed/nonfinite values, claims labeled `verified`, and oversized responses are rejected. A valid response means **schema/provenance checks passed**, not that its physical claim is true. Confidence is not a calibrated probability.

The prompt explicitly separates observed joint motion from unmeasured muscle activation and force. It forbids anatomical or medical conclusions, evaluator changes, tool execution, and generated code. These prompt rules are reinforced by a data-only response contract and no execution/apply path. A person or future independent test executor must evaluate the proposed hypotheses before changing an environment.

## Prepare a request without uploading anything

From the repository root:

```python
from weft.scene_advisor import advise_run

result = advise_run(
    "artifacts/exercise-lab-03/run",
    "artifacts/exercise-advice-new",
    questions=[
        "Which support and joint constraints are missing from this motion, and how would we test them?",
        "Which muscle properties cannot be established from these poses alone?",
    ],
)
```

Run this using `.venv-lab/bin/python`. Inspect `context.json`, `request.json`, and `advisor-call.json`. The request contains source thumbnails; these are written locally and only sent if explicit execution is requested. Keep the prepared directory unchanged and choose a new output directory for a live request.

## Execute one Meta request

The caller must supply `execute=True` and `max_cost_usd` covering the recorded estimate. Configure an explicit **aggregate lifetime budget** first with `weft.meta_access.configure_access(total_budget_usd=...)`. This never resets past charges or reservations. Add `prompt=True` in an interactive terminal to save an existing Meta key with hidden input; do not paste keys into chat or command arguments. Key creation/account setup at Meta is not performed.

Credentials come from the process's `META_API_KEY` / `MODEL_API_KEY`, or the project-only `.local/credentials/meta-api-key` file (mode `0600`, parent directory `0700`). No sibling credentials or dotenv files are read. Hidden prompting refuses noninteractive/echoed fallback. Existing credential files are never silently overwritten. Keys are never included in artifacts, status, the budget ledger, or worker environments.

```python
result = advise_run(
    "artifacts/exercise-lab-03/run",
    "artifacts/exercise-advice-live-new",
    execute=True,
    max_cost_usd=0.06,
)
```

This sends the reviewed class of evidence to the official Meta endpoint. Model: `muse-spark-1.3`, **Standard mode**, not Contributor mode. Its scope is one request, no tools, no automatic retry, a 45-second network timeout, 4,096 maximum completion tokens, `reasoning_effort: low`, and a 128 KiB response limit. Reasoning tokens share the completion allowance; the default request previously exhausted it without an answer. The prompt asks for at most three concise hypotheses. Redirects are rejected instead of forwarding credentials elsewhere. [Documented reasoning controls](https://dev.meta.ai/docs/protocols/chat-completions)

`request.json` preserves the exact payload. `response.json` preserves the bounded raw response, including invalid model output. `hypotheses.json` appears only for schema-valid output and always says `accepted: false`. `advisor-call.json` records source/frame/request/implementation hashes, requested and returned model, response ID, usage, duration, status, cost estimate, and failure class. It never records a proposal as independently verified or applied.

## Cost envelope

Pricing checked **October 28, 2026**: Muse Spark Standard input is $1.25 per million tokens and output is $4.25 per million. The conservative assumed 32,768-input-token envelope plus 4,096 output tokens estimates **$0.058368 per call**; three calls estimate **$0.175104**. This is an application estimate, **not a provider-enforced spending cap**. Both the scene advisor and legacy rolling Meta adapter share the project ledger. Before transport, an exclusive file lock protects a durable reservation against concurrent calls. Interrupted requests and ambiguous failures keep the full estimate charged; only complete returned token usage permits settlement to known cost. Reconfiguring the total preserves all reservations. A known overrun is recorded and prevents requests when the remaining total cannot cover another envelope. Actual input tokenization is not measured before execution. Cached-input discounts are not assumed; recorded cost uses the undiscounted rates. Missing usage is reported as unknown, never free. [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits)

No recurring job, cloud instance, model download, storage service, paid retry, or external log service is created. Local artifact storage remains on the user's computer. Preparation incurs no API charge; local compute/storage cost is not measured as zero.

## Verified state and remaining work

- Deterministic tests cover no-network preparation, provenance exclusions, malformed/budget/credential rejection, private key permissions, thread/process-safe reservations, crash/timeout accounting, secret-free records, no retries, response retention, source integrity, and redirect rejection. Transport tests use explicit mocks; they are **not live model evidence**.
- Offline preparation succeeded against saved rolling, tabletop, and exercise artifacts under `artifacts/meta-advisor-rolling-01`, `artifacts/meta-advisor-planar-01`, and `artifacts/meta-advisor-exercise-01`. Each has 12 observations, three thumbnails, and `request_count: 0`.
- The user added a project-local key. **Live access and schema-valid advice are verified** by `artifacts/meta-body-contact-live-02`: Muse Spark 1.3 returned three hypotheses, 5,965 prompt / 1,691 completion tokens, 8.862 seconds, usage-priced cost $0.014643. Nothing was automatically applied.
- `artifacts/meta-body-contact-live-01` retains the first failed answer: 4,096 completion tokens (4,094 reasoning), empty content, finish reason `length`, cost $0.02486675. The subsequent request explicitly changed reasoning effort and requested concise output; this is not an automatic retry or a controlled latency comparison. These two calls committed **$0.039510**. After the separate toy-van, three-ball and exercise SAM calls plus the fresh exercise advice below, six calls commit **$0.166152 total**, with **$4.833848 remaining** and no unsettled reservations. The ledger preserves conservative micro-dollar rounding; it is not a provider invoice.
- `artifacts/meta-fullbody-construction-live-01` contains one additional construction-only request about the public, attributed fresh squat source. It uses 6,275 input and 1,458 completion tokens, takes 7.062 seconds and costs **$0.01404025**, with no retries. Three suggestions distinguish camera/gravity calibration, body-shape uncertainty and foot support; all remain accepted=false/applied=false with independent verification not performed. Manual labels and evaluation results were excluded. The 16-file seal is `6f27ba31909e0eb660c3b67aeb436c68dda100e747ca7b04ceaa63a2734b13f8`.
- The separate [exercise silhouette request](decoded-masks.md) returned all 68 frames in 8.027 seconds for **$0.0136**, with no retries. Its masks and comparison with saved 2D joints are inspectable offline; neither becomes anatomical truth or changes the simulator.
- [Independent construction checks](meta-body-hypotheses.md) confirm the proposed floor-overlap check within the model, identify a wrong vertical-axis name, and contradict blanket low-confidence ankle language. Camera cause, foot support, real loads and muscle activity remain unidentified. All hypotheses retain `accepted: false`.
- This adapter consumes the current rolling/planar/exercise construction formats; it does not implement general-video segmentation or recover a general physical scene. It does not establish hackathon eligibility, physics accuracy, human biomechanics, or winning performance.
- Next verification: test retained hypotheses against new calibrated physical/contact evidence. Current construction-only checks do not establish physical transfer; repeatedly inspected development clips are not final holdouts.

The subsequent two Physics101 ramp-tracking requests each cost $0.01. After the [frozen repeat comparison](material-transfer.md), eight settled project calls commit **$0.186152**, with **$4.813848 of the approved $5 remaining** and no unsettled reservation. The repeat request returns all 39 post-initialization source frames for both objects in 5.395 s, with no retry. Its observations support a prospective image-motion check, not a validated 3D environment; the frozen accuracy gate still fails.

## Official API references

Current endpoint/model/schema behavior was checked against [Meta Chat Completions](https://dev.meta.ai/docs/protocols/chat-completions), [structured output](https://dev.meta.ai/docs/structured-output), and the [model catalog](https://dev.meta.ai/docs/models). These sources describe API capabilities, not scientific validation of model-generated physical claims.

## Local CLI setup

The user authorized a $5 aggregate development budget and has configured a key. For a fresh project, add an existing key using a hidden prompt; do not overwrite the configured credential:

```sh
.venv-lab/bin/python -m weft meta-configure --budget 5 --key
.venv-lab/bin/python -m weft meta-status
```

Prepare without uploading:

```sh
.venv-lab/bin/python -m weft advise --run artifacts/exercise-lab-03/run --out artifacts/advice-new
```

Once a project key is configured, explicit execution uses the same shared ledger:

```sh
.venv-lab/bin/python -m weft advise --run artifacts/exercise-lab-03/run \
  --out artifacts/advice-live-new --execute --meta-budget 0.06
```

Use a new output directory each time. There is no automatic retry, background spend, account creation, or provider-key creation. The local budget is estimated application-side accounting, not a provider billing hard cap.
