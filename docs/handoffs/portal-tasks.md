# Portal task creation and learning

## Current milestone — November 2, 2026 (PST)

Three parallel lanes implement the pure request/report contract, queued task worker and a new licensed exercise source; the integrator owns portal routing and publication. Frontend ownership follows the contract lane. All use the existing general pipeline. No source/scorer/force gate is relaxed, no policy is installed and no paid request is authorized by these local task actions.

Files: `portal_task_contract.py` validates closed requests and compact reports; `portal_task_jobs.py` owns bounded child publication and source-bound worker execution; `portal_task_options.py` computes options once in the existing background loader. `portal.py`, `portal_worker.py`, `assets/portal.js` and `assets/portal.html` expose the ordinary portal workflow. [User guide](../portal-tasks.md).

HTTP routes require normal owner authentication:

- `GET /api/projects/{id}/task-options`: lazy background admission, then compact cached options.
- `POST /api/projects/{id}/tasks/create`: explicitly authored task child, no training.
- `POST /api/projects/{id}/tasks/train`: separate training/evaluation report child, no installation.

Root status `complete` is terminal but not playable. Created task status `ready` enables its manual sandbox. The supervisor must leave task jobs `queued` until the worker's exclusive start receipt; changing them to `running` beforehand prevents their once-only admission. Requests bind exact parent snapshots and task identities; the worker rechecks before publishing. Cancellation before final publication preserves failures, while later cancellation cannot relabel an already committed result.

## New exercise source

[Basic single leg squat.webm](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm), RickyBennison, CC BY-SA 4.0. Original 349,255 bytes, SHA-256 `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`. Container-only MP4 derivative 352,764 bytes, SHA-256 `19503a197ac4b9db8ec084e705185eba8d7afa11c07a5a12a9c5b67abb59401c`. All 235 decoded frames and exact rational native timestamps/durations are equal; no trimming, pixel conversion or resampling. Preserve original, attribution, license and equivalence receipt in `.local/portal-unfamiliar-source-01/`.

Actual ordinary upload → unknown-inventory simulation request → local preflight reaches `awaiting_consent` in **6.978 seconds**, worker 5.608 seconds /408,698,880 B sampled peak RSS. Project `2fb53469b0d4452782a8e2db088d908a` lives under `.local/portal-unfamiliar-source-acceptance-01/portal`. Result and sealed evidence are in its enclosing acceptance directory. Original source package, installed models, 29 stable core modules and both provider ledgers remain unchanged. Concurrently edited portal orchestration is recorded separately, not falsely included in the stable-core claim.

The ordinary body check succeeds with 79 stride-three samples/references: 2 fully tracked and 77 partially tracked; right shoulder/elbow are usable in only 2/79 samples. Missing joints remain unknown. No detailed-body child is created and no inference is repeated. This qualifies partial-body preparation only: no complete-joint accuracy, equipment world, contact, RL success or physical accuracy. No provider consent or API request occurred. Temporary server/queue closed and workers reaped. Acceptance seal `7aa77ae31884f111330877e8d57cd171b4bc3753d5c80b684291536d4d8228f3` covers 139 members /33,211,866 bytes; see `source-body-facts.json`.

## Verification and next action

**105 distinct Python checks and 25 portal DOM groups pass.** Contract/wrapper/routing: 51; worker: 17, including real analytical task/search/evaluation with source admission explicitly fixture-stubbed; root integration and existing inventory/body-detail regressions: 37. Receipts are in `.local/portal-task-contract-01/`, `.local/portal-task-jobs-01/` and `.local/portal-task-checkpoint-01/`. Independent reviews cover contract, worker, shared routing and UI. No browser visual QA ran.

Actual authenticated import → cached task options → queued task creation → queued training/evaluation → all allowed artifact downloads passes in **45.760 seconds**. Evidence: `.local/portal-task-source-01/result.json`. The original exploration manifest is `b9f448098267c56e7b727c18621e47902da828ab0c4f7a45f3296d6c7306781c`; imported project `a47a63a7c2e944d4a2ebffcd50fc3955`, task `cbc6e2600f8e4058abab4a220de38df6`, report `1923f7c585fa43efa899d12233a715f0` under `.local/portal-task-source-01/portal`. Task SHA `09e8a91d1024b39cf5084ec69be012bd0c60f955aa3dd2e4e52878357a7bae6f` binds an explicitly authored floor target `[1.0015390996363758, 0.30727821541026723, 0.11373623067921489]`, position tolerance .03, speed .1, final hold .5 seconds. It does not replace the original failed hover task.

Training uses seed 1729, one generation/four candidates, 1,502/3,656 reserved controls. Protocol seeds 4101/4102, position variation .025 and velocity variation .1 are frozen before training; evaluation uses 2,772/3,656 controls, 2/2 distinct admitted starts, zero exclusions. Every controller remains **0/2**; selected policy is `authored_unchanged`. Nominal authored/frozen controls complete 15.233 seconds but fail peak angular error (1.063 rad) and terminal placement hold (zero). No policy is installed; changing the target did not establish successful control or solve contact.

Create/train POST responses take 8.7/6.3 ms; cached ready options 0.5 ms. During background source admission, options polling reaches 458 ms, so this is not a claim of uniformly instant or browser-verified responsiveness. Creation worker: 9.306 seconds /971,079,680 B peak; learning worker: 15.769 seconds /860,962,816 B; parent peak 1,148,190,720 B. Portal retains 393,441,195 B. Original/imported/task parents, all captured Python runtime files and provider ledgers remain unchanged. No API or inference calls; temporary server/queue closed and workers reaped.

The created task's training-settings endpoint is separately verified without another training run: `.local/portal-task-options-source-01/result.json`, 10.767 seconds /905,297,920 B peak, exact task/horizon 457, no changed or added portal files, no queue/physics/inference/API, server closed. This checks the UI's actual task-loader gate omitted from the first harness.

Remaining: independently qualify unfamiliar-video geometry/contact and meaningful task learning. Do not repeat unchanged failed training or claim this milestone completes the general product. Preserve both target/scorer histories and every comparison denominator. Publication receipt: `.local/portal-task-checkpoint-01/publication.json` after remote verification.

Git: work on `main`; preserve the separate avatar session's changes in `src/weft/assets/avatar_replay.js` and `tests/avatar_replay_smoke.mjs`. No deployment, persistent server, provider-spend expansion or stash deletion. Relevant verified work may be committed/pushed under existing project authorization. The complete goal remains active.
