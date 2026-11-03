# Weft project guide

Updated November 3, 2026 (PST). [Repository rules](AGENTS.md) ·
[historical studies](docs/HISTORY.md) · [full prior checkpoint][checkpoint].

## Goal

**Video → evidence and assumptions → shared 3D world → interactive RL task.**
One pipeline covers objects, balls and exercises, with optional avatars and
recording/replay. The longer-term goal includes independently tested repairs,
useful learning and physical validation. These are not all completed capabilities.

## Run and use

| Interface | Command from repository root |
| --- | --- |
| Landing page, port 3000 | `npm run dev` |
| Sandbox, port 8765 | `npm run sandbox` |
| Saved demo | `npm run sandbox -- --root .local/portal-unfamiliar-source-acceptance-01/portal` |

The user manages servers. Open the sandbox's complete printed link; keep its token
private. Use `-- --port 3001` / `-- --port 8766` for alternate ports. Never run two
sandbox processes against one project directory. [Python setup](sandbox/README.md)
uses `.venv-lab`; the Node landing page has no dependencies or AI runtime.

1. Upload a video: intake allows 60 seconds, 1,800 frames and 100 MB, with stricter
   checks per reconstruction path. Review assumptions and provider consent.
2. Inspect observations, artifacts and failures; admission does not prove accuracy.
3. Open a supported world, choose manual/reference/learned control where available,
   then **Record & start → Stop & save**. Avatar changes must leave physics unchanged.
4. Inspect/download results. Muse review and controller tests are explicit actions.

State lives in `.local/portal`. Saved examples are absent from fresh clones.
Separate processes/dependencies are not container isolation.

## Implementation and limits

| Area | Implemented | Remaining gap |
| --- | --- | --- |
| Portal | Uploads, bounded jobs, server-key AI, artifacts and sandbox | Reliable unfamiliar-video reconstruction |
| Simulation | Rolling, Pymunk planar and MuJoCo rigid/body/shared worlds | Recovered depth, grasp/contact, materials and balance |
| RL | Explicit tasks, bounded training, baselines and evaluation | Meaningful learned improvement on difficult tasks |
| Meta | SAM tracking, Muse critique and allowlisted controller choice/tests | General autonomous physical repair |
| Avatars | Compatible rigs, state-driven motion, camera framing, recording/replay | Studio skin deformation and arbitrary-model compatibility |

Code: `src/weft/` pipeline/simulation, `src/weft/assets/` viewer, `scripts/` studies,
`tests/` checks, `website/` landing page, `sandbox/` launcher, `hosting/` offline
hosted path, `tools/avatar-assets/` separate authoring package.

Keep observed, inferred, assumed and unknown properties distinct. Root assistance,
illustrative muscles and reference playback are not recovered anatomy, measured
activation, unassisted balance or learned behavior. Exact replay proves software
consistency, not real-world accuracy. Generated views are not new observations.

## Latest recorded evidence

These are November 3 workspace results, not reruns performed during this cleanup.

| Check | Result | Saved evidence under `.local/` |
| --- | --- | --- |
| Continuous avatar camera | All 235 states preserved; robot cropped frames 202→0, built-in panda 192→0; physics unchanged | `demo-camera-20261103/{release,continuous-audit,render-result}.json` |
| Real Muse controller choice | Selected `feedback-1`; unchanged scorer rejected it, score 0.02523263→0.02513716; 936 controls, two starts | `hackathon-controller-choice-20261103-01/{result,comparison,advisor-call,hypotheses}.json` |
| Three unfamiliar videos | **0/3 admitted**; task success not evaluated; 1,050 frames, 46.631 s, 397.7 MB peak, $0 API | `hackathon-unseen-20261103-01/{assessment,seal}.json` |
| Portal storage race | Bounded complete recount; continuous churn refuses rather than omitting files | Runtime checkpoint `5bcd662` |

Primary demo project: `1a5b41db5aa34654a84c83a9dbf5e316` in
`.local/portal-unfamiliar-source-acceptance-01/portal`; recording
`lab/sessions/session-fafeb2ffe374` contains 234 controls / 235 frames (7.8 s).
**Body-position tracking still fails.** The camera-only derivative fixes framing,
not that task result. Equipment omission remains an explicit assumption.

[Source beside robot](.local/demo-camera-20261103/source-and-robot.html) ·
[recording](.local/demo-camera-20261103/recording/avatar-video.mp4) ·
[replay](.local/demo-camera-20261103/recording/avatar-replay.html) ·
[unfamiliar-video report](.local/hackathon-unseen-20261103-01/report.html).
Reload the server/page and reopen attached avatars to pick up viewer changes;
existing recordings remain immutable.

- Muse charge: $0.01911325, rounded to $0.019114 in the ledger; duplicate requests
  made no new reservation. Original source/run/review files remained unchanged.
- New-video admissions failed on pose availability/competing identities; no
  post-outcome retries or relaxed gates. Two clips share a performer.
- Scoped checks: 70 choice-backend, 43 portal/review/worker, 18 unfamiliar-report,
  17 import and 28 portal DOM journeys; camera/retarget checks also passed.
  These are not whole-repository, live-browser or sustained-performance guarantees.
- Older evidence: ramp error improved 61.66→8.51 px but failed its full gate;
  a ball policy passed 8/12 starts versus 12/12 for untrained feedback. Neither
  establishes generalized physical accuracy or a learning advantage.

## Next work

- [ ] Repair generic Studio skin/rig compatibility; verify continuous poses,
  no severe stretching and unchanged physics. Preserve rejected derivatives.
- [ ] Admit a genuinely new supported clip through reconstruction, controls and
  continuous recording without bypassing source identity/geometry checks.
- [ ] Demonstrate a supported improvement under unchanged independent scoring;
  preserve rejected repairs and separate simulated tests from physical evidence.
- [ ] Qualify equipment contact, camera/scale/depth and useful learning against
  matched baselines. Freeze evaluation inputs before tuning and report failures.
- [ ] Record the three-minute demo and prepare judge access/public release after
  authorization. [Script/checklist](hackathon/analysis/demo-plan.md) is current;
  final Apple Notes synchronization remains pending.

Camera framing, the real Muse rejection and the failed three-video study are
complete checkpoints—not completion of the broader product.

## Budgets and custody

- Meta **$5 total**: `.local/meta-budget/ledger.json`. Last observed $0.500710,
  16 settled / zero unknown, November 3, 2026, 10:04 AM PST; recheck before spending.
- OpenAI portal **$5 total**: `.local/portal-openai/ledger.sqlite3`. Avatar authoring
  has separate authorization. Reserve before calls; retain unknown charges.
- Never reset/copy ledgers for more allowance or use ChatGPT/agent-login credentials
  as the product runtime. Keep keys server-side and protected.
- Keep `.local/credentials`, `.local/meshy/ledger.sqlite3`, the ledgers above and
  `tools/avatar-assets/.local/openai/ledger.sqlite3`. Preserve exact evidence paths.
- Repository stays private; hosting stays offline. No submissions, credit claims,
  posts, agreement acceptance or deployment without separate authorization.

## Maintenance and storage

Update this guide in place; do not add overlapping handoffs. Use new directories
for experiments, retain failures and immutable identities, and never change
scorers or success gates to accept a result. Coordinate parallel ownership.

```sh
npm run test:local
node sandbox/run.mjs --check
.venv-lab/bin/python -m pytest tests/<affected_test>.py
git diff --check
```

Cleanup retained 350 old documents through Git ([index](docs/HISTORY.md)); source,
license and evidence records remain separate. Runtime checkpoint `5bcd662` contains
the latest demo work; `db33fa3` retains the untested body-sweep draft.

**Four approved caches deleted November 3, 2026 (PST), about 258 MiB:**
`.pytest_cache`, `.local/swift-cache`, `.local/pip-cache`, and
`tools/avatar-assets/.tools/npm-cache`. No open files were found before removal.
They regenerate. No saved run, credential, budget, environment or evidence was deleted.

Keep `.git`, `.venv`, `.venv-lab`, avatar `.venv`, and installed Blender. `.local`
is Git-ignored, **not backed up by Git**. Archive completed runs with original paths
and verified hashes before removal; never blanket-delete it. The 297 MiB Blender
installer remains a separate cleanup candidate after checking installation/mounts.

[checkpoint]: https://github.com/kintsugi-software/weft/blob/5bcd6625179a5ef36678d2a36214ef72a48b4807/PROJECT.md
