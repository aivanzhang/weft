# Portal timing, candidate testing and reconstruction diagnostics

## Verified checkpoint — November 2, 2026 (PST)

Three parallel lanes are integrated: automatic native timing in raw exercise
uploads, source-qualified failure diagnostics, and bounded Muse-linked controller
comparisons. This is not completion of generalized physical reconstruction.
The portal remains local; no deployment, submission or paid API call occurred.

## Try the portal

```sh
.venv-lab/bin/python -m weft portal
```

The user starts the server. Open its printed authenticated link. The newest ready
`native-timing-exercise.webm` project is `22a5644f9ea648d18e117893e470f561`.
Open its sandbox for reference/manual/learned controls and recording. Failed
verification attempts remain separate, inspectable projects; none was relabeled.

On the original exercise project `4e0f1fe859664bb5ba788214e129451d`, the saved Muse
review now has **Test Muse suggestions** results. All three controller candidates
were rejected; the original simulation and controller remain unchanged.

## Automatic source timing

`exercise_vision.inspect_exercise` optionally reports every decoded frame timestamp
to a callback before sampling/inference, without changing original pose artifacts
or nominal detector timestamps. `source_clock_capture.py` then performs one bounded
FFmpeg presentation-time decode. It does not repeat OpenCV or pose inference.
It uses the bundled executable, strips credentials, permits file protocols and
supported containers only, bounds pixels/frames/log/time, and kills/reaps failures.
Native ticks, signed decoder origin, source/raw-evidence hashes and complete counts
are independently revalidated from retained logs and receipts without decoding.

The body source stores `decoder-clock.json`, `native-clock-capture.json` and
`native-clock-decode.log`, in addition to the prior binding/derived view. These
receipts and their body marker survive world construction, policy packaging,
evaluation and lab loading. Immutable observation provenance detects even removal
of the whole capture chain. Arbitrary log files remain rejected.

Normal portal exercise uploads enable `native_timing=True`. The CLI exposes
`exercise-body --video ... --native-timing` and
`shared-exercise-build --video ... --native-timing`. Saved inputs keep their clocks;
the prior `--native-clock` saved-evidence path remains. Selective-crop capture and
non-control-grid-aligned full-reference endpoints remain explicitly unsupported;
no timestamps, scoring interval or quality gates are silently altered.

### Real upload acceptance

`.local/native-ingestion-04/result.json` records the authenticated upload through
actual local inference, construction, training, evaluation and recording:

- Source SHA `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`;
  235 decoded / 236 metadata frames, 118 inference calls and 1/1000 native ticks.
- Full journey **33.65 s**, worker **30.65 s / 558 MB peak RSS**. Native capture
  takes **0.177 s**. Body preparation records two decodes, not one.
- Seventy HTTP requests: mean **23.8 ms**, maximum **1.395 s** including admission.
- Twelve changed-start development trials: authored and retained learned policy
  each pass 3/3 at equal score 0.8932952; coast/random pass 0/3. **No learned
  improvement, independent physical accuracy or arbitrary-video success claim.**
- `session-36d4a370c3da`: nine actions / ten events replay exactly, including three
  verified policy actions. All **1,382 run files including the manifest** and
  terminal lifecycle metadata remain unchanged.
- Run seal `3735e609a2766ce4e2986324b2e4238070cc150435a2f264366ac1bad9be34c9`.
- `current-runtime-admission.json` separately verifies the ready lab under the
  current workspace runtime, not only the isolated verification snapshot.

This acceptance selected the local exercise route. The earlier paid OpenAI scene
planning path is unchanged; no paid analysis was repeated to test timing.

### Preserved failures and the actual integration fix

Do not delete or relabel these outputs:

1. `.local/native-ingestion-01`: harness confused upload request ID with returned
   project ID; stopped before inference. Interrupted project `3c543c88d0c9426792ba8998c021adee`.
2. `.local/native-ingestion-02`: frozen harness omitted `requirements.lock`;
   source/body/world succeeded, training packaging failed. Project
   `78147f94db924a66bab976239796c7ea` remains failed. The corrected snapshot was
   independently preflighted before further execution.
3. `.local/native-ingestion-03`: actual selector bug after successful training.
   The generator excluded observation-proof paths from its displayed source-view
   hash map; the selector re-added them. The fix validates the same capture proof
   and applies exactly the generator's projection, without relaxing settings or
   gates. Project `69215070f1e44a9988aac8ec5df6a615` stays failed.
4. `.local/native-capture-selection-01`: reuses that complete training without
   new inference/training/evaluation/physics; six recorded candidates qualified,
   original policy retained, 0.72 s. Seal
   `83a01b1e3b736467c7434cfdaa569f4ae133c1290cd4b27ca03e6ab6a98056db`.
   `.local/native-ingestion-04/recovered-evaluation` and `recovered-lab-02` finish
   its evaluation/lab path. The first lab harness omitted its explicit task;
   correction reused the evaluation and exact saved task, not new trials.

## Muse-linked bounded controller testing

`portal_repairs.py` maps supported, evidence-linked control/articulation hypotheses
to three fixed **locally authored** joint-feedback corrections. The numbers are
not Muse measurements or recovered biomechanics. It freezes scene/task/candidates,
runtime, two paired starts, original gates and a strict improvement rule before
running at most eight full-interval trials. It retains all action traces and
rejections. Object/static-support adapters and geometry/contact repair remain
unsupported; required physical measurements remain unresolved.

`portal_repair_jobs.py`, worker mode `repairs`, and authenticated
`POST /api/projects/{id}/muse/repairs` provide one deduplicated background attempt.
It shares the single processing queue, receives no API-key environment variables,
uses a 60-second core / 65-second supervisor limit, retains interrupted evidence
and never applies a candidate automatically. Lifecycle state lives outside the
immutable comparison. Requests read only the bounded sealed report, not traces.
The portal shows scores, original task-gate results, rejection reasons and lazy
downloads for the report, frozen plan, runtime source and manifest.

`.local/portal-repairs-http-01/result.json` verifies the actual HTTP/worker route:

- **4.60 s** end to end; worker **3.97 s / 162 MB peak RSS**; 22 requests, mean
  1.57 ms / maximum 4.72 ms. No inference, provider call or default/policy change.
- 1,872 actions. Baseline score 0.9166642; candidates 0.9148807, 0.9096995,
  0.8986147. All passed original gates but scored worse: **all rejected**.
- Duplicate clicks before and after completion do not rerun or overwrite results.
- All 1,326 original run files including its manifest, terminal status and every
  saved Muse advice file remain unchanged. Comparison seal
  `8e225218aeb0775179751bdb6f73368ea2840a757531d4d2dfd00b5bc6026c76`.
- Initial harness expected a nonexistent advice manifest and stopped before any
  server/job. The corrected harness hashes the actual retained advice files.

The earlier core-only study `.local/muse-repair-loop-01` remains separate development
evidence, not additional independent physical validation or a selected repair.

## Why unfamiliar reconstruction fails

`exercise_reconstruction_readiness.py` reuses existing source quality, temporal
identity and selective-crop logic. It retains every decoded slot and alternate
hypothesis, labels native versus assumed clocks, and reports support-point
visibility without inferring ground or equipment contact. Failed exercise jobs
write a full report and at most 8 KiB summary in `analysis/`. The UI shows coverage,
competing poses, longest gap and why retries may not suffice. Diagnostic failure
never replaces the original construction error; no automatic retry runs.

Actual saved unfamiliar-source study `.local/unfamiliar-reconstruction-02`:
1,107 decoded slots; 353/554 usable samples; longest unusable interval
26.118–32.156 s, 91 samples; competing frames 1004 and 1082. Five supported crop
retries could reach at most 358/554 (64.6209%), still below the unchanged 65% gate
(361 samples). No useful inference rerun is justified. Analysis takes 0.717 s /
160 MB and leaves evidence unchanged. Contact/camera geometry remain unknown.

## Compatibility and verification

The portal preserves exact runtime/policy pins against two fixed trusted releases:
`5819ba4` for the original projects and `64f4bef` for the previous native lab.
Policy-only mismatches are covered. Selection requires sealed metadata and exact
code hashes; the isolated child still performs complete original validation.
Unknown or conflicting versions fail explicitly. Actual admissions and unchanged
seals are recorded in `.local/portal-runtime-dual-release-01.json`.

**334 distinct scoped Python cases and nine DOM test groups pass**, with the exact
receipt set in `.local/native-ingestion-04/test-verification.json`. Additional
agent-run selector/CLI regressions are not included in that deduplicated count.
Browser/device visual QA was not performed. No general whole-repository pass is
claimed; the previously documented decoder allowlist/refinement limitation remains.

## Next work and handoff

- [ ] Reconstruct unfamiliar motion, ground, equipment and contact from adequate
  evidence; diagnostic abstention is not reconstructed geometry.
- [ ] Qualify non-grid source intervals and selective-crop native capture without
  retiming or shortening protected tasks.
- [ ] Extend meaningful repair beyond bounded controller sensitivity; qualify
  source-backed geometry/contact candidates with unchanged independent scoring.
- [ ] Demonstrate improved learned control and broader unseen-source/seed coverage;
  current retained authored-equivalent policies do not establish improvement.
- [ ] Qualify the reference decoder/refinement path independently; do not repin
  the unreviewed producer merely to remove its guard.

All verification servers and child jobs ended. Parallel agents released their
files. Separate avatar work remains owned by its session and must not be staged,
discarded or used to claim these milestones. Preserve all failed runs and ledgers.
Publication receipt: `.local/native-ingestion-04/publication.json` after verified
push; recheck remote/worktree on continuation. Full combined goal remains active.
