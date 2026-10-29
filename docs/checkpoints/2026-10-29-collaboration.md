# Collaboration checkpoint — October 29, 2026

Verified **4:48 PM PDT** before publication. This checkpoint preserves the accumulated local lab/reconstruction work and prepares two non-overlapping coding sessions for one Weft system. It is not a claim that generalized reconstruction is complete.

## Scope

- Track A: Codex, video understanding and scene reconstruction; autonomous work remains paused.
- Track B: agy, simulation/RL/the unified lab. [Complete goal prompt](../../AGY_GOAL_PROMPT.md), [project handoff](../../AGY_HANDOFF.md), [ownership](../parallel-workstreams.md).
- Both sessions must maintain durable critical-context notes with decisions, commands, evidence, failures, live jobs, Git state and next actions.
- The broader product target remains in `goal.md`. [Revised A goal text](../handoffs/track-a-goal.md) is prepared; the runtime control cannot edit an unfinished objective, so its paused old objective has not been falsely marked complete or replaced.

## Verified checkpoint

A clean file snapshot excluded ignored environments, credentials, caches, `.local/` and generated `artifacts/`; it included the candidate source, tests, attributed fixtures and annotation evidence. It used the existing `.venv-lab` dependencies, not a newly installed environment.

- **2,027 distinct Python checks pass**, zero skipped after resolving test-host permission restrictions. Initial sandbox run: 2,004 pass, 22 setup errors and one failure, all from denied localhost socket binding. The targeted loopback run passed all 29 selected checks, resolving the 23 blocked cases; six previously passing checks overlap and are not double-counted. No application-code change was needed.
- All three scripted UI suites pass: `lab_ui_smoke.mjs`, `mask_viewer_smoke.mjs`, `spatial_replay_smoke.mjs`. These are deterministic mock/runtime checks, not browser/device QA.
- At test completion all 233 candidate source/test/dependency files matched the tested snapshot. A later whitespace check removed one extra final blank line from `src/weft/video_io.py`; its Python syntax tree is identical, and the tested original remains in the snapshot. The other 232 files remain byte-identical. `.local/collaboration-eof-cleanup-01.json` records both hashes; no unchanged passing test was rerun for that formatting-only edit.
- Candidate scan found no matches for the checked private-key, GitHub-token, AWS-key or embedded-credential-URL patterns. This bounded scan is not a comprehensive security certification. Secrets and local budget/auth state remain ignored.
- Local evidence: `.local/collaboration-checkpoint-tests-01.xml`, `.local/collaboration-checkpoint-http-tests-01.xml`, `.local/collaboration-checkpoint-verification-01.json`.

No new provider call, reconstruction experiment, model download or deployment was performed for this checkpoint. The hosted site remains intentionally offline by project instruction.

## Git and stash preparation

`origin` was freshly fetched and verified as private `kintsugi-software/weft`, default branch `main`. The pre-checkpoint local/remote revision was `f009536f4f4e11d976561ae2af4c619e0d1d0bda` with neither side ahead.

One stash contained earlier AGENTS/README/dependency changes. Its functional additions are retained and superseded in the current checkout. The exact stash and index commits are also preserved in a verified local bundle before any cleanup:

- Stash tip: `b7c02a424c125bef95ab5f9636a1dfb6a95819ce`.
- Bundle: `.local/stash-before-collaboration-2026-10-29.bundle`.
- SHA-256: `4f5e863574a118611ed688df0f26caf30dc96ebebb0ab91ecd3a309b45239bcb`.
- Required base: `7c330b790332895ce923b466660930f257ca01c2`, already in repository history.

At the time this note was prepared, commit/push and removal of the reviewed stash were still pending. The session handoff and final publication receipt must record their observed outcome. Local evidence/recordings are preserved, not deleted to achieve clean Git status.
