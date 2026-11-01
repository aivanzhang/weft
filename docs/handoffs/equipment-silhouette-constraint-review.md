# Equipment silhouette constraint review

## Scope and ownership

Independent reviewer: `/root/support_learning`. This note is the only owned
tracked path. Reconstruction owns the constraint and estimator integration;
root owns the saved loader, CLI and any bounded evaluation. No source, raster,
model, simulation or network work is part of this review.

The design in [the implementation handoff](equipment-silhouette-constraint.md)
and final consumer implementation are reviewed. The consumer is clear at
`5e6b73eddeab1b22f3d67a29e647f19ac3a60ee9280eadbf26847c347bc75463`;
the estimator integration remains
`59c14428cefa257d07cd718cf874d0f845d50a2abd89211552d7cdb56b7140fc`.
No new tests have been run and no product files have been edited by this reviewer.

## Design findings

The opt-in constraint may only restrict existing `projected_co_motion_candidate`
windows. Image intersection cannot create a candidate or establish contact,
noncontact, grasp, load or support. Keep the baseline's original windows,
coverage, pair fields and counts available, with separate constraint decisions.

Required implementation checks:

- Bind the complete canonical pose and exact saved pose/equipment bytes. The
  estimator can choose `identity_raw`; source SHA equality alone does not prove
  that the annotation used the same wrist point. Compare selected pixels,
  landmark identity, confidence, representation and evidence/observation IDs.
- Require the exact first `floor(0.6*N)` original sampled frame IDs. Retain all
  excluded/unsampled/tail slots without querying their coordinate payloads.
  A broader or unspecified silhouette selection is not equivalent.
- Preserve the verified original/derivative source identities and both native
  and decoder clocks. An alias is not physical identity or a retiming license.
- Qualify every sample in each original candidate window. Any image exclusion
  rejects that candidate; missing/conflicting evidence abstains. Preserve both
  categories and their original evidence, without promoting other outcomes.
- Reuse bounded prefix counts for overlapping windows. Share the estimator's
  original absolute deadline; preparation gets at most five remaining seconds,
  not a new independent budget. Enforce cache and final output limits without
  dropping rows. The `None` branch must avoid the new import and retain exact
  legacy output.

## Saved restoration boundary

Restoring the already-completed composition cache is preferable to repeating
precursor, annotation or silhouette calculations. Initially accept only the
known completed `pose-equipment-silhouette-composition-v1` package layout.

The loader must pin the selected raw members through the root seal, validate
complete bounded caches, and join pose, canonical observations, equipment,
masks, annotations, channels, selection, denominators and saved clock/proof
metadata. Preserve original saved bytes. JSONL separators are not cache-row
payload; any metadata newline normalization must be explicit and must not
replace the original raw identity.

Root's planned loader reuses the saved silhouette reporter's strict `_metadata`
and `_row` validators without HTML, assets or geometry work. It removes one LF
only for in-memory cache restoration and reconciles `cached_row_bytes`; original
files stay unchanged. Review the additional source/cache joins rather than
duplicating the reporter's rational-value and complete-slot checks.

Use the existing full `load_source_equivalence` once when sources differ; never
construct a proof object from its summary. A restored annotation may omit the
live `precursor_custody` qualifier while retaining the exact prior qualification
metadata and cached rows. It must not pretend to have repeated that work. The
consumer still checks each selected point against the actual estimator input.

This is saved diagnostic custody, not a fresh mask computation or physical
validation. Runtime/source hashes establish identity, not truth.

## Next action

The single public CLI acceptance and this lane's saved-output audit are complete.
Preserve the failed first composition, completed second composition and new
support output unchanged. Root owns publication; the UI lane independently owns
link/script inspection. No acceptance job remains owned or running here. Do not
repeat the selector or decode rasters to complete this checkpoint.

## Initial loader/workflow review

Reviewed root's new `equipment_support_snapshot.py`,
`equipment_support_workflow.py`, CLI additions and reported analytical test
definitions without running them. Two findings were sent to root:

1. Recheck the selector's full remaining ten-second reservation **after** the
   selection progress callback, immediately before the estimator starts.
2. Include the verified source-equivalence members in the final bounded source
   identity recheck, not only the seal and nine direct metadata reads. Immutable
   copied proof bytes remain valid, but they do not prove the live proof files
   stayed unchanged until publication.

Both corrections are reviewed and closed: the reservation now follows progress,
and bounded proof-member identities are compared before/after the real proof
loader and again before publication. Root reports focused callback and proof
mutation regressions passing; this reviewer did not rerun them. No additional
loader/workflow blocker was found. The final consumer point/cache joins are
also reviewed as recorded below.

No adversarial test, source job or native process has been started by this
reviewer. Git was neither inspected nor mutated in this review lane.

## Acceptance helper review

Reviewed `.local/run_equipment_support_acceptance_01.py` before its freeze or
execution. Three narrow guard findings were sent to root:

- Tie every Weft import, including lazy imports, to the frozen manifest and
  archive path. Prepending `sys.path` alone allows editable-install fallback.
- Bound child log capture while it runs; checking `communicate()` output size
  only after collection does not bound that allocation.
- Deny every `socket.*` audit event, including DNS/UDP paths, and low-level
  process launch events as well as `subprocess.Popen`/`os.system`.

These three guard findings are reviewed and closed. The additional closed-pipe
cleanup finding is also closed: read failures retain `capture_error`, stop the
child and reach cleanup; the final wait kills and reaps a child that remains
alive. Timeout, capture failure and overflow remain explicit acceptance failures.
The thirty-second acceptance bound includes measured elapsed time; bounded
cleanup can run afterward. No acceptance was executed by this reviewer.

Reviewed helper SHA:
`86ee39ed562b79b14f54f2ee3890e8d9fb40dc91b12c08072850d4ca529016ae`.
Reviewed preparation helper SHA:
`3ed0a113533229869f8a1e4d6c20b88ef6b30c64a03a91b265795beea9e2a251`.
Preparation verifies release02 and the reporter pin, reads only the exact
composition root seal and selected-member sizes, then archives committed
`791127de2a71939ab7d251a8d703b41d7b93652f` plus six explicit overrides.
New exclusive paths, safe regular archive members, file/byte caps and the final
runtime/plan/helper pins preserve the reviewed boundary. Full source payloads
are read only by the later guarded CLI. No preparation was run by this reviewer.

The workflow now records all **loaded** Weft modules with exact package-origin
checks, not a speculative graph of unexecuted imports. Its scope explicitly
does not claim a standalone optional runtime. The report's exact
`window-index.js` filename is admitted without permitting arbitrary paths.
The final loader predicate also rejects missing, non-string or malformed source
hashes before choosing the proof-loader route. Its two reported targeted checks
are reused; no test was rerun for this read-only delta.

## Frozen consumer review

Reviewed consumer `eba1d20a…`, estimator `59c14428…`, the new analytical test
definitions and release `2169ebf2…`; no tests were rerun. Default `None` remains
lazy and unchanged. The consumer binds the exact prefix, source-proof clocks,
named points and original windows, uses prefix counters, and cannot promote a
baseline noncandidate. Missing points take precedence over exclusions while
retaining both counts; neither outcome keeps the candidate.

One generic-API custody gap was reported and root approved its narrow fix:
derive the canonical pose identity from the prepared raw pose bytes, rather
than trusting separate raw/canonical declarations. A coherently replaced raw
pose plus updated raw pin could otherwise retain the old canonical declaration.
The saved snapshot loader already checks this join, so its path was protected.
The final fix derives the bounded raw pose's canonical identity once under the
original absolute deadline, before selected row queries. Its coherent
excluded-field/raw-pin adversarial regression closes the exact finding.

Final consumer release:
`.local/equipment-silhouette-constraint-release-02.json`, SHA
`1783d9037bc619cbea46adf3299155f7b676e3b8d4628ebfbddb07d79b3ba5d4`.
Reconstruction reports the new regression plus one affected success case pass;
prior passing results are reused. No duplicate test was added or rerun by this
reviewer. No remaining core-consumer blocker was found.

## Single saved-data acceptance and audit

Root's public CLI acceptance completed on October 31, 2026 at 9:54 PM PDT:
`artifacts/equipment-support-01`, root seal
`f4a61220aef16b5b5a1ca3459095278aac7a6efd26ca24e3b9a2659f27c2ec08`.
The parent receipt records 1.086575 seconds, 155,926,528 observed peak RSS bytes
and 90 output files totaling 20,626,942 bytes. This was saved-data selection and
reporting, not another pose, raster, media-decoding or native simulation study.

Independent saved audit:
`.local/equipment-support-saved-audit-01.json`, SHA
`566d670e66194adf4f77cfd0dc6adfad9ad35f6e1866823e28ad6e6ae5e7b75f`.
One output inventory/hash pass read 93 files / 20,665,583 bytes, including the
plan, process receipt and frozen runtime manifest. All 89 sealed output members,
37 selected source copies and 43 generator pins match. Copied encoded proof
media was hashed only; no original source paths, decoder, proof loader or product
module was executed. Runtime files were not reread. The 31 consistency checks
are not independent behavioral trials. No failed checker receipt was produced.

Removing only the documented constraint fields reconstructs the exact saved
baseline canonical hash:
`08fdbdf4bc2e277ef552ce087e675113207fcb65cb86dcc2d4df56d8df865459`.
This verifies lossless baseline retention, not a second estimator execution.

Coverage remains 259 source slots (257 decoded plus two metadata-only slots),
129 pose rows, 77 construction rows and 52 excluded later rows. All 516 retained
pose/equipment pairs remain present; the construction prefix has 308 pairs.
Its saved point decisions are 64 intersections, 146 exclusions and 98 unavailable.
The original complete silhouette package retains all 1,036 source/pair slots.

Exactly two original co-motion candidates changed: `sam-2` / `right_wrist` at
frames 20 and 22. Each original seven-point window contains seven conditional
image exclusions, so both candidates are excluded; zero are kept or promoted.
The other 306 construction noncandidates retain their status. Contact, physical
noncontact, support, grasp and load remain unknown; no body or floor is installed.
This exposed development clip does not establish unfamiliar-video generalization
or physical-contact accuracy. Link/DOM/script checks belong to the independent
UI audit and were not duplicated here.
