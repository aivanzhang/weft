# Native-clock reconstruction and preserved portal runtimes

## Verified milestone — November 2, 2026, 1:18 PM PST

The optional saved-evidence path now reconstructs joints and root motion using native
presentation timestamps **before interpolation**. Original video, detector rows,
nominal timestamps, pose identity and quality remain unchanged. This is timing
custody, not recovered camera-capture timing or validated biomechanics.

### Implementation

- `source_clock.py`: bounded complete-frame integer tick table, rational time base,
  signed decoder/native origin, exact source and raw evidence hashes, reversible
  derived view and compact runtime summary. Metadata and decoded counts remain
  distinct. A one-frame discrepancy requires its exact original rounding receipt;
  larger discrepancies require the original matching encoded-frame audit.
- `exercise_body_source.py` / `cli.py`: `exercise-body --evidence-run ...
  --native-clock ... --out ...` preserves raw files and writes separate
  `reconstruction-evidence.json`. Quality/identity gates stay unchanged.
- `observations.py`: schema 4 includes every decoded slot, native/decoder/nominal
  derivation and count audit. Schemas 1–3 keep their existing behavior.
- Root/body/world builders and evidence validators use the same native elapsed
  timeline. Physics retains fixed dt. Person/support packages retain all receipts.
- Shared-exercise gain/residual/feasibility/feedback policies pin timing code and
  retain original evidence, legacy clock, native binding and derived evidence.
  Missing or altered receipts are rejected. Three older diagnostic learners reject
  native input explicitly instead of dropping its custody.
- `portal_lab_runtime.py`: saved portal labs that fail the exact current-runtime
  compatibility check can run in isolated, credential-free workers from fixed
  trusted Git release `5819ba452aa65b492cdf586b53dc827045d0d5ba`. Full exported cache,
  original runtime/policy hashes and dependency checks remain enforced. No saved
  lab code is executed; no pins or scores are rewritten. Requests are serialized,
  size/time bounded and children are closed/reaped. Other incompatible releases
  still fail explicitly; the existing example launcher remains separate.

### Real source and end-to-end evidence

Evidence directory: `.local/native-clock-workflow-01/`.
Source: existing public exercise project `4e0f1fe859664bb5ba788214e129451d`, SHA-256
`ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
One bounded FFmpeg/OpenCV decode took 0.243 s: 235 complete decoded frames,
236 metadata frames, native time base 1/1000, decoder-minus-native +7 ms.
Native elapsed versus nominal cadence differs by at most 0.333334 ms on this clip.
The original 38-member body source remains byte-identical; no pose inference or
provider call was repeated.

- `collection.json`, `native-decode.log`, `native-clock.json`: exact decode command,
  executable/source hashes, complete rows and count/origin evidence.
- `body/`, `workflow/candidate/`: source-qualified construction with 118 reference
  knots and unchanged 1/30 s physics dt. Assembly took 1.82 s.
- `interaction-result.json`: original manual/reference/coast recording
  `session-7e16c1fe041a`, 12 actions plus reset / 14 events, exact replay; all
  407 original run members unchanged. Interaction/export took 1.43 s / 185 MB RSS.
- `lab-complete/`, `complete-lab-verification.json`: package additionally preserves
  all four original/derived clock receipts; source bytes match exactly. This
  packaging-only continuation does not rerun reconstruction or inference.
- `learning-02/`: one generation / population four / two training starts,
  1,872 training control steps, independent changed-start seed 310029442,
  four controllers / 720 executed evaluation steps. Total 7.96 s / 376 MB RSS,
  zero paid cost. Selected original feasible policy; authored and learned each
  pass 1/1 with equal score 0.8932957; coast/random fail. **No learned improvement
  or multi-seed generalization claim.** Random terminates early; failures retained.
- `trained-interaction.json`: ready learned-controller lab, source/inspection
  receipts, nine actions / ten events replay exactly (three policy actions
  independently checked), all 1,362 run members unchanged. 2.19 s.
- Learning workflow seal:
  `320a076b6bf58129f95c125faf461a1dc9e6297eaabc7c4375c7626bb70e0091`.

Try the saved trained lab (user starts the server):

```sh
.venv-lab/bin/python -m weft lab --resume .local/native-clock-workflow-01/learning-02/lab
```

This local artifact is not included in Git and requires the matching runtime.
For a new saved pose package, supply a verified `source-native-clock-v1` table to
`exercise-body`, then use the ordinary `shared-exercise-build --human-run` route.
Automatic clock collection in ordinary portal uploads is **not yet implemented**.

### Existing portal projects preserved

`.local/native-clock-portal-01/result.json` verifies both original exercise and
support child `639d135203db457ead1bb720cead115c` through authenticated loopback HTTP.
Both run under the preserved release, load their policies/inspection, interact,
record and export: 18 actions replay exactly; all 2,650 run members and terminal
status files unchanged. The synthetic object project still opens. 44 requests,
mean 98.7 ms, maximum 1.65 s including startup; complete journey 6.28 s.
Parent peak RSS 214 MB; maximum child RSS 235 MB (not aggregate concurrent RSS).
No external requests/training, no persistent server, all children reaped.

### Verification and preserved failures

488 distinct scoped Python tests pass across these JUnit receipts in `.local/`:
`source-clock-contract-02.xml`, `native-clock-observation-receipt-01.xml`,
`native-clock-shared-regression-01.xml`, `native-clock-body-01.xml`,
`native-clock-integration-01.xml`, `native-clock-final-regression-01.xml`,
`native-clock-policy-closure-06.xml`, `native-clock-policy-closure-07.xml`,
`native-clock-learning-workflow-01.xml`.
Coverage includes bad/irregular clocks, missing slots, signed offsets, raw custody,
policy pins, source wrappers, CLI, preserved-runtime transport and legacy paths.
This is not a passing claim for the entire repository or real-video accuracy.

The unfamiliar source still has 1,107 decoded / 1,113 metadata frames and
0.200635539 s nominal drift. `unfamiliar-clock-check.json` reuses saved native and
decoder tables, verifies their identities, and proves quality remains rejected:
353/554 usable samples (63.7184%), below the unchanged 65% gate. No reinference.

Preserve these harness failures: initial interaction used absent human reference
IDs (fixed by the existing ordered evidence mapping, before any recording); initial
learning callback used two arguments instead of the documented single row
(`learning/` and `learning.log`, failed before training); first complete-lab attempt
used nonexistent `body_seed` rather than `candidate` (no output created).
The sandbox initially denied loopback binding; narrow escalation allowed the same
bounded local test, with no external network use.

Pre-existing reference-input decoder allowlist failure remains: current decoder
SHA `092f328e87eeed03eb40a783c4444d37a37ca313bd4531f53647187bfaba5c78`
is unqualified there. The same failure reproduces under the earlier portal runtime;
its person-region changes are substantive, so no automatic repinning is justified.
Native source refinements/reference-controller trials remain unsupported and fail
closed; their reviewed runtime pins were not weakened.

## Remaining work and next safe action

- [ ] Integrate complete-frame native capture into ordinary video ingestion and
  portal jobs, with exact source/decoder verification and no raw detector rewrite.
- [ ] Qualify reference-input decoder changes independently, then support native
  refinement derivations without bypassing original custody/runtime gates.
- [ ] Verify broader unfamiliar-video perception, ground/equipment geometry and
  contact. Timing alone does not resolve these failures.
- [ ] Connect Muse proposals to bounded typed repairs and independent comparisons;
  retain rejected candidates and leave scoring/final evidence untouched.
- [ ] Extend diversity/multiple-seed evaluation; one development clip is not
  generalized reconstruction, validated physical accuracy or overall completion.

No live jobs at handoff. No paid calls; Meta/OpenAI ledgers unchanged by this
milestone. Hosted site stays offline. Preserve separate avatar-session dirty files.
Publish only owned implementation/tests/docs and record exact remote revision in
`.local/native-clock-workflow-01/publication.json`; recheck remote/worktree on resume.
