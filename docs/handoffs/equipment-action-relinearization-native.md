# Two-round action decision at saved control 50

## State and ownership — November 1, 2026 (PST)

**Attempt 02 completed and abstained: `no_verified_candidate`.** Root executed
the reviewed plan through the approved narrow escalation on November 1, 2026,
at 12:32:37 PM PST; PID 42861 exited 0 and was reaped. All 49 required prefix
actions and all 23 original first-round predictions match. Two rounds used 45
predictions/four solver sweeps, preserved live state and actual accumulated
metrics, and took **no actual control 50**. The exercise remains incomplete.

The executed `plan-frozen-02.json` is under
`.local/equipment-action-relinearization-native-02/` and uses the unchanged
`.local/equipment-action-relinearization-native-01/runtime-02/`. Earlier drafts
and superseded preparations remain preserved; there is no live job.

The initially released helper `e8342cb098c048ef83f2ee9bef19534fb35ad1bf00b258a7c2dff950f0e4a827`
was copied into `runtime/`, seal
`c66d980f8fdad681c65c178ae1be25a37dcb785cf5a65ecc2e3cb262e01f6104`
(293 files / 6,357,036 bytes), immediately before its owner reported a narrow
float32 inward-bound comparison issue outside the planned `.1` difference.
That snapshot is **superseded and unexecuted**, not acceptance evidence. Keep it
unchanged. Final preparation uses distinct `runtime-02/` after the corrected
release/review. No frozen launch plan exists for the superseded snapshot.

Root then caught stale copied velocity-probe wording in `plan-frozen-01.json`
that incorrectly prohibited the approved selected control 50. Plan
`4f30c6edcecbad1abbc86a86988a07db521602a63667fcd1969c10c2f3d26458`
and its original scripts remain **superseded and unexecuted**. Corrected plan 02
states the actual one-decision scope; `worker-02.py` and `supervisor-02.py` change
only the plan filename. Runtime, helpers, inputs, budgets and acceptance criteria
are identical. No additional copy, test or native work was needed.

### Sandbox launch attempt — no native work

Root launched corrected plan
`9f01e2709fe0ce8961a32bba2886a0afa0fd18555d1da4b9c0cc59314a4d7156`
on November 1, 2026, at 12:30:37 PM PST. The sandbox immediately denied the
supervisor's `/bin/ps` child-memory observer with
`PermissionError: [Errno 1] Operation not permitted: '/bin/ps'`.
The supervisor killed/reaped PID 42422, return code -9, after 0.005197417
seconds. Child CPU was 0.002739 seconds, with zero RSS samples, zero log bytes
and no output directory. Final OS child high-water memory was 1,540,096 bytes;
that is not an RSS sample. Root reported no native work reached, and there is no
worker counter artifact to relabel as a completed trial.

The terminal receipt and empty logs stay unchanged under
`.local/equipment-action-relinearization-native-01/`. Fresh attempt 02 changes
only script/output/plan paths and launch metadata. It reuses the exact reviewed
runtime, helper, saved inputs, limits and acceptance checks; no runtime copy,
test, native step or Git action occurred in preparing it. Both the old artifact
output and the new target were confirmed absent before preparation.

Root requested and received **`require_escalated` for the one bounded supervisor
command**, allowing the existing `/bin/ps -o rss= -p CHILD_PID` observer to inspect
its own child. Sampling and the 512 MiB cap stayed unchanged. This resolved an
observer permission failure, not a source/helper or scientific failure; it is
not standing permission for a broader or repeated experiment.

- `contact_reconstruction` owns the helper and focused mocked checks.
- The separate reviewer owns production-source review.
- This lane owns only the new acceptance directory and this handoff.
- Root owns execution, integration, TODO/combined and Git. No product source or
  previous artifact is changed here; the completed depth handoff stays frozen.

## Terminal outcome and evidence

| Observed work | Actual |
| --- | --- |
| Supervisor elapsed / child CPU | 1.259598459 / 1.051142 seconds |
| Child high-water RSS | 138,346,496 bytes |
| Output | Five files / 282,542 bytes |
| Logs | 3,257 stdout bytes / zero stderr bytes |
| Required prefix | 49 exact saved selected controls |
| First-round saved parity | All 23 prediction and value comparisons complete |
| New decision | One projector, two rounds, 45 predictions, four aggregate solver sweeps |
| Native models / data / forwards / resets | 11 / 2 / 96 / 1 |
| Native substeps / copies | 1,504 / 45 |
| Actual selected control 50 | Not attempted; abstention without fallback |
| Forbidden/media/replay/paid work | Zero |

The 1,504 substeps are exactly 784 required-prefix plus 720 detached-prediction
substeps. No new actual task-gate outcome was produced at control 50. This is a
completed bounded negative result, not a successful controller or proof that no
feasible action exists inside the trust region. Root's receipt and saved audit
establish the outcome; substantive saved diagnosis belongs to the contact agent's
separate handoff. No broader search or native follow-up is implied.

| Terminal evidence | SHA256 |
| --- | --- |
| [Manifest](../../artifacts/equipment-action-relinearization-native-02/manifest.json) | `a8bd7d3558d0ad7a73e1971a6463d18bc9c593371eaf2f7d185177f9daf27b3e` |
| [Result](../../artifacts/equipment-action-relinearization-native-02/result.json) | `1620ebfcadab1864f9dc97fa5392d6f50b9945368e1237ef2f327e51c4e83103` |
| [Projection](../../artifacts/equipment-action-relinearization-native-02/projection.json) | `04ea0163c6de37c719c51fb89b2b8578501610781d3ab3a6522c82739448374c` |
| [Required prefix](../../artifacts/equipment-action-relinearization-native-02/prefix.json) | `909d61c7fd99e3d612b0440651a6bfdf1c35d4187c8f5fbaaaa25e4259282761` |
| [Process receipt](../../.local/equipment-action-relinearization-native-02/process.json) | `f4b1575250e39b6c14f97c016e7ea495177876986668f0564e315dd8c5c8911b` |
| [Root saved audit](../../.local/equipment-action-relinearization-native-02/saved-audit-01.json) | `3f8b9ab9885ca33143c3c8157226155a2d018025ec1026f4704e02aa0859b2c8` |

Root independently verified all four sealed members once, exact prefix/parity,
unchanged actual metrics and the no-50 outcome. This lane read only the small
manifest/process records and selected result metadata to document completion;
it did not duplicate the complete readback, simulation or source tests.

## Question and preserved negative evidence

The saved Rollout02 applied 49 verified selected actions, then found no verified
control 50. Its first-round scalar linear approximation underestimated actual
detached equipment speed/spin. A separate signed-velocity diagnostic did not
improve all three saved candidates and was not adopted. Neither failure proves
that no feasible action exists inside the original trust bounds.

This experiment asked one narrower question: can a fresh response estimate at an
**actually measured improved candidate** find a verified control 50, without
moving the original action bounds, root commands, task or scorer? It is an
opt-in programmed search around the pinned v5 base controller, not a newly
learned policy or a new public/default control mode.

## Executed bounded path

1. Reconstruct the required state by applying exactly the 49 saved Rollout02
   selected actions. Do not run prefix projectors. Every before/after state and
   contact hash, snapshot, transition, root bytes and accumulated metric must
   match. Provisional prefix mean crossings remain as originally recorded.
2. At the exact pre-50 state, obtain the same pinned v5 base action; require its
   dtype and all root bytes to equal the saved decision.
3. Create one projector and call `.project(base_action)` once with:

   ```python
   ProjectionConfig(
       candidate_strategy="two_round_relinearization",
       maximum_predictions=48,
       maximum_seconds=5.0,
       difference=0.1,
       trust_radius=0.25,
       maximum_sweeps=64,
   )
   ```

   The five seconds, 48 predictions and 64 solver sweeps are aggregate lifetime
   limits, not fresh allowances per round. There are at most two rounds; a second
   center reuses an already measured response. Every hypothetical action must
   retain base dtype/root bytes and remain inside the **original-base** joint
   trust box. Recentring does not move that box.
4. Compare already produced, completed first-round predictions (up to the saved
   23) against Rollout02: exact actions/status/endpoint/contact hashes and
   available measured values. Do not compare timing or new round metadata. The
   result explicitly records counts and whether all 23 comparisons completed;
   a budget-limited/unavailable partial result remains partial. This consumes no
   additional prediction or simulation.
5. If an action is selected/unchanged, take at most one actual control 50 and
   require exact full endpoint and boundary-contact equality with its verified
   prediction. Retain actual accumulated scorer metrics and transition even if
   a substep gate fails. A passing endpoint prediction does not certify those
   accumulated gates. If unavailable/no verified candidate, take no control 50,
   fallback or retry.

There is no continuation to control 51, complete episode, new source/media work,
training or baseline qualification. Required prefix execution reconstructs the
same state for the changed decision; it is not a repeat full-task experiment.

## Fixed work, resource and spending limits

| Phase | Controls/predictions | Native substeps | Boundary forwards |
| --- | ---: | ---: | ---: |
| Setup | One reset | 0 | 2 |
| Required saved prefix | 49 controls | 784 | 49 |
| New projector | At most 48 predictions | 768 | 48 |
| Selected actual action | At most 1 control | 16 | 1 |
| Maximum total | 50 actual controls + 48 predictions | **1,568** | **100** |

Total native caps remain **11 model compilations, two data objects, one reset
and 48 copies**. Scratch data is reused, and no per-substep forward is added.
The new helper is expected to need no more than 46 predictions; 48 is the hard
ceiling, not a request to spend unused predictions.

The existing supervisor/descriptor guard remains: 40-second worker CPU and
cooperative deadline, 60-second external deadline including five seconds for
owned process-group kill/reap, 512 MiB sampled/high-water RSS, 4 MiB/eight output
files, 2 MiB/result and two 1 MiB logs. RSS/output observation is not hard memory
or filesystem isolation. Worker network, subprocess, rendering, media decode,
training and replay helpers are forbidden. The supervisor records actual usage
and a terminal receipt; there are no automatic retries.

Projected external and recurring cost: **$0**. Local work is bounded above. The
worker verifies one frozen 293-member runtime and the exact saved recording
scene/journal plus Rollout02 trace; it does not read source video or construct a
new source-equivalence proof. Output is result/projection/prefix/plan JSON plus a
terminal manifest, with failure evidence preserved if a check fails.

## Runtime and input custody

The final reviewed runtime was copied from
`.local/equipment-action-controller-native-02/runtime` (293 members / 6,346,589
bytes), seal
`ab0b266943aef665403ef5e2b13939529363a3d7abd25145366e32e6e0820ad7`.
**Only `src/weft/shared_exercise_action_projection.py` differs.** The detached
predictor and all 25 original policy dependencies remain byte-identical. The
runtime does not absorb independently changing avatar, loader, depth or other
controller code. The reused `native-01/runtime-02/` contains 293 members / 6,357,057 bytes, seal
`7e4b72d6df93de507755d61246015d78ef6ddc7189f5481aa61308b1dcddcc02`.
The fixed named-member copy/verification receipt is `runtime-preparation-02.json`.

Final source release:
`.local/equipment-action-relinearization-01/release-01.json`, SHA256
`25cc5596bc298ac7685c2d08e52f400720b08a5ffe55f658325052c6518fc47d`.
Its **47 distinct scoped pure checks** pass; they are not native qualification
and were not rerun here. The independent reviewer and root closed source review.

- Predictor: `f3f8674988467a2129019c1ee6ef9cf4ad39f2fa677a29e4a5ae8e8f0ddbdfbd`.
- Final projector: `1a5182e83c6a4c8981a5ef30e64c7154878e8ae46fec78060054507098894338`.
- Final projector tests: `4497aa21d33d7d840e9000ddbf99cc034eb4497fd7c9f059dd3c074ece357af6`.
- Original projector: `04510c0a46ca13ce397e0d73ff6a5edcebd325d3df3ccd6092e2643b1d90acab`.
- v5 policy: `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`.
- Scene: `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d`.
- Task: `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da`.
- Recording seal: `94ee100c5093ad51880578972aa123cf01b11f75a791362379ea23a053db91cd`.
- Rollout02 seal: `040698301ae85427b79e059cc34c31e9332a51b92ccdcbc5b95aa05817ea04aa`.
- Rollout02 decisions: `ae66fd6282fb7c1107beecd1b152294660ce70135687061e6cbf4517d0168a69`.
- Pre-50 full state: `ca2d641777574ec914d8d430db7d1bd3db11c7beb168c18bdb124b437c0a99a5`.
- Pre-50 snapshot: `883635f2fb2101c73be47fcc01986f7ad7c61762bb1a5749dfaeb110418f25e7`.

The draft derives from the completed velocity-probe worker
`14636e3b568cb000c95868566ef64eea1fb056c314d2dba20c85ae8c69bfbf8f`
and supervisor
`6a2dbb6fbaa42760a33de64f47ad267e2273d5a0eaaeaf8f2ce96ad681fb11dd`.
Its selected-step parity branch is reused from the reviewed Native02 worker.
Signed velocity capture is unchanged and adds no native query; its previous
pure checks are not rerun. `draft-provenance-01.json` records the adaptation.

## Verification and completed launch

`harness-syntax-01.json` records worker/supervisor AST parsing and
compile-to-memory only; those checks did not execute either script. Root reviewed the
native phase/counter/actual-step/failure-retention changes and requested the
narrow first-round saved parity comparison now included. Root also reviewed
explicit compact-diagnostic abstention: missing round metadata is labelled
partial/unavailable, never inferred to be complete parity or given an actual
step. The unchanged syntax result was reused after freeze. Actual terminal
evidence above records the two-round abstention; no passing changed action at
control 50 was obtained.

| Executed file | SHA256 |
| --- | --- |
| `plan-frozen-02.json` | `6ce2135128459539b5e5225e68610260cb578911ff7e0fd349e02b41b6b7609a` |
| `worker-02.py` | `f04f3cd7c08e751f7b5ed368dc3979e5c591c6d0f89874d582228ab8190e6dfd` |
| `supervisor-02.py` | `c292153fface62531cf34291d983c64169b80f6e42c1b4439ab0349b63d1e70d` |
| Reused descriptor guard | `1a37cde2e6f3508e2b136d3b7a6275239b14bbca0e519c4c38462dca4a454673` |

The original attempt's preparation receipts record source/runtime release,
reused syntax verification and the filename/scope correction. Fresh attempt
02's `preparation-receipt-01.json` records the path-only change, previous failure,
unchanged contracts and observer permission need. Root executed this once with
the approved narrow escalation; **do not rerun it**:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  /Users/ivan/kintsugi/weft/.venv-lab/bin/python -B -P \
  /Users/ivan/kintsugi/weft/.local/equipment-action-relinearization-native-02/supervisor-02.py \
  6ce2135128459539b5e5225e68610260cb578911ff7e0fd349e02b41b6b7609a
```

The output is `artifacts/equipment-action-relinearization-native-02/`.
Keep all output/log/receipt paths unchanged. This acceptance is terminal; no
remaining native/test/readback work is owned by this lane. Root owns the scoped
commit and TODO/combined update. Root reports local/origin `main` at `48aa8ff`
after the independent avatar publication; this lane performed no Git operation.

A completed experiment is not a successful full exercise or independent
physical evidence. Actual task/scorer metrics—not endpoint predictions—remain
the authority for any observed gate failure. Unfamiliar-video reconstruction,
physical contact accuracy and complete-interval success remain open.
