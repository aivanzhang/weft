# Public exercise/equipment composition

## Current status — November 1, 2026, PDT

Public composition passes both the authored and real saved-video end-to-end
checks and independent saved inspections. Scoped publication remains in progress. The prior named-hand milestone
is published at `3feeb1dfc129c28cd7d31e07a470b1aaa26ac230`; private remote verified.
Receipt `.local/equipment-hand-publication-01.json`, SHA
`3af2e4b93ee2e97508b61b15f135adb92acf9a17e0d65e00d38f42250870cf09`.
Its 22 owned paths/index were clean and stash empty at publication; 47 unrelated
active avatar entries were preserved. Both earlier source attempts and all saved
audits are terminal. [Completed hand report](../../artifacts/equipment-hand-support-02/index.html).

The new public join connects ordinary `exercise-body`, `tracking-evidence` and
`tracking-masks` outputs to the existing `equipment-support` package. That join
previously lived in a bespoke study script; no new simulator is introduced.

## Ownership and agreed interface

- Reconstruction: new `src/weft/exercise_equipment_evidence.py`, focused tests,
  `docs/handoffs/exercise-equipment-evidence.md`; optional binary retention in
  `segmentation_masks.py`; narrow known-manifest compatibility in
  `pose_precursor_custody.py` and their directly affected tests.
- Root: `cli.py`, new `tests/test_exercise_equipment_cli.py`, this handoff/common
  docs, integration, later frozen acceptance and publication.
- Consumer reviewer: read-only ordinary-parent/failure/source-clock contract
  review; no overlapping writes or source work.
- Avatar work remains a separate session; do not stage, discard or duplicate it.

Callable agreed with producer:

```python
prepare_exercise_equipment_evidence(
    pose_run, equipment_run, masks_run, out,
    human_id=..., equipment_ids=[...], minimum_confidence=...,
    uncertainty_px=..., uncertainty_method=..., uncertainty_note=...,
    source_equivalence_run=None,
    expected_pose_manifest_sha256=None,
    expected_equipment_manifest_sha256=None,
    expected_masks_manifest_sha256=None,
    max_seconds=30., progress=None,
)
```

Successful result: original `pose-equipment-silhouette-composition-v1` profile,
`status=completed`, `sealed=True`, exact `output_path` and `manifest_sha256`.
The CLI prints a shell-quoted, seal-pinned `equipment-support --hand-points`
next command, **only if** the source camera was fixed. It never starts that
command or silently adopts its camera assumption.

`tracking-masks --retain-binary` forwards `retain_binary=True`; absent the flag,
the existing three-positional-argument call and stable mask data/schema stay
exact. Execution times and changed implementation hashes remain truthful.
Retention reuses already-decoded arrays and accounts bytes within the existing
128-MB export/50-million-pixel limits; it does not shrink acquisition coverage.
The composer alone selects at most 512 binary reads/4 MiB. Keep 32-MiB selected
input, 900-pose/1,800-slot, 64-MiB/384-output-file and cooperative stage limits.
Avoid adding dozens of loose generator files to the already tight file count.
Retained construction-mask inventory must not depend on wrist confidence: a
qualified finger can need a mask that neither wrist queried. Count retained
bytes separately from actual wrist geometry reads; never include held-out mask
payloads merely to fill the cache.

An ordinary `exercise-body-source-v1` parent uses `schema_version=1`, unlike the
older hash-only `schema=1` precursor seal. Accept only explicit known versions
and statuses with exact original member bytes/hashes. Do not manufacture an
ancestor manifest. Complete pose evidence can remain inspectable after body
construction fails; preserve the sealed parent failure and never claim body
admission. Original source/proof/clocks, construction prefix, missing/later/tail
slots, authored confidence/ranges and physical unknowns remain unchanged.

## Checks and next action

Root CLI: 23 new and 11 affected checks pass in
`.local/exercise-equipment-cli-tests-01.xml`. They cover explicit settings/source
pins, no acquisition or automatic support route, quoted next commands, failure
versus completed seals and exact default mask dispatch. This is stubbed CLI
verification, not successful producer composition.

Producer has 60 distinct passing analytical cases and closed independent review.
The ordinary 168,516-byte equipment report exposed the initial selected 128-KiB
limit before execution. The reviewed 256-KiB correction has three boundary
passes, retains the 32-MiB aggregate bound, and preserves its earlier failing
test. Core release03 is `.local/exercise-equipment-evidence-release-03.json`, SHA
`de51718e5db12cc6f7f3ba72b3570bbbcc02a91b56abe6d6cfa6f898e34d1658`.
Together with the 34 CLI cases, these are **94 distinct scoped checks**, not a
whole-repository test claim. Actual authored and real public CLI checks are
complete; do not repeat them. Finish disjoint saved audits and the scoped
publication. Provider cost was $0; the existing $5 Meta ledger is untouched.
Hosted site stays offline; no deployment, browser, submission or dev server.
Session `01a18139-c198-70d3-8cfe-e8e79b56fa2a`; complete goal remains active.

## Actual authored public CLI — November 1, 2026, 12:45 AM PDT

Two real public command dispatches, `exercise-equipment-evidence` then
`equipment-support --hand-points --fixed-camera`, pass in **0.657135 seconds**.
Output: `artifacts/exercise-equipment-cli-acceptance-01`, 127 files / 2,968,501
bytes; observed child peak RSS 93,831,168 bytes. All 24 original slots, 23 decoded
slots, 12 pose rows and seven construction rows survive. Seven binary masks are
retained. The failed pose parent stays failed; physical contact stays unknown.

This is an authored analytical fixture, **not a real video**. Its declared
`source.webm` is intentionally absent and its `a`×64 identity is only a test
label. A metadata-only preparation first stopped on that absent member before
creating a runtime/plan or invoking a command. The explicit custody exception
is `.local/exercise-equipment-cli-fixture-custody-01.json`, SHA
`24a5e4b827fc21c0fd9084568694b314a982ea812740bbaf1dbb8e70cd1a92bd`.
All other 57 declared members exist; no media bytes were fabricated or decoded.

Exact receipts:

- Fixture `.local/exercise-equipment-cli-inputs-01/fixture.json`, SHA
  `55e6666d9cacffc91773fa6046fb405534a422e8076c2ccdc94cd308a42ae631`.
- Runtime `.local/exercise-equipment-runtime-01/manifest.json`, SHA
  `c951486dffb7547bd7eea531841747775b8af53c451e6576cb796f8190987231`.
  All 283 parent files match committed `3feeb1d` Git blobs, plus four exact
  reviewed overrides; 284 files / 6,066,043 bytes. No active avatar bytes.
- Plan `.local/exercise-equipment-cli-acceptance-01/plan.json`, SHA
  `7bbd285ca927ba13c0d518f30cb667df4f210a2339237f3c1462e3201859a9d4`;
  execution review `.local/exercise-equipment-cli-execution-review-01.json`, SHA
  `77db817a3a3f1d815f8505a01a42f482842f73162f68b91f527a6ce5a975d89e`.
- Helper `.local/run_exercise_equipment_cli_acceptance_01.py`, SHA
  `be4c34783eadaf1003576e7039c6d3a3b7f5c6ed1f2871d38f83a7e9d25097d8`.
  Exact invocation is retained in the preparation receipt beside the plan.
- Process `.local/exercise-equipment-cli-acceptance-01/process.json`, SHA
  `964539c6f5e19672021ddb71239c2903ea424dc129db912b6fdfc9aa4f341e4b`;
  outer output seal `8302a6dd79e7b67c4713e32169f7f81dff16796d88fc99702f4b31f45deed7d7`.
- Composition seal `5112767062466f17d621f160273093c665c1f0ea8205b763414dd6591ade8b79`.
  Exact command results and stderr progress are saved; no live process remains.

The wrapper enforced a 60-second wall/45-second CPU envelope and bounded
captured output. Aggregate output and 512-MiB RSS were acceptance checks, not
hard live disk/memory isolation. No source/raster decoding, model, physics or
paid API work occurred. Independent saved-only inspection passed: all 127 files,
three seals, source copies, 39 archived composition modules, 48 support generator
pins and eight local links verify. All 24 frames/48 pair views and 56 selected
hand points remain; 1,000 unchanged selections cause no text writes. Audit used
129 cached reads / 2,975,431 bytes / 0.181170 seconds, not a browser or producer
rerun. Receipt `.local/exercise-equipment-cli-saved-audit-01.json`, SHA
`cb4d5699cdc3b70a6d475b7057a08913273c4eb843cbb4bc27206eab8d9e1192`.
This does not establish unfamiliar-video performance or physical accuracy.

## Real saved-video plan — November 1, 2026, 12:54 AM PDT

Prepared, **not executed at this checkpoint**. One original tracking-mask decode
with `--retain-binary`, then public composition and hand support as three
separate bounded CLI stages. A later failure must not repeat a completed mask
decode. Existing 60-second worker and full 257-frame/771-object-slot acquisition
stay unchanged. Expected saved masks: 747 decoded / 24 missing. Pose comparison
keeps 259 original slots, 129 poses and the fixed first 77 construction rows.
Use both original equipment IDs `sam-2`/`sam-3`; confidence 0.5 and two-pixel
ranges remain authored. The supplied whole-video mapping preserves native and
decoder clock origins; no retiming or new physical-contact claim.

- Runtime `.local/exercise-equipment-runtime-02/manifest.json`, SHA
  `e613b8ce23aa5fe4ff019261f3d5d86e6b488024c956e1cf5782a16126a564aa`:
  runtime01 plus only the reviewed 256-KiB equipment-report cap. New core SHA
  `21b32b26a0284ae9b06453474879a45705558cf02c2b797f1dca997013f7eef9`.
- Plan `.local/exercise-equipment-real-01/plan.json`, SHA
  `4abe6e9326a34cbc2e436416fba4cc963ce36153b6b9e608e414383334f70266`;
  helper `.local/run_exercise_equipment_real_01.py`, SHA
  `39dc7cf6592892ef985cfa533bb924361abdd0c02e4f0241072b8444f14bedd3`.
  The preparation receipt stores the exact command; root alone owns execution.
- Metadata discovery `.local/exercise-equipment-real-discovery-01.json`, SHA
  `17d6751013114e6bb8f2691d2143238edab435f91263a7b244d6ca3d148ce11e`.
  Twelve metadata reads / 225,020 bytes; no source array or media processing.
- Ordinary pose seal `e6e88ded526ec582b138dfd90285a0cdde65ddfd33596d255cc23f95afe5529b`;
  equipment seal `acc67c78a8db45d07b855273a255d6e2e77eabd01f3d37c66e1f9e82df15cc40`;
  full proof seal `2c3cf30c05aeddd1d0751ee054993f1c9875bf85beb0f6e7c959f411a3a9e49c`.
  Exact paths/members and installed codec 0.1.0 pins are in the plan.

Bounds: masks 90 seconds / 2,200 files / 160 MiB; composition 45 seconds and
support 30 seconds / 384 files / 64 MiB each. Total 175 seconds / 3,000 files /
320 MiB, including the final manifest. CPU limits are **per process**, not an
aggregate tree budget. Observed 512-MiB RSS and output limits are acceptance
checks, not hard live memory/disk isolation. The exact 6,023,720-byte MP4 fits
the 32-MiB per-file limit. Expected mask size is 33–48 MiB, a forecast rather
than a new measurement; no data is removed to meet it. Cost: $0 provider and
recurring; no new inference, physics, upload, key access or deployment.

## Real public workflow passed — November 1, 2026, 12:56 AM PDT

The reviewed plan completed once with no stage failure. Handle `19379` is
terminal, exit 0. Total **19.166414 seconds**, 2,652 files / 95,193,908 bytes;
observed child peak RSS 203,423,744 bytes. The saved input was decoded only in
the first stage; no new model, physics or paid API call.

| Public stage | Seconds | Files | Bytes |
| --- | ---: | ---: | ---: |
| `tracking-masks --retain-binary` | 15.411572 | 2,054 | 37,126,230 |
| `exercise-equipment-evidence` | 1.369173 | 343 | 22,438,731 |
| `equipment-support --hand-points --fixed-camera` | 2.248697 | 254 | 35,344,677 |

Ready artifacts: [video/mask viewer](../../artifacts/exercise-equipment-real-01/masks/index.html)
and [hand comparison](../../artifacts/exercise-equipment-real-01/support/index.html).
The exact three argument lists, command results, stdout/stderr and separate
process receipts are in `.local/exercise-equipment-real-01/`.

The composition keeps all 259 source slots / 257 decoded frames / 129 pose rows,
77 construction rows and 146 exact binary masks. Original wrist evidence stays
64 intersections / 146 exclusions / 826 unavailable comparisons. All 1,036
pair slots remain. The ordinary legacy pose seal declares no parent success or
failure; the output preserves that unknown status and requests no body admission.
The saved Matroska count audit is used, not rerun. Original source/derivative
clocks and complete proof remain unchanged. Input accounting is 333 unique
members / 17,686,419 bytes / 334 reads, including the 2,633-byte proof-manifest
reread. No excluded data is silently promoted to construction evidence.

Both original motion candidates remain possible under the four-point check;
none is created. This reproduces the earlier study's image result using ordinary
public producers, not a new contact/ground/reconstruction accuracy result.

Exact terminal identities:

- Process `.local/exercise-equipment-real-01/process.json`, SHA
  `d19a5311e6c0bf30a128ef351aa39b65c362b248b6a39e0a3516c7266b871715`.
- Output root seal `87eae41eb9b3e9496010adc03798a92d7b6338896fd3bfa243327206db9e089d`.
- Mask seal `71def14d26bf11078ca32028047ca380f629c2873778ab6c97f6e7134fb8072d`.
- Composition seal `067cd53bd009c7f1e53ac9957d2fa198908ce4df1bcbd2f648deb3779e9efaba`.
- Support seal `29bd5d2123acd5d9c5954194d453d44c753da2d2a17727e85c5584893277019b`.
- Final execution review `.local/exercise-equipment-real-execution-review-01.json`,
  SHA `5fd5f242e65accdaf7f247a49b980c79642718fd28327fbea4ee79bc63cbc40a`.

Root's saved custody audit verifies all outer-to-stage member joins, 188 exact
support source copies, 47 frozen generator pins and eight completed hand files:
249 cached reads / 30,509,752 bytes / 0.038666 seconds. Receipt
`.local/exercise-equipment-real-custody-audit-01.json`, SHA
`80765c2ec9f6905696b094f8d99f3dcbd159176deac8aed73e4b9f75f506e2ca`.

The disjoint report audit checks all 259 frames / 1,036 pair slots, 516 retained
pose pairs, 308 selected hands / 1,232 points and 2,108 window links. All nine
local targets resolve. Four thousand unchanged selections cause zero text
writes; 1,000 changed selections take 9.412 ms with no new nodes in a scripted
DOM, not a browser. Receipt `.local/exercise-equipment-real-support-ui-audit-01.json`,
SHA `9ca4551be15347288a6aa574f9189ab5e2485d9dfe9575601fe000348c2be5ab`;
12 reads / 5,420,211 bytes / 0.312134 seconds. Source/generator/hand directories
were not reread by that audit.

The independent mask/composition audit verifies all 2,397 files / 59,564,961
bytes in 0.216121 seconds: 747 binary masks / 24 missing mask slots, all original
pose/pair schedules, 146 retained construction binaries / 253,054 bytes, exact
parent/proof/clocks and all 40 archived source members. Its final checker used
the wrong leading-slash filter for the package-relative compositor filename;
receipt01 remains failed, SHA
`2d1c0458762bb6f3bda8c5d5f7907efe5abbfe761ac0a4b0a70d603fd37cec0e`.
The narrow amendment verifies the exact generator key using only the saved
manifest and result: two metadata reads / 39,188 bytes / 0.000485 seconds. No
artifact tree, source, archive or processing was repeated. Final receipt
`.local/exercise-equipment-real-saved-audit-02.json`, SHA
`39a95ceaa812f198f419d1ca04350dc43fad066c4d94dea628b895bc78c59b39`.

No failed real acceptance needs recovery. Preserve both runtime snapshots,
the authored missing-media preparation exception, red report-cap regression,
all earlier studies and separate avatar work. The complete goal stays active:
unfamiliar-video reconstruction and independent ground/equipment contact
accuracy remain unfinished; these reports do not change the simulator.

## Publication boundary

Publish only the twelve reviewed workflow code/test/documentation paths on
`main`; preserve the separately owned avatar renderer/test edits. The actual
commit and exact private remote result belong in
`.local/exercise-equipment-publication-01.json`; absence means unfinished.
The latest fetched base is `db72891e7b1c11d9f1934dc3dd5bec24634bf502`.
The index and stash were empty before this scoped checkpoint. No deployment,
branch creation, reset, cleanup or scientific rerun is part of publication.
