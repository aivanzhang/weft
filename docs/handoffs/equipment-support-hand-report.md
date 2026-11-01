# Equipment hand-point inspection

## Status and ownership

Implemented, tested and frozen: **20 distinct focused cases passed**. Independent review cleared both the initial release and hand-only cache-bound delta. Root's public source pilot02 completed, and this lane's independent saved-page audit passed. `/root/support_ui` owns only `src/weft/equipment_support_report.py`, new `tests/test_equipment_support_hand_report.py` and this handoff. Root owns saved workflow/CLI; support-learning owns the restrict-only hand consumer; reconstruction owns the named-point producer. No live job or native/source/browser work in this lane.

## Agreed contract

The existing `build_equipment_support_report(raw, max_seconds=5)` accepts exactly one saved constraint family. The optional `equipment_hand_constraint` has profile `equipment-hand-window-constraint-v1`; the new inspector profile is `equipment-hand-support-report-v1`. Original silhouette reports retain their exact existing bytes. Mixed extensions or a hand result relabeled as the old silhouette profile are rejected.

- Keep the original co-motion baseline and wrist pixel/confidence/box intact. New hand evidence does not rename wrist evidence.
- Four named source points per hand: wrist, pinky, index and thumb. Any missing/unqualified point makes the group unavailable. Otherwise any conditional image intersection keeps point-set image consistency; four exclusions give point-set image exclusion.
- Only original co-motion candidates can change. Original windows, construction split, later rows and complete source-frame coverage remain intact.
- Display source/decoder/derivative clocks and exact point identities separately. Authored pixel uncertainty is not measured hand geometry. No filled hand shape, hull, contact, grasp, support, load or physical noncontact is inferred.
- Saved arithmetic validation only; caller authenticates input seals. No raster reads, producer execution or geometry recomputation.

The result preserves the original `exercise-support-evidence-v1` baseline and adds the distinct constraint, not a renamed v1 artifact. Restore the original statuses/counts and remove only declared additions to recover its exact canonical baseline hash. The report validates all group counts, original window ranges and prefix-count decisions. A zero distance at an excluded half-open cell edge remains exclusion, not intersection.

`hand_rows[i]` corresponds to immutable `construction_frames[i]` / original pose row `i`. Its raw source/header fields are not repaired: a mismatched frame/clock can remain visible when every affected consumer point is unavailable. Qualified points must join original clock/equipment/channel identities; the wrist must retain its original pixel/confidence. Supplied invalid point names can remain raw unavailable records; descriptor slots remain explicit. An unselected hand has four exact named null records marked `producer_record_present:false`, not fabricated observations.

The existing package names and caller-owned sealing remain unchanged: `index.html`, `frames-NNN.html`, `window-index.js`, exact `support-evidence.json`, inspector `report.json`, and exact existing `brand.css`. The report adds no producer/native import or runtime module; `fractions.Fraction` is standard-library saved-distance arithmetic. Root must keep the existing stylesheet in its generator closure.

## Limits and display behavior

Input remains 8 MiB plus optional LF, output 16 MiB / 32 files, HTML 2 MiB, 1,800 original frames / 900 retained poses, and the existing five-second cooperative limit. Reuse 100-frame/1-MiB page limits, selected-pair rendering and twenty window links. Four point-detail rows are created once; selection only updates their text. The complete raw point is available through a four-item selector. Every original source slot, later pose, unsampled frame, window link and raw result remains. Limits compose: oversized evidence is rejected, never truncated. The deadline is cooperative, not an external process kill guarantee.

The **hand-only** added row/prefix cache accepts an explicit historical 2 MiB or current 8 MiB declaration, and actual bytes must fit that declared bound. The legacy wrist cache stays fixed at 2 MiB. Root authorized this bounded correction after its first source pilot hit the 2 MiB consumer cache before producing any report. No cache encoding, row, point, profile or field semantics changed; old saved hand metadata is not relabeled. The 8 MiB complete input/output-result limit still applies. This allows up to 6 MiB more encoded hand cache, not unlimited parsed memory; actual source-process RSS remains root-owned evidence.

## Verification and evidence

```sh
PYTHONPATH=src:tests .venv-lab/bin/python -B -P -m pytest tests/test_equipment_support_hand_report.py -q --junitxml=.local/equipment-support-hand-report-tests-01.xml
```

One run: **18 passed, zero failed/skipped**, 0.477 s pytest time. Receipt `.local/equipment-support-hand-report-checks-01.json`, SHA `051ba03b7af522c412797656c02e10eeb612a8fa6f8db227606c76d72dea37bb`; JUnit SHA `3e85027d8d7435541d68bfdc17257326108030ecafe5c80ed989186cebe87228`. Two standard pytest warnings concern `record_property` with xunit2; the measurements are retained in both XML and receipt. No standalone full stdout log was captured; terminal summary remains in the tool transcript.

Cache-bound correction: two new focused cases passed, 0.335 s pytest time; the prior eighteen were reused, not rerun. Receipt `.local/equipment-support-hand-report-cache-checks-02.json`, SHA `7fc659ef14020ac0d1a4a94f69dbc45961f6ce114518c5d56cadf162bcbacd4a`; JUnit `a16924c089fe6064880bb90fbbc2e369346cd0d9daa6a3dfcfe5fd149fbc372a`. A 1,440-point authored fixture uses 2,287,825 B of hand/prefix cache and passes the explicit 8 MiB declaration; the same data with 2 MiB, an unsupported 4 MiB hand declaration and an 8 MiB wrist declaration all reject before any asset read. Input 2,781,346 B, output 5,741,140 B, preparation 0.103210 s. No source/producer/raster work or evidence removal occurred.

Before edits, exact same-source and derivative fixture outputs were captured in `.local/equipment-support-hand-report-goldens-01.json`, SHA `c0c2d6c1b5d6bcc9ea37b589a215804471f310b1fc2d651a4eb00ab56e9f3c26`. Both complete output byte maps passed. Original reporter bytes remain at `.local/equipment-support-report-before-hands-01.py`, SHA `faab701498fed797aa333e5db66e70d2ff47dadb04c4fbdf31efa6039a8b2e67`. No old passing suite was rerun.

New checks cover exact raw downloads/local links, wrist exclusion alongside finger overlap, all-point exclusion, missing points and unselected hands, raw source-clock/name mismatches retained as unavailable, and ten contradictory profile/group/window/identity/wrist/distance/physical-claim cases. Native/source-producing imports and network are blocked; only local Node scripted DOM work is allowed.

Actual preserved analytical composition: `.local/equipment-hand-constraint-analytic-01/support-evidence.json`, 172,965 B, SHA `ffc9250f664f9a8ff8e6abcbda920c5025d469b0ba6a133fc1846300fa0bf727`. It retains 31 source slots, 15 poses, nine construction poses, 18 selected hand slots and 72 points. The reporter consumed these exact bytes and retained the exact download. Neither producer nor consumer was rerun. This proves their saved schema composes with inspection, not real-source or physical admission. The optional local payload is not shipped; its test explicitly skips when absent.

### Bounded scripted workload

An authored saved fixture retained 1,800 source slots, 200 poses and 960 construction point slots. Input: **1,861,879 B**; output: **4,180,116 B / 23 files**; largest HTML **746,609 B**. Preparation took **0.068211 s**. One thousand changed detail selections took **13.097 ms**, with **zero new nodes**; 3,000 unchanged events produced **zero DOM writes**. Four hand-point rows remained fixed. These are Python/Node mock-DOM observations, not browser responsiveness, RSS, source reconstruction or physics evidence. Local/API cost $0; recurring cost $0.

## Frozen paths

| Path | SHA-256 |
| --- | --- |
| `src/weft/equipment_support_report.py` | `38cecdf4004ec5cd56e3a2e21c2b67c5daf833763a2beb7793b056b23a936747` |
| `tests/test_equipment_support_hand_report.py` | `a7cecc9fafa47c903c0d6a724513f4003745a9d73b9c3b6a368223d3c0e11bc3` |

Finger source pixels/confidence cannot be independently rechecked against original pose or mask files here: the reporter receives only the saved selector result. The selected input seal and consumer's original source checks remain caller-owned. Saved pins alone do not authenticate physical identity or prove contact.

## Remaining

- [x] Complete schema alignment, selected hand-point UI, focused checks and actual saved analytical composition.
- [x] Independent review of initial report: clear, without repeating tests or source work.
- [x] Narrow independent review of the explicit hand-cache-bound correction: clear, with no reruns.
- [x] Root's public pilot02 and this lane's saved-page check, without duplicating source/producer work.
- [ ] Root's separate complete-package custody audit and publication.

## Preserved pilot01 failure and unexecuted plan

Parent reported pilot01 failed in the consumer before output: `complete point/prefix cache exceeds bound`; no slots were dropped and no completed report exists to audit. This is distinct from the passing analytical report checks above. No source audit was attempted.

Draft `.local/audit_equipment_hand_support_saved_ui_01.py` (SHA `eca31ec30de90d0338053cb2cfb92eebda1ed74a2fd437ec0d6616c2e9cae44f`) is Python-AST checked only. Plan `.local/equipment-hand-support-saved-ui-audit-plan-01.json`, SHA `6356c005cb3232b51eaba761b9ecdddf481864254d1f4ee8e94c9fdd8cdbc7dc`, remains **prepared, not executed**. It was superseded by the explicitly amended02 plan below; do not run01 against the absent failed output.

## Actual saved pilot02 page audit

Parent completed `artifacts/equipment-hand-support-02` once, with terminal manifest `a607f6df387c59ae4a833718d783d1a5fc9493d8f2b71847a82cdb2cd17e6c38`. Parent reported 254 files / 35,247,312 B, 2.480205 s process time, observed RSS 202,457,088 B, and no model/decode/API calls. These execution/whole-package figures were not remeasured here.

Only fixed input/output paths changed for the approved02 audit. Helper `.local/audit_equipment_hand_support_saved_ui_02.py`, SHA `09b3a838eff334b9c2ef451f7cd8da6d2d503a34bb5170dbf9c007c5ecda296c`; amended plan `.local/equipment-hand-support-saved-ui-audit-plan-02.json`, SHA `3d8619beb03584539f3a36b9dbaad64eba8d810a8609078e8618eca1068e73fb`.

```sh
.venv-lab/bin/python -B -P .local/audit_equipment_hand_support_saved_ui_02.py a607f6df387c59ae4a833718d783d1a5fc9493d8f2b71847a82cdb2cd17e6c38
```

One execution **passed**, without any checker failure. Receipt `.local/equipment-hand-support-saved-ui-audit-02.json`, SHA `f16b467619507dd583c26986e5a34b2522dc722a41155c57336aa9734de5a0e7`.

- Cached no-follow reads: nine saved manifest/report/raw-result/HTML/JS members, **5,389,424 B**, plus the existing 24,012-B scripted harness. **Ten total reads / 5,413,436 B**, each selected file once, under 32 MiB. Total **0.239465 s**; Node wall **0.171750 s**, within ten-second total/five-second Node bounds. No source, generator, media or raster members were read; support-learning owns the separate whole-package audit.
- Every saved page projection equals the exact raw result. All **259 original frames**, **129 retained poses**, **516 retained pairs**, **1,036 frame/pair views**, **308 construction hand groups**, and **1,232 named points** remain. All **2,108 window-link visits** resolve to the correct sealed page, original frame and evidence ID. Later 208 pair views and 520 no-pose pair views stay distinct.
- Both original co-motion candidates are kept; none is created. Active and original status counts remain equal. Construction hand groups: **89 intersection / 109 exclusion / 110 unavailable**. Named points: **274 intersection / 540 exclusion / 418 unavailable**. Exact restored baseline hash and complete saved group/window arithmetic match. These are image-only conditions, not contact evidence.
- All four selected-point records per construction group were exercised against their exact saved JSON. One thousand changed selections took **9.829 ms**, creating **zero nodes**. Across four pages, **4,000 unchanged selections caused zero DOM writes**. Entire scripted traversal took **65.724 ms**. This is Node mock-DOM evidence, not browser responsiveness or physical validation.

Open `artifacts/equipment-hand-support-02/index.html` for the saved inspector. No report, producer, raster, source, physics, browser or network execution was repeated by this audit. Product/test hashes remain frozen as above.

Do not alter prior sealed v1 artifacts or claim physical qualification from this inspector.
