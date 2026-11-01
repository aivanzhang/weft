# Equipment support comparison report

## Status and ownership

Implemented and frozen. **21 distinct focused checks have passing evidence.** The exact saved analytical selector result was inspected without rerunning its producer. Root subsequently completed the public saved-run CLI once; the saved-page audit below passed independently without repeating that work.

This lane owns only:

- `src/weft/equipment_support_report.py`
- `tests/test_equipment_support_report.py`
- This handoff and ignored `.local/equipment-support-report-*` check receipts.

Root owns the saved loader/workflow and `equipment-support --run … --fixed-camera --out …`; reconstruction owns the selector/constraint. No Git, shared CLI, source evidence, physics, browser, server or avatar changes were made here. No live jobs remain.

## Interface and input boundary

```python
from weft.equipment_support_report import build_equipment_support_report

files = build_equipment_support_report(support_raw, max_seconds=5)
```

The caller supplies exact saved JSON bytes and authenticates their selected source seal. The result keeps the existing `exercise-support-evidence-v1` profile and adds `equipment-silhouette-window-constraint-v1`. The inspector does not import or rerun either producer. A self-declared hash is not independent source authentication.

Returns safe fixed relative-name bytes:

- `index.html`: comparison, complete counts, assumptions and saved identities.
- `frames-NNN.html`: complete original source-frame coverage, up to 100 frames or 1 MiB projected data per page.
- `window-index.js`: bounded construction frame/identity/page index, not an additional evidence source.
- `support-evidence.json`: exact original bytes, including an optional terminal newline.
- `report.json`: inspector counts, page index and raw/stylesheet hashes.
- `brand.css`: exact existing `src/weft/assets/brand.css` bytes. The caller must archive this asset in its generator closure.

No `result.json` or `manifest.json` is returned; those remain wrapper-owned. There are no remote resources or fetches.

## What is checked and shown

- Reconstruct the complete original baseline by restoring pair statuses/counts and removing only the declared constraint additions. Its canonical hash must match the saved baseline pin.
- Preserve the original first-60%-of-retained-poses construction boundary, all later pose rows, all unsampled decoded frames and all metadata-only tails.
- Require every original equipment × left/right wrist pair in stable order, every construction point row and every original window identity/count/clock span.
- Recompute saved window status counts from every point in the unchanged window. Any unavailable point prevents a kept/excluded decision; otherwise an image exclusion excludes the original candidate. All intersections keep it. Noncandidates and later rows cannot become candidates.
- Check original/active counts, selected source/pose/equipment joins, source-equivalence metadata joins and saved cache/work arithmetic.
- Keep source time, pose decoder time and derivative equipment mapping separate. No clock conversion is manufactured.
- Show the original motion-based pair details, new selector decision, this frame's conditional image qualification and unchanged ground-image candidates. Every window member has a link to its canonical frame detail.

The UI renders one selected pair and at most 20 window links. It creates those nodes once, reuses them, handles same-page frame links and skips unchanged selections. Untrusted strings use escaped JSON and `textContent`, never raw markup.

**Kept is not contact. Excluded is not verified physical noncontact.** Support, ground height, load and grasp remain unknown. Authored pixel ranges and the fixed-camera assumption are not calibrated physical measurements. No body/floor is installed and no simulator, task or policy is admitted.

## Bounds

| Boundary | Limit |
| --- | ---: |
| Complete saved input | 8 MiB, plus one optional terminal LF |
| Original source frames / retained poses | 1,800 / 900 |
| Construction poses / equipment IDs / wrist pairs | 540 / 8 / 16 |
| Complete report / file count | 16 MiB / 32 |
| Individual HTML | 2 MiB |
| Existing stylesheet read | 32 KiB |
| Cooperative preparation deadline | `(0, 5]` seconds |

Limits compose: a maximum-width/maximum-row fixture can exceed the byte cap and must fail rather than discard evidence. The deadline is cooperative, not an external process kill guarantee.

## Verification and preserved failure

Receipt: `.local/equipment-support-report-checks-01.json`, SHA `dd34945c0c400e36c3144284dd6704c8c4cca38e9940670a08c0e02589bde685`. It contains exact commands, JUnit hashes, timings and counters.

- `.local/equipment-support-report-tests-01.xml`: 19 passed; one performance fixture failed because its 10,585,232 bytes exceeded the existing 8 MiB input cap. The product correctly rejected it before rendering. This failure is retained.
- `.local/equipment-support-report-tests-02.xml`: corrected legal large fixture and a separate all-16-pairs fixture passed. No cap was raised and no evidence was removed from an input. Unchanged passing cases were not rerun.
- `.local/equipment-support-report-tests-03.xml`: one affected safe-text/selected-window node passed after strengthening its malicious saved-string fixture. The product bytes did not change.

Coverage includes baseline/source mutations, missing rows/points, wrong pair order, window boundaries/counts, unavailable precedence, later-partition leakage, physical claims, work counts, JSON/input/output/deadline bounds, local links, same-page navigation, exact bytes and unknown values. Native/source-producing imports and network are blocked; only the local Node scripted DOM subprocess is permitted.

The actual preserved analytical input is `.local/equipment-silhouette-constraint-analytic-01/support-evidence.json`: 52,979 bytes, SHA `ce3c8a7055cb8b23df8aff5e719493146cb9a146f30c60047872f03afb46ece9`. It contains 31 source frames, 15 retained poses, 9 construction poses and 18 original/kept candidates. All were inspected and the exact download was retained. No producer call was repeated. This optional local payload is not shipped as a repository fixture; its composition test explicitly skips if it is absent.

### Measured bounded workload

One authored saved-data workload used 1,800 source frames, 900 poses and 10,800 retained pairs (12 pairs/frame). Input: **8,158,918 bytes**. Output: **16,675,115 bytes**, 23 files, maximum HTML 613,190 bytes. Python preparation took **0.254716 seconds**.

In the Node mock DOM, 1,000 selected-detail updates took **7.458 ms**, created **0 new nodes**; 1,000 unchanged selections caused **0 text writes**. A separate smaller legal fixture retains all 16 named pairs. These are analytical saved-data/script measurements, not browser rendering, source admission, native physics or RSS evidence.

## Frozen identities and next step

| Path | SHA-256 |
| --- | --- |
| `src/weft/equipment_support_report.py` | `faab701498fed797aa333e5db66e70d2ff47dadb04c4fbdf31efa6039a8b2e67` |
| `tests/test_equipment_support_report.py` | `6bbb38eb71081b7ea05a89bd0acfa15172e07d17a1838ae113b8517562ebf326` |
| `src/weft/assets/brand.css` (read only) | `53ed61a8f32aab25c9fb6caf01aeb2ee17bbf092c411c2c6429f2c6ab65d8156` |

Root owns publication and the separate all-member/candidate-arithmetic audit. Preserve any failed output in place and use a new output directory. Do not repeat the selector or source study merely to regenerate this report.

## Actual saved CLI page audit

Root reported one completed public CLI run at `artifacts/equipment-support-01`: 1.086575 seconds, observed child peak RSS 155,926,528 bytes, 90 files / 20,626,942 bytes. Its exact terminal manifest SHA is `f4a61220aef16b5b5a1ca3459095278aac7a6efd26ca24e3b9a2659f27c2ec08`. Those execution/whole-package figures are root evidence, not repeated by this lane.

Independent saved-page audit **passed once**:

```sh
python3 -B .local/audit_equipment_support_saved_ui_01.py
```

- Audit helper SHA: `b777136a7b89ba69b524438c5af2106ccfa83dbfb8c71402c4dfb5c2a0b14f37`.
- Receipt `.local/equipment-support-saved-ui-audit-01.json`, SHA `f7ff2de68308b1d367233724e292cafd56af8f58c881091852810f6eacb5ad26`.
- Exact manifest and selected page/JS/report member pins verified; every local link resolves to a sealed member. The raw download link and report pin match `support-evidence.json` in the seal. Its large raw bytes and source/generator members were **not reread**; another lane audits those.
- Cached no-follow reads: seven saved HTML/JS/JSON files totaling **731,446 bytes**, plus the existing test harness source; **8 reads / 755,458 bytes total**. No source media, report/selector/native execution or network. Audit elapsed **0.106096 seconds**; Node wall time **0.094413 seconds**, within 4 MiB / 5-second Node / 20-second outer bounds.
- All **259 original frames / 1,036 frame-pair views** inspected: 129 retained poses, 77 construction poses / 308 construction point-pairs, 52 later poses / 208 later-pair views, and 520 pair views at original frames without a retained pose. All 516 retained pairs preserve exact original/active statuses, reasons, clocks and saved metadata.
- **2,108 window-link visits** checked against the actual saved construction frame/identity/page index, including same-page navigation. Unavailable/later states stayed distinct; no window, coordinate or zero value was invented. Embedded JSON was safely escaped and displayed only as text.
- On the actual first page, 1,000 selected-detail updates took **9.842 ms**, with **0 new nodes**. Across three pages, 3,000 unchanged updates caused **0 text writes**. All scripted page traversal took **30.396 ms**.

This is saved-data/scripted DOM evidence, **not browser QA, source authentication beyond the selected seal, exact physics replay or physical validation**. Contact/support/grasp/load remain unknown. No checker failure or product change occurred in this audit. The renderer and focused test hashes above remain unchanged.
