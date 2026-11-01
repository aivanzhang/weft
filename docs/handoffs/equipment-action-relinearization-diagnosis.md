# Saved two-round response diagnosis

Updated **2026-11-01 (PST)**. Diagnosis owner: `/root/contact_reconstruction`; execution/Git owner: `/root`. **The native two-round experiment completed but still abstained before control 50. No action passed.** This diagnosis reads the sealed result/projection once, then uses a bounded derived cache. No native replay, production edit, optimization, parameter sweep or Git operation occurred here.

The [native handoff](equipment-action-relinearization-native.md) owns complete execution, permission-failure and resource receipts. Root verified the completed run: **45 predictions, 49 exact prefix controls, two solver sweeps per round, no actual control 50 and no live/task change**, in **1.259598 seconds / 138,346,496 bytes peak RSS**. The first 23 predictions match the previous saved result. Actual task gates still pass at control 49, but the interval remains incomplete.

## What improved and what failed

Worst normalized violation is `max(0, max(value / original_limit) - 1)`. Zero would indicate no positive violation before the existing numerical tolerance; smaller positive values are not passing actions.

| Measured action | Worst violation | Equipment speed | Angular speed |
| --- | ---: | ---: | ---: |
| Original base | 2.82345147 | .47983709 | 2.86758861 |
| Round-2 center | 1.12338066 | .30479411 | 1.59253549 |
| Best of all 45 measurements | **1.02969863** | **.30445479** | **1.44507254** |
| Required limits | — | .15 model units/s | .75 rad/s |

The best measurement is evaluation 24 / prediction call 25: a round-2 finite difference on joint index 1 by −.1. Its largest original-base change is **.09892332**, inside `.25`. This is an observed result, not a recommendation to hard-code that joint. It modestly improves the center but still fails both velocity gates. The lowest speed anywhere among all 45 measurements remains **.25955757**, from the first round; these samples do not establish a lower bound for untested actions.

Both local solvers converged in two sweeps. The fresh round-2 model again predicts a feasible full candidate that fails when actually simulated:

| Round-2 scale | Linear speed / actual speed | Linear spin / actual spin | Worst limit-normalized model error |
| --- | --- | --- | ---: |
| Full | .14999985 / .25965936 | .46445256 / 1.84031854 | 1.83448797 |
| Half | .22739698 / .29701557 | 1.02849403 / 1.70578127 | .90304966 |
| Quarter | .26609555 / .30265067 | 1.31051476 / 1.58372341 | .36427821 |

More solver sweeps do not address this measured discrepancy. The full candidate's original-base maximum change is **.11014253**, so it does not saturate the `.25` box. Neither observation proves that a feasible action exists or that the box is infeasible.

Only the same `sam-2` speed/angular-speed constraints fail. All measured displacement/orientation, other-object velocity, human tracking and floor constraints pass. Sampled and boundary human-floor penetration values are **zero** for all predictions. Saved endpoint/substep contact-cache **hashes** do not reveal contact pairs or forces, so they cannot establish a changed contact set or explain the response mismatch. No equipment-contact causal claim is supported.

## Probe scale and next justified proposal

The fresh differences are `.1` in one coordinate. Candidate movement from the **round-2 center** is substantially smaller:

| Candidate scale | Maximum joint change | Joint-vector norm | `.1` / maximum change |
| --- | ---: | ---: | ---: |
| Full | .03763497 | .05889945 | 2.657 |
| Half | .01881748 | .02944973 | 5.314 |
| Quarter | .00940874 | .01472486 | 10.628 |

This supports **one fixed-center `.025` finite-difference sensitivity diagnostic**, not another solver implementation yet. If root separately approves/freezes it, measure at most 19 smaller one-sided probes around the exact saved round-2 center, retaining the original-base bounds/root bytes and every original gate. Fit their local response and compare against the **same three already saved candidate actions/responses**. No new candidate optimization or actual control 50 is necessary. Require each candidate's worst normalized prediction error to improve; keep all constraint errors, not only favorable rows.

The `.025` proposal is one declared numerical-scale check, not a search over sizes. Current `.1` samples cannot determine the smaller derivatives. Smaller probes may remain inaccurate or noisy at nonsmooth contacts. Improvement would justify only a separately reviewed controller experiment—not feasibility, promotion, or generalization. Running the whole projector with `.025` would also change round 1 and its chosen center, so it would **not** isolate this question. No new method or execution occurred in this diagnosis. Root has separately assigned fixed-center diagnostic preparation; execution still awaits reviewed frozen bounds.

## Evidence and release

- Native plan: `6ce2135128459539b5e5225e68610260cb578911ff7e0fd349e02b41b6b7609a`.
- Native seal: `a8bd7d3558d0ad7a73e1971a6463d18bc9c593371eaf2f7d185177f9daf27b3e`.
- `result.json`: `1620ebfcadab1864f9dc97fa5392d6f50b9945368e1237ef2f327e51c4e83103`.
- `projection.json`: `04ea0163c6de37c719c51fb89b2b8578501610781d3ab3a6522c82739448374c`.
- `.local/equipment-action-relinearization-native-02/saved-diagnosis-01.json`: `85d18a7dec1890d0e6da11baf364df50cc269ef9704bed0e641f6b45bd14960c` (13,801 bytes).
- Derived `saved-diagnosis-input-01.json`: `7d9ab54612a59e185053ed1c62120973e351ca00044b545b1aac358acf4f2137` (153,942 bytes). Reuse it instead of reading the sealed payloads again; `analyze_saved_round2.py` records the deterministic calculations.

Parent publication may include this diagnosis and the implementation handoff's current-status update. Production source/tests and all previous frozen outputs remain unchanged. No live handle is owned here. Root owns any subsequent bounds, implementation, execution and Git action. External spend remains **$0**.
