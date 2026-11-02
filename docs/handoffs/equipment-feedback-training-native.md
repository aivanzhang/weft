# Equipment-feedback training pilot

## Current state

**Completed and verified; no controller improvement.** Root ran the frozen pilot exactly once on November 1, 2026, finishing at 4:06:37 PM PST. All eight full-length trials completed, but **0/4 candidates passed both training starts**. The unchanged v5 anchor was correctly selected. Root-owned supervisor session `87535` is terminal; PID `88067` was reaped with exit 0. No job remains live. The [policy/driver release](equipment-feedback-learning.md) retains its reviewed 38-check source/test pins.

This was one training-only test of equipment-conditioned joint/root residuals around the exact saved v5 base. It used the original world, not the later automatic-frame construction. No default/public policy route, independent evaluation, physical transfer or unfamiliar-video claim follows.

## Actual task result

Every trial passed all five body gates and all `sam-3` equipment gates. Every trial failed all four `sam-2` stability measurements: accumulated displacement, orientation change, speed and spin. Each candidate therefore violated **8 of its 26 individual gate terms** across the two starts. No engine error, incomplete interval or missing observation explains this result.

| Candidate | Worst normalized gate excess | Mean excess | Original mean reward | Successful starts |
| --- | ---: | ---: | ---: | ---: |
| 0 — unchanged v5 | 6.499293 | 1.214913 | 0.745010 | 0/2 |
| 1 | 13.597569 | 2.973980 | 0.738282 | 0/2 |
| 2 | 9.126146 | 1.728837 | 0.732387 | 0/2 |
| 3 | 17.247676 | 3.718545 | 0.765064 | 0/2 |

Normalized excess is `(measurement − limit) / limit` above the original physical tolerance; zero passes. Candidate 3 earned higher reward while worsening equipment violations, so the unchanged feasibility-first selector correctly rejected it. Nonzero weights or higher reward are not task success.

For the selected anchor, worst `sam-2` values across starts were **0.105749 displacement** (limit 0.03), **0.650340 rad orientation** (0.15), **0.722086 speed** (0.15) and **5.624470 rad/s spin** (0.75). Length/speed use the declared simulator model units, not measured physical calibration. Initial-state hashes matched across all four candidates for each original seed. All **508 anchor actions** were byte-exact v5 outputs.

## Actual verification and resources

- **2,032 returned controls / 32,512 substeps / 8 resets**; 16 model builds, one data allocation, 2,056 forwards and zero copies. Actual counts equal the reviewed caps; no extra episode, baseline, replay or source-proof pass occurred.
- **8.495484 seconds**, **313,229,312 bytes peak RSS**; producer 7.497313 seconds, saved verification 0.598423 seconds. Output: **342 files / 34,960,745 bytes**. External cost: **$0**.
- The worker verified all **338 child members**, **2,032 action/result pairs** and **8 journals**. Actual journal bytes: **10,017,217**, below the 13,360,912-byte projection and 24 MiB cap. The environment was released before the 34,786,466-byte saved map.
- During this one frozen workload, guarded current-state v5-plus-equipment sampling averaged **0.120 ms**, maximum **0.208 ms**, over 2,032 samples. This is a measured training-path timing, not UI latency or a comparative speedup.
- A separate saved interpretation read only **175,778 bytes** of result/process/seal/plan metadata. It recomputed every candidate's 26 terminal gate terms, ranks and stable selection, and checked counts/seed/seal joins. It reused the worker's complete journal/map proof: **no journal, runtime/source payload, prior training cache, holdout, native call or source proof was repeated**.

| Receipt | SHA256 |
| --- | --- |
| Frozen plan | `647c8e0920c1523aecab28e1a663f0d358d4b3ba9f5ac7017bc99e11ec42d553` |
| Outer seal | `df47a705ef6c2ba5045fd81ca70d90c3c5bb8f7ecbf1fa439ba7fc01a322a8a1` |
| Native result | `452b569cd1ef5ed100946bbf823980f5bdfd158aa53136d32866a3bd0bd5b44c` |
| Training seal | `3fa5d1ef8758fbfa240f5489642726718435a7700ea04172292661d70a2277da` |
| Terminal process | `2ded18af0797787316a8e2793199c267e878b4099d07cabd2480c50da96359be` |
| `saved-interpretation-01.json` | `6e50f1e1456e4403625b318e1d2e034ba37d6712bf9f97d14753a6793bb90319` |

## Frozen inputs and preparation

- Scratch: `.local/equipment-feedback-training-native-01/`; completed output: `artifacts/equipment-feedback-training-native-01`.
- Construction: `artifacts/fullspan-exercise-source-01`, seal `f300b46cfb71697ed2ec4cdf4746c96ceda7bf0995a77412f0835d274b6d055f`; 192 declared members, 67,977,626 bytes. Preparation checked named stat identities and small metadata only. Production must still perform its original complete admission, source proof and final unchanged-input reread.
- Base: `artifacts/split-limb-feedback-native-01/selected/policy.json`, SHA `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`.
- Scene/task canonical identities: `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d` / `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da`. Scorer, geometry, source clocks/partitions, reset protocols, gates and legal action authority are unchanged.
- Runtime: all 296 members of `.local/equipment-root-search-native-01/runtime` remain exact, plus only the released actor, policy and driver. New inventory: **299 files / 6,480,605 bytes**, manifest SHA `fe2ecbf773423dbacb71c2ad8213bccd8bedc1a0d768c32a3d8df714c7337e12`. The original 25 policy dependencies and three new source pins are checked before work. The existing `src/weft` layout and three project files support the unchanged implementation snapshot.
- Pure metadata probe: **0.298016 seconds / 109,887,488 bytes peak RSS**, zero native/forbidden attempts. The optional import-time `sysctl` probe was blocked before execution. Receipt: `metadata-receipt-01.json`. It computes resource metadata, not policy or physics acceptance. Harness syntax was AST-parsed only.

## Exact work envelope

One generation × four candidates × two original starts × 254 controls: **at most 8 episodes, 2,032 controls and 32,512 substeps**. The zero-weight unchanged-v5 anchor is one of those four candidates, not another baseline episode. CEM seed `2026110101`; original training seeds `1263521311` and `137031775`, original human jitter 0 / 0.05. Incomplete/engine-failed trials remain ineligible; mean/peak/contact/equipment gates are neither widened nor bypassed. The original full-interval feasibility-first ranking selects among complete candidates.

| Phase | Models | Data | Resets | Forwards | Substeps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Producer | 14 | 1 | 8 | 2,056 | 32,512 |
| Saved arithmetic verification | 2 | 0 | 0 | 0 | 0 |
| Total cap | 16 | 1 | 8 | 2,056 | 32,512 |

The 14 producer model builds are: source world/task admission 2; v5 admission 2; task/engine construction and mass 5; prepared snapshot validator 3; final policy/base validation 2. Four nominal resets use two forwards each; four human-jitter resets use four each. Each control uses one boundary forward. Copies/forecasts are forbidden. Attempted and returned operations are counted separately, including source admission/proof, policy samples, resets, actual controls and saved arithmetic samples.

One **30-second cooperative producer** plus a **≤10-second saved check**, inside **40 seconds CPU / 60 seconds external wall / 512 MiB observed RSS**. Worker reserves five CPU seconds and stops guarded work at 45 elapsed seconds; supervisor reserves five wall seconds for cleanup. These are bounded observed resource checks, not hard memory/disk isolation. External cost: **$0**; one worker, no retry, media, API, server, holdout query or extra episode.

## Data retention and saved check

The frozen metadata calculation reserves **13,360,912 raw journal bytes** across all eight full-horizon episodes: 6,543 bytes per ordinary attempt/result pair plus 8,192 terminal-failure bytes per episode. Maximum projected file: 1,670,114 bytes. Existing caps remain **24 MiB aggregate journals, 32 KiB per line, 8 MiB per journal**. Every action intent, current observation, v5 features/action, equipment features/residual, actual result/metrics and rejected candidate is retained without quantization or shortened coverage.

The pre-training package estimate is 327 files / 24,662,854 bytes, including 291 copied implementation/project members. That estimate uses named source/runtime metadata and the previous exact-source proof-report size; the real producer recomputes and validates its proof. Package + complete projected journals + 8 MiB metadata reserve is below **64 MiB**, and the combined output remains bounded to **128 MiB / 512 files**, with 1 MiB per log.

After producer cleanup, explicit garbage collection and a weak-reference check released the environment before reading the new output once into a **≤64 MiB** map. The saved arithmetic verifier reused the qualified original source/task/base pins, verified the child seal, and reconstructed CEM selection, same-state actor arithmetic, original task-metric accounting and all journals. It performed two cold policy model validations but **no physics replay and no second source-proof/package-admission pass**. One journal was decoded at a time; the map was then released. Actual peak RSS is recorded above.

The outer receipt reuses the verified child seal's member hashes instead of hashing all output payloads again. Failures preserve partial training outputs and explicitly mark incomplete inventory/verification; no retry or promotion is implied. Hot-path sample, step and journal times are measured during the authorized work only.

## Review closure and next hypothesis

Root and independent reviews closed before the run on worker `c8248603e571bcfd54a1284ca262d68eae40bfc4861152854898eede248aef3f` and supervisor `56ca65ddcd6add190c30923c546ef2bb1959c754743013ce35a0330c3e8182d2`; `review-01.json` records independent closure. Both drafts are preserved. Draft 02 changed only its worker pin after the failure-seal fix: hashes of child members are included in the outer seal only after the complete bounded member-hash pass, never merely because a child manifest was readable.

**Proposed, not implemented or authorized to run:** isolate early equipment-conditioned root assistance with a tied signed root-away gain over the entire interval. Set all joint and closing-speed coefficients to zero; tie the three root proximity/away coefficients to `+0.08` for one candidate and `−0.08` for its paired directional control. This reuses the existing authored search scale and actor/features/authority, with no task change. Compare the two original full starts against this saved same-world zero anchor using the unchanged feasibility rank. The present CEM entrypoint does not accept fixed candidate lists; any explicit fixed-weight study path requires separate review rather than silently replacing its sampler.

Why this is distinct: this pilot sampled only three simultaneous 16-coefficient changes; all worsened equipment stability, so it cannot isolate root direction from joint response. A tied signed pair tests one reusable state-dependent direction from the beginning, unlike the earlier failed late two-control root polls. Terminal summaries do not establish when contact began or prove this mechanism will help. Keep it a bounded hypothesis, not a controller promotion, additional CEM sweep or infeasibility claim. Root must scope/freeze any next experiment; this lane performed none.
