# Guided equipment frame selection

## Current state — November 1, 2026, 3:41 PM PST

Owner: root in thread `01a18139-c198-70d3-8cfe-e8e79b56fa2a`. This extends the
existing offline setup and report, not a separate reconstruction pipeline.
Core inference/producer/constructor work belongs to `reconstruction_next_step`;
root owns the two configuration modules, this handoff and the focused workflow
tests. Setup/report checks make no native call. Separate root-owned native build
acceptance is recorded below; no source acquisition, paid request or server ran.

The new per-object choice is explicit and mutually exclusive with `source_frame`:

```json
{"object_id":"sam-2","frame_selection":{"method":"construction_box_consensus_v1"}}
```

The saved request/config retain that unresolved choice. Setup calls the shared
bounded construction-only selector once, saves all receipts in
`frame-selection.json`, and pins them in metadata/the terminal manifest. Manual
choices still use the original shape and do not call the selector. No default
choice changes. A selector abstention produces an inspectable `needs_input`
package without a configuration or fallback frame.

The offline report joins receipt/source/body hashes, original clocks, complete
slot metadata, original body references and selection eligibility. It does not
repeat inference or authenticate source payloads. A ready automatic request must
have a selected receipt for every automatic object. The complete receipt is
downloadable, not embedded in the HTML. The builder independently checks its own
selection; this preview is not world admission, recovered depth or contact.

The editor offers **Choose a frame myself** (initial mode) and **Choose from
consistent observations**. The latter disables the manual frame field and names
its uncertainty. Mixed manual/automatic choices and saved requests round-trip
without invented frame numbers. Eligibility counts are computed once; no network,
timers, repeated full scans or generated nodes accompany unchanged events.

## Verified evidence

**86 distinct focused cases pass**. The initial combined run had 74 passes
and one incorrect test expectation about the source archive's member prefix. The
archive correctly stored `equipment_box_consensus.py` without a directory. Only
that assertion changed; its targeted rerun passed. Product code did not change
between runs, and the initial failure remains saved.

Independent review subsequently closed three consistency fixes: saved exclusion
reasons must agree with known source metadata; a selected receipt cannot violate
its own saved policy statistics; original decoded counts must match the equipment
view. The report still cannot verify in-image box bounds without coordinates.
Derivative container metadata cannot extend original-frame receipts. Eleven new
cases cover these boundaries. `checks-03` retained a fixture failure: the existing
upstream decoder schema correctly rejected five metadata frames for three time
stamps. The corrected test preserves that rejection and separately tests the
report-only metadata distinction. No source validator was relaxed.

- Runner: `.local/equipment-frame-config-01/run_checks.py`; 30 CPU/40 wall seconds,
  512 MiB reported high-water threshold, 8 MiB per-file limit; native imports and
  network denied. Only existing five-second mocked-DOM Node children allowed.
- Initial: `checks-01.json`, SHA
  `9c6980dcafb21a5e343fccab4093e6020654897445d381cb463eec0a326f0b75`;
  3.644262 seconds / 200,097,792 B, seven Node children, no denied operations.
- Targeted correction: `checks-02.json`, SHA
  `817b0830e1801105511b38b98c1e4079fe0d239f3f292d7729162f803a529692`;
  0.231295 seconds / 62,930,944 B, no Node child.
- Review checks: `checks-03.json` SHA
  `cee1c98fb6706054110d20cd3516dbe8bf587072222cdb6e228cfb6f2d4e9ffb`
  (12 passes / one fixture failure); targeted `checks-04.json` SHA
  `aefad5db9871f7f659e5bf1996b0fdaf72ca1cbf8109202a2f1dc5a01ef83132`
  passed that corrected case; final `checks-05.json` SHA
  `2af7f7fc2f72ee2ea5b087a16283d7674c52999e9b96c1a4edb881688169188b`
  passed the new scalar join plus the derivative path against the final core.
  No unchanged UI check was rerun.
- At 1,800 frames, report preparation took 0.071679 seconds for 2,082,802 input
  bytes / 2,395,269 output bytes. Ten thousand unchanged frame-mode events took
  7.4325 ms with zero writes/new nodes. Existing changed-frame navigation took
  5.022625 ms for 1,000 updates. These are bounded scripted DOM measurements,
  **not browser, device or sustained-use performance evidence**.

Source pins at these checks:

| File | SHA-256 |
| --- | --- |
| `src/weft/shared_exercise_config.py` | `4970b36c63e1b3a54dfad1d3c4ea6537280255b92afbf95981b077d987019067` |
| `src/weft/shared_exercise_config_report.py` | `7cdfb71a6f337f2e5770f13d1590991ccca4eae03a4bff40ec4a4e3853b06bde` |
| `tests/test_shared_exercise_config_consensus.py` | `7086a39e5e376430a5394b3c0c89a042b257934062829e17a299563088df3dfc` |

Core selector/config pins are recorded in each process receipt. The shared
selector has separate core verification; do not count that as independent
physical evidence or rerun unchanged passing checks.

## Actual saved exercise qualification

Independent source review is closed. One ordinary public setup resume completed
on the retained exercise inputs, changing only sam-2/sam-3 from manual frame8 to
the automatic method. It selected **sam-2 frame 28 and sam-3 frame 18** from all 118
eligible observations each, spanning 8.266 source seconds. Both complete259-slot
receipts remain. Camera/depth/body/target choices and the original snapshot are
unchanged. No policy threshold was tuned to this clip.

- [Actual report](../../artifacts/equipment-frame-config-real-01/inspection/index.html)
  and output `artifacts/equipment-frame-config-real-01`, seal
  `12e60e6d10f44cac4e7b7a350288d56013b0599d93931b173d08ac3173cdde3e`.
- Runtime **1.126666 seconds /143,327,232 B**. One numerical selector batch;
  zero model/native/media/API calls. Ordinary saved-source custody copied/read
  the required sources; this is not a new video decode or physical observation.
- Root audit `.local/equipment-frame-config-real-01/saved-audit-01.json`, SHA
  `a5f658b1a05770361bb17d501d37cc09415201b72485dcc51c64d844f22e6a14`:
  all 190 sealed members,57,988,108 B including manifest,176 unchanged source
  copies and 25 local HTML links verified without rerunning inference.
- Frozen plan02 SHA
  `c1238dba0753230c0d30a0f1aa1b4ee6e2f86176cac614382f72ecff65b1cd4a`;
  `.venv-lab/bin/python -B .local/equipment-frame-config-real-01/run-02.py PLAN_SHA`.
  Do not rerun into the completed output. Plan01/worker remain unexecuted;
  preflight corrected their callback signature before any source run.
- The separate metadata-only selector plan is superseded/unexecuted to avoid
  duplicate inference. Root subsequently executed the native build twice with
  distinct corrected inputs/code: attempt01 rejected duplicate receipt metadata;
  attempt02 admitted the proposal and full-reference candidate, then stopped at
  the final lab's aggregate file reservation. Neither is a ready integrated lab.

## Native build continuation

Attempt01 remains at `artifacts/equipment-frame-consensus-native-01`, seal
`376c5f2732ad3692711c9cbac2b3a03e15e908b86527c416b8eece51fec7362a`.
Its500KB scene-size rejection is a packaging limit, not physical infeasibility.
The tested compact reference points to the complete immutable receipt in the
source candidate package; a lab retains that package under `run/source-run/`.
Neither scene nor proposal limit increased, and all observed slots remain.

Attempt02 is preserved at `artifacts/equipment-frame-consensus-native-02`.
The proposal and full254-control candidate passed normal admission, with exactly
the same full proposal/receipts intended for comparison. Final lab creation
stopped before lab native work:131 existing files +1536 generic lab reservation
+8 terminal reserve exceed the1600-file assembly maximum. The byte allowance fits.
Root observed4.763526seconds/284,835,840B and a reaped child; no native job remains.
The reviewed fix passes a smaller enforceable lab allowance, without removing
evidence or increasing either parent limit. Saved audit verifies all138output files, the132-member failed assembly and
70-member ready candidate: `.local/equipment-frame-consensus-native-02/saved-audit-01.json`,
SHA`cd988daae291d0391d178479edca9e516803f0e0f9b9d12c6b5ccb14f15f24b1`.
The actual compact scene is479,848B, distinct from the earlier mocked analytical
476,719B result. Complete proposal/receipts and protected body/task fields match.
Attempt03 subsequently completed; the earlier failed artifacts remain unchanged.
Pose feasibility and this setup report are not physical validation.

## Ready native build and offline report

The actual saved-input public build passed in **5.793494 seconds /381,583,360 B**:
`artifacts/equipment-frame-consensus-native-03/build/lab`. It has 121 kinematic
reference poses and a 254-control task, with manual/authored control and no
learned policy. Exactly 32 migration controls ran; no training or extra rollout.
The build seal is `3623cb0004e4c1dd19a8a7fdaa2f58e6cac23e563cfa78480c296f98d267f97c`;
lab run seal `1d4a32d609cfbb9647827116ddcb2a44f13067bbb307387f7aef18783481c900`.

Root audit `.local/equipment-frame-consensus-native-03/saved-audit-01.json`, SHA
`de9b8a172f8734e1590dcaccf616cc47f3eb468fa8442b2ecf0314dd76999c2a`, verifies all
543 output files, 400 lab members, exact prior proposal/receipt/scene bytes,
unchanged original gates, source clocks and all ten archived runtime overlays.
The lab uses 404 files /54,766,231 B within its 1,461-file /192 MiB allowance;
the parent remains within 1,600 files /384 MiB. Source records resolve through
`run/source-run/frame-selection.json`; they are not duplicated at lab root.

`exercise_build_report.py` now links complete selection records. Its first actual
inspection stopped before report output: full inline receipts made the candidate
display summary 316,480 B, over its unchanged 128 KiB bound. Native03 stays ready.
The fix replaces only the display's inline receipt with a selected-frame summary,
canonical receipt hash and explicit source-proposal location. Full proposal and
receipt downloads remain byte-identical; all other equipment fields remain.
No source validation or inference is performed by this presentation step.

- **15 distinct focused cases pass** (13 new, two affected). Checks02 receipt
  `.local/equipment-frame-report-01/checks-02.json`, SHA
  `357fdbddcb6e8293f38897db5dd1a1994af5be15b18539f8d8c230700b669687`.
  Checks03 reran only the two cases affected by naming `source_artifact` explicitly;
  SHA `698e946f6f168d651fe9b103a4b9e4956ec0a5a885995e0b30975b4ba905437e`.
  Native calls are zero; an import-time OS probe was blocked before execution.
- Source pin `13d62867297dddca42e09d1c8e3892d67b6382e5b7e2e358e3d22a4a17c666fd`;
  test pin `ca6cb7ae7623e951640be552f6b0a7d5f8883bc9d00529dae89a92fe93032983`.
- Failed inspector01 is preserved under `artifacts/equipment-frame-inspect-01`
  and matching `.local/`: 0.308849 seconds, 112,295,936 B, five files /751 B,
  no report, native/model/proof/media calls. PID82366 is reaped. Its old supervisor
  description says build02; actual arguments and pinned source identify build03.
- Corrected [inspection02](../../artifacts/equipment-frame-inspect-02/inspection/index.html)
  completed in 0.368607 seconds /133,693,440 B with unchanged 45-second,
  512 MiB observed-memory and 132 MiB aggregate output bounds. No native rebuild.
  All 35 files /12,839,451 B, 25 copied artifacts and 45 local HTML links verified;
  actual display candidate is 6,417 B, with complete receipt downloads unchanged.
  Audit `.local/equipment-frame-inspect-02/saved-audit-01.json`, SHA
  `feb6ba966807e8b707f7994ce53c4f952211a5a07a333964c1b5ee9a5661bb46`.
  Report seal `328fd70d0d0ba76d49bda0b18be558e29bef588ef30068d672366d063bfd49d5`;
  outer seal `969d59e48f1f5b350ad02d90f927730ecb711419c28894c61328ad45a56966c0`.
  Frozen plan `ca270d76ea4dcc63ea38b0d21fe906a6d6399cbdca4e90f0fd4eb11b3de3b17f`;
  PID84232 is reaped. Browser/device interaction was not exercised.

## Remaining work and next safe action

Source review, native acceptance, offline inspection and saved-artifact audits
are complete. The exact-path checkpoint publication receipt is
`.local/equipment-frame-build-publication-01.json`. Next, complete the separate
training-driver review and bounded pilot preparation; no native rebuild is needed. Automatic selection is
not guaranteed on unfamiliar videos and does not establish physical accuracy.
The separate equipment-feedback learning driver remains independent work.

The latest root-owned published checkpoint is `640b39d`; standing approval applies
without another merge question. Preserve independent avatar work and all failed
runs. No live handle remains in this workflow lane. Meta budget is unchanged.
