# Camera-body shared runtime

Updated October 31, 2026, 2:32 PM PDT. Owner: root. Status: native runtime verified;
source import/task/recording/lab integration remains separate, not a ready lab.
Base: `12cd7a59ca3cc0f6c4fa0897971a659f892069e0` on `main`.

## Boundary and ownership

- Root owns `src/weft/shared_world.py`, `tests/test_camera_shared_world.py` and this note.
- Reconstruction owns the new `camera_shared_candidate.py` importer/tests/handoff.
- Contract lane owns `camera_shared_contract.py` and its analytical tests/handoff.
- Review lane owns the calibrated-foot consumer correction/tests/handoff. It found
  a missing consistency check between retained source points and derived foot axes/gaps.
- Avatar edits remain unrelated and must not enter runtime snapshots or commits.

`shared-world-v6` admits only unchanged, grounded `two-view-root-body-v1` humans
in the existing single-model `SharedWorldEnv`. Camera body `dt`, shape, rotation,
roots, ground and source rows remain unchanged. The shared control interval is
`1/source_fps`, including fractional footage rates; it is not the body's stored
integration setting (normally 30 Hz; an original 60-Hz body is also verified).
A versioned coupling binds this distinction.

World-to-local translation is `[-origin.x, -origin.y, -ground.height_m]`, with
identity rotation/unit scale, zero additional shared placement and no floor fit.
Supplied camera metres, inferred body dimensions and authored mass remain labeled.
Legacy profiles retain their camera rejection and exact 30-Hz feasibility grid.

The engine permits zero to twelve static objects in its existing object envelope;
none are recovered by this runtime change. Dynamic equipment, calibrated feet,
independent task/support scoring, learning and lab/recording packaging remain
unavailable until their own explicit adapters are integrated. Old exercise and
foot task adapters must continue to reject this profile.

## Verified checks

Six initial native cases use analytical two-view inputs, not a saved real video:

1. Compare every original native site/collision geometry, subtree mass, joint
   range, source target and knot velocity; verify the full fractional-rate grid.
2. Compare seeded states exactly for three controls and their repeat, with root
   channels zero and inspection unable to perturb live physics.
3. Preserve 18-unit travel in snapshots and a body-follow rendering; retain the
   actual fractional clock when an action terminates partway through physics.
4. Retain/reject initial ground penetration without changing the floor/body.
5. Reject legacy profile relabeling, unsupported task/feet/learning and retiming.
6. Reject an authored static obstacle intersecting the body in the same model.

All six plus two affected legacy schema/mass checks passed at **2:27 PM PDT**:
0.897526 s, seven guarded controls, nine output files/180,239 bytes, observed
child peak RSS 138,592,256 bytes. This is a bounded software check, not sustained
UI performance. Frozen committed export plus three exact overrides:

- Runtime `.local/camera-shared-native-runtime-01`, manifest
  `0495ca791cf89b80cce3e7c2673e28d9763b2fc483f7899c45298514072b89b1`.
- Plan `.local/camera-shared-native-plan-01.json`, SHA
  `5e07ab8b6e297c988b75ad2bde8e5056e2db17fb4379dcc8c9f3e0b1195b676a`.
- Receipt `.local/camera-shared-native-process-01.json`, SHA
  `f5e3369a26327c0f102484cd3bba222d4cdd97c6b97199bc8085879ece52785e`.
- JUnit `.local/camera-shared-native-tests-01.xml`, SHA
  `c2ce7c1e4798354fbc58cdf00b7fcc36f0d475886f4eb8255a99c07344e12e92`.

Read-only review found two compatibility gaps, both corrected:

- Existing coupled policy validation/action assembly now reject v6 before native
  compilation or legacy controller work. Two pure regressions pass (0.410382 s),
  `.local/camera-policy-admission-tests-01.xml`; frozen runtime manifest
  `2d0c55bbcea10c63da2bb0edd5a19eaeb30e59c91e5a6f0eaf5de19eddbbbae2`.
- The old 30-Hz body-period guard is now legacy-only. One fresh native check
  preserves an originally 60-Hz body under 29.97-Hz control, comparing all retained
  native joint positions and subtree mass: 0.540516 s, zero controls, seven files/
  107,072 bytes. Receipt `.local/camera-body-period-process-01.json`; runtime
  manifest `05edffcfcb81b579b1dee5b5e8385e1ae09b1889ede44e43790415b6d2ac2a42`.

The eight unchanged passing checks were not repeated. Final engine SHA
`83a01927f04c308d834f2bb813a08cd353c25f373aaa2a2f98845aab5c79e4ae`;
learning adapter SHA
`a647111c87c5a07109330172216aa3f9af1b91df36039e3bfc7e1a5704b73c3d`.
Protected exercise scorer remains
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
All jobs are terminal. Each had 60-second wall/45-second CPU bounds, 32 MiB/1,024
output limits and no source decoding, encoder, inference, network, provider,
training or browser activity. Bodies remain inferred/authored; contacts are simulated.

## Actual saved-source admission — October 31, 2026, 2:39 PM PDT

The reviewed camera importer ran once on the original sealed jogging camera-body
bundle. A fixed-eight-substeps bug was caught by independent review before this
run: 29.97-Hz footage now uses nine substeps, without changing the timestep bound,
body timing or source timestamps. Forty-one distinct pure importer cases remain
covered; the two reproduced fractional-rate failures and three corrected focused
passes are preserved in the [importer handoff](camera-shared-candidate.md).

`artifacts/camera-shared-admission-jog-01/candidate` is **constructed and sealed**:
54 files/26,145,833 bytes, manifest
`29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`.
All 240 source slots and 112 original body poses remain; the full original body
interval is a 238-action horizon at `1001/30000` seconds, with body `dt=1/30` unchanged.
Body raw SHA `849ece268cc6b091cf85460f3e22ecee172aa5323c4db88dac3e221e367d0867`;
scene SHA `5626deea3a709cf1a20e996b1f0bb5c89e0f61d2382e7a50ad78e54628be8666`.

The shared-model zero-physics audit checks 112 source poses and 239 control-grid
poses: **351 feasible geometry checks**, no initial overlap or floor penetration.
This does not mean contact or useful balance: the previously documented floating
camera-body gap remains. No action, rollout, learning or task evaluation ran.
The source scene still has no attached calibrated feet or recovered equipment.

Frozen runtime `.local/camera-shared-source-runtime-01` is committed `6a29850`
plus only importer `095555cd8af80eaf527b3d0e66367776e71a350ab76f0c3975d6c415fc34fa67`;
manifest `b68c25a5960c75d6e4e1b2cb87aabc4059001a020a468b8a6cd77df63d73e37a`.
Plan `.local/camera-shared-source-plan-01.json`, SHA
`f1e9246be4a96d6277badd15aa96837bbe0b362f708542a4ab91c9e3ff099cd5`;
process/result pins are in `.local/camera-shared-source-process-01.json`.
Elapsed: 0.738501 s importer /0.820989 s worker /1.250118 s parent;
observed child peak RSS 202,326,016 bytes. Exactly 352 native forwards and 352
kinematic resets, **zero physics steps**; 57 outer files/26,280,097 bytes.

The importer verifies eighteen selected original files/20,641,152 bytes, not all
242 historical bundle members. Secondary media/rights remain unavailable and
explicit. Primary media is copied exactly without decoding; no new source/model/
provider work. All jobs are terminal. Do not rerun this unchanged successful input.

Independent saved-only audit: **72 checks passed**, receipt
`.local/camera-shared-source-saved-audit-01.json`, SHA
`0f05eae277d4c754e4c4347cdccfc12cc8f3a6ea1f9d9f12a463542629b3ae19`.
It read each candidate file once. An auditor-only truncated metadata output and
redundant hashing from cached bytes are disclosed; neither is a source/candidate
failure or reason to rerun physics. Conservative total: 35,064,850 bytes/72 reads.

The existing CLI now exposes `camera-shared-build --run ... --out ...
--max-seconds 30`; help parsing passed. This thin entry point calls the verified
importer without starting a server or claiming a lab. The saved-source run above
used the API, not a second CLI invocation. [User instructions](../local-lab.md#import-a-camera-based-body-experimental).

Separately, the reviewed foot consumer retains all 240/480/1,440 source/foot/endpoint
slots but remains **unavailable at original frame 0**: the right heel has
`inconsistent_image_rays`. Six negative conditional source gaps remain. No later
start or invented heel replaced it. [Exact saved-only check](calibrated-foot-consumer-jog.md).

## Remaining product work

The importer preserves both available camera/evidence documents, root
triangulation and ground, original primary media/rights, all source slots and
missing secondary media/rights explicitly. Candidate construction is not physical
qualification. A subsequent independently pinned task/recording/lab integration must
make controls and exact replay usable without weakening the protected scorer.
Calibrated feet and useful ground/equipment contact remain unverified.
