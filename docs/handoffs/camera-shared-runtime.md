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

## Remaining product work

The importer must preserve both available camera/evidence documents, root
triangulation and ground, original primary media/rights, all source slots and
missing secondary media/rights explicitly. Candidate construction is not physical
admission. A subsequent independently pinned task/recording/lab integration must
make controls and exact replay usable without weakening the protected scorer.
Calibrated feet and useful ground/equipment contact remain unverified.
