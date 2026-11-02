# Restore camera-lab evidence inspection

## Verified fix — November 2, 2026, PST

Both the new motion-prior lab and the earlier learned camera lab returned
`Invalid saved lab evidence: exercise scene binding` from ordinary inspection.
The read-only reader recognized shared-world v1–v3 but not the camera v6
contract, so all fourteen evidence categories were hidden. This is a product
integration defect, not failed reconstruction, corrupt runs or new physical data.

`lab_evidence.py` now recognizes the camera scene/task and reuses the existing
pure saved-admission validator before displaying learned counts. It binds exact
policy bytes, admission, original/execution runtime identities, task, root
assistance, evidence inventory and training/evaluation manifests. The archived
research labels remain explicit. Training-only and untrained labs cannot gain
evaluation counts. The reader neither admits a controller nor reruns evaluation.

`lab_inspection.py` retains only the already-read ≤64 KiB policy bytes, reads the
additional ≤128 KiB admission metadata once, and links existing camera/geometry
artifacts without parsing their large arrays. It restores the camera task's
declared timeline without turning unsampled frames into measurements. Existing
8 MiB metadata/128 KiB output bounds and background-worker/cache behavior remain.
The earlier formats retain their existing behavior; unsupported formats do not
become supported just because their version number is higher.

## Try the fixed evidence workflow

From the repository root, open the same motion-prior lab with the tested runtime
overlay (only the two read-only reader modules changed):

```sh
PYTHONPATH="$PWD/.local/camera-lab-evidence-recovery-01/motion-runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/aspset-motion-native-01/lab
```

Enter `evidence` for all fourteen categories, `evidence 12` for learning, and
`evidence 5` for physical relationships. If the first command says loading,
enter it again after the background read completes. Nothing advances until
`step`. The motion example has no learned policy; its force/contact accuracy
remains unavailable, not zero error or a pass.

For the previously trained example, substitute runtime directory
`.local/camera-lab-evidence-recovery-01/learned-runtime/src` and lab
`artifacts/reconstructed-camera-learning-01/lab`. Its saved counts are
learned/authored **3/3**, coast/random **0/3**, labeled exposed-development
simulation. Missing completion/reward/physical-transfer measurements remain
unknown here; this reader intentionally does not parse the archived trial arrays.
These local evidence/runtime directories are not bundled into a fresh Git clone.

## Acceptance and bounds

**116 Python checks pass**, including 23 new camera cases: roots on/off,
training-only/evaluated/untrained labs, admission/policy/runtime/hash conflicts,
wrong research labels, missing/oversized metadata, incompatible profiles,
impossible success counts, no source mutation and complete missing-frame slots.
Native/numerical/model/network/subprocess imports/calls are forbidden in the
new analytical cases. Receipt:
`.local/camera-lab-evidence-recovery-01/tests-01.xml`.

Both actual console subprocesses accepted `evidence`, `evidence 12`,
`evidence 5`, `status`, and `quit`. They each compiled eight models, allocated
one native state, made four geometry forwards and **zero physics steps**, with
stepping/rendering/media decode/network calls prohibited. All fourteen categories
rendered as terminal text. No new recording, training, source fit, server or
browser was started.

| Saved lab | Console wall / worker peak RSS | Saved inspection read | Output JSON |
| --- | --- | ---: | ---: |
| Motion prior | 1.550 s /195,674,112 B | 10.962 ms | 16,006 B |
| Learned camera | 2.671 s /333,922,304 B | 27.134 ms | 29,670 B |

Two workers maximum, each bounded to 20 s wall/15 s CPU/512 MiB sampled RSS,
32 model loads/64 states/1,024 forwards and zero steps. These local checks do not
establish browser/device responsiveness. Both processes are terminal/reaped,
exit 0. Original status/run manifests stayed unchanged; all **1,308 sealed run
members** and every non-overlaid runtime member verify after close. No budget,
credential, API or deployment activity occurred.

The motion run keeps 47 reference poses and 138 unsampled slots; the trained
run keeps 47 and 141, including its three trailing source frames. Both tasks
declare 184 controls /3.68 seconds. Native-clock synchronization is not rechecked
or newly claimed by metadata inspection.

## Resume and ownership

Root owns `lab_evidence.py`, `lab_inspection.py`,
`tests/test_camera_lab_evidence.py` and these guides. Preserve the separately
owned avatar renderer/tests/tools. Plans, exact overlays, before/after summaries,
stdout/stderr, terminal handles and read-only verification are in
`.local/camera-lab-evidence-recovery-01`; no live root process remains.
Publication receipt: `.local/camera-lab-evidence-publication-01.json` after push.
Previous root checkpoint is `3c666c7`; standing approval covers the scoped fix.

The [continuous-velocity source diagnostic](continuous-reference-screen.md) is
rejected and not installed. This fix restores inspection; it does not supply
missing force/contact evidence or complete the broader goal. Next connect
source-bound force diagnostics to ordinary saved evidence without modifying
existing runs/scorers, and pursue source-qualified geometry/force-aware
construction plus different-subject/view and independent physical validation.
