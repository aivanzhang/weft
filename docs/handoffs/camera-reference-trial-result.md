# Camera fitted-reference trial result

## Result

The real jogging camera-angle proposal now completes the existing matched
body-reference trial, export and exact-replay path. This verifies delivery and
controller execution, **not improved control quality or physical accuracy**.

The run completed on October 31, 2026, at approximately 8:02 PM PDT. Both
controllers passed the unchanged authored simulator gates, with explicit
three-axis root assistance. The fitted controller scored worse against the
original task:

| Saved measure | Original authored reference | Fitted reference |
| --- | ---: | ---: |
| Mean reward | 0.8262091657 | 0.8021709909 |
| Joint-angle RMSE, radians | 0.1457363253 | 0.1578925997 |
| Root RMSE, model units | 0.0219746792 | 0.0223122619 |
| Maximum floor penetration, model units | 0 | 0 |
| Original task success | true | true |

The earlier source-image fit improved mean endpoint error from
12.7921541242 to 9.8793089935 pixels. That is a different objective from
tracking the original reference in simulation. No contact, balance, biomechanics,
physical transfer or learned-policy claim follows from either result.

## Frozen evidence

- Trial: `artifacts/camera-reference-real-01/trial-03`.
  Manifest SHA256:
  `ca80907ca85c741b7c1dcc46d1ad096df64132cb1b7248ff40043737408cbec3`.
- Native acceptance receipt:
  `.local/camera-reference-trial-real-acceptance-03.json`, SHA256
  `caba38960f4e16b94374ff0907d2bbdb7b85acdcd73214a62c3ab3894d677abb`.
- Source proposal: `artifacts/camera-reference-real-01/fit-02/proposal`, seal
  `12ed22060e2dc699d26139121b73605d95c31a76625c54817395014eae3df8b0`.
- Original source candidate:
  `artifacts/camera-shared-admission-jog-01/candidate`, seal
  `29fef74cb9ed1c0bf454b78f1a32b4d845277461ed8948a70d620da0dd08d1db`.
- Exact recipe: `trial-03/run/reference-controller.json`, raw SHA256
  `aaec5420f3d0e6f0fa2bc98aa9d9ee31419bd80b3fdcf6c8df614adce1eea962`.
  Its canonical document hash is separately recorded by the producer; raw and
  canonical hashes must not be interchanged.
- Frozen runtime: `.local/camera-reference-real-runtime-02/manifest.json`,
  SHA256 `7dbaaa8174534631079f814435d75871128436bf6f711cb08edca4938fb381eb`.

The actual scene remains `shared-world-v6`; its canonical hash is
`5626deea3a709cf1a20e996b1f0bb5c89e0f61d2382e7a50ad78e54628be8666`.
The unchanged task hash is
`76a36eb74de5b18455011ce1cd20d9caaba6759a6135fd286e329b7fdac8a122`.
Original body/root/camera/floor/source bytes and protected scorer identity remain
bound. The overlay is not an installed body; scene learning availability remains
false and task learning scope remains unqualified.

## Verified execution and coverage

- One matched trial, seed 1729, no jitter: 238 authored and 238 fitted controls.
  The two full initial snapshots are identical.
- Each recording contains 239 events: one start and 238 steps, no in-trial reset.
  Controls are 19 joints plus the explicitly authorized three root channels;
  object forces remain forbidden.
- Both exports have saved full-state replay proofs. The fitted proof verifies
  regeneration of all 238 fitted actions under the frozen recipe.
- All 238 paired joint-action vectors differ; maximum absolute difference is
  1.1303645624. This compares the recorded diverging trajectories, not two actors
  evaluated on an identical state. Later root feedback values can consequently
  differ without changing root authority.
- All 240 original source slots are retained. There are 120 sampled pose slots
  and 112 supported body references; 120 unsampled source clock values remain
  unavailable, not interpolated. Surface terms remain unavailable for all 112
  references, with zero available surface terms.
- Control period remains `1001/30000` seconds; original body period remains
  `1/30`. The full supported interval ends at frame 238 / 7.9412666667 seconds,
  the original final body reference. It does not fabricate a body reference for
  the final video slot 239.
- Native acceptance: 13.6414 seconds; 476 live and 476 replay controls at nine
  substeps each explain exactly 8,568 physical substeps. No fitting, training,
  source inference, source video decode or API spending occurred in this trial.
  The saved receipt retains the blocked optional MuJoCo startup subprocess
  attempt; no execution permission was added.

## Independent saved-only audit

Final receipt: `.local/camera-reference-trial-saved-audit-03.json`, SHA256
`ebc24a5379dd01629b8229d8d67fb6d48760a4f277d0303f3eb0aa3ccf4ae8e5`.
Reproducible checker: `.local/audit_camera_reference_trial_saved_03.py` with its
preserved cache/view and prior receipts. No further audit run is needed while
these inputs remain unchanged.

The audit checked seals, exact selected aliases, source/task/recipe/runtime
identity, complete action labels and clocks, matched starts, final metrics and
saved replay proofs. Each journal was streamed once. Original reads were
32,617,432 bytes across 43 files in 0.294 seconds. Including cached analysis and
the narrow checker repairs, the conservative data-read bound was 133,191,655
bytes, below 128 MiB. No product imports, simulation, replay, media decode or
source work ran during the audit.

Checker receipts 01 and 02 remain failed and preserved. Corrections addressed
auditor assumptions: Weft canonical JSON uses default spaces, skipped manifest
keys are relative, unsampled clocks/admissions remain null, and the source hash
is carried in the outer camera snapshot rather than duplicated in inner legacy
metrics. Completed checks were reused; neither journal was reread.

Limits: media/viewer members were joined by their manifest pins, not decoded or
rehash-scanned; runtime pins were joined to the previously frozen manifest;
native exact replay was authenticated from its saved proof, not rerun. Per-step
task clocks and first/last actual-engine clocks were checked. The cache omitted
intermediate engine scalar extraction; no second journal pass was added merely
to repeat the native replay proof. Root owns viewer inspection separately.

## Interactive lab acceptance

The separately implemented lab attachment passed its one native acceptance on
October 31, 2026, at 8:27 PM PDT. The prepared console lab is
`artifacts/camera-reference-play-01/lab`. It preserves the exact original scene,
task and fitted recipe; no refit, training or source inference was needed.

- Outer seal:
  `61298ff790531bf99486c6f340214c732b489595594bc9b27938c9451873f8a8`.
- Immutable lab run seal:
  `b4375d1fdc321a99ed6098546213a9a90dbfff0728463192b0a3152d4aa9f6f5`.
- Session: `session-06fb54b1d283`; 248 events / 246 actions / one reset.
  Actions comprise two manual, two coast, two original-authored and 240 fitted
  controls. After the reset, 238 fitted controls complete the original interval.
- One full-state replay verifies all actions and all 240 fitted decisions.
  Live plus replay execution accounts for 492 controls / 4,428 substeps.
- Saved receipt inspection performed no native work; a fresh lab resume passed.
  The 48 original recipe modules remain exact inside the 57-module execution
  closure. Added lab adapters do not replace or repin original recipe code.
- Parent process: 11.9313 seconds, 387 files / 90,131,979 bytes; observed peak RSS
  530,661,376 bytes, not a hard memory-limit claim. No paid calls occurred.

Independent saved audit: `.local/camera-reference-play-saved-audit-01.json`,
SHA256 `340fbf2f05720aecd732fccf54323dd6ecea6d547ef2e9414ec38386db73d15e`.
It read the new journal exactly once (5,888,604 bytes), checked all mode labels,
reset/source clocks, original source/task/recipe custody, and the new replay
receipt. Total data reads, including cached analysis, were 12,211,418 bytes;
23 original files were read, including two small final manifest aliases.
Measured successful read/analysis phases totaled approximately 0.067 seconds.
The parent's complete file-hash map and frozen runtime were reused; no media,
native replay or source study was repeated. The initial incorrect runtime
manifest filename lookup is preserved separately in
`.local/camera-reference-play-saved-audit-failure-01.json`; correction preceded
the sole journal stream and did not alter acceptance artifacts.

The interactive result proves usability and reproducibility, not better task
performance. The preceding matched comparison remains the quality evidence.

## Next safe action and ownership

Use the frozen runtime while the working tree contains unrelated reconstruction
changes; the exact recipe correctly rejects a changed implementation:

```sh
PYTHONPATH=.local/camera-reference-play-runtime-01/src .venv-lab/bin/python -B -m weft lab-console --resume artifacts/camera-reference-play-01/lab --seed 19
```

The console supports `mode fitted`, `mode reference`, manual motors, coast,
reset and recording. Do not repin the recipe, replace body geometry or relabel
fitted control as learned. A new runtime compatibility change needs its explicit
contract and bounded acceptance.

Root owns publication and further native work. These audits have no live handle
and made no product or Git changes. Earlier failed fit/workflow/report artifacts
remain immutable; saved report recovery and successful trial/lab runs do not
relabel those failures. Root separately checked the saved viewer; this audit did
not duplicate that check.
