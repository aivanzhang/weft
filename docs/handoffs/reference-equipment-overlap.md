# Saved reference–equipment overlap

## Completed — November 1, 2026, 6:13 PM PST

The actual public inspector and independent saved audit pass. Open the
[new offline report](../../artifacts/reference-equipment-inspection-01/inspection/index.html).
**52 of 121 saved reference poses overlap `sam-2`**, across 115 negative-distance
contact records. The first is reference index 20 / source frame 42 / declared
source time **1.4 seconds**; largest penetration is **0.0987466991 model units**
(the explicit world/model scale here is one). `sam-3` has no recorded overlap in
these poses. Do not interpret that as clearance between poses or physical proof.

Contact-record counts: right forearm collider `right_elbow_geom` 64; left forearm
`left_elbow_geom` 37; right upper-arm `right_shoulder_geom` 12; left upper-arm
`left_shoulder_geom` 2. These are **records, not unique poses**. Raw collider IDs,
first source identities and complete exact snapshots remain downloadable. Forces
computed for kinematic poses are deliberately not interpreted as load.

The earlier preflight remains `feasible` under its original floor/static and
initial-placement scope. The report now displays that scope separately from
“Sandbox prepared; reference overlaps equipment.” Lab status remains ready, task
success remains unassessed by the inspector, and no task/scorer/default changes.
This demonstrates a reconstruction consistency problem, not an impossible task
or a proof that all controllers fail. It supports addressing body/equipment
geometry jointly before further controller-only sweeps.

Public report: **0.141270 seconds**, 31 files/17,078,083 B. Complete supervised
acceptance: **0.241444 seconds**, 32 files/17,089,844 B, **76,562,432 B** OS child
high-water RSS, 0.201929 CPU seconds. The single sampled RSS value missed the
short-lived peak; use the high-water result, not that sample. PID 9415 exited zero
and was reaped. All 307 pinned input identities stayed unchanged; no denied
operations, native/media imports, physics, source-proof rerun, API or server.
These are bounded offline backend measurements, not browser/sustained-use QA.

Root independently verifies all 30 report-seal members, 28 selected source
artifact joins, 51 local HTML links and all 115 overlap records/52 pose identities.
Read 17,091,404 B including the process receipt; no second geometry evaluation,
original source inference, video decode or simulation. Cached interpretation:
`.local/reference-equipment-inspection-01/saved-metadata-01.json`.

| Evidence | SHA-256 |
| --- | --- |
| Acceptance result | `427ab225a9a33d037a140f0672692a3ec4e6b111e6950186622944a0630e81ff` |
| Report seal | `f3a3cc6dc09ccd917c84e8bbff6abb9a0a31e53467aa73228a181e1a33e0891c` |
| Report summary | `25752241e105439ac0529c8da3e1c73287dd51298d6f2b3fd4f311b286632cc3` |
| Process receipt | `1691cdb141cc6f17d36c7c3b2f7f2786e884f503530da0a799950cbb2fdb3343` |
| Independent audit | `cdedd3f783f042799822b191ba3383168a667535b46edae2dee3fb3c2da3e8a7` |
| Latest 36-case receipt | `440b44680844364b15b8797b8a77f6842c1f908591c53b0b1f2ae47a547ca46d` |

For a different completed/stopped build, use the existing command with a new output:

```sh
.venv-lab/bin/python -m weft shared-exercise-inspect \
  --run path/to/saved-build --out artifacts/new-build-inspection
```

Do not rerun this acceptance or overwrite its output. No browser was opened.
Native snapshot generation is not part of this command. Inspection is descriptive;
current physical accuracy, successful full exercise learning and unfamiliar-video
coverage remain incomplete. Root owns exact-path publication under standing
approval; receipt `.local/reference-equipment-publication-01.json` will record the
actual commit/remote identity. Preserve all independent avatar changes and artifacts.

## Original preparation — November 1, 2026, 6:09 PM PST

Previous goal turn made progress: published temporal actor and eight full-interval
negative trials at `a90d97d`. This turn returns to reconstruction consistency.
The original automatic-world preflight reports feasible floor/static geometry
and initial equipment placement. It does **not** audit dynamic-equipment overlap
through the reference. The already-sealed 121 kinematic snapshots contain later
human/equipment overlaps before any physics action, so controller-only tuning
does not address the entire mismatch. This is saved simulator geometry evidence,
not real contact, identified depth, anatomy, load or proven control infeasibility.

## Implementation and ownership

Root owns:

- `src/weft/reference_equipment_report.py`: pure bounded inspection of all saved
  references. Checks source-frame/time, body angles/root/origin, nominal equipment
  placement/geometry, zero-control state, contact identity/classification/counts
  and explicit scale. Reports unique affected poses separately from contact records,
  first source-linked overlap and maximum penetration per object; at most 12 examples,
  with omitted counts and complete original records preserved. No native imports.
- `src/weft/exercise_build_report.py`: integrate this diagnostic into the existing
  public offline inspector, copy its exact three scene/reference/lab-binding inputs,
  and explain the old preflight's narrower scope. Runtime admission and task success
  remain unassessed; a ready lab remains ready. Unknown/oversized inputs do not become
  zero error. Full-reference inspection currently covers v1/v2 shared worlds; other
  profiles explicitly remain unavailable. No schema/physics/task/scorer change.
- `tests/test_reference_equipment_report.py` and one wording expectation in
  `tests/test_exercise_build_report.py`. Preserve all independent avatar files.

**132 distinct scoped checks now pass:** 36 new core/report cases, 94 unchanged
passing legacy cases from the first guarded run, plus two original fresh-process
import assertions executed separately. The first guard rejected those tests'
child-process launches; preserve its two failures and exit 1. Exact child test
code was extracted, not relaxed, and run in fresh sanitized processes with 5 CPU
seconds, 4 MiB per file and no network/subprocess/native imports: 0.033196 and
0.034405 seconds. No unchanged passing legacy suite was repeated. Root then
added root/origin joins and three regressions; only all 36 affected new cases were
rerun successfully. Receipts are in `.local/reference-equipment-report-01/`.

## Frozen saved-build acceptance contract

Root owns one inspector-only run. Input is
`artifacts/equipment-frame-consensus-native-03/build`, seal
`3623cb0004e4c1dd19a8a7fdaa2f58e6cac23e563cfa78480c296f98d267f97c`.
Its `lab/run/human-reference.json` is 4,193,073 B, SHA
`056fed932c18d3c87f2753296ae43af7b2e852e87d91f473863b9c4106d85d7f`.
Scene identity remains `6510c4a0f6c2c1034fa6d90d071ca4e6870f9a4f6cf250883b4651485aa40c4b`.
The source video, original lab, reference preflight, scorer and failed control
studies are immutable. No new source processing or replay is needed.

Reuse the exact 299-file e5 runtime with only the two frozen report modules and
the unchanged palette. Four direct base dependencies match the tested worktree
byte-for-byte. Output: `artifacts/reference-equipment-inspection-01`, scratch
same suffix under `.local/`. Limits: report 30/cooperative 40/CPU 30/external 60
seconds, 512 MiB observed RSS, 32 MiB/64 files, 1 MiB per log. Expected output
about 20 MiB. No native/media imports, network, subprocess, training, source-proof
rerun or API work. Cost $0 external/$0 recurring. Preserve any failure; no retry
or cleanup is implicit. Existing descriptor guard and tested existing-ancestor
mkdir no-op shim remain intact. Root reviewed; peer agents remain at usage limits.

Frozen plan `ef718194c3b49fa43704a1ee15f5b656b7cea5c754eda3090d6b5efb18a41587`;
worker `5ca23f230e5ad9a94991880e905593a871a6da3128fdd7848bba23eb3a14d81a`;
supervisor `31b08c5c9437057535f2468b1776b2d90d1b56e4f54d1e854d30533e207111f2`.
Helper `078f7e4457e0fe34a75887ff73217d09f2350952dfcfba5859ba283e9d661cee`;
inspector `4a6fab1eec5c1979342b5bef64796d117c0a961a9e31e3c496d2669bbf7c9dd4`.

The independent count, link, identity and resource audit is complete above.
Only owned tested paths may be published; the full goal remains active.

## Next safe work

Keep the source-linked mismatch visible while investigating a source-qualified
joint body/equipment geometry candidate. Do not silently move the floor, carve
an unseen handle, overwrite references or weaken the task. Existing old-world
arm refinement is already a negative result: 303→216 penetrating pairs, maximum
0.104829→0.098129, 33 pointwise worsenings; saved at
`artifacts/source-arm-equipment-01`, seal
`500b953c66dcb0bab092d01be43d2aea252635e4268fc59cde30a7d3af276426`.
Its late right-forearm samples have weak model confidence, not measured depth.
The new report is not permission to repeat that unchanged solve or the eight
failed temporal-controller trials. A new candidate needs an explicit changed
input/hypothesis and preserved source/image/floor/control evidence boundaries.

## Source search boundary

A brief metadata search also considered Berkeley MHAD. The publisher's HTTP/HTTPS
root and HTTPS data endpoint were inaccessible through the web tool; no matched
licensed acquisition unit was established. The [author's paper](https://www.ferdaofli.com/_files/ugd/aff495_37414c2479a84a0babc69e6e3f69560c.pdf?index=true)
describes synchronized calibrated modalities but does not itself supply the
missing downloadable package/rights verification. The search also rediscovered
the already-known [MoVi access restriction](https://www.biomotionlab.ca/movi/);
no restricted data/code download, account, agreement or message was attempted.
This is not an exhaustive source screen or a whole-goal blocker. Current useful
work is the actual local consistency diagnostic above, not repeated access polling.
