# Separate characters and new-animal checks

## What changes

Each independent character needs its own generation job, model, skeleton,
motion clips, review and downloads. A scene groups those jobs; it does not weld
their meshes. A failed child must not be hidden by a successful sibling.

The accepted panda example reused a combined panda-and-shoulder-dragon source.
It does **not** prove separate asset generation or independent dragon motion.
That source and every earlier result remain unchanged. Copying it into two files,
renaming its parts, or describing a 2D preview as a separate rig is not a fix.

Single-asset planning now explicitly refuses multiple independently animated
characters and directs the request to separate jobs. This is model guidance,
not a reliable natural-language entity detector. Use explicit asset cards for
the deterministic boundary; automatic decomposition of an arbitrary paragraph
into separate characters is not yet implemented.

## New verification requests

| Asset | Appearance | Requested motion |
| --- | --- | --- |
| Barn owl | Cream/brown feathers, broad distinct wings | Take off, flap both wings, travel forward |
| Cream rabbit | Short coat, long ears, four distinct legs | Repeated forward hops with hind-leg and foreleg articulation |

Neither request reuses the panda, dragon, references or an existing source mesh.
Each requests one 4K geometry/texture generation. Normal rig, deformation,
actual weighted-part motion, native/exported appearance, video and fallback
checks still apply. Root travel alone cannot establish wing or leg articulation.
Flight is authored animation, not aerodynamic simulation. No seamless loop or
Weft runtime compatibility is assumed.

The request files are in `outputs/different-animals-preflight-v1/`. They are
prepared inputs, **not generated assets**. Analytic scene/UI tests likewise do
not establish generated-animal quality.

## Use

Start Studio yourself from the Weft root:

```sh
./tools/avatar-assets/run studio --port 8766
```

Open `http://127.0.0.1:8766/`, select **Two separate assets**, and give each card
one subject, its appearance and its actions. Any reference should contain only
that card's subject; a shared image of two creatures can recreate fused geometry.
Saving creates separate jobs without paid calls. Prepare and Start each child
through its existing allowance checks. Each child's progress and downloads stay
separate. The group is not an accepted synchronized scene.

For a reusable, no-call command-line draft:

```sh
./tools/avatar-assets/run scene-prepare \
  examples/requests/owl-rabbit-separated-v1.json \
  --out outputs/my-animal-pair-v1
```

Use a new output folder every time. The CLI saves exact child request drafts,
aggregate estimates and source hashes; it does not create Studio jobs or call
providers. The schema accepts 2–3 subjects; the initial Studio form accepts two.
See `examples/requests/panda-dragon-separated-v1.json` for a two-asset request
that retains the shoulder-takeoff relationship as **unsupported**, not executed.

The actual CLI checks used `outputs/owl-rabbit-scene-plan-v1` and
`outputs/panda-dragon-scene-plan-v1`. Both verified nine artifact hashes. They
contain no geometry and started no child jobs or paid calls.

## Local verification

- 12 scene-contract checks: exact per-subject actions, separate output paths,
  invalid/oversized inputs, source hashes, symlink/traversal refusal, duplicate
  source/model rejection and explicit unsupported attachment requirements.
- 57 Studio checks: separate job creation, HTTP authorization, reference binding,
  reserved save slots, interrupted-save recovery, no dispatch on save, per-child
  progress/downloads and existing lifecycle/UI compatibility.
- 44 existing pipeline/prompt checks passed after the single-actor guidance change.
- Three subsequent polling checks: 20 known-scene refreshes issue 20 scene-detail
  reads and no duplicate selected-job reads. Library reads are unchanged; this
  measures analytic request counts, not live browser latency.

The [saved evidence bundle](../evidence/separate-assets-v1/manifest.json) verifies
42 copied artifacts. These are analytic tests and no-call preparations, not a
live browser run or successful generation. The UI accepts two explicit cards;
automatic paragraph decomposition and coordinated attachment motion remain open.

## Costs and execution boundary

Two new sources require 70 Meshy credits: 35 per asset. The last inspected
canonical allowance has 8 remaining, so these require 62 additional approved
credits. No credit purchase, paid generation retry or automatic budget increase.
The tested ledger path can apply 35 and 27 credits to the exact child jobs,
using the 8 existing base credits without changing either 35-credit job ceiling.
Thirteen focused checks cover partial grants, isolation, retained uncertain
charges and concurrent reservations. No real grant has been applied.

Six expected OpenAI calls (planner, rig/motion author and reviewer per asset)
are estimated at $1.197 using 50,000 input and 8,000 output tokens per call.
This is an estimate, not a guarantee. The existing $15 total cap includes
$13.375724 of settled usage plus retained unknown-charge holds, leaving
$1.624276. Calls must run sequentially because each reserves $0.929376 before
execution. The existing cap can stop a run before completion even if the
estimated final bill fits. No hold is released without billing evidence.

The concrete animal trial proposes three calls and zero repairs per animal.
It does not change the existing server configuration or authorize calls merely
by saving requests. Existing ledgers, exact-job grants and explicit Start gates
remain authoritative. No new paid call has been made for these examples yet.

## Separate delivery versus coordinated scenes

A group of independently downloadable models is not yet a synchronized scene.
Static placements may be recorded as authored assumptions. A dragon attached
to a shoulder, releasing at a given frame, flying and returning requires a
verified attachment/timeline implementation; saving that requirement does not
execute it. Unsupported relationships must remain explicit.

For a fused existing source, hidden contact surfaces and independent meshes
cannot be recovered faithfully by relabeling. Generate each subject separately
or use a separately reviewed authored derivative, clearly labeled as such.
Never claim that manual separation proves automatic text-to-separate-models.
