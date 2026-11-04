# Demo comparisons

The public catalog in `src/weft/assets/demo_examples.json` contains ten source/result
pairs: rolling ball, bodyweight squat, incline press, deadlift, jumping jacks and floor
transition, overhead squat, waiters squat, Sots press, lunge and twist, single-leg squat.

The ball uses the existing verified one-dimensional model. Both sides play at quarter
speed; `previews/manifest.json` records the presentation export. Older controller and
character clips remain archived but are no longer advertised in the catalog.

The nine human previews use saved reconstructed segment endpoints, not a newly
trained policy. A neutral articulated body, fixed camera, matte materials and floor
shadows make the motion easier to inspect. Display interpolation and a spherical head
marker change presentation only. Equipment, contact forces and full-body balance are
not recovered. These are kinematic reference replays, not validated biomechanics.

Each `motion-showcase/ID/receipt.json` records reference and source hashes, source
window, render settings and output hashes. Original scored artifacts remain unchanged.
Source attribution is retained in the catalog and shown in the studio. Each derived
source excerpt retains its source licence; CC BY-SA excerpts remain CC BY-SA 4.0.

Rebuild in a new output directory with the lab environment and archived runs present:

```sh
.venv-lab/bin/python scripts/export_motion_showcase.py \
  --hosted-catalog /path/to/walkthrough-catalog --out /tmp/weft-motion-preview
```

The exporter uses local offscreen MuJoCo rendering and makes no provider calls.
Local reference paths are declared in the script; their hashes identify the evidence.
The committed previews work without the archives or rendering dependencies. Only the
selected comparison loads media; changing examples releases the previous pair.

## Studio walkthroughs

`studio/` contains the explicitly exported, read-only evidence for the result-first
studio pages: compact skeleton samples, source and presentation receipts, and
selected construction/task/outcome records where those stages actually ran.
The viewer interpolates saved endpoints for display; its camera and playback
controls do not run a controller or alter recorded evidence.

Rebuild the motion exhibits into an empty directory with
`scripts/export_motion_studio.py --hosted-catalog /path/to/walkthrough-catalog --out /tmp/new-studio-exhibits`.
The earlier controller/appearance exhibits use `scripts/export_studio_walkthrough.py`.
Both scripts reuse saved evidence without provider calls. The server permits only
filenames explicitly listed in `portal_studio.py`; private run models and stage
bundles are separate authenticated endpoints.
