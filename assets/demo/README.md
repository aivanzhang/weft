# Demo previews

Three distinct examples pair source footage with a saved result:

- **Object motion:** the verified rolling model, evaluated through the existing
  physics implementation. Both sides show the primary observed pass at quarter
  speed. This is a presentation export, not a new reconstruction or evaluation.
- **Body control:** the recorded assisted squat controller, cropped to its front
  camera. The robot is a visual stand-in for the simulated body.
- **Character transfer:** the post-swap portion of that movement, shown with the
  generated panda. Appearance changes; the recorded physics does not.

`previews/manifest.json` records source hashes, intervals, crops, playback changes
and output hashes. Original scored artifacts remain unchanged. Source attribution
is retained in the curated catalog and displayed in the studio.

Rebuild with the lab environment and the archived source sessions present:

```sh
.venv-lab/bin/python scripts/export_demo_previews.py --out /tmp/weft-preview-build
```

The exporter requires an empty destination and makes no provider calls. The
committed previews let the curated demo work without those archived sessions.
