# Temporal identity source labels

Frozen assistant-authored visual estimates from **27 original frames across four
exercise clips**. The manifest retains 54 shoulder/hip pair observations: 39 with
approximate source-visible directions and 15 explicitly unobservable. These are not
independent expert or clinical ground truth.

Annotation SHA-256:
`89f70a41d934989343f7836bb224208484164f9781e0a4c82b8353124be4b120`.

Original frames were inspected before new identity-candidate outputs. No raw detector
outputs were inspected for newly selected negative windows. The existing jumping-jack
frame-330 diagnosis is already exposed development evidence, not a holdout. Other
frames/clips also have prior development use, documented per window in the manifest.

- `annotations.json`: frozen pixel centers, visibility, source-facing descriptions,
  pair directions, uncertainty, timestamps, checksums, and source provenance.
- `*-<frame>.png`: original decoded source frames, unchanged pixels.
- `*-originals-*.png`: source frames assembled with identifying title strips.
- `*-annotated.png`: approximate anatomical left points in magenta, right in cyan,
  left→right pair vectors in yellow. Hidden points remain absent, not guessed.
- `lunge-source-selection.png`: resized overview used to select real-turn windows.

## Attribution and reuse

| Clip | Author | Source | License |
| --- | --- | --- | --- |
| jumping-jacks-burpees | Taco Fleur / Cavemantraining | [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| lunge-twist | Taco Fleur / Cavemantraining | [Kettlebell Lunge and Twist](https://commons.wikimedia.org/wiki/File:Kettlebell_Lunge_and_Twist.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| single-leg-squat | RickyBennison | [Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| kettlebell-pushup | Taco Fleur / Cavemantraining | [Kettlebell crush grip push-up](https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Excerpts and adaptations retain CC BY-SA 4.0. Changes: frames extracted, title strips
and separate annotation marks added, selection overview resized. Annotation data is
offered under CC BY-SA 4.0. Source sidecar provenance is copied into the manifest.

See [interpretation and scoring contract](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/temporal-identity.md).
