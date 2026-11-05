# Body-orientation development labels

Nine assistant-authored visual estimates from original video frames, frozen before
candidate body-orientation scoring. These are not independent human-expert labels,
clinical measurements, recovered anatomy, or a final holdout.

`annotations.json` records original video and frame checksums, frame indices,
approximate hip/shoulder centers, visibility limits, and fixed 15° axis tolerances.
Coordinates use original image pixels: x right, y down, including letterboxing.
The circular marks show approximate endpoint uncertainty, not measured confidence
intervals. Clothing and occluded far-side joints limit accuracy.

- `*-<frame>.png`: unchanged decoded source frames, without marks or resizing.
- `*-originals.png`: original frames assembled with identifying title strips.
- `*-annotated.png`: the same frames with estimated torso axes and uncertainty marks.

## Attribution and reuse

| Clip | Author | Source | License |
| --- | --- | --- | --- |
| squat | FitnessScape | [Squat demonstration](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| single-leg-squat | RickyBennison | [Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| kettlebell-pushup | Taco Fleur / Cavemantraining | [Kettlebell crush grip push-up](https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Source excerpts and adaptations retain their listed licenses. Changes: individual
frames extracted, contact-sheet title strips added, and separately marked estimates.
Full source provenance is copied into `annotations.json`. The annotation data is
offered under CC BY-SA 4.0. This small convenience sample is development evidence,
not a representative accuracy study.

Frozen annotation SHA-256:
`a017fff83778a19cb19ceedc403a7b98cdedf37b40ded01f8a4e1b6bb9c08a01`.

See [grader and interpretation](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/body-fidelity.md).
