# Source-only limb-direction development labels

Twelve frames, 96 segment slots: 84 approximate visible directions and 12 explicitly
unavailable source observations. Labels were authored by the assistant from decoded
original video frames before inspecting candidate limb predictions. They are not
independent expert annotations, clinical measurements, or a final holdout.

`annotations.json` is frozen at SHA-256
`f051db850105a77e0350cdfd7128f84d340ad1a18dafcfeda9c6f4ab722290bd`.

- `*-<frame>.png` are original decoded source frames, without annotations or resizing.
- `*-annotated-*.png` show approximate segment endpoints: anatomical left in magenta,
  right in cyan. Absent segments mean source visibility was insufficient.
- The source overview and selection contact sheets document frame selection, not
  candidate predictions. These sheets are resized; use original PNGs for coordinates.

Coordinates are original pixels, x right and y down, including letterbox bars.
Each frame retains shoulder, elbow, wrist, hip, knee, and ankle entries on both sides.
Occlusion or inseparable projected overlap uses `xy: null`, not guessed coordinates.
Tolerances were fixed from visual endpoint uncertainty before candidate scoring:
`degrees(atan2(sum(endpoint_uncertainty_pixels), segment_length_pixels))`, clamped to
15–35°. This heuristic is not a calibrated confidence interval.

## Attribution and reuse

| Clip | Author | Original | License |
| --- | --- | --- | --- |
| jumping-jacks-burpees | Taco Fleur / Cavemantraining | [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| kettlebell-pushup | Taco Fleur / Cavemantraining | [Kettlebell crush grip push-up](https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| squat | FitnessScape | [Squat exercise demonstration](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |

Original excerpts and derivatives retain these licenses. Changes: individual frames
extracted, contact sheets assembled, identifying strips and separately marked visual
estimates added. Annotation data is offered under CC BY-SA 4.0. Full provenance and
source hashes are in the annotation document. See [grader](https://github.com/aivanzhang/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/limb-fidelity.md).
