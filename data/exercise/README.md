# Exercise video fixtures

The originals are retained unchanged. These sources show recorded people, not synthetic actors. Pose landmarks and derived body geometry are model estimates, not measured muscle activation, force, or anatomical scans. No endorsement by the creators is implied.

| File | Creator / source | License | Use |
| --- | --- | --- | --- |
| `single-leg-squat.webm` | RickyBennison, [Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Full-body, fixed-camera example; distant subject and occlusion remain challenges |
| `squat.webm` | FitnessScape, [Squat demonstration](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Limited-input example: feet are cropped |
| `kettlebell-pushup.webm` | Taco Fleur / Cavemantraining, [Kettlebell crush grip push-up](https://commons.wikimedia.org/wiki/File:Kettlebell_crush_grip_push-up.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Horizontal torso, loaded hands and far-side occlusion |
| `kettlebell-flip-juggling.webm` / `.mp4` | Taco Fleur / Cavemantraining, [Kettlebell Flip Juggling](https://commons.wikimedia.org/wiki/File:Kettlebell_Flip_Juggling.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Full-span MP4 adaptation: identical decoded YUV frames, audio omitted, explicit 3 ms origin translation; two competing pose samples prevent current body admission |
| `lunge-twist.webm` | Taco Fleur / Cavemantraining, [Kettlebell Lunge and Twist](https://commons.wikimedia.org/wiki/File:Kettlebell_Lunge_and_Twist.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Torso twist, loaded motion and ambiguous duplicate detections |
| `jumping-jacks-burpees.webm` | Taco Fleur / Cavemantraining, [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Lateral limb motion, jumping and floor transitions |

Source URLs, SHA-256 identities, and verified local timestamps are in each `.source.json`. The explicit `weft fetch-exercise-sample` command verifies existing files without redownloading them. Derived overlays retain attribution and the source license; the share-alike requirement applies to adaptations of the CC BY-SA videos, not automatically to unrelated Weft software.

Use these as development inputs. They were selected and inspected during implementation; none is a final holdout or a physical intervention test.

The flip-juggling clip is a separately acquired development source, not an entry
in the sample-fetch registry. Its `.mapping.json` preserves all 458 frame pairs
and native times; `.source.json` records attribution, changes and hashes. The
MP4 adaptation remains CC BY-SA 4.0. See the [portal acceptance](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/handoffs/visible-person-and-selected-muse.md).

The three additional clips vary exercise and setting, but share a creator/performer. They expand motion coverage, not demographic coverage. See [exercise-diversity findings](https://github.com/kintsugi-software/weft/blob/d0338eb926ef97ba05d59418ade028461ccbc9ae/docs/exercise-diversity.md) for preserved successes, failures and model limitations.

The separate [Autelica jog-turn case](autelica-jogturn.source.json) credits **Autelica (autelica.com)** under CC BY 4.0. Its unchanged video and reconstructed multiview reference are stored once in `artifacts/multiview-jog-source-01`; the source record points there without duplicating media. This moving-camera development case is not in the original 15-video benchmark or the sample-fetch registry. [Sandbox, reference limitations and results](../../docs/multiview-jog.md).
