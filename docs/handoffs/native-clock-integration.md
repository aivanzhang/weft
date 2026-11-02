# Native-clock integration: next implementation

## Read-only diagnosis — November 2, 2026 (PST)

No native-clock fix is implemented by this note. The broad goal requires exact
source time and identity; nominal FPS currently introduces drift before pose/root
reconstruction. Native-clock support must not hide pose-quality failures.

## Coordinated change

1. Add a bounded versioned clock binding for source/evidence hashes, every decoded
   frame index, native integer ticks/time base, decoder times, signed constant
   origin and distinct container/decoded counts. Reject missing/reordered or
   nonmonotonic rows, drifting offsets and unsupported count discrepancies.
2. Add an optional binding to `prepare_exercise_body_source_from_evidence`.
   Preserve raw pose/landmark rows, status/identity/model timestamps and source
   bytes. Produce a separate derived reconstruction view linked by exact frame ID.
3. Feed that view into both spatial and root reconstruction **before** interpolation,
   gap checks, velocities or horizon calculation. Retiming final `human.json` is
   not sufficient. Existing runtime supports nonuniform target knots.
4. Version observation-clock validation to retain distinct metadata/decoded counts
   with the encoded-frame audit. Never set counts equal just to pass schema 3.
5. Update body/reference validators and shared-world clock construction together;
   distinguish native source timestamps from fixed physics timestep. Do not rewrite
   any completed recording, evidence package or scorer.
6. Integrate complete-frame clock collection into ordinary ingestion after the
   saved-table path is qualified. If detector VIDEO inference used nominal times,
   derived native-time reconstruction must preserve that fact.

## Exact current substitutions and tests

| File/function | Current constraint |
| --- | --- |
| `exercise_vision.inspect_exercise` | Detector receives rounded frame/FPS milliseconds; saved time is frame/FPS. Decoder time retained only on sampled rows. |
| `exercise_body_source._validated_pose` | Requires nominal row times and exports nominal `source-clock.json`. |
| `root_motion.build_root_reference` | Requires frame/FPS before root interpolation/gap checks. |
| `spatial_pose.build_spatial_reference`, `human.build_spatial_human` | Reconstruct from those nominal pose times. |
| `observations.adapt_exercise` | Nominal clocks; schema 3 full-time support still requires equal metadata/decoded counts. |
| `shared_world_human._scene`, `shared_world.validate_shared_world` | Explicit nominal clock method and fixed clock field set. |
| `whole_body_reference_inputs` | Revalidates nominal raw/source/reference clocks. |

Extend corresponding body-source, root-motion, observations, reference-input and
shared-world tests. Require legacy byte-equivalence, successful analytical
irregular-clock construction, constant-velocity preservation and actual-time gap
boundary checks. No repeated inference or silent solver/gate relaxation.

## Saved unfamiliar source

Reuse `.local/unfamiliar-exercise-input-20261102-01/native-clock.json` and
`.local/unfamiliar-exercise-body-01/decoder-clock.json` after verifying their saved
source/audit bindings. There are 1,107 decoded frames versus 1,113 metadata frames;
native ticks 0…37093 at 1/1000, six longer intervals, decoder time=native + 7 ms.
Nominal drift reaches 0.200635539 s. Preserve all of these distinctions.

This input still has 353/554 usable samples (63.7184%) and competing hypotheses;
it must fail the unchanged 65%/identity gates. Use it to qualify timing and rejection,
not to claim successful body reconstruction. Use an analytical irregular-time
fixture for the positive path, distinctly labeled as software evidence.

No `source_clock.py` was previously delivered. Constant rate/offset alignment cannot
restore irregular intervals. `source_equivalence.py` expects a derivative/YUV
bundle and excludes the required negative native-minus-decoder offset; inventing
such a bundle would be incorrect. Root owns implementation; the diagnosis agent
performed no edits, jobs, inference, provider calls or Git mutations.
