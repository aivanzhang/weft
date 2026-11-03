# Project overview

Weft is a local development environment for building and testing simulations from
video evidence. It connects observation, reconstruction, interactive control and
evaluation while retaining the assumptions behind each result.

[Quick start](README.md#get-started) · [Sandbox setup](sandbox/README.md) ·
[Avatar Studio](tools/avatar-assets/README.md)

## How it works

```mermaid
flowchart LR
    Video[Video] --> Evidence[Observations and assumptions]
    Evidence --> World[Simulation]
    World --> Task[Actions and RL task]
    Task --> Evaluation[Independent evaluation]
    Evaluation --> Proposal[Bounded repair proposal]
    Proposal -. tested candidate .-> World
```

1. **Observe.** Decode the video and retain pose/object estimates, timestamps,
   confidence, missing observations and competing interpretations.
2. **Construct.** Build a supported world with explicit camera, geometry, contact
   and control assumptions. Unsupported inputs retain an inspectable failure.
3. **Interact.** Apply actions, reset state, select compatible avatars and record
   sessions. Learned control is available for supported task profiles.
4. **Evaluate.** Compare predictions and outcomes using fixed criteria. Meta
   models can contribute tracking, evidence review and bounded test selection;
   proposals do not automatically become accepted repairs.

Reinforcement learning (RL) means learning actions from feedback. Recorded pose
playback and programmed reference controllers are different modes, not learned policies.

## Capabilities and limitations

| Area | Available | Not yet established |
| --- | --- | --- |
| Video intake | Uploads, processing stages, evidence inspection and failure reports; transient competing pose hypotheses are screened instead of rejecting the clip | Multi-person videos (refused with reasons); reliable reconstruction across arbitrary videos |
| Simulation | Rolling, planar, rigid-object and articulated-body profiles; shared MuJoCo worlds | Accurate recovery of hidden depth, materials, grasp/contact and unassisted balance |
| Interaction | Manual controls, explicit tasks, reset, recording and action replay; disclosed root assistance | Physical accuracy of changed-action predictions |
| Learning | Bounded training, baselines and evaluation on supported profiles | Consistent improvement over simple controllers |
| Meta integration | SAM tracking, Muse review and constrained controller-choice tests | General autonomous physical repair |
| Avatars | Built-in appearances, compatible-rig imports and state-driven motion | Automatic compatibility for arbitrary models; Studio skin weights (bounded repairs attempted and rejected; robot remains the qualified avatar) |

Intake supports videos up to **60 seconds, 1,800 frames and 100 MB**, with further
checks for each reconstruction path. Keep observed, inferred, assumed and unknown
properties distinct. Illustrative muscles and assisted body controls are not
measured muscle activity, recovered anatomy or validated biomechanics.

## Evaluation snapshot

Development observations from November 3, 2026; these are small, scoped studies,
not an external benchmark or a guarantee for new inputs.

- Rerunning the frozen three-video evaluation after the transient-hypothesis screen:
  **2/3 unfamiliar videos now complete the full pipeline** (admission, body, world,
  lab and a 450+ step playable rollout). The multi-person clip stays refused with
  recorded reasons. Evidence: `.local/hackathon-unseen-20261103-05/`.
- With disclosed three-axis root assistance (the primary route's default), the
  recorded demo world and the new sots-press video **pass every frozen task gate**
  (root tracking 0.018 vs the 0.15 gate). Unassisted tracking still fails and
  remains the honest hard mode. Evidence: `.local/tracking-study-01/`.
- The overhead-squat video completes and replays but fails one gate (peak angular
  1.00 vs 0.70): its squat-bottom chart transition is ill-conditioned. A bounded
  joint-speed screen now bridges single-sample reference spikes with audited
  zero-confidence proposals; the residual near-bound transition is preserved.
- A real Muse-selected controller candidate was tested and rejected under the
  unchanged scoring rules. No accepted physical repair was demonstrated.
- Three bounded Studio-panda skin-weight repairs (filter, re-authoring, probe-guided
  consensus) all failed posed-stretch validation; the robot remains the qualified
  avatar. Tooling and evidence: `tools/avatar-assets/avatar_assets/skin_weight_repair.py`,
  `.local/studio-skin-repair-01/NOTES.md`.
- In a separate ball task, a learned policy passed 8/12 changed starts while
  a fixed feedback controller passed 12/12. This did not show a learning advantage.

Exact replay verifies software consistency. It does not, by itself, validate the
reconstructed physics. Evaluation inputs must remain separate from tuning data.

## Architecture

| Location | Responsibility |
| --- | --- |
| `src/weft/` | Evidence processing, reconstruction, simulation and task evaluation |
| `src/weft/assets/` | Portal, lab viewer and avatar rendering |
| `scripts/` / `tests/` | Reproducible studies and contract checks |
| `website/` / `sandbox/` | Independent landing-page and sandbox entry points |
| `tools/avatar-assets/` | Separate character-authoring environment |
| `hosting/` | Hosting implementation and deployment configuration |

The landing page uses Node.js; the sandbox uses an isolated Python dependency
environment. This is process/dependency separation, not a container security
boundary. API keys stay server-side, and paid operations use persistent spending
ledgers. See [API configuration](sandbox/README.md#api-configuration).

## Roadmap

- Extend admission beyond single-person clips and qualify unassisted tracking;
  resolve the ill-conditioned chart transitions behind the overhead-squat near-miss.
- Constrain camera, scale, geometry and equipment contact using stronger evidence.
- Re-skin the Studio avatar against its existing rig in the authoring environment;
  bounded weight-space repairs are exhausted and documented.
- Demonstrate useful learning and supported repairs against fixed independent baselines.
- Improve end-to-end usability, reproducibility and resource measurements.

## Development

Run checks that cover the component being changed:

```sh
npm run test:local
node sandbox/run.mjs --check
.venv-lab/bin/python -m pytest tests/<affected_test>.py
```

Keep recorded runs immutable, retain failed attempts, and use new output directories
for new experiments. Local project data is Git-ignored: back up needed recordings,
credentials and spending history separately. Dataset and asset attribution remain
with their source records.
