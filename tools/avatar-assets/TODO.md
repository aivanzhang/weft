# Avatar Studio roadmap

The next milestone is reliable motion across independently generated characters,
with consistent appearance and verified integration into Weft.

## Current status

- Selected saved-source workflows produce rigged assets and reviewed motion media.
- Separate-character requests support individual progress and downloads.
- Recent owl and rabbit source models generated successfully, but neither passed
  final rig/action validation. Generated geometry is not a completed animation.
- Source-repair helpers have component tests; their integration into the full
  Studio retry workflow remains incomplete.
- An asset that passes authoring review still needs qualification under Weft's
  simulation-driven motion.

## Priorities

| Priority | Completion criteria |
| --- | --- |
| General rig and motion repair | Repair incompatible geometry and weights without weakening deformation or action checks |
| Reliable continuation | Resume from verified source assets, preserving lineage, failed attempts, and cumulative costs |
| Independent characters | Separate models, rigs, actions, and downloads; explicit handling of unsupported fused sources |
| Export consistency | Requested motion remains clear in native renders, browser previews, and exported assets |
| Weft integration | Compatible mappings follow RL state without changing simulation controls, collisions, or rewards |

## Validation

For each candidate, retain the source identity and compare native and exported
results across representative poses. Require meaningful requested motion, no
severe deformation, working previews/downloads, and unchanged evaluation criteria.
Keep unsupported results visible rather than substituting a different action.

A completed milestone means independently controllable characters with reviewed
motion and verified consumer mappings—not only generated meshes or passing unit
tests. Automatic cast decomposition, synchronized interactions, and universal
model compatibility remain outside the current acceptance claim.

[Setup and usage](README.md) · [Weft architecture and status](../../PROJECT.md)
