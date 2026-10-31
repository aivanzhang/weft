# Avatar authoring and browser library

The reusable avatar workflow now lives in [tools/avatar-assets](../tools/avatar-assets/README.md), in this repository rather than a separate checkout. It includes editable scene recipes, a local CLI, the reusable [authoring skill](../tools/avatar-assets/skills/avatar-asset-workflow/SKILL.md), detailed panda/dragon, robot, fox and lamp models, rendered previews, and provenance.

## Run locally

From the Weft root on the migrated machine:

```sh
tools/avatar-assets/run gallery
```

Open **http://127.0.0.1:8765/** yourself. The gallery starts paused, supports static/reduced-motion views, and loads its dependencies and assets locally. It does not start Weft physics. Use Ctrl+C to stop it. For a fresh checkout, follow the tool's pinned Python and browser setup; its separate environment avoids a Pillow version conflict with Weft.

## Preserved boundaries

- The detailed general/PBR models include real skin weights and four skeletal clips. Panda and fox have short geometric fur. They do not establish Muse-quality parity or recovery of the reference's unseen surfaces.
- These detailed models exceed the constrained Weft importer profile. Do not substitute them silently for the lower-detail matte models.
- The earlier [Weft-compatible panda and robot bundles](../tools/avatar-assets/docs/weft-delivery-v1.md) preserve explicit node IDs and mappings. Frozen-importer acceptance is recorded; actual simulation alignment remains unverified.
- The [consumer contract](avatars.md), importer, retargeting, physics, actions, rewards and recording are unchanged by this move. No new budget, provider call or deployment is involved.
- All historical local outputs and rejected attempts moved with the tool. Only the documented portable release is tracked in Git; runtime installations and intermediate frames remain local.

See the [relocation record](../tools/avatar-assets/docs/relocation.md), [handoff](../tools/avatar-assets/TODO.md), and [quality evidence](../tools/avatar-assets/docs/quality-pass-v2.md) for exact paths, commands, results and limits.
