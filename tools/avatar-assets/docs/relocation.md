# Relocation into Weft

The user authorized moving the work and current session into Weft, committing and pushing directly to `main`, and removing the old standalone checkout after verification. This supersedes the former ownership boundary; it does not replace Weft's goal or authorize physics changes.

Migration and cleanup are complete. Implementation commit `05eb853` was verified on private remote `main` before the former checkout was removed. The new CLI was checked again after deletion.

## Locations and preservation

- New tool root: `/Users/ivan/kintsugi/weft/tools/avatar-assets`.
- Former root: `/Users/ivan/kintsugi/avatar-assets`.
- All original outputs, rejected attempts, source snapshots and local runtimes move under the new tool root. The original Downloads ZIP stays unchanged.
- Historical absolute paths inside immutable manifests, logs and source snapshots retain their original spelling. Resolve the old root to the new root when inspecting them; do not rewrite hash-bound evidence.
- Source code, reusable skill, recipes, final detailed model packs, portable media, original reference, legacy Weft-compatible models/mappings and selected verification records are versioned in Weft. Blender, Python/Node dependencies and full intermediate frames remain local and ignored.
- The original standalone repository had no commits or remote. Its Git metadata is retained in a local migration backup, not embedded as a second repository.

`evidence/weft-migration-v1.json` records the relocation, file preservation checks and verification. The local migration receipts retain the full inventory; the remote receipt does not claim that ignored files are backed up remotely.

## Use

From the Weft root:

```sh
tools/avatar-assets/run gallery
```

Then open `http://127.0.0.1:8765/` yourself. Stop any old gallery terminal with Ctrl+C and start the command above; no persistent server is started during migration.

The tool environment stays separate from Weft's `.venv` and `.venv-lab`. Existing local dependencies are reused without paid generation or downloads. On another machine, follow the tool README's pinned Python and browser setup.

## Portable media

Portable media packs copy the existing portrait, encoded animations, fallback images and editable render settings without re-rendering. Their new manifests retain the exact source manifest and list omitted native frames/logs. The original full packs remain unchanged locally. This is a distribution subset, not new animation or a replacement of the original evidence.

### Create another portable media pack

```sh
.venv/bin/python scripts/package_portable_media.py outputs/my-media-v1 --out outputs/my-media-portable-v1
.venv/bin/python -m avatar_assets verify outputs/my-media-portable-v1
```

The packer verifies all original hashes, including omitted frames, before copying. It is bounded to 512 MiB input, 64 MiB output, 4,096 source files, 256 retained files and 60 seconds of cooperative runtime. Sixteen packaging tests cover tampering, oversized input, path/symlink attacks, mutation during copying and exact deterministic manifests. No generated asset is decoded or re-encoded.

## Integration boundary

The detailed fur/PBR editions are browser assets, not compliant replacements for Weft's constrained renderer. The older matte panda and robot deliverables retain their explicit mappings; their actual simulation alignment is still unverified. See [delivery details](weft-delivery-v1.md), [quality limits](quality-pass-v2.md), and Weft's [consumer contract](../../../docs/avatars.md).

Session assignment uses the installed app server's supported `thread/metadata/update` and `thread/settings/update` operations, not SQLite edits or a new conversation. The directory setting applies to subsequent turns; remaining migration commands use explicit paths.
