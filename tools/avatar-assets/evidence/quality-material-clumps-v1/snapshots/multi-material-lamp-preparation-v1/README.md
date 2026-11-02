# Prepared multi-material lamp finish

**Prepared only.** The parent owns the one native execution and must explicitly run:

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python outputs/multi-material-lamp-preparation-v1/execute.py --execute
```

Source: `outputs/lamp-hd-build-v2/model.glb`, SHA256
`774eae69f51ae1a0cd49c61dd19abeab19b847ba8b72f393ad409b4582500949`.
New output: `outputs/multi-material-lamp-finish-v1`. Neither an existing output
nor an existing execution directory may be reused. No automatic retry.

## Exact scope

`finish_worker.py` calls the unmodified public
`avatar_assets.finish_generated.finish(config_path, output)`. Its sole native
runner call is transported to the parent. The parent checks the exact frozen
Blender argv, current directory, output/log paths and original requested limits,
then directly owns the real native child. The real runner receipt returns to
the public function unchanged; its original postprocessing, validators and
manifest writer still run. No validator or success result is substituted.

- Native: 120 seconds, 2,048 MiB sampled direct-child RSS, four threads,
  128 MiB output; original public limits are tightened, never widened.
- CPU worker: preflight and postprocessing each 30 seconds / 512 MiB. It is
  blocked while the parent runs native work. Combined result/log output is
  capped at 128 MiB, CPU log at 16 MiB. Each owned cleanup gets at most 7 seconds.
- The parent owns both children. There is no outer worker timeout wrapping a
  separately detached native child. SIGTERM/interrupt unwinds the native runner
  cleanup and then the CPU cleanup. Hard OS termination still requires inspecting
  the preserved PID/monitor records, never blind replay.
- Both processes receive a small local-tool environment, without provider or
  unrelated project credentials. No provider, AI-role, browser or server call.

`prepared.json` pins 99 source files, the original lamp/manifest/rig contract,
the sampling regression test, controller/config files and Blender executable.
The pinned scripts are copied for provenance; public finishing runs the same
live source bytes after action-time hash checks. The original `plan.json`
remains unchanged; `prepared.json` records the completed source freeze.

## Material and motion expectations

The source has two different base-color images and nine constant-color
materials, plus two normal images. Both mapped normal scales are implicitly 1,
so the recipe's `normal_strength:1` has no declared normal-scale delta.
`source-contract.json` retains every source material JSON object, image-byte
hash, texture/sampler binding, node, skin and four animation names.

After an actual successful finish, independently compare images or prove
decoded-pixel equivalence; compare base-color constants, metallic/roughness,
alpha and UV routes; and check named rig/rest transforms and sampled motion.
Importer/exporter repacking or resampling is not byte preservation without that
proof. The lazy sampling report must say skipped with no color-dependent
consumer. No shared texture is manufactured and no fur is requested.

Portrait and pose images are **256px integration drafts**, not the final 720px
quality evidence. Existing clips remain the four original authored lamp actions;
no action may be relabeled as newly generated. Even a public
`needs_visual_review` result is not a fresh avatar, quality approval, Studio
delivery, seamless loop or Weft compatibility claim.

## Receipts

Future `execution/` retains the start record, CPU PID/log, bounded IPC messages,
controller monitoring, native PID/monitor/log pointers, the exact native job
receipt and final `receipt.json` described by `receipt-schema.json`. Native output
keeps its own public `job.json` and manifest or preserved failure. A failed or
uncertain execution remains terminal for this preparation; never rerun it.

Preparation checks compile both controller files without executing them and
verify all frozen hashes. No actual native/CPU finisher run has occurred yet.
