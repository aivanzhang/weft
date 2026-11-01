# Equipment proposal configuration extraction

Recorded: 2026-11-01 01:15:05 PDT. Implemented and analytically verified; root owns integration/publication. No live job or source study.

## Stable API

Import `CONFIG_PROFILE`, `MAX_CONFIG_BYTES`, `MAX_EQUIPMENT`, `EquipmentProposalError` and `validate_proposal_config` from [equipment_proposal_config.py](../../src/weft/equipment_proposal_config.py). The module imports only `copy`, `math` and `re`; it performs no I/O, model admission, geometry or solver work.

[equipment_proposal.py](../../src/weft/equipment_proposal.py) re-exports the same objects under the existing public names. The unchanged contract remains `source-equipment-proposal-config-v1`, 64 KiB at caller byte admission, and at most 12 equipment objects. Validation returns a detached copy without normalizing supplied numeric types. This validator does not itself serialize or enforce the caller's byte cap.

The optional XY translation preserves the original list-only, two-coordinate, exact `int`/`float` (not boolean), finite `[0, 6]` checks. Invalid coordinates still raise plain `ValueError` with the original message; an all-zero vector raises `EquipmentProposalError` with code `invalid_frame_translation`. Existing incidental `TypeError` for an unhashable target and `OverflowError` for huge integers are intentionally unchanged.

## Custody and scope

- The proposer source archive adds `equipment_proposal_config.py`. Its other numerical functions/classes are AST-identical to the pre-extraction version; only the validator/error definitions move.
- [exercise_scene_inputs.py](../../src/weft/exercise_scene_inputs.py) imports the pure validator and archives the new module while retaining the old proposer entry. Its function/class bodies are unchanged.
- The guided configuration owner imports this module directly and pins it in that new workflow. Configuration readiness is not body, camera, floor, collision, contact or task admission.
- No historical source package, lab, policy or runtime is repinned. Existing archive identities remain exact; current code changes do not imply compatibility with an older frozen runtime.

## Verification

41 focused checks passed, zero failures/errors/skips, in 0.2245 s command wall time. [Tests](../../tests/test_equipment_proposal_config.py) compare the frozen original validator and original pure NumPy vector arithmetic with the extraction, including boundary types, maximum object/frame values, detached output, exception messages/codes, model-free import, alias identity and both snapshot dependencies.

Command: `.venv-lab/bin/python -m pytest -q tests/test_equipment_proposal_config.py --junitxml=.local/equipment-proposal-config-tests-01.xml`.

Receipt: [.local/equipment-proposal-config-checks-01.json](../../.local/equipment-proposal-config-checks-01.json), SHA `202041f4158da346115207c8ac0adc660f284f08e0daec95b086d62f805efbf2`. Original source snapshots are preserved under `.local/equipment-proposal-config-before-01.py` and `.local/equipment-proposal-scene-inputs-before-01.py`. No unchanged native suite, source decode, inference, simulation or API call ran.

Frozen product/test pins:

| File | SHA256 |
| --- | --- |
| `src/weft/equipment_proposal_config.py` | `b11f91380d878781f9362a2b3446ea1a3c072a06f7300a8a7d5942ccf220a3f7` |
| `src/weft/equipment_proposal.py` | `b8309206e00cc60a45f6a1f9c78872fe839971c23b0f4c9ef2c39cd3407be750` |
| `src/weft/exercise_scene_inputs.py` | `e8fed61e8dbab9ff4d9b7d922ce5d6fcd4584e3de890c02cd15810717980c59a` |
| `tests/test_equipment_proposal_config.py` | `c9e65586d0b789ef60732248abbe6d99a5adc958244210b4d9895aee1342f3f2` |

## Next safe action

Integrate the guided helper against these exports; independently review its selected source/frame/proof joins and truthful readiness labels. Reuse the 41 passing checks while these inputs remain unchanged. Root coordinates any separately planned end-to-end acceptance and Git publication; this extraction does not authorize a source or physics run.
