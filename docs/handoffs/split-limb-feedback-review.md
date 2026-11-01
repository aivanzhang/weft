# Split-limb feedback integration review

Reviewed November 1, 2026, 3:16 AM PST. **No concrete blocker found** in the scoped integration. This was read-only code review plus inspection of existing check receipts: no tests, native models, simulation, source arrays, media, network or paid work were executed.

## Checked boundary

- Explicit `feedback-residual-split-limbs` selects policy v5 and its 5-by-2 coefficients. Training samples, saved CEM regeneration, action diagnostics, final policy and changed-start evaluation all use that same profile. The actor's 19 joint channels and optional three authored root channels are unchanged.
- Omitted profile remains v4 with 3-by-2 coefficients; legacy CLI defaults remain unchanged. Existing authored, coast and random baselines retain their arithmetic. Implementation/runtime hashes necessarily change; this review does not claim old archived packages have the new runtime identity.
- Original task/reference interval, independent scorer/gates, source custody, matched training starts, separate evaluation seeds, action budgets, residual bounds and saved-only selection remain unchanged. No evaluation results enter training.
- Static-support workflow/front-door combinations reject the new family. Camera tasks remain outside this task contract: the camera scene permits no dynamic equipment, while the legacy non-person-only task requires at least one dynamic object. No camera or support adapter was added.
- The public build/train routes add only the explicit profile choice. The split backend passes the profile to the existing learner; it does not replace the optimizer or shorten the task.

## Reviewed file identities

| File | SHA-256 |
| --- | --- |
| `src/weft/shared_exercise_feedback_learning.py` | `ddab3fc81ce41c49fa6c0b7bddb857be8893659834311850de1b7bba5c678ccd` |
| `src/weft/exercise_learning_pipeline.py` | `af7b53870e59dba672a24e61e8105835f6c5753f36497fcef09eacb4e3e579cd` |
| `src/weft/cli.py` | `def7198950db6a50b0cd3892af61b56f2bc5c341bba3e5c2812935acc6e9b876` |
| `scripts/build_sandbox.py` | `cd1d4efe284b6c79f56e3f397f84d0930e40382c697fa4de454edcf64819493e` |
| `src/weft/shared_exercise_feedback_residual.py` (actor dependency) | `3aefc6ef0e5be09fc3e710ffd277dc200c44bbc0b6051f852a13795f6a89268c` |
| `tests/test_split_limb_feedback_learning.py` | `b576cbf421f856b2962a232f7fe322626de55ecd251097ec49ceee7b9b5a1772` |
| `tests/test_feedback_workflow_options.py` | `cde81244718d125ade5ed065f021de724da950146bb889b17fdefa80c6f132d5` |

Existing root evidence was reused, not rerun:

- `.local/split-limb-feedback-root-tests-02.xml`: 42 passed, 1.885 s; SHA-256 `49f9c567605d25a7509bcc1fc1428811b6c4646489ae79f32c079f0b667cf10f`.
- `.local/split-limb-feedback-root-tests-03.xml`: two static-support rejection checks passed, 0.190 s; SHA-256 `843a3ec1e8a6fa98d287f4cd56c0019aeeff424555b64b3b3128c44bf63ee99c`.
- Root retains the first check attempt's fixture-only native-function denial and obsolete mock-environment failure. Neither is reclassified or deleted by this review.

## Ownership and next action

Only this review note was written. Root owns integration, frozen runtime preparation, native acceptance and publication; actor and UI owners retain their separate paths. There is no live review job or handle. Root may proceed with its separately bounded, frozen native acceptance. This review and the analytical checks do not establish learned improvement, physical transfer or unfamiliar-video reconstruction accuracy.
