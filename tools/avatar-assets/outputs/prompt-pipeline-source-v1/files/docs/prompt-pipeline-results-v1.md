# Prompt pipeline: actual results v1

Evidence through **2026-10-31 22:25:50 PDT**. Paths below are relative to `tools/avatar-assets/` unless stated otherwise.

## Result and scope

Two different subjects completed the prompt → planner → rig/motion author → native build/reload → separate visual critic → bounded repair → media → portable pack → browser workflow. Both started with **previously generated, unrigged Meshy geometry**, not a hand-authored finishing recipe. These runs verify automatic downstream authoring and delivery from text plus an existing mesh; they do **not** test a new paid text-to-Meshy request in the same run.

| Delivered result | Dragon | Robot |
| --- | --- | --- |
| Immutable job | [agent-dragon-v3](../outputs/agent-dragon-v3/delivery.json) | [agent-robot-v3](../outputs/agent-robot-v3/delivery.json) |
| Accepted attempt, zero-based | 2, after two repairs | 1, after one repair |
| Real skinned GLB | [model-2/model.glb](../outputs/agent-dragon-v3/model-2/model.glb) | [model-1/model.glb](../outputs/agent-robot-v3/model-1/model.glb) |
| Editable source | [source.blend](../outputs/agent-dragon-v3/model-2/source.blend) | [source.blend](../outputs/agent-robot-v3/model-1/source.blend) |
| Exported vertices / bones | 36,322 / 9 | 45,386 / 11 |
| Four real 3D clips, each 4 s | `idle`, `look_around`, `wing_flap`, `milestone_level_up` | `idle`, `look`, `wave`, `celebrate` |
| Media preset | `preview`: 720×720, 4 s, 96 frames/clip | `draft`: 256×256, 2 s, 48 frames/clip; full 4 s curves retimed |
| Portrait | Opaque RGB WebP, 1600×1600 | Opaque RGB WebP, 512×512 |
| Compact media | [media-portable](../outputs/agent-dragon-v3/media-portable/manifest.json) | [media-portable](../outputs/agent-robot-v3/media-portable/manifest.json) |
| Actual browser test | [Passed, 11.760 s](../outputs/browser-agent-dragon-v1/browser-test.json) | [Passed, 10.610 s](../outputs/browser-agent-robot-v1/browser-test.json) |

Every clip is silent H.264 MP4 at 24 fps, with measured image changes, a first-frame static fallback and retained source PNGs. Loop and transparency claims remain explicit: videos/portraits are opaque, and seamless loops are **not certified**. The files are not repeated stills or stock animations. Full sources and rejected attempts remain local; portable packaging omits full frame sequences without deleting or re-encoding them.

## Generation versus local work

The source inspections report **zero skins and zero animations**. Each source has two 4096×4096 JPEG texture images and one 2048×2048 image. Source resolution is not proof of visual quality, and the robot's draft media is not an HD render.

| Reused generated source | SHA-256 |
| --- | --- |
| [Dragon raw GLB](../outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb) | `a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb` |
| [Robot raw GLB](../outputs/workshop-robot-meshy-v1/stages/texture/downloads/model.glb) | `5f00602dd727429277ae53c0e8940ba9162ea119e643ba554f081c00cc1b7e8c` |

Meshy supplied the earlier shape and textures. The new agents proposed subject-specific bones, skin-weight regions and action curves. Trusted local code validated those proposals, built real skins, preserved prompts/configurations, independently reloaded exported GLBs, rendered media and enforced the critic's result. Neither successful run received a manually written finishing recipe.

The planner, author and critic use separate ephemeral `codex-cli 0.166.0` calls. Tool execution, browser access, plugins, MCP servers, web search and nested agents are disabled; responses are schema-constrained data. Successful receipts record `tool_activity: false`. Native programs run through the existing bounded runner. The request left `agent_model` null, so these records identify the CLI/default selection but **do not pin an underlying model version**. Exact agent prompts, answers, usage, process limits and thread handles are in each job's `agents/ROLE/receipt.json`, `prompt.txt`, `answer.json` and invocation files.

## Preserved experiments and actual usage

All six job directories remain unchanged. Wall time below is the saved request creation → last terminal/checkpoint timestamp, not summed CPU time. Agent seconds sum actual call-process receipts. The failed v1 parser call's usage comes from its retained `turn.completed` event, not a successful receipt.

| Job | Terminal result | Calls | Input / output tokens | Agent seconds | Job wall seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| [agent-dragon-v1](../outputs/agent-dragon-v1/status.json) | Checkpointed adapter/parser error; no avatar delivered | 1 | 15,074 / 487 | 13.470 | 13.614 |
| [agent-dragon-v2](../outputs/agent-dragon-v2/unsupported.json) | Unsupported: insufficient wing skin motion and repair evidence | 3 | 53,415 / 2,484 | 43.771 | 60.540 |
| [agent-dragon-v3](../outputs/agent-dragon-v3/delivery.json) | Complete, critic accepted attempt 2 | 7 | 186,779 / 9,115 | 152.240 | 757.159 |
| [agent-robot-v1](../outputs/agent-robot-v1/unsupported.json) | Unsupported after bounded repair; motion/deformation not accepted | 5 | 104,925 / 5,906 | 101.722 | 137.100 |
| [agent-robot-v2](../outputs/agent-robot-v2/unsupported.json) | Unsupported after two repairs; wrist strip still stretched | 7 | 179,583 / 7,734 | 135.330 | 190.132 |
| [agent-robot-v3](../outputs/agent-robot-v3/delivery.json) | Complete, critic accepted attempt 1 | 5 | 127,090 / 5,475 | 94.021 | 288.304 |

Aggregate: **28 real agent calls, 666,866 reported input tokens and 31,201 output tokens**; 16,640 cached-input and 1,638 reasoning-output tokens are separately reported fields, not additional totals. Summed agent-process time is 540.554 s. Account-specific AI dollar charges are unknown; these calls are not described as free.

These experiments made **zero new Meshy or Meta calls**. The canonical Meshy ledger remains at the earlier 15 submissions / 167 reported and reserved credits. No account, subscription, purchase, deployment or recurring server was created. Each successful v3 request had `max_credits: 0`, `max_agent_calls: 7`, `max_repairs: 2`, and a 240 s per-agent deadline. A new job is not a renewed paid allowance.

The initial dragon v1 parser rejected a runtime notification as tool activity; the original failed/uncertain receipt remains preserved and was not silently replayed or rewritten as success. Dragon v2 had only 13 positive-weight vertices per wing and failed motion validation. Robot v1's uniform five snapshots missed alternating wrist extremes; robot v2's improved evidence exposed the remaining defect. These failures are part of the result, not accepted assets.

## Improvements exercised by the later jobs

- **Coordinate guides:** normalized front X/Z and side Y/Z grids use exact imported mesh bounds, including tails/horns. Native camera projection is checked against the annotation coordinates. Bone placement remains an authored assumption.
- **Actual key-pose evidence:** up to eight authored key poses replace fixed integer-time snapshots. Larger designs use bounded pose-error sampling. This changes evidence selection, not the motion curves.
- **Actionable repairs:** author retries receive the previous rejected design, per-state skin-motion measurements, weight coverage and contact sheets. Robot v3 also receives stretched-edge positions and bone influences, rather than only a generic distortion warning.
- **Fail-closed orchestration:** cached inputs/stages are hash-bound; agent uncertainty does not trigger automatic paid retries; failed stages and reviews remain immutable; native process exit must be resolved before repair. Separate author and critic roles do not constitute independent human approval.

### Robot deformation: accepted does not mean zero error

The [read-only wrist diagnostic](../outputs/robot-wrist-diagnostic-v1/report.json) examined robot v2's exported mesh at all 97 wave keys. It found 27 root-weighted vertices on 52 root/hand boundary edges; the worst measured edge grew by **0.159200 m**, from 0.004217 m to 0.163417 m. Its source GLB/config hashes and original normalized coordinates are recorded, and input bytes remained unchanged.

Robot v3's accepted [finish report](../outputs/agent-robot-v3/model-1/finish-report.json) still records a **0.0226335 m sampled excess edge length** during `wave`, across `Torso`/`UpperArmR` influences. The critic found no *major visible* distortion in the supplied images; it did not prove zero deformation. V2's all-key diagnostic and v3's sampled authored poses are different measurements, so these numbers are **not a matched percentage-improvement benchmark**.

## Native, browser and resume verification

| Measured stage | Dragon v3 | Robot v3 |
| --- | ---: | ---: |
| First planner event → delivery event | 757.037716 s | 195.415325 s |
| Accepted native build | 12.173 s; 1,204,420,608 B sampled peak RSS | 13.461 s; 1,248,362,496 B |
| Independent native GLB reload | 1.565 s; 561,463,296 B | 1.564 s; 560,119,808 B |
| Full media render | 534.994 s; 1,177,485,312 B | 55.215 s; 640,729,088 B |
| Video/media packaging | 8.819 s | 1.046 s |
| Browser initial library readiness | 2.455 s | 2.316 s |
| Browser test sampled process-tree RSS | 2,696,626,176 B | 2,735,898,624 B |

The recorded execution-stage span differs from request-creation-to-terminal time above: the robot's prepared request existed before its run started. Neither measure is a hardware-independent latency guarantee.

Native jobs exited successfully with `process_exit_verified: true`; neither media run needed a frame continuation. Browser tests loaded the actual GLB and every declared clip, played/paused/sought each animation and video, decoded portraits and state fallbacks, checked reduced motion without GLB/video fetching, injected GLB/video failures, and checked a 390 px layout. Both report **zero unexpected errors and zero external requests**, with owned browser/server exit verified. Expected injected failures remain in their reports. Chrome 156.0.8078.76 used software WebGL; these short observations are not sustained FPS results. Browser RSS is summed sampled process-tree RSS and can double-count shared pages.

Completed-job [resume evidence](../outputs/agent-resume-proof-v1/manifest.json) reports `resumed_without_generation: true` and zero generation calls: dragon **0.744259 s** as observed by the original command tool; robot **0.592917 s**, with all five agent receipt hashes unchanged. The dragon observation was recorded without rerunning an already passing check.

The [v3 local test log](../outputs/prompt-workflow-tests-v3.txt) records **323 tests passed in 12.545 s**. Those analytic/schema/safety fixtures test software gates, not generated characters, visual quality or physical validity. Actual model/media/browser evidence above is separate.

## Exact commands

From the Weft repository root, view the two completed jobs without a provider call:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-agent-v1.json
```

Open `http://127.0.0.1:8765/` yourself; the command starts a user-managed local server. The combined catalog preserves the completed jobs' exact asset paths and changes only display titles. Individual catalogs remain in each job.

The following request-equivalent CLI commands were reconstructed directly from the saved v3 `request.json` values, **not from hand-authored rig recipes**. Their historical output directories already exist: to make a new run, change only `--out` to a new unused directory. A new run consumes new agent usage. Add `--prepare-only` to inspect its immutable request/estimate first without calling an agent or provider.

```sh
./tools/avatar-assets/run generate \
  --prompt 'Create a friendly upright emerald dragon with amber eyes, orange horns and back spines, a cream belly, two small wings and a long tail. Keep the scaled body intact and animate gentle idle, looking around, standing wing flaps and a happy bow with a tail flourish. No flight physics or facial animation.' \
  --style 'Rounded friendly stylized 3D dragon with emerald scales, amber eyes, orange horns and a cream belly' \
  --states idle,look_around,wing_flap,milestone_level_up \
  --reuse-model outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb \
  --preset preview --max-credits 0 --max-repairs 2 --max-agent-calls 7 --agent-timeout 240 \
  --out outputs/agent-dragon-v3

./tools/avatar-assets/run generate \
  --prompt 'Create a friendly blue and cream workshop robot with copper metal details, a rounded torso, two arms and two planted feet. Animate a gentle idle, looking left and right, a small friendly hand wave and a modest celebratory nod. Preserve the existing hard-surface design; no props, facial animation or walking.' \
  --reuse-model outputs/workshop-robot-meshy-v1/stages/texture/downloads/model.glb \
  --preset draft --max-credits 0 --max-repairs 2 --max-agent-calls 7 --agent-timeout 240 \
  --out outputs/agent-robot-v3
```

Robot state IDs/style were chosen by its planner because the saved request leaves those inputs empty. Its generated actions were not substituted with the dragon's named states.

These exact completed-job resume commands make no new model/provider calls while the pinned backend and outputs are unchanged:

```sh
./tools/avatar-assets/run generate --resume outputs/agent-dragon-v3
./tools/avatar-assets/run generate --resume outputs/agent-robot-v3
```

Historical browser commands, run from `tools/avatar-assets/`; use new output names for a later verification:

```sh
node browser/tests/pipeline.mjs outputs/browser-agent-dragon-v1 outputs/agent-dragon-v3/gallery.json
node browser/tests/pipeline.mjs outputs/browser-agent-robot-v1 outputs/agent-robot-v3/gallery.json
```

## Canonical identities and remaining limits

| Artifact | SHA-256 |
| --- | --- |
| Dragon delivery | `0a897035f23b5d68aae87ffea2756ec119974cd7a7692fef0984789d387980cc` |
| Dragon selected GLB | `71484ae9183e78415096fc252f5c58d00eeecf8a6bca4a8d472b9ba60789de00` |
| Dragon selected model manifest | `34f696204794cd7f048862b697172608f82a12708261d1d14ce3c93b8607e954` |
| Dragon media manifest | `419f90ffbf4b4ff150a68eab57e7acee666b8812846d967b2da65547aac8d120` |
| Robot delivery | `487ce897f5eba38387a976904cb66ccb678cfceeaea7d0e9fd810b50b5b178f1` |
| Robot selected GLB | `e2e1c4e73a83773673a9343d3ae0baeddea84dbace9ac61ac7ee203a9ec89d3e` |
| Robot selected model manifest | `61cd4572faea439b6ad4e5ce12f3e08f8c5b140a72f0eeebd4bfe8bc64e764f6` |
| Robot media manifest | `6f55749827926a45c3a368e1b05f5eb5660b47af38038035bcbebac7d8ba2c62` |

Each delivery receipt binds the delivery, plan, catalog and selected review. Pack manifests bind their artifacts, including editable sources and copied backend code. Exported node IDs/rest transforms and local bone controls are retained in each selected model's `exported-nodes.json` and `rig-contract.json`; they are **not Weft joint mappings**.

### Selected rig IDs and rest pose

These are the independently AI-authored v3 rigs, not the earlier assisted 19-bone dragon or provider robot rig. Both exports use meters, glTF **+Y up / +Z forward**, right-handed. Their design coordinates use normalized Blender **+Z up / −Y forward** bounds. Rest/bind pose is the default exported node hierarchy plus inverse bind matrices, not a recovered anatomical neutral pose. Preserve all parent transforms and rotations when loading.

| Selected export | Authored height | Critical node IDs | Skin 0 joint order |
| --- | --- | --- | --- |
| Dragon | 1.6 m | Mesh `9`; armature `10`; root bone `8` | `[8,6,1,0,2,3,4,5,7]` |
| Robot | 1.2 m | Mesh `11`; armature `12`; root bone `10` | `[10,7,0,3,2,1,6,5,4,8,9]` |

Dragon bone node IDs: `Head=0`, `Neck=1`, `ArmLeft=2`, `ArmRight=3`, `WingLeft=4`, `WingRight=5`, `Chest=6`, `Tail=7`, `Root=8`. [Exact nodes/transforms](../outputs/agent-dragon-v3/model-2/exported-nodes.json), [rig contract](../outputs/agent-dragon-v3/model-2/rig-contract.json), [agent-authored design](../outputs/agent-dragon-v3/model-2/sources/finish-design.json).

Robot bone node IDs: `Head=0`, `HandL=1`, `ForearmL=2`, `UpperArmL=3`, `HandR=4`, `ForearmR=5`, `UpperArmR=6`, `Torso=7`, `LegL=8`, `LegR=9`, `Root=10`. [Exact nodes/transforms](../outputs/agent-robot-v3/model-1/exported-nodes.json), [rig contract](../outputs/agent-robot-v3/model-1/rig-contract.json), [agent-authored design](../outputs/agent-robot-v3/model-1/sources/finish-design.json).

`JOINTS_0` indexes the listed skin-joint array; it is not a direct node ID. Both skin records use inverse-bind accessor `7`. These contracts map bone names to local controls only, with no Weft body-frame mapping. The dragon is standalone, not the Muse panda's shoulder attachment; no such attachment is delivered here.

The result is a reusable bounded pipeline with two actual delivered examples, not universal prompt success. The records explicitly leave `human_approval`, `quality_equivalence_verified`, `weft_runtime_verified` and `loop_verified` false. Hidden geometry, scale, bone placement and skin weights remain authored/generated assumptions; rights are provenance records, not independent commercial-rights verification.

No locomotion, flight/contact physics, recovered fingers, facial animation, interactive props or hair dynamics was demonstrated. Neither a scaled dragon nor a metal robot requires fur, so these tests do **not** establish the original Muse panda/fur-quality target. Small-view critic acceptance does not settle close-up quality, all-frame collisions or zero stretch. The far dragon wing is partly occluded; the robot critic also notes that exact preservation of its original source was not independently established by its review images. These HD/PBR models exceed Weft's 20k-vertex importer limit, and no faithful unchanged-fidelity Weft conversion is claimed.

Next safe action: inspect the completed combined gallery. A new subject uses a new prepared job; fresh Meshy generation needs its own authorized ledger allowance. Preserve these outputs and unsupported attempts. Local hashes are not a remote backup or publication.
