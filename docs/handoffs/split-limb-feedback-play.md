# Split arm/leg feedback: bounded lab acceptance

## State

**The separate fresh-process resume passed.** Attempt05 had completed controls and exact export/replay, then failed during its second load; that whole attempt remains failed. Its completed actions/export were not repeated. The existing lab can now be opened with the exact frozen runtime below. No browser, server, native operation, producer or training job ran in this preparation lane. All failed attempts and partial outputs remain preserved.

The existing native comparison is terminal and reaped: `artifacts/split-limb-feedback-native-01`, outer seal `cb6249e97ec4208507ab5d11685113cb925a41a7b0b8e822c44e1bbcc0cb4e1e`. Root's custody audit is `.local/split-limb-feedback-native-01/terminal-custody-audit.json`, SHA `cd553ecb66364af2355172e0eed9a3c37cdf45576d8ef7e92656df1b4dbef53e`; it is reused, not repeated.

All four evaluated controllers passed **0/3** starts. The learned controller completed 3/3 intervals but its reported mean reward was 0.745110 versus authored 0.747771. This play check does not promote that policy or perform another evaluation. Full task failure remains visible even if interaction/export/replay all work.

## Exact preparation

- Worker: `.local/split-limb-feedback-play-01/worker.py`, SHA `67b692ad3a90ca16ff08e680997298aed6419a64c22be17f51953a1e340c7455`.
- Released review plan: `plan-draft-03.json` in that directory, SHA `3be988da294a83d236236e87f109a2603b42f046fad84708ce743f6796c9ca7a`. That draft remains unmodified; root's subsequent frozen execution plans are separate.
- New execution runtime: `runtime/manifest.json`, SHA `67f99aece68eed6b4088a226aa58491d5e3a212a89da15d8a40beaba306e7f77`: 291 files, 6,312,490 bytes. It copies 290 exact parent members and changes only `shared_exercise_lab.py` to root's reviewed `41059ca8a3e9a1a7882089560668e4090e7dc7fa0f13fa2a356a0094383ce36a` packaging fix. **All 25 policy-runtime files remain exact.** Original runtime `e1dcc006…` and all original results are untouched.
- Original construction: `artifacts/fullspan-exercise-source-01`, seal `f300b46cfb71697ed2ec4cdf4746c96ceda7bf0995a77412f0835d274b6d055f`.
- Selected policy: native comparison `selected/`, seal `e3e13a5ab6a3ddfd3c989b2406e5fc97afb623d76400716795e862565b5379f3`; exact policy `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`.
- Evaluation: comparison `evaluation/`, seal `6834d7590a207874833225be0945eae8a986a78d4698a9fedcdf6f73f4357f95`.

### Attempt04 runtime candidate

`.local/split-limb-feedback-play-04/runtime` is a new exact copy of play01's snapshot, with only `shared_exercise_lab.py` replaced by root's `382f2bd3dec89d87bbf2446261654d345fc88cee796d9d1bf4769b5cde6cdb1c`. Its manifest is `6acc810d058c7bf422110e7f411738265d259ff4c6d11728705670318fa034f5`, covering **291 files / 6,312,858 bytes**. All 25 policy-runtime hashes remain identical. Old snapshots were not mutated.

`worker.py` in preparation04 is byte-identical to the reviewed worker above. It already verifies the exact plan/manifest byte count, so no hardcoded-size or protocol change is needed. The preparation receipt is `.local/split-limb-feedback-play-04/runtime-preparation.json`, SHA `030d17c55b969deb311fcfec29f5bf015276dcf337eb6c0976752c989465a6e7`: 293 bounded reads / 6,384,819 bytes, 0.033916 seconds, AST validation only, no native/source work. That receipt covers preparation only; root's subsequent attempt04 result is recorded below.

### Released attempt05 preparation

Root executed runtime04 afterward; its completed public assembly can be opened independently of the failed combined process. Preparation05 reuses **the identical runtime04** without copying or changing it:

- `.local/split-limb-feedback-play-05/worker.py`: `009b6b376819ece827273b82bf35b2dec2474ca847b51ef744c9ef02dba5e468`.
- `plan-draft-01.json`: `6a83066b26fdb51b7565d87d1baf216e3167d6b749e315dacec132e9eff98818`; deliberately unexecutable pending root review/freeze.
- `preparation-01.json`: `3c9b062c0d8234bf376f5c45f92f1b2bb0f2639af99bf7a0e9177687b08558ec`; worker AST parsed, 17 existing guard/counter/helper ASTs unchanged, no copy or worker execution.

The worker pins attempt04's ready run, three lifecycle files and original `prepare.json`. It uses the existing no-follow, exclusive, 1 MiB streaming `copy_package_file` to copy **1,195 lab files / 173,861,851 bytes** into new `artifacts/split-limb-feedback-play-05/lab`. Every manifest-listed member must match its SHA, regular-file identity and size, with a 64 MiB member bound and exact aggregate inventory. The original `prepare.json` is retained verbatim as provenance; a separate `copy-ready-lab.json` describes the new copy, not a second assembly. No payload-byte map is retained. The unchanged `Lab` loader then validates the copy before controls.

Ten flushed stdout checkpoints bracket copy, initial summary, live actions, export and resume. Each records phase, elapsed time, normalized process peak RSS and attempted/returned counters; maximum 12 checkpoints / 32 KiB. The complete journal display objects are released before resume. Bounds, frozen source/task/policy/scorer and action counts are unchanged. This matches the ordinary build-then-open process lifecycle; it is not evidence of lower memory use until measured.

`ready-metadata-01.json` pins the five selected ready documents (166,135 bytes), not source payloads. A preparation display script initially treated `prepare.json.files` as a member map rather than a count; `preparation-inspection-failure-01.json` preserves that checker error. Four small metadata files were reread once to cache their exact bytes; the run manifest was first read in the continuation. No failed producer/native result was rewritten.

### Completed read-only resume

Root executed `.local/split-limb-feedback-resume-01/worker-v2.py` (`43cd1b29fe28742f12928a3985d56c24a3146121124022295e5f8032f78503d6`) after independent review. Actual frozen plan: `0620d4bfe4669f140b2f1bb1322562dcd18b7b8e30e3c399deabdd6d92b2a95f`. Earlier draft02 (`ccac6ae29bd3bd6748781a3b6bed8baebacdeeb5cd7fa5edb45f6108d8d1c394`), preparation receipt02 (`b77bfab4c5e3d2383368ed752f86686431533306549d275f9c8d94e5d03b172e`) and first worker/draft remain preserved. V2 rejects changed existing stat identities before reading. Runtime04 and the supervisor were unchanged; this lane performed preparation/AST checks only.

The new process opens **the existing attempt05 lab read-only once**, obtains its actual summary, and checks the saved recording receipt. Initial admission is bounded to the previously observed 28 model constructions, 1 data object, 2 forwards and 1 reset; physics steps, controls, replay, export, decoding and training are forbidden. Ordinary `Lab.summary` admission verifies the run, with exact before/after manifest inventory and stat checks; there is no redundant 173 MB source hash pass. The saved receipt helper verifies the scene/recording/journal/evidence. Only `video.mp4`, `preview.gif` and `replay.html` receive one additional streaming hash pass, with no decoding.

The separate `artifacts/split-limb-feedback-resume-01` contains the proof and resumed summary: **5 files / 849,450 bytes**, seal `5616e03a64afd55b2639f6915a562a10caa15101e92097a65585e9d7a40bea9b`. Result SHA: `5c1b76f8034a12b4fc9884c6cf47da11c628c189b5baf44cf5f984120c361f92`. Root reports exit0/reaped in **2.885009750 seconds**, peak RSS **399,474,688 bytes**, under the unchanged 512 MiB ceiling. Process receipt SHA: `95c877eb638d33fc923fc7d1379dcef43228d125c41e26c445a9aa1000c1f538`.

Observed work was exactly **28 models, 1 data object, 2 forwards and 1 reset; zero new controls, substeps, replay, decoding or forbidden calls**. The verified run contains **1,192 files / 173,860,491 bytes**; recording inventory contains **9 files / 19,843,650 bytes**, covering all six sealed recording members plus manifests/evidence. Source, controller and task remained unchanged. Root's three-file / 17,036-byte terminal metadata audit is `b1b049b460316d4f173d6cd628f446de87f4278423ed7a8ca62aad056b3e6c48`. These terminal facts are root-reported; this lane did not reread artifacts. Attempt05 is not relabeled as a passing whole run, and no outer05 seal is invented.

Preparation metadata is pinned in `metadata-02.json` (`31337e6255f996cc25bec65b960a1b3a34aaaf6d8700592d9dabbf5f80100f6b`). A checker incorrectly compared access time as part of file identity; its failure is preserved in `metadata-failure-01.json`. Recovery uses device/inode/size/mtime/ctime. Actual reads were **2,685,334 bytes from original metadata plus 1,654,152 bytes reread from the local display cache**, exceeding the amended 3 MiB preparation limit. Root recorded that efficiency miss and allowed only 256 KiB further small metadata; 35,561 bytes were used through draft02. No large capture was reread afterward, and no source payload or native work occurred.

`input-metadata-01.json` records the six selected metadata reads: 893,885 bytes. The source scene/task metadata and 192-member stat inventory come from the existing native preparation, not a new source read. The worker checks those source stats before and after, without rereading source payloads for that guard. Ordinary public admission still hashes/copies exact opaque source media as required; decoding source media remains forbidden.

## Protocol and bounds

The local worker adapts `scripts/studies/check_shared_exercise_policy_lab.py` and the prior person-feedback guard; it does not execute that older 120-second driver. Attempt04 completed public `shared-exercise-lab` with original construction, selected policy, exact task and independent evaluation. Attempt05 reuses that sealed assembly and exercises the existing `Lab` control methods in a fresh process; it does not assemble or train again.

At seed **29**: record 3 manual actions (motor index 0 at +0.1), 3 coast, 3 authored, 3 learned; reset once to the identical nominal start; then complete the original **254-step** learned interval. Expected totals: **266 live + 266 replay controls, 257 learned actions, 1 reset, 268 events, 8,512 native substeps**. Equipment-force commands remain zero. The exporter owns the sole exact action/state replay; the helper does not replay again.

Maximums remain 600 controls, 9,600 substeps, 50 seconds cooperative work, 55 CPU seconds and **60 seconds total supervised time including cleanup**, 400 MiB output, 2,000 files, one worker and one export thread. The initial 400 MiB child-RSS estimate was insufficient during assembly; root authorized **512 MiB for attempt03**, matching the completed native comparison ceiling. This is a measured allowance correction, not a memory optimization. Root enforces RSS, output and process-group deadline/reaping. No internal retries or automatic shortened interval are allowed.

The exact existing descriptor-aware `file_guard.py` from native preparation restricts writes to the new owned output. Native counters charge positional/keyword `nstep` before dispatch and separately count returns. Every loaded Weft module must match the already hashed frozen runtime and its unchanged file identity. Final output hashing/sealing is deadline-checked; final size/file counts include the manifest. Worker timing before sealing is explicitly labeled; root's process elapsed is authoritative.

Existing export requires one output-only writer/capture at the new session's `video.partial.mp4`: at most 268 frames, 269 verification reads and 8 MiB per decoded frame. This is necessary output verification, not source decoding. Source image/video decode, inference, native renderer, network/API, training/evaluation, avatars and old artifact mutation are denied. Partial export/failure evidence remains preserved.

## Packaging finding and retained failures

`packaging-preflight-02.json` (SHA `b055533efe6f2290f6a2e29d6b37bc9af2b142439ed70bfc4fff08c9f167640d`) found a mandatory lower bound of **1,151 members / 146,617,504 bytes**, already above the old 1,024-file cap. The full static union is 1,192 run files including its manifest, plus 3 outer lab files. Fixed stored bytes are 168,773,639 before root's 90-byte comment/cap delta; generated contract/source verification/121-reference metadata and manifests remain uncomputed. About 32.55 MB remains under the unchanged 192 MiB lab byte cap.

Root changed only the lab file cap to **1,536**, with three reported targeted passes: 1,035-file prepare/resume, 1,545-file rejection and unchanged byte-limit rejection. No policy/scorer/source repinning or retraining is required. The initial selected-as-source draft is preserved unexecuted; it duplicated the training package unnecessarily. Original-source drafts are also retained.

The first static checker assumed the wrong source-equivalence key and failed before publication. `packaging-preflight-failure-01.json` preserves it. Its continuation uses the actual `bundle_path`; it repeated eight small metadata reads and 673 stat calls because the first in-memory cache was lost, then saved `packaging-metadata-cache-01.json` for reuse. This was an auditor error, not a producer/native failure.

### Supervised attempts reported by root

1. **Attempt01:** the RSS monitor's `/bin/ps` call was denied by the managed sandbox. The supervisor stopped after 0.003855 seconds and killed/reaped its owned child with return code −9. Output was 0 files / 0 bytes; no worker receipt or native counters exist. Root preserved `process.json`, stdout and stderr in preparation01. Runtime, worker and helpers were unchanged. Root then requested narrowly scoped permission for the same sanitized supervisor to monitor its own child, rather than disabling monitoring.
2. **Attempt02:** assembly exceeded the planned 400 MiB RSS limit. Sampled and OS high-water RSS were **439,042,048 bytes** (about 419 MiB); the supervisor stopped/reaped the child with return code −9 after **3.605 seconds**. The partial output contains **1,193 files / 173,870,144 bytes**, with no worker receipt. No interaction, replay or task-success result is inferred from it. The complete saved input closure fit the corrected file cap, but its in-memory cost exceeded the estimate.
3. **Attempt03:** the same assembly-memory failure occurred at the 512 MiB ceiling. Sampled and OS high-water RSS were **543,997,952 bytes**; root killed/reaped the child after **4.119725 seconds**. Partial output contains **1,193 files / 173,870,516 bytes**, again with no worker receipt. Root did not raise the RSS limit again.
4. **Attempt04:** public assembly completed, but the combined process exceeded the unchanged ceiling while opening the lab afterward. Root killed/reaped it after **6.701952958 seconds**, at **546,226,176 bytes** RSS; partial output contains **1,197 files / 173,875,849 bytes**, without a worker receipt. Completed `prepare.json` is `d133948ba47deed76a2d393275b71cc6eb0ba9691cc097f143df20176b010c77`; ready `lab/status.json` is `c4905a7d977390a00a0069957ccd89d058a3abd80b5e57f11c2c28c46123506e`; immutable `lab/run/manifest.json` is `36fefcb67333f51e212860555e1af95f4e594a52b3bf17688bcf11f324a372af`. These establish assembly completion, not control/replay success.
5. **Attempt05:** fresh initial summary passed at **4.019508 seconds / 395,706,368 bytes** peak RSS. All 266 live controls / 4,256 substeps completed at 5.225984 seconds. Export completed at **11.159369 seconds / 516,505,600 bytes** peak RSS, verifying 532 total live/replay controls, 8,512 substeps, 1 exact replay, 268 video frames written/decoded and 269 verification reads. The second same-process resume then reached **623,230,976 bytes** RSS; root killed/reaped it after **12.120609 seconds**. Output is **1,211 files / 195,374,332 bytes**. There is no terminal worker result or outer05 seal. Process receipt is `1a44df38c87cb68752cc8cf5d665ebc0b72e2a101fe59a02f61f691b60aafc01`; its nine-stage stdout log is `c73c4375fd67e3ec6d62a7b1529e3239fe37fe732443905ca949fa0a6d750e33`.

### Retained attempt05 results

Session: `artifacts/split-limb-feedback-play-05/lab/sessions/session-1dcf521d278e`. Recording seal: `94ee100c5093ad51880578972aa123cf01b11f75a791362379ea23a053db91cd`; receipt seal: `32a1bd9067828fa272f7304ccd73ff5a9119a5812c7b4bfd683156c59de3b7af`; receipt: `63ccde9df4e82fd3f0bf0c84a226ad8d39094a6226dc278aa5138fe0980f9618`. Exact export reports 266 actions, 257 learned actions, 1 reset and 268 events. Its SHA is `f26d5ae838469c883c09ee7aaf1a9dc521a818f68b5428174a55af8ead3fe935`.

Actual final task outcome is **completed interval, success=false**, failing `equipment_stability:sam-2`; mean reward is **0.7453869733917114**. The separate evaluation remains 0/3 for every controller. Initial summary SHA is `48bf14fe956395048de47e5803e44f8f594fbd1cda6ce2f3278ca48ffcb044d2`; final summary is `1a57c0f182ccbe3682896719719e00cb039eb5c876aa7fd2e91b60344ff33d61`. They are captures from a failed whole-process attempt, not a new success seal or policy promotion.

Root identified overlapping ownership: preparation retained all written payload maps while full output verification loaded the same package again. The attempt04 patch captures the file count, releases those maps and attachment bytes, then calls the unchanged output loader; returned counts and identities stay the same. Assembly subsequently completed, but combined build/open still exceeded RSS. Attempt05 changes process lifecycle rather than raising memory or weakening validation.

The intended 532 controls / 8,512 substeps, source, task, scorer, policy, 60-second time limit, 512 MiB RSS ceiling and 400 MiB output bound remain unchanged for attempt05. Attempt timing/RSS facts are root-reported; this lane read only the selected completed-assembly metadata, not partial payload trees, and did not repeat execution.

## Open the verified saved lab

From the repository root:

```sh
PYTHONPATH="$PWD/.local/split-limb-feedback-play-04/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab \
  --resume artifacts/split-limb-feedback-play-05/lab
```

This uses the original pinned task and policy, with manual, authored and learned controls kept distinct. Opening starts at the original nominal state; the completed recording remains separately available. The learned controller has not passed this task or outperformed authored control in the saved comparison. This command was not launched by this preparation lane.

Root owns the user guide and publication. The independent saved-UI audit is in progress at this handoff; no browser QA or UI-audit pass is claimed here. It must join exact05 captures and completed recording seals to the separate resume proof. All failed whole-process outcomes remain visible. No additional jobs are needed in this lane. New external or recurring spend: $0.

## Final interface check

The [actual saved-UI audit](split-limb-feedback-ui.md) now passes the initial,
completed-trial and fresh-resume summaries, policy identity, 22 controls and
recording counts. It retains failed training/evaluation/live-task status and
does not reclassify failed05. No physics, media or browser work was repeated.
Two unchanged generic status labels still assign on every update; the scoped
policy/replay fields avoid redundant assignments. Publication outcome belongs in
`.local/split-limb-feedback-lab-publication-01.json`; absence means unfinished.
