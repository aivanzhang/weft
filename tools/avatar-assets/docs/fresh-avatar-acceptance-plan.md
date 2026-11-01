# Fresh text-to-avatar acceptance plan

Prepared **2026-11-01, 2:04 PM PST**. **Planning only: no job has been prepared,
no AI/provider call made, and no allowance granted for these cases.** Finish the
current reused-robot development run and freeze its resulting implementation
before preparing these requests. Recheck estimates if that implementation changes.

These are three fresh text-to-mesh tests of the [hands-off completion
criteria](hands-off-studio.md), not three promised successes. Existing generated
meshes, reference images, hand-written finishing configurations and manually
corrected agent answers are excluded. Each case requests one meaningful custom
action plus a portrait and static fallback; this deliberately bounds rendering
and does **not** constitute three new four-state packs. Prior assets stay intact.

## Exact inputs — not yet executed

Commands below are future **no-call preparation** commands, run from
`tools/avatar-assets` only after the implementation is frozen. Do not run them
during the current development job. All use a new output directory, text-only
generation, one local repair at most and a persistent five-agent-call ceiling.
The planner must preserve each action string verbatim and keep the supplied
state ID; it writes the generation prompt and subject-specific rig/finish design.

### 1. Short-fur dancing character

```sh
./run generate \
  --prompt 'An original friendly panda-like character with a round ivory belly, black ears, black eye patches, black arms and legs, warm brown eyes, a short muzzle and a small black tail. Compact upright proportions, distinct hands and feet, and a relaxed neutral pose with clear space between arms and torso and between the feet. A clean smooth undercoat beneath short soft fur, without carved fur spikes. No clothes, props, companion, pedestal, text or scenery.' \
  --style 'Premium stylized 3D character with dense soft short fur, readable individual fiber tips, subtle directional grooming, clean black-to-ivory boundaries and expressive glossy eyes. Rounded believable forms; not a plastic figurine or a painted-fur texture. Preserve the same identity and palette in every view and motion.' \
  --states dance \
  --action 'Perform a cheerful rhythmic dance with three alternating side-to-side beats, small whole-body hops and coordinated arm swings and head tilts. The arms and head must move independently of whole-body travel. Finish back in the starting pose. This is an authored toy dance, not a physics simulation.' \
  --preset preview --max-credits 35 --max-repairs 1 \
  --max-agent-calls 5 --agent-timeout 240 \
  --out outputs/fresh-fur-dancer-v1 --prepare-only
```

Acceptance emphasis: readable rhythm and independent body articulation; continuous
coat coverage, clean eye/nose exclusions, source-consistent black/ivory patches
and no floating flecks, bald regions or stretched fur. Native strands and the
portable GLB coat must each pass their own full-resolution review. The case asks
for an original panda-like subject, not exact reconstruction of the Muse image.

### 2. Flying winged creature

```sh
./run generate \
  --prompt 'An original small teal dragon with an amber belly, two broad orange membranous wings, four short legs, a long tapered tail and two small ivory horns. A friendly narrow muzzle and bright gold eyes. Neutral standing pose with both wings partly open, clearly separated from the torso, legs and tail. Smooth small-scale skin and thin intact wing membranes. One creature only; no rider, companion, clothes, props, pedestal, text or scenery.' \
  --style 'Polished stylized creature with fine teal scale detail, supple orange wing membranes, subtle material variation, clean edges and restrained highlights. No fur or hair. Keep anatomy, wing size, colors and scale consistent throughout the animation.' \
  --states flight \
  --action 'Take off from the starting position, fly upward and sideways while both wings visibly flap through at least three beats, then descend and return to the starting position. The wings must articulate relative to the body while it travels; standing wing flaps or moving the whole dragon without wing motion do not satisfy this flight.' \
  --preset preview --max-credits 35 --max-repairs 1 \
  --max-agent-calls 5 --agent-timeout 240 \
  --out outputs/fresh-winged-flight-v1 --prepare-only
```

Acceptance emphasis: actual rise, travel, descent and independent wing
deformation; readable wing area and intact membranes throughout. Fused or missing
wings cannot be hidden by root travel, invented unseen anatomy or a stationary
flap substitution. Takeoff/landing are authored visual motions, not measured lift,
balance, ground contact or collision correctness.

### 3. Articulated desk lamp

```sh
./run generate \
  --prompt 'An original articulated desk lamp with a heavy circular cream ceramic base, two slender matte teal metal arm links, visible brass hinge joints and a rounded cream ceramic lampshade. The arm is bent in a relaxed neutral position with clear mechanical gaps around each hinge and space between the shade and arm. No face, limbs, visible cable, separate objects, lettering, pedestal or surrounding furniture.' \
  --style 'High-quality stylized product object with smooth satin ceramic, matte painted metal and restrained brass highlights. Precise rigid mechanical forms, clean material boundaries and a warm cream and teal palette. No fur, hair or organic skin; no animated light or material changes.' \
  --states mechanical_scan \
  --action 'Keep the circular base fixed in place throughout. Sweep the articulated arm left and right while the lampshade nods independently twice, then return to the initial pose. Motion must occur at the mechanical joints, without rocking the base, stretching the metal links or bending the ceramic shade.' \
  --preset preview --max-credits 35 --max-repairs 1 \
  --max-agent-calls 5 --agent-timeout 240 \
  --out outputs/fresh-articulated-lamp-v1 --prepare-only
```

Acceptance emphasis: independent mechanical controls and rigid-looking materials,
with a genuinely stationary weighted base. No whole-object rocking in place of
joint movement. This is a different articulated object, not a recolored humanoid
robot. The light itself need not switch on, cast an animated beam or change color.

## Cost and resource envelope

### Meshy credits — verified against current API documentation

The local adapter currently pins `meshy-7.1`, standard topology with remeshing,
`geometry_resolution: "4k"`, `texture_resolution: "4k"`, target 35,000 triangles,
and **no provider rig or animation**. The API documentation supports this text
model and a separate preview/refine sequence; an older web-app/search summary
is not authoritative for the API. [Text-to-3D API](https://docs.meshy.ai/en/api/text-to-3d)

| Work | Per case | Three cases |
| --- | ---: | ---: |
| Geometry preview: 20 credits + 5 Ultra geometry surcharge | 25 credits | 75 credits |
| 4K texture refinement | 10 credits | 30 credits |
| Provider rigging, motions, image generation and paid retries | 0 | 0 |
| **Total / paid submissions** | **35 credits / 2** | **105 credits / 6** |

These rates match `meshy.validate_config`, `meshy.stage_request` and
`pipeline._source` at this checkpoint. Requested resolutions/counts are generation
settings, not proof of delivered texture detail or mesh quality.
[Official API pricing](https://docs.meshy.ai/en/api/pricing)

The existing read-only project receipt reports **202 credits used/reserved out of
210 authorized; 8 remain**. Thus the arithmetic shortfall is **97 credits**.
However, the implemented `Ledger.authorize_job` only appends an immutable grant
equal to a prepared job's entire ceiling. Each fresh geometry plan is capped at
35, and a grant cannot subsidize another job. **Without changing that mechanism,
the executable approval envelope is three job-specific 35-credit grants: 105
additional credits, leaving the old 8 unused.** This document grants nothing.
Do not change the ledger, reset history or treat `--max-credits 35` as approval.

No dollar conversion is asserted: the account's actual credit acquisition price,
subscription allocation and current balance were not accessed. Public purchase
documentation does not establish this account's marginal price. Spending already
held credits is different from purchasing more. No account, subscription, credit
purchase or auto-recharge change is included.
[Credit purchase documentation](https://help.meshy.ai/en/articles/9992062-how-do-i-purchase-credits)

### AI roles, local work and retention

| Item | Aggregate across the three proposed jobs |
| --- | --- |
| Normal first attempts | 9 AI calls: planner + author + post-media critic for each case |
| Current route with one repair per case | At most 15 AI calls: 3 planners + 6 authors + 6 post-media critics; no separate pre-critic exists in this version |
| Configured call ceilings | 15 calls total, each at most 240 seconds; failed/uncertain calls retain their consumed allowance |
| Token/image planning envelope | At the 15-call cap, approximately 360,000 input text tokens and 15,000–120,000 output tokens using the estimator's per-call assumptions; up to 120 selected image inputs. These are planning estimates, not tokenizer-enforced token limits or measured usage |
| AI account cost | Existing authenticated CLI; account-specific billing/remaining quota unknown. Not zero-cost simply because no API key is supplied |
| Native builds and exported-view renders | At most 6 of each; each stage has a 600-second / 2 GiB sampled direct-child RSS limit. Initial inspections are up to 300 seconds each |
| Final video rendering | Up to 6 attempts at 1,200 seconds each; an eligible wall-time failure may have one 600-second hash-bound frame continuation, not a new Meshy attempt |
| Audit / reload / packaging | Per attempt: weighted action audit 60 seconds / 768 MiB; native reload 180 seconds / 2 GiB; media packaging 180 seconds / 768 MiB; HD evidence assembly 120 seconds / 768 MiB |
| Media volume | One 720×720, 24 fps, 4-second clip per case: 96 delivered frames each / 288 total if all pass. Up to 576 rendered clip frames over six full attempts, before any preserved partial work |
| Disk retention | 2 GiB per-job aggregate stage-start allowance, or 6 GiB nominal across the three; retain failed attempts, logs, downloads, source snapshots, editable Blender files, renders, media and compact copies. This guard is not an atomic disk quota; reserve additional local headroom before execution and measure actual output |
| New hosted compute/storage/services | None; no recurring cloud charge introduced. Local electricity, disk occupancy and AI account usage are not priced here |

The estimator assumes up to roughly 24,000 text input tokens and typically
1,000–8,000 output tokens per call. Actual safeguards bound bytes, image sizes,
runtime and calls; receipts must record actual model identifiers/usage. Images
add account-dependent usage. No unpriced retries or switch to another paid
provider is included.

Run **one expensive pipeline/native stage at a time** on this workstation while
the fresh cases are measured; do not overlap the current robot job. Independent
read-only validation/preparation can proceed concurrently. The native monitor
samples direct-child RSS, not the whole process tree; report that scope. One
pipeline invocation stops starting new stages after 3,600 seconds. Resuming does
not reset attempts/calls; it opens another bounded watch window. The stage caps
above are ceilings, not a fresh-case latency prediction or a guaranteed combined
completion time.

Meshy's published medians are 75 seconds for geometry and 80 seconds for 2K/4K
refinement, with Ultra geometry adding up to a minute; queue and processing vary.
These are provider observations, not measurements for these subjects. Our
900-second watcher may checkpoint an unfinished task. Preserve and poll its
existing ID; never resubmit merely because the local watch expired.
[Task lifecycle and processing times](https://docs.meshy.ai/en/api/task-lifecycle)

The provider normally retains API assets for **at most three days**, and signed
URLs expire. Download successful results promptly into the existing bounded
local store and preserve byte hashes; do not treat a URL as archival storage.
Retain local accepted and rejected evidence until the user authorizes cleanup.
No remote publication or paid storage is part of this test.
[Asset retention](https://docs.meshy.ai/en/api/asset-retention)

## Fixed acceptance and unsupported gates

All gates remain in force through repair; a failed test is not made successful by
changing its subject, exact action, surface material, budget or review threshold.

1. **Fresh provenance:** request → planner → fresh provider geometry/texture →
   actual source inspection → agent-authored finish → evidence → critic → delivery
   is traceable by hashes and task/call IDs. Preserve exact prompts, schemas,
   model/settings, rights assumptions and all rejected attempts. No handwritten
   per-case bones, weights, masks, motions or finishing JSON may be inserted.
2. **Intent:** each state retains its exact `user_request` and ID. Structured roles
   and requirements cover every requested action component. Stationary roles use
   `stability`, not a positive-motion check; a whole-clip role cannot be both stable
   and moving. Neither renaming nor a weaker alternative counts as acceptance.
3. **Actual 3D:** bounded GLB loading, explicit normals/materials, real skeleton,
   normalized skin weights, named animation and independent reload/motion proof
   must pass. The weighted audit allows at most 100,000 vertices, 64 joints and
   1,024 probes, excludes reserved groom/card surfaces from anatomy, and samples
   the full 97-frame endpoint-inclusive four-second action. These limits are not
   Weft's stricter importer contract.
4. **Necessary numeric action checks:** articulation needs root-relative skin
   displacement ≥0.5% of measured role scale and meaningful weighted coverage;
   rhythm needs ≥3 reversals, ≥2 independently active controls and ≥2% scale
   displacement. Travel needs ≥3% scale; takeoff needs rise and travel ≥5%; landing
   needs descent ≥5%. Stability needs at least 8 directly target-weighted vertices
   (weight ≥0.5), with world displacement ≤min(0.5% role scale, 5 mm), not merely
   root-relative stillness. Missing evidence fails. These thresholds do **not**
   prove semantic success, joint realism, collisions, contact or physics.
5. **Material-specific quality:** use actual inspected bounds/regions, not a panda
   template. Fur requires a real native groom and portable representation with
   measured exclusions and regional direction/length, not painted fur or counts
   alone. Scales/metal/ceramic keep their appropriate appearance without added fur.
   Reject plastic-looking coat, colored flecks, holes, seams, texture smearing,
   rigid-material stretching and identity drift that remain visible at final size.
6. **Independent HD evidence:** actual native **and exported** front/side/back/
   three-quarter views, informative state poses at least 720 pixels per pose,
   weighted full-motion audit, and decoded samples of the final encoded media.
   A large canvas holding small thumbnails is insufficient. The critic must cite
   supplied hash-bound evidence for identity, appearance, framing, deformation
   and every action requirement. Critic acceptance is model assessment, not human
   approval or Muse-quality equivalence.
7. **Deliverables:** 1600×1600 portrait; 720×720 silent H.264 clip at 24 fps for
   four seconds; exact first-frame static/reduced-motion fallback; editable source,
   animated GLB and portable-media package with reproducible manifests. Verify
   actual format/dimensions/duration/decoded frames, visible action changes,
   framing and declared background. Ending at the first pose does not establish
   a seamless loop. No transparency, loop or native/browser parity claim without
   corresponding evidence.
8. **Browser and durable lifecycle — separate proof:** the overall goal needs one
   actual authorized Studio UI/job-to-output run with real downloadable assets,
   refresh/resume, playback, seek, reduced motion and failure fallback checked.
   The running reused-robot case can satisfy this UI proof; it cannot satisfy a
   fresh-generation case. These three cases may stay CLI-driven within their
   stated caps. Fixture-only UI results are insufficient. Do not start another
   user-managed server or duplicate a live worker; preserve process-exit receipts.

An explicit unsupported result is correct when the actual mesh lacks usable
appendages/joints, contains inseparable fused contacts, cannot meet coat/material
quality within bounds, or exhausts the fixed repair allowance. Hidden mesh
separation/capping, long-hair/cloth dynamics, facial performance, arbitrary prop
interactions and physical flight are not automatically solved by this backend.
Do not claim these from a 2D preview or moving root. A source-generation failure
or uncertain call is retained without automatic paid regeneration; unresolved
ownership is a resumable/inspection state, not a success.

## Next safe action and evidence to retain

After the current robot development run and reusable portable-fur work are
resolved, freeze source hashes and relevant tests, then prepare these exact
requests in their new folders. Review the real generated plans and estimates
before the parent session requests the required concrete allowance. Bind any
future approval to the actual generation plan IDs/digests; **do not grant from
this draft or silently consume another job's grant**. Start with one authorized
case to measure real resource use, then continue the other ready cases within the
same aggregate envelope. An unacceptable result does not buy a replacement.

Record per case: request/plan/source hashes, fresh provider task IDs, charged and
reserved credits, AI model/call usage, all design/repair/evidence manifests,
actual wall/RSS/output measurements, terminal process handles, critic result,
media audit, static fallback, final gallery/download bindings and any unsupported
reason. Record actual allowance receipts separately. Preserve the exact failed
case if later generic code improvements require a newly versioned acceptance run.
No source archive, approved asset, Weft physics/importer or shared budget is edited.

**Completion claim remains pending.** Three successful fresh cases would support
this tested range of subjects/actions, not universal prompt reliability, guaranteed
Muse parity or a measured success rate over arbitrary requests. Failed or reused
examples must not be counted as fresh quality successes.
