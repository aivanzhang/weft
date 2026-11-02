# Temporally supported depth branches — verified checkpoint

November 2, 2026, PST. Previous turn is **progress**: `20be23c` is published,
remote-verified, with 213 Python /73 mocked UI checks, one frozen source
candidate and exact actual recording. The candidate still fails one source
and two grid floor checks. The full combined goal remains active.

## Frozen hypothesis and separate contract

A local .30-rad correction cannot reach either exact image-compatible ankle
depth at source frame84. Do not expand the existing v1/v2 correction bounds.
Add a separate opt-in branch-proposal stage after unchanged v2 fitting. Derive
both forward intersections of each original distal image ray with the sphere
around its fixed knee/elbow, using its fixed distal length. Keep all rays,
missing intersections, duplicate roots, candidates and rejection reasons.

At most four views/eight raw roots. Axial change is bounded to [-pi, pi]; the
existing physical hinge range remains [0, 2.6]. This is a discrete geometric
proposal, not a widened local optimizer. Fixed root/upper axes/lengths/ground,
complete fits, source observations and independent floor/task gates remain.
Every candidate is compared against **all** qualified views. No reference
annotation, ground height, floor score or contact target participates.

Only complete, multi-/two-view-supported fitted limbs can provide temporal
anchors. Precompute their world distal directions once; neither partial fits
nor newly selected branches become anchors. Require nearest bracketing anchors
within .25 s on each side, with nonnegative direction dot product. Interpolate
their directions by source time and normalize. This is an explicit smooth-motion
assumption, not another observation or calibrated statistical confidence.

Require lower aggregate weighted normalized pixel SSE than the existing v2
pose, lower distance to the temporal direction, and distance at most .5 distal
lengths. Rank image-compatible distinct branches by temporal distance, including
runners just beyond that temporal cutoff; require a runner-up gap of at least
.25 lengths when multiple branches remain image-compatible. Retain
ambiguous/rejected cases unchanged. Deduplicate coincident endpoints only for
ranking, preserving every raw root and contributing view. Bound processing to
20 s per body and 900 reference poses; use sorted anchor lookup, not all pairs.

New method/source custody must preserve the original v2 correction history
and bind the before-branch reference, selected changes, anchor identities and
all assumptions. Selected depth remains an **inferred hypothesis** with zero
independent 3D-measurement confidence. Keep unsupported training/coupling guards.

## Work, ownership and acceptance

- [x] Implement and qualify pure geometric branches, abstention and all-view
  selection: known roots, tangency/misses, invalid/behind-camera cases, noisy
  competing views, ambiguous anchors, image resizing and hinge/axis invariants.
- [x] Integrate explicit CLI/body/lab/recording provenance, preserve legacy
  numerical behavior, and test rejected source/anchor/contract tampering.
- [x] Freeze at most one new source candidate after controlled qualification,
  using the same original ASPset evidence/cameras/ground and v2 uncertainty.
  Pin runtime and per-run CPU/time/memory/native/output bounds before execution.
- [x] Audit all candidates, source/image/floor slots and actual recording;
  preserve failures, update guides/TODO and publish only root-owned paths.

Root owns these source/body/lab paths; preserve the separate avatar session's
dirty renderer, renderer tests and prompt work. Start revision `20be23c`;
all preceding root native/test/recording jobs are terminal/reaped. Reuse the
saved source packages; no source acquisition/decode/inference, API/key/ledger
access, server/browser, deployment or paid work is planned. Existing dependencies
only. Do not rerun prior unchanged passing studies. Follow the complete goal,
not just this single reconstruction case, when deciding completion.

## Controlled qualification — November 2, 2026, 2:10 AM PST

234 distinct Python cases pass across the latest applicable receipts, plus
74 mocked UI checks. Receipts: `.local/distal-depth-core-tests-01.xml`,
`.local/distal-depth-geometry-tests-02.xml`,
`.local/distal-depth-integration-tests-02.xml` and
`.local/distal-depth-ui-tests-01.log`. The integration receipt is 148/148,
17.709 s; it includes actual recording/export/exact replay and native selected
branches with two, three and four cameras. These are controlled software
checks, not real-source depth accuracy or browser QA.

Preserve the first core receipt's three failing fixture expectations: its
negative-hinge target produced nearly tangent, ambiguous alternatives and
correct abstention. The fixture's target bend correction changed from .5 to
.9 rad to test a separated supported branch. No selection threshold or
production bound changed. All 24 pure-geometry cases then passed. The first
three positive native cases also passed in
`.local/distal-depth-integration-tests-01.xml`; the broader receipt supersedes
that subset, not an additional three cases.

Frozen plan: `20be23c` plus only root-owned runtime overlays, pin one new
`artifacts/aspset-depth-branches-body-lab-01` candidate, and use the same
original evidence, cameras, ground and explicit v2 uncertainty. Construction
is capped at 30 s external /25 s CPU /768 MiB sampled RSS, six model compiles,
64 data allocations, 1,000 static forwards, zero dynamics/decode/inference/API,
64 MiB per file and 400 MiB/3,000 files post-exit. No paid/provider work.
Do not widen the frozen branch thresholds after seeing this source result.

## Real-source result — November 2, 2026, 2:25 AM PST

**No branch selected.** The same original ASPset `04ac-0003` evidence produces
24 attempts, 70 rays and 52 distinct forward alternatives. Forty-four rays
miss their fixed-knee/length sphere. Fifteen poses lack complete supported
observations on both sides; nine have no qualified alternative. Fifty branches
worsen all-view image agreement; the remaining two at frame84 worsen the
assumed temporal direction. Preserve all results and thresholds.

At frame84, v2's temporal distance is `.20638521820408479` distal lengths.
The two exact-image alternatives have distances `1.044353611337787` and
`.37506933502787715`. Neither improves the frozen temporal criterion, even
though their nominal single-camera image error is effectively zero. This
is evidence/model disagreement, not recovered depth or physical impossibility.

Body reference angles, source images, all 163 complete fits, all 24 v2 pixel
fits and 259 v2 trial losses remain unchanged. All 6,768 original image slots
remain: 1,620 compared, 72 low-confidence/outside and 5,076 unsampled. Image
means stay left/mid/right `10.4368426411/6.6739925583/6.3521329987` pixels.
The original floor contract/check order stays exact: **1/47 source poses and
2/111 grid poses fail**, worst penetration `.049339568330757165` model units.
`left_knee_geom` names the shin capsule, not the location of its knee center.
No default/policy promotion, new training, physical validation or changed scorer.

### Frozen inputs and construction

- Runtime: `20be23c0b91c4e995064bef9fa146960fb2bf9dd` plus eight owned overlays,
  313 pinned members, `.local/aspset-depth-branches-lab-01/runtime`.
- Plan: `.local/aspset-depth-branches-lab-01/plan.json`, SHA-256
  `67ca9244e228ac28a6607c211025222d8d52b9286519ae35cb5c3eaea56fbdef`.
  Worker SHA-256 `ed3d390b42ba331045d38fbf99bdc84f4e557996a96e74f6d81f6ee889a45262`.
- Original source package `artifacts/aspset-camera-body-input-01`, seal
  `0e63ca94b07bd68b1e76bae8fe2e8987737c9e4289f84013519f07cdb457739b`;
  third-camera input `artifacts/aspset-third-camera-input-01`, seal
  `8abfc496985b8538d07ae7b33d741dd33a21e658842f2076818fbd477e28a928`.
- Manual ground `artifacts/aspset-manual-ground-01`, seal
  `bdf11ee7ddd7b74fce3420c6ee36902acc35b9c57d15117536671bf6745f23f2`.
  Mid evidence package `artifacts/aspset-person-region-body-02/mid`, seal
  `2a98b5ad478e6035c6222825bdb7f1ea22941841f1b83bb82649a6657e7082ca`.
- Unchanged `examples/partial-pixel-uncertainty.json`, file SHA-256
  `6644c04918654879fd4f9e4261451f864bdd052b5cdc6e4c2a2aa0ab50d48057`:
  assumed image fraction `.0025`, angle scale `.3`, temporal scale `.45`.
  These are not calibrated noise/confidence.
- Candidate `artifacts/aspset-depth-branches-body-lab-01/run`: **383 members**,
  seal `6deb3c0b9f5171b531f06d6443f4bcb6df6a7b0491ad62f0bf3ee612353d6402`.
- Construction worker: **1.687583208 s /151,453,696 B RSS**; supervisor
  **1.737285542 s**, exit0/reaped. Four compiles, 53 data allocations,
  253 static forwards; zero physics, source decode, inference or API.
  Output including lifecycle: 386 files/44,040,653 B. No provider cost.

Exact construction command, with a **new** output directory on any future run:

```sh
.venv-lab/bin/python -m weft camera-body \
  --run artifacts/aspset-camera-body-input-01/left \
  --camera artifacts/aspset-manual-ground-01/camera-0.json \
  --ground artifacts/aspset-manual-ground-01/ground-0.json \
  --root-view artifacts/aspset-camera-body-input-01/root-view.json \
  --additional-view artifacts/aspset-third-camera-input-01/view.json \
  --body-view both --fit-limbs --partial-segments --fit-partial-pixels \
  --partial-pixel-uncertainty examples/partial-pixel-uncertainty.json \
  --partial-depth-branches --out artifacts/my-new-depth-branches-lab
```

Do not rerun this unchanged failed source hypothesis. The saved bounded
supervisor is terminal and deliberately rejects duplicate execution.

### Independent audit

`scripts/studies/verify_aspset_depth_branches.py` uses saved JSON, independent
camera equations, cubic radial inversion and polynomial ray/sphere roots.
It does not call the production branch proposer or optimize a new candidate.
Temporal directions are checked against native saved limb positions; all
roots, all-view errors, rejection/selection decisions, upper-bone invariants,
source-image slots and original floor gates are checked. The preceding v2
audit is verified and its exact unchanged fit histories reused.

- Report `artifacts/aspset-depth-branches-audit-01/report.json`, four sealed
  members, seal `007aa69154aedfc6783adf61089fbc0f06ca2430c4798391cbe2f1732884c2d8`.
  It checks **768** baseline/candidate/prior-audit members and 46 identical
  source/settings files, with zero new native/inference/optimizer/API calls.
- Successful controls `.local/aspset-depth-branches-audit-02`: plan SHA-256
  `860cba2ab0633541abc5b10983bfece4c4be879834d508ab017da58928b1b64a`, worker
  `22b411922f2945dc6dc655b3294efd33360fa57b09deb7c2b2c1314e76b47111`.
  **.341321833 s /75,595,776 B RSS**, supervisor .389840000 s, exit0/reaped.
  Limits: 30 s external /25 s CPU /512 MiB RSS, 16 MiB/file and total output,
  eight output files; no network, subprocess, simulation/video/inference imports.
- Preserve `.local/aspset-depth-branches-audit-01`: first control attempt
  stopped before output because importing the old audit helper indirectly
  imported `cv2`. The verifier now uses direct JSON/hash/math dependencies;
  the import guard and all scientific rules remain unchanged.

### Actual recording and after-close integrity

Controls `.local/aspset-depth-branches-interaction-01`; plan SHA-256
`39aba8ad55dfd8017906d7f8e4e75cb8c257bb19947fb78479bcee05b9fe528e`, worker
`3f7896e983cdbd05ff6dbaefe8109173be65b4379b23e72043c2e22c15f5bb05`.
Recording `artifacts/aspset-depth-branches-body-lab-01/sessions/session-7a63eab3930f`,
six sealed members, seal
`f2752fdd309ef4f5a2811b9cfa3027e7bbbd85f609020cdf5ec252c44d1114b5`.

**39 actions /one reset /41 exact states, rewards, flags and video frames**.
Two compiles/two data allocations, 145 forwards, 1,104 physics calls,
41 video writes/42 reads. Worker **1.047033167 s /162,086,912 B RSS**,
1.078799250 s including close; supervisor 1.191857000 s, exit0/reaped.
Seven recording files including manifest: 1,913,830 B. Bounded by 45 s
external /35 s CPU /768 MiB RSS, 6 compiles/16 allocations/1,000 forwards/
3,000 physics calls/64 video writes/256 reads/64 MiB output.

Sixty in-process action/state/JSON calls: median `.843417365` ms,
p95 `.896429759` ms, maximum `.917416997` ms, maximum response 35,512 B.
These are not HTTP/browser/painting or sustained-use measurements.
`after-close.json` verifies all 383 immutable run members and both lifecycle
metadata hashes: status remains **ready**, result **manual_candidate**.

## Ownership, publication and next safe action

Root owns the two new depth modules, six body/CLI/lab/UI integration paths,
two new Python test files, the UI test, this audit script, guide, TODO and
combined handoff. Preserve all separate avatar renderer/tests and
`tools/avatar-assets/` edits. Stash/index were empty before publication;
remote fetch completed at 2:25 AM PST. Check the publication receipt for the
final revision and remote/owned-path verification:
`.local/aspset-depth-branches-publication-01.json`. Its absence means publication
is unfinished. No live root construction, interaction or audit job remains.

This checkpoint is **progress, not completion** of the combined goal. Next,
qualify a distinct noise-aware articulated-geometry model that accounts for
uncertain joint positions/shape across time, rather than widening this fixed
sphere selector or selecting a depth on ground clearance. Establish controlled
identifiability/abstention, source bindings and unchanged independent gates
before another original-source candidate. The remaining coupled-world contact,
unfamiliar-video reconstruction, learning and independent physical-accuracy
work still belongs to this session. No single clip or software test completes it.
