# Parallel reconstruction and imported-world Muse integration

Recorded November 2, 2026 (PST). Main session owns integration and publication;
separate avatar work remains untouched. Full product goal is **incomplete**.

## Shared body shape and ground

`global_limb_camera_fit.py` fits two symmetric thigh/shin factors shared across
all native frames, jointly with leg angles and a bounded camera rotation. The
simultaneous solve preserves original source pixels/confidence, roots, floor,
clock, torso, arms and task gates. Geometry edits preserve exact compiled masses,
inertias, motors and contact settings; standard density-based XML rebuilding
would not. A fresh model audits the candidate; it is not installed automatically.

One new saved-source study, `.local/global-limb-camera-01`, takes 0.519 s,
77,037,568 B peak RSS and 2,563 kinematic calls. No inference, API or rollout.
Source-floor failures improve **13→6/34**, interpolation-grid failures **30→13/80**,
and source image error **9.936→9.129 px**, without worsening any protected endpoint.
The original .025 floor gate still fails: **rejected, unpromoted, not installed**.
The fitted proportions are hypotheses, not recovered anatomy.

- Frozen plan: `7f2d1f142b1f914c10f652f2ece9eab04bcb5c089f0d8a7d1efab1f534a35a88`.
- Original result: `543f214fe6813607066c24c50a43bd9c3e6e8c23abdc449a1a8c3c24001b89e5`.
- Pure detached parameter checks were added after review and applied to the saved
  candidate without rerunning the fit or changing its result. Preserve both.
- Six remaining source failures affect the left shin. Fourteen rejected trial
  proposals exceed the unchanged correction-rate bound; others regress protected
  image endpoints or floor gaps. This is not proof of global infeasibility.

Next: include the original temporal-rate and support inequalities in the joint
proposal solve, instead of filtering unconstrained proposals only afterward.
Use a new bounded study; do not widen gates or repeat this identical search.
Only a fully qualified candidate can proceed to an explicit fixed-inertia runtime
contract and independently protected dynamics/task evaluation.

## Automatic equipment role proposals

`equipment_autoproposal.py` derives conditional body/equipment roles from exact
source-frame torso containment and complete object observations. It preserves
every raw identity, slot, relationship and partition. Existing constructor math
then creates the proposal, full-reference world and untrained lab. The producer
independently rederives the role report/config and retains it in sealed
`input-bindings.json`, carried through ordinary world/lab packaging.

```sh
.venv-lab/bin/python -m weft shared-exercise-auto-build \
  --run BASE_WORLD --evidence-run OBJECT_EVIDENCE \
  --source-mapping VERIFIED_MAPPING --out NEW_DIRECTORY
```

This requires an existing source-qualified body world and saved object evidence;
it is **not raw-video automatic reconstruction**. Roles remain hypotheses;
camera distance, box depth/support, masses and target intent remain authored.
Ambiguity produces an inspectable abstention and nonzero CLI exit.

The analytic separable case reaches the real ordinary strict constructor and
sealed lab. The one real kettle source-only pass retains all **771 slots / three
raw IDs**: sam-1 torso containment 1.0, sam-2/sam-3 0.0; 118 common frames out of
121 body references. It still abstains because source identity-ambiguity
relationships remain unresolved. No world/config promotion, fitting, physics,
decode, inference or API. Wall 0.721 s; peak RSS 110,247,936 B.
Evidence: `.local/equipment-autoproposal-01`, tests
`.local/equipment-autoproposal-tests-02.xml` (61 focused cases).
The latest eight focused checks in `equipment-autoproposal-tests-03.xml` add
separate-document role inference through the real constructor and withheld
sentinel coverage; they overlap the earlier suite rather than adding eight cases.

## Imported mixed-world review

The playable **Equipment candidate.webm**, project
`47dcca08e7a6461295993941fb9162bc`, now has its exact original sealed body package
attached beside its immutable run. `portal-attach-evidence` requires a stopped
portal, verifies source/body/world correspondence and preserves existing review
history. Action-time validation precedes every review; listings read only small
pinned metadata. Imported comparisons/repairs remain blocked.

The historical selective-crop proof uses fixed repository validator release
`0658b6c67d398ac563f25783b42b8d9b24aba0db`, matching all ten original recovery
module pins. A bounded subprocess executes only fixed trusted repository code,
not artifact-supplied code; no provider credentials or network are passed.
This does not expand the interactive lab runtime allowlist or rewrite old proof.

Exact source: `4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`.
Original imported run seal:
`32e1d2b6ef4a0a747e3ccb5058f9ff5050f1880108e6d8bb0f9dec496cb55cc2`.
Body: `artifacts/pose-source-recovery-acceptance-01/kettlebell-pushup/body`.
No scene human, task, source or scored run is rewritten.

The first no-call request preparation exceeded the existing 20 KB context limit;
its failure is preserved. The bounded fix samples fewer evenly spaced source
observations, includes endpoints and explicit selected/available counts, and keeps
all world assumptions plus complete original evidence. It does not enlarge the
request limit or omit equipment to fit. Live acceptance is recorded separately
under `.local/imported-muse-live-01`; never repeat an existing request.

### Live review and preserved worker failure

One actual Muse Spark 1.3 review completed on **November 2, 2026, 3:46 PM PST**,
using the project API key and shared persistent Meta ledger. Eight of 121 source
observations plus three thumbnails fit the unchanged 20 KB context limit; both
equipment objects and all current world assumptions remain. The prepared and
executed request hashes match. Repeated queue submission is deduplicated.

Cost **$0.016165**, total Meta committed **$0.282827**, remaining **$4.717173**,
12 lifetime reservations / zero unsettled. OpenAI allowance unchanged. Queue
journey 21.70 s, review 19.62 s, worker peak 228,687,872 B. All **844 imported run
members and 16 pre-existing status files** remain unchanged. Worker and queue
are stopped; no training, physics, inference, applied repair or deployment.

Muse returns three unaccepted hypotheses about ground/object support,
articulation and equipment geometry, explicitly retaining unverified contact,
depth, muscle activation and alternative explanations. Its text is a proposal,
not an independent anatomical or physical finding.

The first queue attempt failed **before any API request** because its supervisor
imposed a 45/46-second CPU ceiling while the worker attempted to set 360 seconds.
`portal_worker.py` now preserves tighter inherited CPU/output limits. Twenty-two
worker/repair checks pass, including a real constrained child process. Original
failure files remain in `muse-preflight-failure-01/`; explicit reversible recovery
and unchanged 11-reservation ledger are recorded in
`.local/imported-muse-live-01/preflight-recovery.json`. No failed paid request was
retried, no history was erased, and the one-call allowance was not reset.

Further receipts: `.local/imported-muse-attachment-01/result-02.json` preserves
900 original body/lab members; `.local/imported-muse-live-01/result.json` records
live cost, binding, exact request and unchanged run checks. The canonical request
hash is `7feab4d6f91028a431336365d0b3119acac807a172ecb503eae53079458b5861`.

Three exact attachment downloads are enabled through pinned receipts/manifests:
the attachment receipt and prepared context/request snapshots. Root verified all
three on the actual project, rejected body/parent-path/call-receipt access, and
confirmed imported checks/repairs still fail server-side. No body directory is
exposed. Registration preserves prepared bytes without decoding or calling an
API; evidence `.local/imported-muse-live-01/artifact-registration.json`.

## Verification and publication

The combined scoped verification comprises **245 distinct Python cases and 13
scripted UI journeys**. Successful overlapping tests are counted once. Source
fits, real source-role analysis and the paid review were not repeated to refresh
checks. The UI checks are not browser/device visual QA. Preflight file/test
hashes: `.local/parallel-reconstruction-preflight-01.json`.

All lane workers, validation children and portal queues are stopped. Main matched
fetched remote before publication; stash list was empty. Publish only the 24
owned paths listed in that preflight, preserve the separate avatar session's
dirty files, and verify the final remote hash in
`.local/parallel-reconstruction-publication-01.json`. No new branch, deployment
or repository cleanup is needed. The full goal remains active.

## Remaining product work

- [ ] Source-qualified unfamiliar body/ground/equipment contact with original
  timing, identities and gates preserved; current candidates still fail.
- [ ] Resolve equipment identity alternatives using additional source evidence
  or explicitly admitted conditional branches, without deleting raw identities.
- [ ] Integrate this saved-evidence route into mixed-video upload. Prepared lab
  import and analytic construction do not complete the upload requirement.
- [ ] Qualify geometry/contact repairs and expose accepted candidates as separate
  labs; AI critique is not automatic application or physical validation.
- [ ] Finish independent full-task learning and physical/coverage validation.
  Do not train the rejected floor candidate or count abstentions as successes.

The existing portal remains usable via `.venv-lab/bin/python -m weft portal`.
The user manages the server. No hosted deployment, browser operation or submission.
