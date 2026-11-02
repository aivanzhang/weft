# ASPset third-view reconstruction

## Active — November 1, 2026, PST

Previous goal turn is **progress**: `06889e5` publishes reusable manual stereo
ground calibration, 121 passing checks and the frozen-ground two-view body.
Remote main was independently verified; root-owned work/index/stashes were clean.
The new body retains one source and three interpolated floor failures. Source
frame 80 has no usable two-view left knee/ankle, so the failing leg retains a
monocular prior. Full goal remains active; ground/contact and learning are not
complete. Preserve the independent avatar session's current edits.

Changed hypothesis: the publisher's third camera can supply independent image
support for occluded limbs. Do not change the already frozen ground or evaluation
limits. Published C3D remains assessment-only; no reference coordinates in any
new inference, camera/body construction, correspondence choice or floor fitting.

## Bounded acquisition plan

The original saved 32 MiB gzip prefix identifies the complete tar header for
`ASPset-510/trainval/videos/04ac/04ac-0003-right.mkv`: 17,804,318 bytes, data offset
30,913,536. The member is incomplete in the old prefix and was never extracted.
Header inspection is saved at `.local/aspset-third-view-header-01.json`.

Owner: root. Download exactly compressed bytes **33,488,896–50,331,647 inclusive**:
16 MiB new bytes plus a 64 KiB overlap. One request, no retries, identity encoding,
exact HTTP 206/range/total-length/ETag checks, overlap byte comparison, existing
publisher mirror host allowlist. Reuse the original prefix; do not download the
23.7 GB parent archive or repeat the left/mid downloads. Full parent checksum
remains unverified. Source rights remain the saved publisher/archive CC0-1.0.

Acquisition output `.local/aspset-third-view-acquisition-01`; maximum response
35 s, socket timeout 10 s, external 45 s. No inference, media decode, physics,
reference data, API key/ledger or downloaded code execution. One-time external
and recurring cost $0; retained incremental compressed bytes about 16.1 MiB.
After overlap verification, bounded decompression (52 MiB) may copy only the
complete named right-video and matching camera member into a new source package.

## Next work and acceptance

- [x] Verify complete member bytes/provenance, then decode this view once per
  necessary independent native/decoder-clock check. Preserve every frame and
  timestamp; do not assume matching metadata proves physical synchronization.
- [x] Inspect source pixels before choosing an explicit fixed person region;
  use the same Full/VIDEO/raw/capacity-two/stride-four inference settings as the
  existing views. Keep all raw poses and missing/competing/unsampled slots.
- [x] If supported, add reusable bounded multi-view geometry and integrate it
  with the existing camera/body lab. Use only image-supported rays, preserve
  view/source bindings and fixed guards, and report conflicts/missing evidence.
- [x] Freeze any new body before unchanged floor/source assessment; retain
  failures and original candidates. No control/ground tuning or success claim
  from a source-fit change alone.

At the initial plan, no root job had started. No new branches, server, browser,
deployment or paid call. Follow standing merge/push approval for verified scoped
work; never stage, stash or discard the avatar session's files.

## Source and inference results

Acquisition completed in 0.553528 s with exact HTTP 206, expected ETag and all
65,536 overlap bytes verified. Range SHA-256:
`67f1a1cfbcd87f3e992128d1b017717c6365790aa4b84b9b686359cf6a36d104`.
The response/socket deadlines were enforced cooperatively; no separate external
acquisition supervisor was used. New source `data/aspset/04ac-0003-right-01`:
seal `02cb72e5e893fc37cb8ac823ee7d7f4f04b4b65b3008796ddfc6a0ea08d9c67b`,
right video `828fc6ecfd4ff42270de5785ce55e00a40342c57e227efba4a0a5daed4c6a7f5`,
camera `3725b642a66a00dbc66a059b0497667e2fc8130d4f195a3e3f9e5ea47a9309c5`.
Full parent integrity is still unverified. Only the complete named train clip and
camera were copied; no reference poses or test unit were extracted.

Native FFmpeg hashes and OpenCV decode-to-EOF both retain 188 frames, 4K/50 fps,
native 1/1000 timebase, zero native/decoder offset. Packaged timestamps match the
existing views; independent physical synchronization remains unverified. Process
`.local/aspset-third-media-process-01` completed/reaped in 6.203344 s; worker
5.961139 s, peak 286,785,536 B, FFmpeg child peak 462,454,784 B. Existing left/mid
decodes were not repeated. Source thumbnails 0/60/120/180 were inspected before
freezing a new explicit person region `[1720,800,2280,1440]` in original pixels.

`artifacts/aspset-third-person-region-01` seal:
`bce514c9790825dca3961f7379202e5130f7b6f2294d520fface4aac06170a21`.
Same Full model/VIDEO/raw/capacity-two/stride-four settings as earlier views;
one detector, 47 inference calls, 188 decodes and 189 read attempts including EOF.
One person at all 47 samples: **46 tracked, one low-confidence**. Worker
4.074369 s /395,231,232 B; supervised 4.152416 s, completed/reaped. Two optional
subprocess attempts were denied, not executed. No API, physics or reference read.

Next implementation: bounded 3–4-view geometry with at least two qualifying rays
per joint, every source/view slot retained, and no best-pair selection when an
available view conflicts. Preserve exact two-view behavior. Extend existing
camera root/body/lab contracts with distinct multi-view method identities and
per-joint contributor provenance, leaving shared-world/learning adapters that
do not support the new method unavailable. Test analytic missing/conflicting
views and source/clock binding before one new frozen-ground native construction.

## Integration checkpoint and native run plan

Reused the existing `multiray_pose.triangulate_pixels` bounded pixel solver;
discarded the draft duplicate ray solver. Original `tests/test_multiray_pose.py`
was restored byte-for-byte after a draft test used that occupied filename. New
coverage lives in `test_multiview_pose_extra.py`, `test_multiray_body.py` and
`test_multiview_extra_lab.py`; the original solver and tests are unchanged.
The early 47-check run retained one failed expectation: conflicting views can
exhaust the fixed optimizer as well as fail reprojection. Both retain the
proposal and reject the estimate. No solver gate was loosened to pass the test.

146 integration checks passed (`.local/multiview-extra-integration-tests-01.xml`),
including native manual recording/exact replay; the mocked lab UI suite passed.
After explicit roundoff preparation was added, 40 affected checks passed
(`.local/multiview-extra-rotation-tests-01.xml`). The legacy two-view entry stays
unchanged. Multi-view mode records nearest-proper solver rotations separately
from the original publisher matrices (maximum element correction 1e-5), because
the publisher's six-decimal rotations exceed the existing pixel solver's 1e-8
orthogonality gate. This is numerical preparation, not camera fitting or a
physical-calibration claim. All original source matrices remain unchanged.

Third-camera input `artifacts/aspset-third-camera-input-01`, seven members,
seal `8abfc496985b8538d07ae7b33d741dd33a21e658842f2076818fbd477e28a928`.
Reproducer `scripts/studies/prepare_aspset_third_camera.py` verifies the frozen
source/image/ground packages and applies the same `B @ Q` world rotation and
declared mm-to-m translation scale. No source video decode, new inference,
C3D input, ground change or first/second camera change.

Next bounded job: `.local/aspset-third-camera-lab-01`, output
`artifacts/aspset-third-camera-body-lab-01`. Freeze committed runtime plus only
the root-owned integration files; exclude the avatar session's dirty changes.
Call the existing `camera-body` CLI, original left source/root-view/ground,
one `--additional-view`, `--body-view both`, no partial-segment mode. Maximum
30 s external /25 s CPU, 768 MiB sampled RSS, six model compilations, 64 data
allocations, 1,000 static forwards, zero dynamics/inference/media/network/API.
Local one-time and recurring provider cost $0; no Meta ledger/key access.
Require all 47 original poses and 188 source slots; assess unchanged floor
limits after sealing. Failed construction remains a result, not permission to
drop a pose, move the floor or tune the solver. All preceding processes reaped.

## Native construction result and interaction plan

Three-view lab sealed successfully: `artifacts/aspset-third-camera-body-lab-01/run`,
377 members, `fe2b84b7b8f7c26ab1f912f87962a2e7ef13eb58d88bcc035911514e98365857`.
Worker 1.687127 s /150,847,488 B; supervisor 1.737852 s, completed/reaped.
Four compiles/53 data allocations/253 static forwards, no dynamics/media/model/API.
All 47 body poses and 188 original slots remain. Left-arm image support improves
26→47 poses; left-leg 19→22. Frame80 knee/ankle now use mid/right observations,
but overall source floor failures worsen 1→4 and control-grid failures 3→7.
Maximum penetration grows .066662→.092148. Ground, thresholds and source bytes
are unchanged. This is a retained unpromoted regression, not resolved contact.

Next bounded independent checks: saved-only source/projection/geometry audit,
and one actual manual recording/replay in `.local/aspset-third-interaction-01`.
Interaction reuses the frozen native runtime: ≤45 s external/35 s CPU/768 MiB
sampled RSS, six compiles/16 data allocations/3,000 physics calls/1,000 forwards;
39 declared manual actions plus exact replay and 60 measured action/state calls.
At most64 video writes/256 verification video reads, no original-source decode,
network, inference, API, browser or server. Preserve the immutable run and put
new recordings only under `sessions/`; report dynamics failures, not erase them.
Provider cost $0. No other root job is live before this launch.

## Verified checkpoint

All jobs completed and were reaped. No server, browser, deployment, API/key/ledger
access or default promotion. The manual-recording worker completed in 1.049960 s
/160,792,576 B; supervisor 1.131605 s. Two model compilations/two data allocations,
1,104 physics calls/145 forwards, 41 video writes/42 verification reads, no source
video decode. Immutable `run/`, `status.json` and `lab-config.json` stayed exact.

Recording `artifacts/aspset-third-camera-body-lab-01/sessions/session-c3c8e7838e80`:
39 actions, one reset, 41 states/video frames; every state/reward/completion flag
replayed exactly. Seal:
`e46b7b8934e4093295a115cfceb79fa306055b850ab34ca8794199fd74e81ebc`.
Sixty in-process action/state/JSON calls measured median 0.831896 ms, p95 0.880456 ms,
max 0.915750 ms; maximum 35,426 B response. Not browser/HTTP or sustained-idle QA.
Exact plan/worker/runtime linkage, logs, process receipt, latency samples and
replayed hashes are in `.local/aspset-third-interaction-01`.

Saved-only audit `artifacts/aspset-third-view-audit-01`, three files, seal:
`209543b97c40cdb1ae8e1082ec30f96e562fddb4d6f6e989b9142eee56df5cbe`.
It independently checks 776 manifest members across baseline/candidate/ground/
third-camera inputs; all 6,204 joint slots and 4,247 accepted point projections.
Maximum recomputed projection difference 5.083e-13 px; maximum recorded rotation
preparation 4.445e-7 per element. Joint slots: 1,461 estimated /84 inconsistent
rays /4,659 missing image evidence. No missing/conflicting slot is promoted.
The audit retains 564 model/image comparison slots, not independent accuracy.
At frame 80, estimated ankle heights are +.0943/+.0978 m above conditional ground,
while model ankle centers are −.0136/−.0379 m; 3D model/image differences .11194/
.14007 m. This supports investigating the fixed-body fit, not shifting ground.
Audit took 0.178527 s without inference, solving, physics or media work.

Final scoped check union: **163 distinct passing Python cases**, taking the
latest outcome per test from `.local/multiview-extra-core-tests-01.xml`,
`multiview-extra-integration-tests-01.xml` and
`multiview-extra-rotation-tests-01.xml`. **69 mocked UI checks** pass in
`.local/multiview-extra-ui-tests-01.log`. The initial conflicting-ray expectation
failure is preserved; existing optimizer limits and legacy solver tests remain
unchanged. No browser/device QA claim.

Reproduce with new output names:

```sh
.venv-lab/bin/python scripts/studies/prepare_aspset_third_camera.py \
  --out artifacts/aspset-third-camera-input-new
.venv-lab/bin/python -m weft camera-body \
  --run artifacts/aspset-camera-body-input-01/left \
  --camera artifacts/aspset-manual-ground-01/camera-0.json \
  --ground artifacts/aspset-manual-ground-01/ground-0.json \
  --root-view artifacts/aspset-camera-body-input-01/root-view.json \
  --additional-view artifacts/aspset-third-camera-input-new/view.json \
  --body-view both --out artifacts/aspset-third-camera-body-lab-new
.venv-lab/bin/python scripts/studies/verify_aspset_third_view.py \
  --out artifacts/aspset-third-view-audit-new
```

The verifier intentionally checks the exact frozen original packages, not the
new reproductions. Construction runtime is frozen at
`.local/aspset-third-camera-lab-01/runtime`: committed `f05dc2f` plus eight exact
root-owned file overlays listed in `plan.json`. No dirty avatar code was copied.
UI, body methods, CLI/config and tests are reviewed together. Commit/push only
owned paths; publication receipt `.local/aspset-third-view-publication-01.json`
will record exact local/remote identities and preserve foreign avatar work.

### Next safe work

- [ ] Explain fixed-body versus supported image-joint discrepancies, preserving
  all observations, frozen ground and the failed candidate. Qualify any changed
  geometry hypothesis on construction evidence before fresh independent checks.
- [ ] Continue coupled equipment/contact, full-interval learning and unfamiliar
  video coverage; these multi-camera manual controls do not close those gates.
- [ ] Refresh Git/remote/ownership before new work; no duplicated inference,
  source acquisition or unchanged controller/ground sweeps.

The complete goal remains active. Passing software, recorded replay or a wider
calibrated-input envelope is not generalized reconstruction or physical transfer.
