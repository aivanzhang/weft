# Saved pose precursor custody

## Metadata check complete — October 31, 2026, 7:29 PM PDT

Owner: reconstruction lane. This new note and ignored
`.local/pose-precursor-custody-01/` are the only edited paths in this step. The
released silhouette/pose/camera/collector modules remain frozen. No live job,
native call, media/raster decode, model inference, network/API request or paid
work occurred. No Git mutation. New qualification implementation is proposed
below but is not authorized or started yet.

## Exact three-member result

Root approved exactly the already named kettlebell run manifest,
`pose-landmarks.json` and `identity-audit.json`, bounded to3 source files/8MiB/10s.
The check read those3 original files once:2,051,472 B in0.013358s. Paths share
`artifacts/mobile-exercise-e2e-10/kettlebell-pushup/run/`.

| Member | Observed bytes | SHA256 |
| --- | ---: | --- |
| `manifest.json` | 10,122 | `e6e88ded526ec582b138dfd90285a0cdde65ddfd33596d255cc23f95afe5529b` |
| `pose-landmarks.json` | 2,024,499 | `e60c7257dead659a5d4add202e2c45c0196556140978b59903b489f24bc7e264` |
| `identity-audit.json` | 16,851 | `2c74ba5a3c842a9bb8921f89f4b9b50763e813b652f310f4f1eebd1575ea7d5b` |

Both member hashes exactly match the supplied run manifest. Its schema1 seal
stores path→SHA only: sizes above are observed copy sizes, not manifest claims.
The raw precursor hash also equals the previously retained audit/raw-artifact
pin. The audit source and all129 raw frame IDs match source
`4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc`.
The supplied manifest was not independently authenticated against another
ancestor in this three-file check; custody is not physical authentication.

The first helper incorrectly required a manifest byte-size field and saved a
terminal failure. That failure was diagnosed from the copied manifest only.
A separate saved-only inspection verified the actual hash-only schema and
finished metadata inspection in0.010289s, with **zero further original reads**.
All source copies and the first failed receipt/helper/seal remain unchanged.

## Observed semantics; no pixel qualification yet

- The precursor wrapper contains only`frames`, `model`, `scope`; it lacks
  source SHA, fps, dimensions, declared/decoded counts and top-level identity
  declaration. No missing field was synthesized.
- It retains129 rows from original frame0 through256, with all recorded decoder
  times.128 rows have exact integer-one model hypotheses and33 landmarks;
  frame0 is exact integer-zero/empty landmarks/`no_pose`.
- Statuses:2`tracked`,125`tracked_partial`,1`low_confidence`,1`no_pose`.
  Identity labels:128`single_model_hypothesis_not_verified_person_count`,
  1`no_model_hypothesis`. No other inspected identity/crop derivation marker is
  present. That absence does not independently authenticate raw identity.
- The audit explicitly remains`mode: temporal`, method
  `short-island-bilateral-continuity-v1`, with0 decisions and0 repaired IDs.
  It retains90 identity-continuity-unavailable rows, including original frame0.
  No repairs is not proof of identity; the audit's own scope leaves the initial
  global left/right assignment unverified. Its unavailable reasons remain in
  the exact copied audit and inspection.
- No landmark coordinates were evaluated, selected, compared to silhouettes or
  emitted as qualified annotations. Original frame0 stays unavailable. This is
  development source evidence, not a fresh holdout or accuracy validation.

## Immutable local evidence

Commands:

```text
python3 .local/pose-precursor-custody-01/check.py
python3 .local/pose-precursor-custody-01/inspect_saved.py
```

| File under `.local/pose-precursor-custody-01/` | SHA256 |
| --- | --- |
| `plan.json` | `4a0c431ca3dd443158f84f23fe5a1a51c69f41d94eeba4956319aff0a9a87b71` |
| `check.py` | `286cbb2caff124350f211bbecaa65baa381eeda1865464399f10518da6d95979` |
| `result.json` — preserved helper failure | `f810b5b09732f9bb53dcb140f940ea0fd427b6b22c87c6f0d61c0f3952aa6aba` |
| `manifest.json` — original local seal | `952d20d1d56a76342749a41b129706ec145fe886f22280d9deafe1d9be3260ca` |
| `inspect_saved.py` | `fdb8b850025499dc66d7944b81930c4fa13e75131cded8d5d0ef20a5f7d29b0c` |
| `saved-inspection.json` | `43a001757bc948ccf811009bfd96623d6ec3edbffe1f532a3485c69319a714b9` |
| `saved-inspection-manifest.json` | `0f1c1648f846eae66004d711051d5cd396cfd109b44115b299dbb55afe7545d7` |

The second seal links the first seal and new inspection without rewriting its
terminal failure. Exact copied source members live under`inputs/` and are
included in the original local seal. No tests or source job were rerun.

## Smallest useful qualification proposal (awaiting root)

The remaining context is the already known sealed full pose document
`run/evidence.json`, SHA
`df21a959787fc098711f4762151cc72c3d53bb87e4199fd2e901fc465a1b4ec8`.
Its metadata was read in the earlier bounded discovery, but its bytes were not
copied into this three-member package. A separately approved single read bounded
to4MiB/5s could pin it against the already copied run manifest and compare its
complete frame/source/clock/model metadata with the precursor. No source media
is needed. The old candidate human observation pins a different pose document;
that mismatch must not be silently accepted.

Proposed new paths (confirm absent before implementation):
`src/weft/pose_precursor_custody.py`,
`tests/test_pose_precursor_custody.py`, plus this note. For actual usability,
request a narrow explicit optional`precursor_custody` argument in the existing
`pose_surface_annotations.py`; its default raw guards and serialized outputs
must stay unchanged. Do not implement a disconnected alternate pose annotator.

The new pure prepared proof would take **exact** manifest, original precursor,
audit and full-context pose bytes. It would preserve all four byte identities,
context`mode: temporal`, audit mode/config/unavailable records and full original
denominators. Per-row admission requires exact source/frame/clock/model joins
and equality of the original precursor's unchanged image landmark/score payload
with the supplied context row, not merely zero repairs. Crop/identity-derivation
markers, unknown labels, changed arrays, contradictory clocks or missing
members remain unqualified; no temporal label is removed or rewritten.

The annotation adapter would accept only that typed proof pinned to its exact
context bytes, label admitted coordinates as matched saved precursor estimates
with **unverified model/anatomical identity**, and embed all custody/uncertainty
provenance in the ordinary annotation document. Existing observation adapters
can create a **new** canonical document from the exact context, with a new raw
document identity; the mismatching older candidate observation stays unchanged.
No source confidence becomes a pixel-error bound and no audit-unavailable row
becomes authenticated identity. All original frame/missing/excluded slots and
explicit study partitions survive.

Suggested pure bounds:4 supplied documents/16MiB,≤900 pose rows/1800 source
slots,5s cooperative preparation,≤2MiB prepared proof/cache, no per-query whole
evidence copies. Test exact unchanged precursor, derived/unknown/crop/clock
contradictions, hash-only seal semantics, missing context, integer counts and
uncorrected-versus-authenticated identity distinctions. No inference/native/
raster study follows automatically; root reviews the new contract first.
