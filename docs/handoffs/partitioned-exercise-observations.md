# Partition-safe exercise observations

## Frozen for review — October 31, 2026, 8:21 PM PDT

Owned paths: `src/weft/observations.py`, new
`tests/test_partitioned_exercise_observations.py`, and this note. Root owns
CLI/lab/runtime/publication. Completed precursor files and its handoff remain
frozen at `.local/pose-precursor-qualification-release-03.json`. No live job.

Implemented change: optional `construction_frames=None` in existing
`adapt_exercise`. Default output remains exact. An explicit ordered selection
must identify original sampled frames; it is not inferred from favorable pose,
contact or scoring results. Only selected rows reach `_record`, deep-copy and
measurement generation. Excluded sampled slots export typed identity/clock/
status/reason metadata, not coordinates. Original source/canonical identity and
every source slot remain; later/missing frames are never silently dropped.

Reuses the current canonical schema and validators. Opt-in bounds are
900 pose rows/selected IDs and1800 original source slots, with existing64MB
canonical output limit. Full-document fingerprinting remains an opaque custody
operation over the original JSON; it does not admit excluded coordinate values
for inspection or measurements. The caller still retains exact original input
bytes. The ordinary synchronous exporter is not a hard-isolated worker.

Verification passed: narrow excluded-coordinate/deep-copy sentinels, exact default-byte
goldens, malformed selection/metadata guards, source timing/count preservation,
and an analytical composition through existing precursor annotations. No new
source read, saved-source annotation/raster run, inference/native/API/network or
trial rerun. Native/model/media/network/subprocess work blocked in pure tests.

The completed kettlebell development pilot found75 exact arrays in77 fixed
prefix requests, preserving2 unavailable poses and all259 source slots. It did
not establish a matching canonical observation or correct person/contact
identity. This integration must create a new document from the exact source,
not relabel the older observation that pins another canonical pose. Actual
source-to-annotation-to-silhouette acceptance remains a separately reviewed
next step; no standalone custody study will be repeated.

`observations.py` belongs to the completed fitted trial's runtime closure. Root
will exclude this active edit from its imminent native runtime. Do not repin
recipes, reinterpret old trial evidence or rerun the trial for this feature.

## Public contract and downstream use

```python
observations = adapt_exercise(
    exact_parsed_source,
    construction_frames=predeclared_original_sampled_frame_ids,
)
```

The optional value is a list of at most900 strictly increasing distinct exact
integer sampled frame IDs. Empty selection is valid and exports metadata only.
Bool/float IDs, unsampled selections, duplicate/unordered source rows, foreign
IDs or conflicting original nominal clocks fail. Source model-hypothesis count
is exact integer0–128 or null; missing count does not establish one person.
Decoder timestamps remain supplied, not synchronized/recovered here. Pose and
identity labels are bounded scalar strings or null; structured sentinel
payloads fail. Known derivation marker **names** survive without their arrays.

The default `None` branch bypasses all new selection logic. Existing canonical
source fields, original nominal/decoder times, model depth/world-unit warnings,
authored scene relationships and measurements remain unchanged. The new branch
leaves selected evidence IDs/payloads intact. For excluded sampled frames it
emits canonical observations with `partition="unassigned"`, unknown visibility,
empty measurements and no evidence references. It does not fabricate a new
pose record or pretend metadata is an original full pose payload.

`provenance.construction_selection` has profile
`explicit-exercise-construction-selection-v1`, the exact selected list,
`source_slots` and complete counts. Its slots retain original frame/evidence ID,
nominal and decoder clocks, status/identity labels, hypothesis count, marker
names and exclusion/missing reason. For the normal whole-clip case, decoded
observations remain complete and any container-only tail remains in this slot
ledger plus original source counts. Selected-interval inputs retain absolute
IDs and all original metadata slots up to the explicit1800-slot bound; slots
outside the decoded interval remain unknown. Nothing shifts the source start.

The complete source JSON is still fingerprinted by existing canonical hashing.
Exclusion means **no coordinate access for copying, transformation or
measurement**, not that excluded bytes are never parsed/hashed for custody.
Changing an excluded value changes the full input pin without exposing that
value in the exported evidence. Exact original source bytes remain the
caller's immutable retained input. The canonical output is a new document;
no prior candidate or source pose pin is rewritten.

Existing `load_world_evidence` and `prepare_pose_surface_annotations` consume
this schema directly; no new registry, namespace or simulator is introduced.
The analytical integration uses actual precursor custody and both explicitly
named wrist channels with an authored2px range/.5 confidence threshold. Only
the two selected frames reach qualification; the excluded frame's coordinate
sentinel remains unqueried, all7 source slots/14 channel slots remain, and
contact/identity stay unknown. These authored test settings do not authorize
or calibrate a future real-source pixel-error bound.

## Verification and retained failure

Before editing, current `observations.py` was verified byte-identical to
committed `9cbc2836c5fef361b378e932e4c0e8b57b42b26d`, SHA
`321a4505e8d5e5a2e1c340c3602e88e24b0f1d8c76bf5c19cafd123f53bab849`.
An analytical row that forbids coordinate access/deep-copy proved that the old
export-before-partition workflow reaches excluded original-row deep-copy.
The saved baseline also captures four exact default serialized hashes: ordinary
pose, original no-pose plus authored scene, fractional fps, and selected source
interval. Tests retain these golden constants, not a dependency on a local
baseline file or importing the old implementation.

```text
.venv-lab/bin/python -m pytest tests/test_partitioned_exercise_observations.py -q -k 'excluded_rows_never or all_excluded_selection' -o junit_family=xunit1 --tb=short --junitxml=.local/partitioned-exercise-observations-red-01.xml
.venv-lab/bin/python -m pytest tests/test_partitioned_exercise_observations.py -q -o junit_family=xunit1 --tb=short --junitxml=.local/partitioned-exercise-observations-green-01.xml
```

| Receipt | Result | SHA256 |
| --- | --- | --- |
| `.local/partitioned-exercise-default-baseline-01.json` | Four serialized goldens + old excluded-copy reproduction | `c490b133d4a144aebf5f0e900c88592facd3aea4b251aa2a04202652a9c08b38` |
| `.local/partitioned-exercise-observations-red-01.xml` | Two expected failures: selection keyword not implemented yet | `b770ce0cdd0a6a0d859031ae182dbda37a598ce0edc220a628bddc606c6a8b28` |
| `.local/partitioned-exercise-observations-green-01.xml` | 35 focused cases pass, .181s | `f58a32b51af3a4c10b8b31adeb80e3265561d84584fa046ae2112acc0f7bb5c5` |

No unchanged legacy suite was rerun. The35 cases cover guarded selection before
fingerprinting, source row/count envelopes, metadata type and source/clock
contradictions, all-excluded behavior, nonmutation, exact default bytes,
fractional clock and absolute interval identity, source-start unavailability,
opaque canonical pin coverage and the actual precursor→annotation join.

One maximum-row metadata workload exported900 saved pose rows/1800 original
slots in0.0757235s. It emitted1,160,400 B, retained1799 decoded observations and
all1800 source slots, but copied only the two selected pose payloads. This is
one bounded analytical responsiveness measurement, not a hard timing/memory
guarantee or real-source accuracy result. Full input parsing/fingerprinting
inherits existing trusted-exporter behavior; no new hostile-input worker or
hard isolation is claimed. The existing64MB canonical document cap is unchanged.

## Frozen paths and next action

| Path | SHA256 |
| --- | --- |
| `src/weft/observations.py`,44,887 B | `dbbfa6406f7f78a33fe492990842937dfd6f1ac6cd5fc98e376143869b35b38f` |
| `tests/test_partitioned_exercise_observations.py`,13,826 B | `379f56072230e6f2927d6a708468f022dc06e5a208cd7a165cfd24e1b2fcc625` |

All three final path hashes, receipts and baseline identity are recorded in
`.local/partitioned-exercise-observations-release-01.json`. Scoped whitespace
checks passed. No Git mutations, source reads, raster/native/inference/API
calls, old recipe repins, trial reruns or edits to completed precursor files.
All processes are terminal; no live handles. Root owns integration/publication.

After review, the next actual acceptance should reuse the already saved
qualified kettlebell prefix through **this ordinary canonical exporter →
ordinary pose annotations with explicit precursor custody → existing equipment
silhouette analysis**, preserving the supplied derivative proof and every
missing/occluded/excluded slot. It must use a new immutable output, explicit
wrist names, predeclared authored uncertainty/confidence and bounded mask
loads. No such source/raster run has been started or newly authorized here.
Image overlap/exclusion still cannot establish physical contact/noncontact,
grasp, load, anatomy or reliable unfamiliar-video reconstruction.
