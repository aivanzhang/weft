"""Actual pure selector→prepared-mask constraint, without source/model work."""

import builtins
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import time

import pytest

from test_multiview_landmark_evidence import local_only
from test_equipment_silhouette_evidence import fixture as silhouette_fixture
from test_exercise_support_evidence import prior
from weft.exercise_support_evidence import estimate_exercise_support


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(*, raster=b'\1' * 9, count=31, pose_change=None):
    module = importlib.import_module('weft.equipment_silhouette_evidence')
    selected = list(range(0, count - 1, 2))[:int(.6 * len(range(0, count - 1, 2)))]
    def torso(pose):
        for row in pose['frames']:
            for index, xy in {11: (.3, .15), 12: (.7, .15), 23: (.35, .75), 24: (.65, .75)}.items():
                row['landmarks'][index].update(x=xy[0], y=xy[1])
        if pose_change:
            pose_change(pose)
    def exclude_later(document):
        for row in document['observations']:
            if row['partition'] != 'unsampled' and row['frame'] not in selected:
                row.update(partition='unassigned', visibility='unknown', evidence_ids=[], measurements={})
    annotations, args, raster_calls, _ = silhouette_fixture(module, count=count, raster=raster,
        pose_change=torso, observation_change=exclude_later)
    prepared = module.prepare_equipment_silhouette_evidence(annotations, construction_frames=selected, **args)
    pose = json.loads(annotations.pose_document)
    selector = {'camera_prior': prior(pose['source_sha256']), 'equipment_observations': args['equipment_document'],
                'equipment_ids': args['equipment_ids']}
    baseline = estimate_exercise_support(pose, **selector)
    return pose, selector, prepared, baseline, raster_calls


def constrained(baseline, prepared):
    return importlib.import_module('weft.equipment_silhouette_constraint').constrain_equipment_support(
        baseline, prepared, deadline=time.monotonic() + 10)


def recovered(result):
    baseline = deepcopy(result)
    extra = baseline.pop('equipment_silhouette_constraint')
    baseline['counts']['equipment_status_counts'] = extra['baseline_equipment_status_counts']
    for row in baseline['rows']:
        for pair in row['equipment_pairs']:
            pair['status'] = pair.pop('co_motion_baseline_status')
            pair.pop('silhouette_constraint')
    return baseline


def changed_frame(prepared, frame, change):
    rows = list(prepared._rows)
    row = json.loads(rows[frame]); change(row)
    rows[frame] = encode(row)
    return replace(prepared, _rows=tuple(rows))


def changed_metadata(prepared, change):
    metadata = prepared.describe(); change(metadata)
    return replace(prepared, _metadata=encode(metadata))


def test_existing_selector_opt_in_keeps_supported_windows_and_exact_baseline(tmp_path, monkeypatch):
    pose, args, prepared, baseline, calls = fixture()
    before = encode(baseline)
    original_loads = len(calls)
    # A restored cache needs no live qualifier or annotation point queries.
    prepared = replace(prepared, pose_annotations=replace(prepared.pose_annotations, precursor_custody=None))
    monkeypatch.setattr(type(prepared.pose_annotations), 'frame', lambda *_: pytest.fail('Repeated annotation query'))
    result = estimate_exercise_support(pose, **args, equipment_silhouette=prepared)
    extra = result['equipment_silhouette_constraint']
    assert extra['counts']['baseline_candidates'] > 0
    assert extra['counts']['kept_candidates'] == extra['counts']['baseline_candidates']
    assert extra['counts']['excluded_candidates'] == extra['counts']['unavailable_candidates'] == 0
    assert encode(recovered(result)) == before == encode(baseline)
    assert extra['baseline_canonical_sha256'] == sha256(before).hexdigest()
    assert result['coverage'] == baseline['coverage']
    assert [row.get('window') for row in result['rows']] == [row.get('window') for row in baseline['rows']]
    assert len(calls) == original_loads and extra['work']['raster_calls'] == extra['work']['pose_annotation_queries'] == 0
    assert extra['work']['selected_frame_queries'] == len(extra['construction_frames']) == 9
    assert all(extra[name] == 'unknown' for name in ('contact', 'physical_noncontact', 'grasp', 'load'))
    target = os.environ.get('WEFT_SILHOUETTE_CONSTRAINT_FIXTURE_DIR')
    output = Path(target) if target else tmp_path / 'analytical-payload'
    if target:
        assert output.parts[:1] == ('.local',) and '..' not in output.parts
    output.mkdir()
    for name, raw in {'support-evidence.json': encode(result), 'baseline.json': before,
        'pose.json': prepared.pose_annotations.pose_document, 'equipment.json': prepared.equipment_document,
        'mask-metadata.json': prepared.mask_document, 'silhouette-metadata.json': prepared._metadata,
        'silhouette-frames.jsonl': b'\n'.join(prepared._rows) + b'\n',
        'annotation.json': prepared.pose_annotations.annotation_document(),
        'annotation-metadata.json': prepared.pose_annotations._metadata,
        'annotation-frames.jsonl': b'\n'.join(prepared.pose_annotations._rows) + b'\n',
        'human-observations.json': prepared.pose_annotations.input_documents[0][1]}.items():
        (output / name).write_bytes(raw)


def test_mask_hole_excludes_only_original_candidates():
    _, _, prepared, baseline, _ = fixture(raster=b'\1\1\1\1\0\1\1\1\1')
    result = constrained(baseline, prepared)
    extra = result['equipment_silhouette_constraint']
    assert extra['counts']['excluded_candidates'] == extra['counts']['baseline_candidates'] > 0
    assert extra['counts']['kept_candidates'] == extra['counts']['unavailable_candidates'] == 0
    for before, after in zip(baseline['rows'], result['rows']):
        for old, new in zip(before['equipment_pairs'], after['equipment_pairs']):
            assert new['status'] == ('image_silhouette_exclusion' if old['status'] == 'projected_co_motion_candidate' else old['status'])
    assert encode(recovered(result)) == encode(baseline)


@pytest.mark.parametrize('kind', ['missing', 'pixel', 'confidence', 'clock', 'entity', 'human_observation', 'evidence', 'representation'])
def test_one_unavailable_or_conflicting_point_abstains_every_original_covering_window(kind):
    _, _, prepared, baseline, _ = fixture()
    def change(row):
        point = row['pairs'][0]
        if kind == 'missing': point.update(status='unavailable', reason='mask_occluded', intersects=None)
        if kind == 'pixel': point['pixel'] = [36., 33.]
        if kind == 'confidence': point['source_confidence'] = .7
        if kind == 'clock': point['equipment_time_s'] += .001
        if kind == 'entity': point['human_entity_ref']['object_id'] = 'another-person'
        if kind == 'human_observation': point['human_observation_id'] += '-other'
        if kind == 'evidence': point['mask_evidence_id'] += '-other'
        if kind == 'representation': point['representation'] = 'identity_derived'
    result = constrained(baseline, changed_frame(prepared, 8, change))
    assert result['equipment_silhouette_constraint']['counts']['unavailable_candidates'] > 0
    for old, new in zip(baseline['rows'][:9], result['rows'][:9]):
        for a, b in zip(old['equipment_pairs'], new['equipment_pairs']):
            if a['status'] != 'projected_co_motion_candidate':
                assert b['status'] == a['status']
                continue
            left, right = b['silhouette_constraint']['window_point_range']
            includes = left <= 4 < right and a['hand'] == 'left_wrist'
            assert b['status'] == ('unavailable' if includes else a['status'])
    assert encode(recovered(result)) == encode(baseline)


def test_identity_raw_baseline_origin_cannot_reuse_context_point_by_pixel_equality():
    _, _, prepared, baseline, _ = fixture()
    baseline['rows'][4]['landmark_origin'] = 'identity_raw'
    result = constrained(baseline, prepared)
    point = result['equipment_silhouette_constraint']['point_rows'][4]['pairs'][0]
    assert point['status'] == 'unavailable' and point['reason'] == 'source_point_origin_not_qualified'
    assert result['equipment_silhouette_constraint']['counts']['unavailable_candidates'] > 0


def test_unavailable_takes_precedence_over_exclusion_but_retains_both_counts():
    _, _, prepared, baseline, _ = fixture(raster=b'\1\1\1\1\0\1\1\1\1')
    prepared = changed_frame(prepared, 8, lambda row: row['pairs'][0].update(status='unavailable', reason='mask_occluded', intersects=None))
    result = constrained(baseline, prepared)
    pair = result['rows'][4]['equipment_pairs'][0]
    assert pair['status'] == 'unavailable'
    counts = pair['silhouette_constraint']['window_image_counts']
    assert counts['unavailable'] == 1 and counts['exclusion'] > 0


def test_non_candidate_baseline_never_promoted_by_silhouette():
    def moving(pose):
        for index, row in enumerate(pose['frames']):
            row['landmarks'][15]['x'] += .05 * (index % 2)
    _, _, prepared, baseline, _ = fixture(pose_change=moving)
    result = constrained(baseline, prepared)
    for old, new in zip(baseline['rows'], result['rows']):
        for a, b in zip(old['equipment_pairs'], new['equipment_pairs']):
            if a['status'] != 'projected_co_motion_candidate':
                assert b['status'] == a['status']
                assert b['silhouette_constraint']['status'] in ('not_candidate', 'excluded_later')
    assert encode(recovered(result)) == encode(baseline)


@pytest.mark.parametrize('kind', ['source', 'pose', 'raw_pose', 'equipment', 'mask', 'prefix', 'scope', 'contact'])
def test_incompatible_global_input_bindings_reject_before_selected_queries(monkeypatch, kind):
    _, _, prepared, baseline, _ = fixture()
    api = importlib.import_module('weft.equipment_silhouette_constraint')
    def change(meta):
        if kind == 'source': meta['source_sha256'] = 'f' * 64
        if kind == 'pose': meta['pose_provenance']['pose_canonical_sha256'] = 'f' * 64
        if kind == 'raw_pose': meta['pose_provenance']['pose_raw_sha256'] = 'f' * 64
        if kind == 'equipment': meta['equipment_document_sha256'] = 'f' * 64
        if kind == 'mask': meta['mask_document_sha256'] = 'f' * 64
        if kind == 'prefix': meta['construction_frames'].append(18)
        if kind == 'scope': meta['canonical_frame_selection']['partition_consistency_scope'] = 'all_frames'
        if kind == 'contact': meta['contact'] = 'verified'
    prepared = changed_metadata(prepared, change)
    original = api._decode
    rows = {id(raw) for raw in prepared._rows}
    def decode(raw, *args):
        assert id(raw) not in rows, 'Invalid global binding reached frame cache'
        return original(raw, *args)
    monkeypatch.setattr(api, '_decode', decode)
    with pytest.raises(ValueError): constrained(baseline, prepared)


def test_coherently_repinned_raw_pose_must_derive_the_declared_canonical_identity(monkeypatch):
    _, _, prepared, baseline, _ = fixture()
    api = importlib.import_module('weft.equipment_silhouette_constraint')
    pose = json.loads(prepared.pose_annotations.pose_document)
    pose['frames'][-1]['landmarks'][15]['x'] += .01
    raw = encode(pose)
    pose_meta = prepared.pose_annotations.describe()
    pose_meta['provenance']['pose_raw_sha256'] = sha256(raw).hexdigest()
    annotation = json.loads(prepared.pose_annotations.annotation_document())
    annotation['provenance'] = encode(pose_meta['provenance']).decode()
    annotation_raw = encode(annotation)
    pose_meta['annotation_sha256'] = sha256(annotation_raw).hexdigest()
    annotations = replace(prepared.pose_annotations, pose_document=raw,
                          _metadata=encode(pose_meta), _annotation=annotation_raw)
    meta = prepared.describe()
    meta['pose_provenance'] = pose_meta['provenance']
    prepared = replace(prepared, pose_annotations=annotations, _metadata=encode(meta))
    assert meta['pose_provenance']['pose_canonical_sha256'] == baseline['pose_evidence_canonical_sha256']
    assert sha256(raw).hexdigest() != baseline['pose_evidence_canonical_sha256']
    original = api._decode
    rows = {id(row) for row in prepared._rows}
    def decode(value, *args):
        assert id(value) not in rows, 'Unjoined raw pose reached selected frame cache'
        return original(value, *args)
    monkeypatch.setattr(api, '_decode', decode)
    with pytest.raises(ValueError, match='canonical pose identity'):
        constrained(baseline, prepared)


def test_excluded_frame_payloads_are_never_decoded(monkeypatch):
    _, _, prepared, baseline, _ = fixture()
    api = importlib.import_module('weft.equipment_silhouette_constraint')
    selected = set(prepared.describe()['construction_frames'])
    forbidden = {id(raw) for index, raw in enumerate(prepared._rows) if index not in selected}
    original = api._decode
    def decode(raw, *args):
        assert id(raw) not in forbidden, 'Excluded frame payload decoded'
        return original(raw, *args)
    monkeypatch.setattr(api, '_decode', decode)
    result = constrained(baseline, prepared)
    assert result['coverage'] == baseline['coverage']
    assert result['equipment_silhouette_constraint']['work']['selected_frame_queries'] == 9


def test_native_origin_and_decoder_clock_mismatch_abstains_without_retiming():
    _, _, prepared, baseline, _ = fixture()
    changed = changed_frame(prepared, 8, lambda row: row.update(pose_decoder_timestamp_s=row['pose_decoder_timestamp_s'] + .014))
    result = constrained(baseline, changed)
    points = result['equipment_silhouette_constraint']['point_rows'][4]['pairs']
    assert all(point['reason'] == 'source_frame_or_clock_differs' for point in points)
    assert result['rows'][4]['decoder_timestamp_s'] == baseline['rows'][4]['decoder_timestamp_s']


def test_default_none_is_lazy_and_identical(monkeypatch):
    pose, args, _, baseline, _ = fixture()
    # Published 791127d produced these exact bytes from this analytical fixture.
    assert sha256(encode(baseline)).hexdigest() == 'adabae52b761e440a7b3e9db84a6ea12e9ac4b7d170dfb489fb92bec64652417'
    original = builtins.__import__
    def imported(name, *values, **kwargs):
        assert name != 'weft.equipment_silhouette_constraint', 'Default imported the optional consumer'
        return original(name, *values, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    assert encode(estimate_exercise_support(pose, **args, equipment_silhouette=None)) == encode(baseline)


@pytest.mark.parametrize('kind', ['exact', 'missing', 'unverified', 'alias', 'clock'])
def test_actual_loaded_derivative_proof_preserves_both_clocks(tmp_path, kind):
    from test_exercise_support_evidence import _equivalent_equipment
    data, _, proof = _equivalent_equipment(tmp_path)
    module = importlib.import_module('weft.equipment_silhouette_evidence')
    annotations, args, _, _ = silhouette_fixture(module, supplied_pose=data, proof=proof, raster=b'\1' * 9)
    prepared = module.prepare_equipment_silhouette_evidence(annotations, construction_frames=[0], **args)
    pose = json.loads(annotations.pose_document)
    baseline = estimate_exercise_support(pose, camera_prior=prior(pose['source_sha256']),
        equipment_observations=args['equipment_document'], equipment_ids=args['equipment_ids'],
        equipment_source_equivalence=proof)
    if kind == 'missing': prepared = replace(prepared, source_equivalence=None)
    if kind == 'unverified': prepared = replace(prepared, source_equivalence=proof.inspection())
    if kind == 'alias':
        prepared = changed_metadata(prepared, lambda meta: meta['canonical_frame_selection'].update(source_aliases={}))
    if kind in ('missing', 'unverified', 'alias'):
        with pytest.raises(ValueError): constrained(baseline, prepared)
        return
    if kind == 'clock':
        prepared = changed_frame(prepared, 0, lambda row: row['pairs'][0].update(equipment_time_s=row['pose_decoder_timestamp_s']))
    result = constrained(baseline, prepared)
    extra = result['equipment_silhouette_constraint']
    assert extra['input_bindings']['source_equivalence_manifest_sha256'] == proof.manifest_sha256
    assert result['rows'][0]['equipment_source_mapping'] == baseline['rows'][0]['equipment_source_mapping']
    assert result['rows'][0]['equipment_pairs'][0]['object_time_s'] != result['rows'][0]['decoder_timestamp_s']
    if kind == 'clock':
        assert extra['point_rows'][0]['pairs'][0]['reason'] == 'qualified_pair_identity_or_clock_differs'
    else:
        assert extra['counts']['point_status_counts'] == {'intersection': 2}
    assert encode(recovered(result)) == encode(baseline)


def test_extra_constraint_requires_exact_equipment_document():
    pose, _, prepared, _, _ = fixture()
    with pytest.raises(ValueError, match='exact equipment observation'):
        estimate_exercise_support(pose, camera_prior=prior(pose['source_sha256']), equipment_silhouette=prepared)


def test_cache_exhaustion_retains_no_partial_result(monkeypatch):
    _, _, prepared, baseline, _ = fixture()
    api = importlib.import_module('weft.equipment_silhouette_constraint')
    before = encode(baseline)
    monkeypatch.setattr(api, 'MAX_CACHE_BYTES', 1)
    with pytest.raises(ValueError, match='cache bound'):
        constrained(baseline, prepared)
    assert encode(baseline) == before


def test_original_and_final_deadlines_are_enforced(monkeypatch):
    _, _, prepared, baseline, _ = fixture()
    api = importlib.import_module('weft.equipment_silhouette_constraint')
    with pytest.raises(TimeoutError):
        api.constrain_equipment_support(baseline, prepared, deadline=time.monotonic() - 1)
    original = api._encode
    clock = time.monotonic
    late = [False]
    monkeypatch.setattr(api.time, 'monotonic', lambda: clock() + (6 if late[0] else 0))
    def encode_late(value, *args):
        result = original(value, *args)
        if 'equipment_silhouette_constraint' in value:
            late[0] = True
        return result
    monkeypatch.setattr(api, '_encode', encode_late)
    with pytest.raises(TimeoutError): constrained(baseline, prepared)


def test_cached_overlapping_windows_require_one_point_pass(tmp_path):
    _, _, prepared, baseline, _ = fixture(count=301)
    started = time.perf_counter()
    result = constrained(baseline, prepared)
    elapsed = time.perf_counter() - started
    extra = result['equipment_silhouette_constraint']
    assert extra['work']['selected_frame_queries'] == 90
    assert extra['work']['point_slots'] == 180
    assert extra['work']['prefix_counter_entries'] == 91 * 3 * 2
    assert extra['counts']['kept_candidates'] == extra['counts']['baseline_candidates'] > 0
    assert elapsed < 2.
    (tmp_path / 'performance.json').write_text(json.dumps({'construction_frames': 90, 'point_slots': 180,
        'window_queries': extra['work']['window_queries'], 'seconds': elapsed, 'output_bytes': len(encode(result)),
        'cache_bytes': extra['work']['serialized_point_cache_bytes'] + extra['work']['prefix_counter_buffer_bytes']}))
