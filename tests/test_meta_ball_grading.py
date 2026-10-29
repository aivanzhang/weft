"""Analytic controls for source-mapped, coverage-aware offline grading."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('check_meta_balls', Path(__file__).parents[1]/'scripts/studies/check_meta_balls.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def manual():
    return {'source_sha256': study.SOURCE_HASH, 'objects': list(study.NAMES),
        'observations': [{'frame': frame, 'centers': [[20,30],[40,50],None if frame in (2580,2760) else [60,70]],
                          'visibility': ['visible','visible','occluded' if frame == 2580 else 'out_of_frame' if frame == 2760 else 'visible'],
                          'identity_status': 'inferred after contact'} for frame in study.LABEL_FRAMES]}


def frames():
    return [{'frame': f, 'objects': [{'frame': f, 'object_id': str(i), 'status': 'missing',
                                    'box_xyxy_halfopen': None} for i in (1,2,3)]} for f in range(181)]


def put(rows, source_frame, object_id, center):
    obj = rows[source_frame-study.START]['objects'][object_id-1]
    obj.update(status='returned', box_xyxy_halfopen=[center[0]-2,center[1]-2,center[0]+2,center[1]+2])


def test_all_manual_slots_missing_are_not_zero_error():
    result = study.grade(frames(), manual())
    assert result['manual_slots'] == 24 and result['eligible_postinit_center_slots'] == 17
    assert result['missing_eligible_center_slots'] == 17 and result['returned_eligible_center_slots'] == 0
    assert result['conditional_rmse_px'] is None and result['conditional_max_error_px'] is None
    assert result['postinit_frames'] == 140 and result['postinit_object_slots'] == 420
    assert result['all_three_returned_postinit_frames'] == 0
    assert all(row['scored_error_px'] is None for row in result['rows'])


def test_initialization_preinit_and_unknown_absence_are_never_accuracy_credit():
    rows = frames()
    for frame in (2580,2620):
        for identity in (1,2,3):
            put(rows, frame, identity, [200,200])
    put(rows,2760,3,[200,200])
    result = study.grade(rows,manual())
    assert result['returned_eligible_center_slots'] == 0
    assert result['conditional_rmse_px'] is None
    assert result['predictions_on_labeled_absence'] == 1
    assert result['predictions_on_unknown_visibility'] == 1
    assert sum(row['period'] == 'pre_initialization' for row in result['rows']) == 3
    assert sum(row['period'] == 'initialization' for row in result['rows']) == 3


def test_sparse_error_remains_conditional_and_unlabelled_coverage_separate():
    rows = frames()
    put(rows,2640,1,[23,34])
    for identity in (1,2,3):
        put(rows,2646,identity,[100,100])
    result = study.grade(rows,manual())
    assert result['returned_eligible_center_slots'] == 1 and result['eligible_center_coverage'] == 1/17
    assert result['conditional_rmse_px'] == 5 and result['within_5px_label_tolerance'] == 1
    assert result['all_three_returned_source_frames'] == [2646]
    assert result['all_three_returned_postinit_coverage'] == 1/140
    assert result['postinit_returned_object_slots'] == 4


def test_all_eligible_centers_compare_by_frozen_identity_not_nearest_neighbor():
    rows, labels = frames(), manual()
    for label in labels['observations']:
        if label['frame'] > study.INITIAL:
            for i,center in enumerate(label['centers']):
                if center:
                    put(rows,label['frame'],i+1,center)
    exact = study.grade(rows,labels)
    assert exact['returned_eligible_center_slots'] == 17 and exact['conditional_rmse_px'] == 0
    put(rows,2640,1,[40,50])
    swapped = study.grade(rows,labels)
    assert swapped['conditional_rmse_px'] > 0
    assert swapped['rows'][6]['scored_error_px'] == pytest.approx(800**.5)


@pytest.mark.parametrize('mutation', [
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(frame=1),
    lambda rows: rows[0]['objects'].pop(),
    lambda rows: rows[0]['objects'][0].update(object_id='2'),
    lambda rows: rows[0]['objects'][0].update(frame=1),
    lambda rows: rows[0]['objects'][0].update(box_xyxy_halfopen=[0,0,2,2]),
    lambda rows: rows[0]['objects'][0].update(status='returned',box_xyxy_halfopen=[0,0,float('nan'),2]),
    lambda rows: rows[0]['objects'][0].update(status='returned',box_xyxy_halfopen=[0,0,427,2]),
    lambda rows: rows[0]['objects'][0].update(status='ambiguous',box_xyxy_halfopen=None),
])
def test_malformed_or_incomplete_predictions_fail_closed(mutation):
    rows = frames()
    mutation(rows)
    with pytest.raises(ValueError):
        study.grade(rows,manual())


def test_prior_trace_excludes_rejected_hypotheses_and_uses_same_box_center():
    trace = {'frames': [{'frame': f, 'objects': [{'object_id': name, 'status': 'not_detected',
                'box_xywh': [18,28,4,4]} for name in study.NAMES]} for f in range(2580,2761)]}
    trace['frames'][60]['objects'][0]['status'] = 'tracked'
    result = study.grade(study.prior_frames(trace),manual())
    assert result['returned_eligible_center_slots'] == 1 and result['conditional_rmse_px'] == 0
    assert result['rows'][6]['prediction_center_px'] == [20,30]
    trace['frames'][0]['objects'][1]['object_id'] = study.NAMES[0]
    with pytest.raises(ValueError):
        study.prior_frames(trace)


def test_bad_manual_mapping_or_missing_visible_center_fails():
    labels = manual()
    labels['observations'][0]['centers'][0] = None
    with pytest.raises(ValueError):
        study.grade(frames(),labels)
    labels = manual()
    labels['objects'].reverse()
    with pytest.raises(ValueError):
        study.grade(frames(),labels)


def test_required_evidence_must_be_inside_seal(tmp_path):
    study.write_json(tmp_path/'report.json',{'status':'fixture'})
    study.seal_suite(tmp_path)
    study.write_json(tmp_path/'unsealed.json',{})
    with pytest.raises(ValueError,match='manifest'):
        study._sealed(tmp_path,['unsealed.json'])


def _saved_fixture(out, *, direction='forward'):
    import shutil
    inputs, tracking = out/'inputs', out/'tracking'
    inputs.mkdir(parents=True)
    (tracking/'call').mkdir(parents=True)
    shutil.copyfile(study.__file__,inputs/'check_meta_balls.py')
    study.write_json(inputs/'freeze.json',{'video_sha256':'fixture-sha'})
    study.write_json(inputs/'manual.json',manual())
    for name in ('prior-regraded','prior-trace','prior-comparison','video-frame-mapping-and-pixels'):
        study.write_json(inputs/f'{name}.json',{})
    study.write_json(inputs/'report.json',{'status':'fixture'})
    study.seal_suite(inputs)
    prompt = '<40f>' + ','.join(f'{i+1}<|box;x1={x};y1={y};x2={x+w-1};y2={y+h-1};w=426;h=240|>'
                               for i,(x,y,w,h) in enumerate(study.BOXES))
    study.write_json(tracking/'request.json',{'metadata':{'propagation_direction':direction},
        'input':[{'content':[{'type':'input_text','text':prompt},{'type':'input_video','video_url':'fixture'}]}]})
    study.write_json(tracking/'call/call.json',{'status':'completed','video_sha256':'fixture-sha',
        'initialization':study.PROMPTS,'request_sha256':study.digest(tracking/'request.json')})
    study.write_json(tracking/'call/segmentation.json',{'status':'completed','initialization':study.PROMPTS,'frames':frames()})
    study.write_json(tracking/'video.json',{'source_sha256':'fixture-sha','frame_count':181,'width':426,'height':240})
    study.write_json(tracking/'report.json',{'status':'fixture'})
    study.seal_suite(tracking)


def test_offline_grade_verifies_nested_evidence_and_seals_once(tmp_path):
    out = tmp_path/'study'
    _saved_fixture(out)
    result = study.grade_saved(out)
    assert result['verified_files'] > 10
    comparison = study._json(out/'comparison.json')
    assert comparison['sam']['eligible_postinit_center_slots'] == 17
    assert comparison['sam']['conditional_rmse_px'] is None
    assert comparison['new_api_calls'] == 0 and comparison['default_change'] is False
    with pytest.raises(ValueError,match='already exists'):
        study.grade_saved(out)


def test_offline_grade_rejects_sealed_wrong_prompt_direction(tmp_path):
    _saved_fixture(tmp_path,direction='both')
    with pytest.raises(ValueError,match='forward-only'):
        study.grade_saved(tmp_path)
    assert not (tmp_path/'manifest.json').exists()


def test_offline_grade_rejects_unsealed_result_mutation(tmp_path):
    _saved_fixture(tmp_path)
    study.write_json(tmp_path/'tracking/call/segmentation.json',{'status':'changed'})
    with pytest.raises(ValueError,match='integrity'):
        study.grade_saved(tmp_path)
