"""Fresh-input automatic choices defer to actual body references without launching perception."""

import json

import pytest

from test_exercise_scene_inputs import inputs, preflight, edit_equipment, write, module
from weft.equipment_proposal_config import CONSENSUS_METHOD


def automatic(paths):
    config = json.loads(paths[2].read_bytes())
    config['equipment'][0].pop('source_frame')
    config['equipment'][0]['frame_selection'] = {'method': CONSENSUS_METHOD}
    write(paths[2], config)


@pytest.mark.parametrize('native_clock', [False, True])
def test_raw_preflight_defers_exact_automatic_request_and_archives_dependency(tmp_path, monkeypatch, native_clock):
    paths = inputs(tmp_path, monkeypatch, native=native_clock); automatic(paths)
    before = paths[2].read_bytes(); result = preflight(paths).receipt()
    row = result['selected_equipment'][0]
    assert row['frame_selection'] == {'method': CONSENSUS_METHOD}
    assert row['frame_selection_status'] == 'deferred_until_body_references'
    assert 'source_frame' not in row and 'evidence_id' not in row and 'time_s' not in row
    assert paths[2].read_bytes() == before
    assert set(result['implementation_sha256']) == set(module.IMPLEMENTATION_FILES) | {'equipment_box_consensus.py'}
    assert result['work']['inference_calls'] == result['work']['physics_steps'] == 0


@pytest.mark.parametrize('kind', ['roi', 'unknown_box', 'missing_box', 'no_candidates', 'unknown_id'])
def test_raw_auto_unknown_or_authored_candidates_cannot_reach_inference(tmp_path, monkeypatch, kind):
    paths = inputs(tmp_path, monkeypatch); automatic(paths)
    if kind == 'unknown_id':
        config = json.loads(paths[2].read_bytes())
        config['equipment'][0]['object_id'] = config['target_object_id'] = 'missing'
        write(paths[2], config)
    else:
        def change(document):
            if kind == 'roi': document['evidence'][2]['payload']['geometry_status'] = 'authored_initial_ROI_translated_by_appearance_match'
            elif kind == 'unknown_box': document['observations'][2]['measurements']['box']['status'] = 'assumed'
            elif kind == 'missing_box': document['observations'][2]['measurements'] = {}
            else:
                for row in document['observations']:
                    row.update(visibility='not_detected', measurements={})
        edit_equipment(paths, change)
    with pytest.raises(ValueError): preflight(paths)


def test_raw_auto_does_not_consume_withheld_roi_payload(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch); automatic(paths)
    def change(document):
        document['observations'][2]['partition'] = 'audit'
        document['evidence'][2]['payload']['geometry_status'] = 'authored_initial_ROI_forbidden_if_consumed'
    edit_equipment(paths, change)
    assert preflight(paths).receipt()['selected_equipment'][0]['frame_selection_status'] == 'deferred_until_body_references'
