import json
from pathlib import Path

import pytest

from weft.ball_benchmark import CASES, create_analytic_probe, run_analytic_probes
from weft.io import digest


def test_independent_analytic_probes_cover_motion_contacts_and_abstention(tmp_path):
    results = run_analytic_probes(tmp_path / 'probes')
    assert {row['name'] for row in results} == set(CASES)
    assert all(row['expectation_met'] for row in results)
    rows = {row['name']: row for row in results}
    for row in results:
        if row['status'] == 'reconstructed':
            assert row['tracking_rmse_px'] < 1
            assert max(row['analytic_replay_rmse_px'].values()) < 10
    assert rows['gravity_flight']['development_replay_rmse_px'] < 6
    assert rows['gravity_bounce']['development_replay_rmse_px'] < 12
    assert rows['elastic_contact']['detections'] == 108
    assert rows['brief_occlusion']['excluded_frames'] == 0
    assert rows['brief_occlusion']['reconstruction_coverage']['missing_observation_slots'] == 4
    assert rows['brief_occlusion']['reconstruction_coverage']['simulation_frames'] == 45
    assert rows['fixed_camera_texture']['camera_check']['result'] == 'no_large_motion_detected'
    assert rows['constant_motion']['camera_check']['result'] == 'unknown_insufficient_background_evidence'
    assert 'Moving camera' in rows['camera_translation']['error']
    assert 'Ambiguous identity' in rows['same_color_ambiguity']['error']
    gravity = json.loads((tmp_path / 'probes/gravity_flight/run/scene-initial.json').read_text())
    assert abs(gravity['objects'][0]['velocity'][1]) < .01
    camera = json.loads((tmp_path / 'probes/camera_translation/run/camera-motion.json').read_text())
    assert camera['samples'][-1]['center_shift_image_width'] > .01


def test_probe_provenance_and_outputs_are_pinned_and_not_overwritten(tmp_path):
    result = create_analytic_probe('constant_motion', tmp_path / 'source')
    provenance = json.loads((tmp_path / 'source/provenance.json').read_text())
    truth = json.loads(Path(result['truth']).read_text())
    assert result['source_sha256'] == digest(result['video'])
    assert provenance['physical_evidence'] is False
    assert truth['used_by_reconstruction'] is False
    config = json.loads(Path(result['config']).read_text())
    assert 'truth' not in config
    with pytest.raises(FileExistsError):
        create_analytic_probe('constant_motion', tmp_path / 'source')
    with pytest.raises(ValueError, match='Unknown'):
        create_analytic_probe('not-a-case', tmp_path / 'bad')
