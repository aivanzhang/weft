import json
import sys

import numpy as np
import pytest

from test_camera_pose import fixture
from test_multiview_root import CLOCK, package_inputs
from weft.camera_body_lab import prepare_camera_body_lab
from weft.cli import main
from weft.foot_reference import build_foot_reference
from weft.io import digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.mobile_human import calibrate_floor
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.whole_body import WholeBodyEnv


def package(tmp_path):
    source, camera, ground, config = package_inputs(tmp_path)
    roots = [[.2+x, .3+x/4, 5.] for x in np.linspace(0, .6, 10)]
    evidence, third_camera, _, _ = fixture(roots=roots, translations=[[.8, -.3, 0.]]*10)
    evidence['source_sha256'] = third_camera['source_sha256'] = 'c'*64
    for row in evidence['frames']:
        row['id'] = f"{'c'*64}:frame:{row['frame']}"
    write_json(tmp_path/'third-evidence.json', evidence)
    write_json(tmp_path/'third-camera.json', third_camera)
    extra = tmp_path/'third-view.json'
    write_json(extra, {'schema_version': 1, 'name': 'third', 'evidence': 'third-evidence.json',
                      'camera': 'third-camera.json', 'clock_provenance': CLOCK})
    return source, camera, ground, config, extra


@pytest.mark.parametrize('mode,limb_fit,partial,pixels', [(None, False, False, False), ('both', False, False, False),
    ('both', True, False, False), ('both', False, True, False), ('both', True, True, False), ('both', True, True, True)])
def test_extra_view_cli_keeps_source_ground_and_exact_manual_recording(tmp_path, monkeypatch, mode, limb_fit, partial, pixels):
    source, camera, ground, config, extra = package(tmp_path)
    if pixels:
        for name in ('second-evidence.json', 'third-evidence.json'):
            evidence = json.loads((tmp_path/name).read_text())
            for row in evidence['frames']:
                for joint in (15, 27): row['landmarks'][joint]['visibility'] = .1
            write_json(tmp_path/name, evidence)
    before = verify_artifacts(source)
    out = tmp_path/'lab'
    args = ['weft', 'camera-body', '--run', str(source), '--camera', str(camera), '--ground', str(ground),
            '--root-view', str(config), '--additional-view', str(extra), '--out', str(out)]
    if mode:
        args += ['--body-view', mode]
    if limb_fit:
        args += ['--fit-limbs']
    if partial:
        args += ['--partial-segments']
    if pixels:
        args += ['--fit-partial-pixels']
    monkeypatch.setattr(sys, 'argv', args)
    main()
    run = out/'run'; verified = verify_artifacts(run)
    body = json.loads((run/'human.json').read_text())
    assert body['camera_reference']['method'] == ('camera-body-partial-pixels-v1' if pixels else 'camera-body-length-fit-v2' if limb_fit and partial else
        'camera-body-length-fit-v1' if limb_fit else 'multi-view-body-v2' if partial else
        'multi-view-body-v1' if mode else 'multi-view-root-body-v1')
    assert (run/'ground.json').read_bytes() == ground.read_bytes()
    assert (run/'camera.json').read_bytes() == camera.read_bytes()
    scene = json.loads((run/'scene.json').read_text())
    assert scene['human']['camera_view_count'] == 3
    assert scene['human'].get('partial_segments', False) is partial
    assert ('partial_pixels' in scene['human']) is pixels
    if pixels:
        assert scene['human']['partial_pixels']['fitted_group_poses'] == 20
    assert 'Multi-view' in scene['human']['mobility']['root_depth']
    assert not scene['learned_policy_available'] and not (run/'policy.json').exists()
    for name, original in [('config.json', extra), ('evidence.json', tmp_path/'third-evidence.json'),
                           ('camera.json', tmp_path/'third-camera.json')]:
        assert digest(run/'inputs/two-view-root/extra-0'/name) == digest(original)
    with pytest.raises(ValueError, match='independent trajectory'):
        WholeBodyEnv(body)
    with pytest.raises(ValueError, match='transformed-foot'):
        build_foot_reference(body, {}, {})
    with pytest.raises(ValueError, match='must not be replaced'):
        calibrate_floor(body)
    lab = Lab(out)
    try:
        lab.load_session()
        with pytest.raises(ValueError, match='no learned policy'):
            lab.control('/api/step', {'policy': True})
        lab.control('/api/record/start', {})
        for action in (1, 7):
            lab.control('/api/step', {'action': action, 'steps': 2})
        lab.control('/api/reset', {'seed': 4})
        stopped = lab.control('/api/record/stop', {})
        lab.export_future.result(timeout=15)
        recording = out/'sessions'/stopped['directory']
        replay = CameraHumanSession(run)
        try:
            for event in load_recording(recording)['events']:
                if event['type'] in ('start', 'reset'):
                    replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
                assert replay.snapshot() == event['snapshot']
        finally:
            replay.env.close()
        assert verify_recording(recording)['video_frames'] == 6
    finally:
        lab.close()
    assert verify_artifacts(source) == before and verify_artifacts(run) == verified


@pytest.mark.parametrize('damage', ['no_root', 'too_many', 'partial', 'duplicate', 'clock', 'during_copy'])
def test_extra_view_failures_never_publish_ready_lab(tmp_path, monkeypatch, damage):
    import weft.camera_body_lab as module
    source, camera, ground, config, extra = package(tmp_path)
    out = tmp_path/'lab'; extras = [extra]; partial = False; mode = 'both'
    if damage == 'no_root': config = None
    elif damage == 'too_many': extras *= 3
    elif damage == 'partial': partial = True; mode = 'dimensions'
    elif damage == 'duplicate': extras *= 2
    elif damage == 'clock':
        evidence = json.loads((tmp_path/'third-evidence.json').read_text())
        evidence['frames'][1]['time_s'] += .01
        write_json(tmp_path/'third-evidence.json', evidence)
    elif damage == 'during_copy':
        copy = module.shutil.copyfile
        def mutate(src, dst):
            result = copy(src, dst)
            if str(src).endswith('source.mp4'):
                (tmp_path/'third-evidence.json').write_text('{}')
            return result
        monkeypatch.setattr(module.shutil, 'copyfile', mutate)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_camera_body_lab(source, camera, out, ground_path=ground, root_view_path=config,
                                body_view=mode, partial_segments=partial, additional_view_paths=extras)
    assert not (out/'status.json').exists()
