import json
import re
import sys

from PIL import Image
import pytest

from test_stereo_ground import fixture
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.stereo_ground_package import prepare_stereo_ground


def inputs(tmp_path):
    cameras, annotations, _ = fixture()
    camera_paths, image_paths = [], []
    for index, camera in enumerate(cameras):
        camera_path = tmp_path/f'camera{index}.json'; write_json(camera_path, camera)
        image_path = tmp_path/f'image{index}.png'
        Image.new('RGB', (camera['width'], camera['height']), (50+index, 60, 70)).save(image_path)
        annotations['views'][index]['image_sha256'] = digest(image_path)
        camera_paths.append(camera_path); image_paths.append(image_path)
    annotation_path = tmp_path/'annotations.json'; write_json(annotation_path, annotations)
    return camera_paths, image_paths, annotation_path


def test_actual_cli_saves_exact_inputs_valid_report_and_bound_calibrations(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    cameras, images, annotations = inputs(tmp_path)
    originals = {p: p.read_bytes() for p in cameras+images+[annotations]}
    out = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['weft', 'stereo-ground', '--camera', str(cameras[0]), '--camera', str(cameras[1]),
        '--image', str(images[0]), '--image', str(images[1]), '--annotations', str(annotations), '--out', str(out)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'conditional_plane'
    verification = verify_artifacts(out)
    assert verification['verified_files'] == 20
    assert verification['manifest_sha256'] == result['manifest_sha256']
    assert (out/'annotations.json').read_bytes() == originals[annotations]
    for index in range(2):
        assert (out/f'input-camera-{index}.json').read_bytes() == originals[cameras[index]]
        assert (out/f'input-image-{index}.png').read_bytes() == originals[images[index]]
    html = (out/'index.html').read_text()
    for link in re.findall(r'href="([^"]+)"', html):
        assert (out/link).is_file()
    assert 'unavailable_no_assessment_points' in html
    assert 'not error bars' in html
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    with pytest.raises(ValueError, match='must be new'):
        prepare_stereo_ground(cameras, images, annotations, out)
    assert verify_artifacts(out) == verification


@pytest.mark.parametrize('damage', ['image_hash', 'image_dimensions', 'image_corrupt', 'extra_camera', 'oversize_annotations'])
def test_invalid_inputs_do_not_publish_an_output(tmp_path, damage):
    cameras, images, annotation_path = inputs(tmp_path)
    if damage == 'image_hash': images[0].write_bytes(b'wrong')
    elif damage in ('image_dimensions', 'image_corrupt'):
        if damage == 'image_dimensions': Image.new('RGB', (64, 64)).save(images[0])
        else: images[0].write_bytes(b'not an image')
        labels = json.loads(annotation_path.read_text()); labels['views'][0]['image_sha256'] = digest(images[0]); write_json(annotation_path, labels)
    elif damage == 'extra_camera': cameras.append(cameras[0])
    elif damage == 'oversize_annotations': annotation_path.write_bytes(b' '*(128*1024+1))
    out = tmp_path/'result'
    with pytest.raises((ValueError, OSError)):
        prepare_stereo_ground(cameras, images, annotation_path, out)
    assert not out.exists()


def test_rejected_geometry_keeps_inputs_report_and_no_ready_calibration(tmp_path):
    cameras, images, path = inputs(tmp_path)
    labels = json.loads(path.read_text()); labels['points'][0]['pixels'][1][1] += 30.; write_json(path, labels)
    out = tmp_path/'rejected'; result = prepare_stereo_ground(cameras, images, path, out)
    assert result['status'] == 'ground_unavailable'
    assert not (out/'camera-0.json').exists()
    assert not (out/'ground-0.json').exists()
    assert verify_artifacts(out)['verified_files'] == 16
    assert (out/'annotations.json').read_bytes() == path.read_bytes()


def test_point_names_cannot_inject_html(tmp_path):
    cameras, images, path = inputs(tmp_path)
    labels = json.loads(path.read_text()); labels['points'][0]['id'] = '<script>bad()</script>'; write_json(path, labels)
    out = tmp_path/'result'; prepare_stereo_ground(cameras, images, path, out)
    text = (out/'index.html').read_text()
    assert '<script>' not in text
    assert '&lt;script&gt;' in text


def test_cli_rejection_exits_unsuccessfully_but_keeps_the_report(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    cameras, images, annotation_path = inputs(tmp_path)
    labels = json.loads(annotation_path.read_text()); labels['points'][0]['pixels'][1][1] += 30.; write_json(annotation_path, labels)
    out = tmp_path/'rejected'
    monkeypatch.setattr(sys, 'argv', ['weft', 'stereo-ground', '--camera', str(cameras[0]), '--camera', str(cameras[1]),
        '--image', str(images[0]), '--image', str(images[1]), '--annotations', str(annotation_path), '--out', str(out)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'ground_unavailable'
    assert verify_artifacts(out)['verified_files'] == 16
