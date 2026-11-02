"""Keep manual ground labels, source images and conditional camera outputs together."""

from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

from weft.io import digest, local_time, write_json
from weft.stereo_ground import _estimate_prepared, validate_stereo_ground


def _bytes(path, maximum):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum+1)
    if not raw or len(raw) > maximum:
        raise ValueError('Stereo-ground input is empty or exceeds its byte limit')
    return raw


def _report(result, annotations, images, cameras):
    rows = []
    for point in result['points']:
        errors = point['reprojection_px']
        rows.append(f"<tr><td>{escape(point['id'])}</td><td>{point['partition']}</td>"
                    f"<td>{escape(point['status'])}</td><td>{'unavailable' if errors is None else f'{max(errors):.3f} px'}</td></tr>")
    figures = []
    for index, (name, camera) in enumerate(zip(images, cameras, strict=True)):
        marks = []
        for point in annotations['points']:
            x, y = point['pixels'][index]; radius = point['uncertainty_px'][index]
            marks.append(f'<rect x="{x-radius}" y="{y-radius}" width="{2*radius}" height="{2*radius}" fill="none" stroke="#fff" stroke-width="2"/>'
                         f'<circle cx="{x}" cy="{y}" r="5" fill="#111" stroke="#fff" stroke-width="2"/>'
                         f'<text x="{x+14}" y="{y-14}" font-size="28" fill="white" stroke="black" stroke-width="1" paint-order="stroke">{escape(point["id"])}</text>')
        view = annotations['views'][index]
        figures.append(f'<figure><figcaption>{escape(view["name"])} — original frame {view["frame"]}, {view["time_s"]:.3f} s</figcaption>'
                       f'<svg role="img" aria-label="Source image with manual ground labels" viewBox="0 0 {camera["width"]} {camera["height"]}">'
                       f'<image href="{name}" width="{camera["width"]}" height="{camera["height"]}"/>{"".join(marks)}</svg></figure>')
    sensitivity = result['sensitivity']
    sensitivity_text = 'Unavailable: construction did not pass.'
    if sensitivity is not None:
        sensitivity_text = (f"{len(sensitivity['scenarios'])} one-coordinate perturbations; complete: {sensitivity['complete']}. "
                            f"Maximum angular change: {sensitivity['maximum_angle_degrees']} degrees. "
                            f"Maximum center-plane distance: {sensitivity['maximum_center_distance_m']} m. "
                            'These are sensitivity samples, not error bars or bounds on combined/camera errors.')
    links = ['<a href="annotations.json">Exact labels and assumptions</a>', '<a href="estimate.json">Geometry and every sensitivity sample</a>']
    if result['status'] == 'conditional_plane':
        links += [f'<a href="camera-{i}.json">Reframed camera {i}</a> / <a href="ground-{i}.json">matching ground</a>' for i in range(2)]
    return ('<!doctype html><meta charset="utf-8"><title>Weft — conditional ground</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#214f50;background:#f5eadc}'
            'svg{width:100%;height:auto}figure{margin:24px 0}td,th{text-align:left;padding:8px}pre{white-space:pre-wrap}a{color:#965644}</style>'
            f'<h1>Manually guided ground estimate</h1><p><strong>{escape(result["status"])}</strong></p>'
            '<p>This uses matching ground points you identify in two calibrated images. It does not detect ground automatically, '
            'measure gravity, prove contact, or adjust a body to fit. Camera scale, marker footprints and plane assumptions remain uncertain.</p>'
            f'<p>Separate source-label check: {escape(result["assessment"])}. Physical validation: unavailable.</p>'
            '<h2>1. Inspect the source labels</h2><p>Dots show the selected points. Squares show the declared pixel ranges. '
            'The image-to-video association is supplied provenance; this command does not decode or verify the original videos.</p>'
            + ''.join(figures) + '<h2>2. Check the geometry</h2><table><tr><th>Point</th><th>Use</th><th>Geometry</th><th>Largest image error</th></tr>'
            + ''.join(rows) + '</table><pre>' + escape(json.dumps(result['checks'], indent=2)) + '</pre>'
            + '<h2>3. Inspect uncertainty and outputs</h2><p>' + escape(sensitivity_text) + '</p><ul>'
            + ''.join('<li>'+link+'</li>' for link in links) + '</ul><p>Outputs do not replace the current lab or install gravity. '
            'Reframed cameras preserve original source clocks and pixels; downstream body construction must pass its own unchanged checks.</p>')


def prepare_stereo_ground(camera_paths, image_paths, annotation_path, out):
    started = time.monotonic()
    if len(camera_paths) != 2 or len(image_paths) != 2:
        raise ValueError('Supply exactly two cameras and their two source images')
    target = Path(out)
    if target.exists() or not target.parent.is_dir():
        raise ValueError('Stereo-ground output must be new under an existing parent')
    module_dir = Path(__file__).parent
    implementation = {name: (module_dir/name).read_bytes() for name in (
        'cli.py', 'stereo_ground.py', 'stereo_ground_package.py', 'camera_pose.py', 'camera_ground.py', 'multiview_pose.py', 'io.py')}
    raw_cameras = [_bytes(path, 4 << 20) for path in camera_paths]
    raw_annotations = _bytes(annotation_path, 128 << 10)
    cameras = [json.loads(raw) for raw in raw_cameras]
    annotations = json.loads(raw_annotations)
    # Validate bounded numerical inputs before accepting associated image bytes.
    prepared = validate_stereo_ground(cameras, annotations)
    raw_images = [_bytes(path, 12 << 20) for path in image_paths]
    image_names = []
    for index, (raw, camera, view) in enumerate(zip(raw_images, cameras, annotations['views'], strict=True)):
        if sha256(raw).hexdigest() != view['image_sha256']:
            raise ValueError('Source image bytes differ from the annotation identity')
        with Image.open(BytesIO(raw)) as image:
            if image.size != (camera['width'], camera['height']) or image.format not in ('JPEG', 'PNG'):
                raise ValueError('Source image header must match the original camera raster; only JPEG/PNG supported')
            image_names.append(f'input-image-{index}.'+('jpg' if image.format == 'JPEG' else 'png'))
            image.load()
    result = _estimate_prepared(cameras, annotations, prepared)
    target.mkdir()
    for index, raw in enumerate(raw_cameras):
        (target/f'input-camera-{index}.json').write_bytes(raw)
    (target/'annotations.json').write_bytes(raw_annotations)
    for name, raw in zip(image_names, raw_images, strict=True):
        (target/name).write_bytes(raw)
    for index, (camera, ground) in enumerate(zip(result['cameras'], result['grounds'], strict=True)):
        write_json(target/f'camera-{index}.json', camera)
        write_json(target/f'ground-{index}.json', ground)
    write_json(target/'estimate.json', result)
    (target/'index.html').write_text(_report(result, annotations, image_names, cameras))
    implementation_dir = target/'implementation-source'; implementation_dir.mkdir()
    for name, raw in implementation.items():
        if (module_dir/name).read_bytes() != raw:
            raise RuntimeError('Stereo-ground implementation changed during construction; output remains unsealed')
        (implementation_dir/name).write_bytes(raw)
    write_json(target/'implementation.json', {'files': {name: sha256(raw).hexdigest() for name, raw in implementation.items()},
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version})
    write_json(target/'receipt.json', {'profile': 'stereo-ground-package-v1', 'status': result['status'],
        'created_at': local_time(), 'seconds_before_seal': time.monotonic()-started,
        'input_bytes': sum(map(len, raw_cameras+raw_images))+len(raw_annotations),
        'source_video_decodes': 0, 'saved_image_decodes': 2, 'model_calls': 0, 'physics_calls': 0, 'api_calls': 0,
        'limits': {'points': 16, 'camera_bytes_each': 4 << 20, 'image_bytes_each': 12 << 20,
                   'annotation_bytes': 128 << 10, 'sensitivity_scenarios': 128},
        'enforcement': 'Input and operation-count bounds; no separate hard process-time or RSS supervisor'})
    write_json(target/'manifest.json', {'schema': 1, 'created_at': local_time(),
        'files': {p.relative_to(target).as_posix(): digest(p) for p in sorted(target.rglob('*')) if p.is_file()}})
    return {'profile': 'stereo-ground-package-v1', 'status': result['status'], 'output_path': str(target),
            'report': str(target/'index.html'), 'manifest_sha256': digest(target/'manifest.json'),
            'assessment': result['assessment'], 'physical_validation': 'unavailable'}
