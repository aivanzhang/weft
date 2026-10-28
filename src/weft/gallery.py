"""Small, evidence-backed records shared by the public gallery and upload results."""
import json
from pathlib import Path


def result_record(directory):
    root = Path(directory)
    read = lambda name: json.loads((root / name).read_text())
    evidence, report = read('evidence.json'), read('report.json')
    initial, selected = read('world-initial.json'), read('world.json')
    episodes = [evidence['generation'] + evidence['development_validation']] + evidence['audit_episodes']
    segments = []
    for index, rows in enumerate(episodes):
        dt = rows[2]['time_s'] - rows[0]['time_s']
        segments.append({'start': rows[0]['time_s'], 'end': rows[-1]['time_s'],
                         'x0': rows[0]['x_px'] / evidence['width'],
                         'v0': (rows[2]['x_px'] - rows[0]['x_px']) / evidence['width'] / dt,
                         'initialization_dt': dt, 'primary': index == 0})
    return {'title': 'Rolling motion', 'duration': evidence['frame_count'] / evidence['fps'],
            'width': evidence['width'], 'height': evidence['height'], 'segments': segments,
            'models': [{'label': 'Reconstructed', 'world': selected,
                        'error_px': report['reconstruction']['after']['rmse_px']},
                       {'label': 'Initial fit', 'world': initial,
                        'error_px': report['reconstruction']['before']['rmse_px']}],
            'source_sha256': evidence['source_sha256'],
            'manifest_sha256': __import__('hashlib').sha256((root / 'manifest.json').read_bytes()).hexdigest()}
