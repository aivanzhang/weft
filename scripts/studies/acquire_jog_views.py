"""Acquire the remaining CC BY jogging views; downloaded JavaScript is read-only evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


FILES = [
    ('https://autelica.com/sample/files/drone2.mp4', 'drone2.mp4', 16_000_000),
    ('https://autelica.com/sample/files/drone3.mp4', 'drone3.mp4', 18_000_000),
    ('https://autelica.com/sample/', 'source-page.html', 1_000_000),
    ('https://autelica.com/sample/files/LICENSE.txt', 'LICENSE.txt', 20_000),
    ('https://autelica.com/js/sample.js?v=mvsz9x31', 'provider-viewer.js', 1_000_000),
]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    started = time.monotonic()
    write_json(out/'pre-run.json', {'created_at': local_time(), 'requests': FILES, 'workers': 2,
        'maximum_seconds_per_request': 35, 'automatic_retries': 0, 'maximum_total_bytes': sum(row[2] for row in FILES),
        'projected_api_cost_usd': 0, 'paid_requests': 0, 'rights': 'Autelica jog-turn sample CC BY 4.0 only; hands/pilot data excluded',
        'construction_views': ['drone1', 'drone2'], 'development_projection_view': 'drone3',
        'timing_rule': 'Unresolved until packaged clocks are checked; do not blindly reapply recovered sync offsets',
        'reference_boundary': 'Provider 3D joints, UV detections and contacts are excluded from construction. Calibration itself is reconstructed, not independently measured.',
        'scope': 'Additional-input development experiment, not a video-only result or a final holdout',
        'inspection_frames': [0, 120, 200, 239], 'remote_code_execution': False})

    def fetch(item):
        url, name, maximum = item; tick = time.monotonic()
        command = ['curl', '--fail', '--location', '--proto', '=https', '--max-redirs', '2', '--max-time', '35',
                   '--max-filesize', str(maximum), '--silent', '--show-error', '--write-out', '%{http_code}',
                   url, '--output', str(out/name)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=38)
            status, http, error = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            status, http, error = 124, None, 'Bounded download timed out; partial bytes retained'
        path = out/name
        return {'url': url, 'file': name, 'retrieved_at': local_time(), 'status': status, 'http_status': http,
                'error': error, 'bytes': path.stat().st_size if path.exists() else 0,
                'sha256': digest(path) if path.exists() else None, 'seconds': time.monotonic()-tick}

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(fetch, FILES))
    report = {'status': 'acquired' if all(row['status'] == 0 for row in rows) else 'incomplete',
              'sources': rows, 'total_bytes': sum(row['bytes'] for row in rows),
              'runtime_seconds': time.monotonic()-started, 'paid_requests': 0, 'remote_code_execution': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
