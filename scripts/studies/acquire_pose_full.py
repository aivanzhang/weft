"""Acquire one versioned official local model and its source documentation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


DOCUMENTS = [
    ('https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker', 'official-guide.html', 2_000_000),
    ('https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf', 'model-card.pdf', 4_000_000),
]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('full', 'heavy'), default='full'); args = parser.parse_args()
    model_name = f'pose_landmarker_{args.variant}'
    files = [(f'https://storage.googleapis.com/mediapipe-models/pose_landmarker/{model_name}/float16/1/{model_name}.task',
              model_name+'.task', 15_000_000 if args.variant == 'full' else 40_000_000)] + DOCUMENTS
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'requests': files, 'workers': 2,
        'maximum_seconds_per_request': 35, 'automatic_retries': 0, 'maximum_total_bytes': sum(row[2] for row in files),
        'projected_external_cost_usd': 0, 'recurring_resources': 0,
        'purpose': f'Explicit local {args.variant.title()}-model comparison, not automatic model or threshold selection',
        'default_change': False, 'remote_code_execution': False})
    started = time.monotonic()

    def fetch(item):
        url, name, maximum = item; tick = time.monotonic()
        try:
            process = subprocess.run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
                '--max-redirs', '2', '--max-time', '35', '--max-filesize', str(maximum), '--silent', '--show-error',
                '--write-out', '%{http_code}', url, '--output', str(out/name)], capture_output=True, text=True, timeout=38)
            status, http, error = process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            status, http, error = 124, None, 'Bounded request timed out; partial bytes retained'
        path = out/name
        return {'url': url, 'file': name, 'retrieved_at': local_time(), 'status': status, 'http_status': http,
                'error': error, 'bytes': path.stat().st_size if path.exists() else 0,
                'sha256': digest(path) if path.exists() else None, 'seconds': time.monotonic()-tick}

    with ThreadPoolExecutor(max_workers=2) as executor: rows = list(executor.map(fetch, files))
    success = all(row['status'] == 0 for row in rows)
    report = {'status': 'acquired' if success else 'incomplete', 'sources': rows,
              'total_bytes': sum(row['bytes'] for row in rows), 'runtime_seconds': time.monotonic()-started,
              'paid_requests': 0, 'remote_code_execution': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))
    if not success: raise SystemExit(1)


if __name__ == '__main__':
    main()
