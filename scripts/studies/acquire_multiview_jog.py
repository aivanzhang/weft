"""Acquire one bounded public CC BY sample; no accounts or external execution."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


BASE = 'https://autelica.com/sample/'
FILES = [('', 'source-page.html', 1_000_000), ('files/LICENSE.txt', 'LICENSE.txt', 20_000),
         ('files/metadata.json', 'metadata.json', 20_000),
         ('files/calibration__cameras.json', 'cameras.json', 100_000),
         ('files/calibration__extrinsics.npz', 'extrinsics.npz', 200_000),
         ('files/motion__body_coco17.npz', 'body-coco17.npz', 500_000),
         ('files/qc__error_report.json', 'provider-error-report.json', 100_000),
         ('files/drone1.mp4', 'drone1.mp4', 16_000_000)]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/'acquire_multiview_jog.py')
    started = time.monotonic()
    write_json(out/'pre-run.json', {'created_at': local_time(), 'requests': FILES, 'base_url': BASE,
        'workers': 2, 'maximum_seconds_per_request': 35, 'automatic_retries': 0, 'paid_requests': 0,
        'rights': 'CC BY 4.0 for autelica_sample_jogturn only; hands/pilot data excluded',
        'purpose': 'Fresh moving-camera full-body development input with a third-party multiview reference',
        'evidence_limit': 'Camera poses, timing offsets, metric scale and 3D joints are reconstructed, not independent optical-marker ground truth',
        'planned_source_inspection_frames': [0, 120, 239],
        'evaluation_boundary': 'Video-only baseline must not receive provider joints, calibration or quality scores'})

    def fetch(item):
        relative, name, maximum = item
        tick = time.monotonic()
        result = subprocess.run(['curl', '--fail', '--location', '--proto', '=https', '--max-redirs', '2',
            '--max-time', '35', '--max-filesize', str(maximum), '--silent', '--show-error',
            '--write-out', '%{http_code}', BASE+relative, '--output', str(out/name)],
            capture_output=True, text=True, timeout=38)
        target = out/name
        return {'url': BASE+relative, 'file': name, 'retrieved_at': local_time(),
                'status': result.returncode, 'http_status': result.stdout,
                'error': result.stderr, 'seconds': time.monotonic()-tick,
                'bytes': target.stat().st_size if target.exists() else 0,
                'sha256': digest(target) if target.exists() else None}

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fetch, FILES))
    report = {'status': 'acquired' if all(r['status'] == 0 for r in rows) else 'incomplete',
              'sources': rows, 'total_bytes': sum(r['bytes'] for r in rows),
              'runtime_seconds': time.monotonic()-started, 'paid_requests': 0,
              'physical_accuracy_verified': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
