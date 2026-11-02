"""One released CPU-only quantization diagnosis and independent derived-model proof."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, require, sha256, write_json


def main():
    require(not (HERE / "execution-started.json").exists(), "CPU proof already attempted")
    prepared = read_json(HERE / "prepared.json")
    for row in prepared['frozen_files']:
        require(sha256(HERE / row['path']) == row['sha256'], 'Frozen CPU source changed')
    command = [sys.executable, str(HERE / "postprocess.py")]
    write_json(HERE / "execution-started.json", {'command': command, 'prepared_sha256': sha256(HERE / 'prepared.json')})
    job = run_bounded(command, HERE / 'cpu.log', HERE, timeout=60, memory_mb=768, output_mb=128)
    write_json(HERE / 'cpu-job.json', job)
    passed = job['exit_code'] == 0 and not job['failure']
    write_json(HERE / 'manifest.json', {'schema_version': 5,
        'status': 'analytic_quantization_and_mikk_proof_passed' if passed else 'cpu_proof_failed_preserved',
        'passed': passed, 'new_native_jobs': 0, 'provider_calls': 0, 'quality_approved': False, 'artifacts': artifacts(HERE)})
    print(json.dumps({'passed': passed, 'elapsed_seconds': job['elapsed_seconds'],
        'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes'], 'exit_code': job['exit_code']}))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
