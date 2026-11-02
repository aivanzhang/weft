"""Real attachment-stage integration only; never invoke generation or AI roles."""
import json
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets import pipeline
from avatar_assets.safety import artifacts, read_json, sha256, write_json


class NoCalls:
    def __getattr__(self, name):
        raise AssertionError('No generation-pipeline AI role is authorized: ' + name)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Component controller received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    request = read_json(HERE / 'prepared-final.json')
    for item in request['inputs']:
        path = ROOT / item['path']
        assert path.is_file() and not path.is_symlink()
        assert sha256(path) == item['sha256'], item['path']
    assert sha256(HERE / 'check-stage.py') == request['driver_sha256']
    job = ROOT / request['job']
    pack = ROOT / request['pack']
    config = ROOT / request['config']
    design = read_json(config)
    source = {'path': design['source'], 'sha256': design['source_sha256']}
    plan = {'states': design['states']}
    worker = pipeline.Pipeline(job, agents=NoCalls())
    started = time.monotonic()
    try:
        with pipeline._lock(job):
            worker.guard()
            worker.event('portable_attachment', 'running', attempt=0)
            report = worker._portable_attachment(0, pack, plan, source, config)
            worker.event('portable_attachment', 'complete' if report['passed'] else report['status'],
                         attempt=0, scope='Isolated real stage, not a full pipeline delivery')
            before = artifacts(job / 'portable-root-audit-0')
            receipt_before = sha256(job / 'portable-root-audit-0.json')
            cached_started = time.monotonic()
            cached = worker._portable_attachment(0, pack, plan, source, config)
            cached_seconds = time.monotonic() - cached_started
            assert cached == report
            assert artifacts(job / 'portable-root-audit-0') == before
            assert sha256(job / 'portable-root-audit-0.json') == receipt_before
            summary = worker._attachment_summary(0, report)
            write_json(job / 'attachment-summary.json', summary)
    except BaseException as error:
        worker._fail_active_stage(error)
        raise
    for item in request['inputs']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    result = {'schema_version': 1, 'scope': 'Actual saved-fox final GLB and real locked pipeline attachment stage; not complete Studio or fresh generation',
              'status': report['status'], 'passed': report['passed'], 'roots': report['roots'],
              'metrics': report['metrics'], 'state_sample_counts': {s['id']: len(s['times']) for s in report['states']},
              'report_sha256': sha256(job / 'portable-root-audit-0/report.json'),
              'audit_manifest_sha256': sha256(job / 'portable-root-audit-0/manifest.json'),
              'controller_receipt_sha256': receipt_before, 'cached_report_equal': True,
              'cached_pack_bytes_unchanged': True, 'cached_validation_seconds': cached_seconds,
              'total_stage_seconds': time.monotonic()-started, 'source_pins_unchanged': True,
              'new_provider_calls': 0, 'new_pipeline_role_calls': 0, 'quality_approved': False}
    write_json(job / 'component-result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
