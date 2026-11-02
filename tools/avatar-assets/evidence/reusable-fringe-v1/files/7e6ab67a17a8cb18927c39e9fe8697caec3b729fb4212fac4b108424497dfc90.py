"""Execute one frozen, local-only full finish after the adapter proof passes."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import require, sha256, write_json


def main():
    cfg = json.loads((HERE/'execution.json').read_text())
    require(cfg['status'] == 'released_single_attempt', 'Native lane is not released')
    require(not (HERE/'execution-started.json').exists(), 'One attempt only')
    for row in cfg['sources']+cfg['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Frozen input changed: '+row['path'])
    require(sha256(cfg['blender']) == cfg['blender_sha256'], 'Native tool changed')
    require(not (ROOT/cfg['output']).exists(), 'Preserve the existing output')
    write_json(HERE/'execution-started.json', {'execution_sha256': sha256(HERE/'execution.json')})
    allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
    allowed.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ.clear()
    os.environ.update(allowed)
    from avatar_assets.finish_generated import finish
    start = time.monotonic()
    error, result = None, None
    try:
        result = finish(ROOT/cfg['config'], ROOT/cfg['output'])
    except Exception as failure:
        error = str(failure)
    unchanged = all(sha256(ROOT/row['path']) == row['sha256'] for row in cfg['sources']+cfg['inputs'])
    receipt = {'result': result, 'error': error, 'elapsed_seconds': time.monotonic()-start,
               'frozen_inputs_unchanged': unchanged, 'new_generation_calls': 0, 'automatic_retry': False,
               'native_receipt': cfg['output']+'/job.json', 'quality_approved': False}
    write_json(HERE/'execution-receipt.json', receipt)
    print(json.dumps(receipt))
    require(error is None and unchanged, 'Preserve failed full-finish evidence; no automatic retry')


if __name__ == '__main__':
    main()
