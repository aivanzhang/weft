"""One panda diagnostic replay after JSON-only witness serialization correction."""

import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'outputs/fringe-fidelity-independent-preparation-v1'
sys.path.insert(0,str(PRIOR/'producer-sources'))
from avatar_assets.runner import run_bounded


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


if __name__=='__main__':
    manifest=HERE/'prepared.json'
    if digest(manifest)!=sys.argv[1]:raise ValueError('Prepared audit hash mismatch')
    pins=json.loads(manifest.read_text())['files']
    for row in pins:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Audit preparation changed')
    request_path=HERE/'panda-request.json';request=json.loads(request_path.read_text())
    output=ROOT/request['output'];outer=output.parent;outer.mkdir()
    job=run_bounded([ROOT/'.venv/bin/python',HERE/'audit.py',request_path,output],outer/'audit.log',outer,
                    timeout=60,memory_mb=512,output_mb=32)
    with (outer/'job.json').open('x') as handle:json.dump(job,handle,indent=2,sort_keys=True);handle.write('\n')
    for row in pins:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Audit preparation changed during execution')
    print(json.dumps({'exit':job['exit_code'],'seconds':job['elapsed_seconds'],
          'rss':job['sampled_peak_rss_bytes'],'path':str(outer.relative_to(ROOT))}),flush=True)
    if not job['process_exit_verified']:raise RuntimeError('Unresolved owned audit process')
