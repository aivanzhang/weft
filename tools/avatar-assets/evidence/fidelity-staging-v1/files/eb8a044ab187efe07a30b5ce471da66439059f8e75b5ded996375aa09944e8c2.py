"""One bounded independent raw export audit; no restore/native/model calls."""

import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/fringe-fidelity-independent-preparation-v1/producer-sources'))
from avatar_assets.runner import run_bounded


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


if __name__=='__main__':
    manifest=HERE/'prepared.json'
    if digest(manifest)!=sys.argv[1]:raise ValueError('Prepared hash mismatch')
    files=json.loads(manifest.read_text())['files']
    for row in files:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Pinned audit input changed')
    request=HERE/'request.json';out=(ROOT/json.loads(request.read_text())['output']).parent;out.mkdir()
    job=run_bounded([ROOT/'.venv/bin/python',HERE/'check.py',request],out/'audit.log',out,
                    timeout=30,memory_mb=512,output_mb=32)
    with (out/'job.json').open('x') as handle:json.dump(job,handle,indent=2,sort_keys=True);handle.write('\n')
    for row in files:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Pinned audit input changed during execution')
    print(json.dumps({'exit':job['exit_code'],'seconds':job['elapsed_seconds'],'rss':job['sampled_peak_rss_bytes'],
          'output':str(out.relative_to(ROOT))}),flush=True)
