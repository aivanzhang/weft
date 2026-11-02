"""Exactly one bounded CPU producer per case; preserves any first failure."""

import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'producer-sources'))
from avatar_assets.runner import run_bounded


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,value):
    encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with path.open('x') as handle:handle.write(encoded)


if __name__=='__main__':
    manifest=HERE/'prepared.json'
    if digest(manifest)!=sys.argv[1]:raise ValueError('Prepared hash mismatch')
    pins=json.loads(manifest.read_text())['files']
    for row in pins:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Preparation changed')
    for case in ('panda','fox'):
        request_path=HERE/f'{case}-producer-request.json';request=json.loads(request_path.read_text())
        outer=(ROOT/request['output']).parent;outer.mkdir()
        job=run_bounded([ROOT/'.venv/bin/python',HERE/'produce.py',request_path],outer/'producer.log',outer,
                        timeout=30,memory_mb=512,output_mb=128)
        save(outer/'job.json',job)
        for row in pins:
            if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Preparation changed during execution')
        print(json.dumps({'case':case,'exit':job['exit_code'],'seconds':job['elapsed_seconds'],
              'rss':job['sampled_peak_rss_bytes'],'path':str(outer.relative_to(ROOT))}),flush=True)
        if not job['process_exit_verified']:raise RuntimeError('Unresolved owned child; later work stopped')
