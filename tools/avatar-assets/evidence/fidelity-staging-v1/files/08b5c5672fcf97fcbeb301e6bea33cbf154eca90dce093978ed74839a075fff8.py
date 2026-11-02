"""Run the frozen experimental producer; independent audit lives in a different file."""

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'producer-sources'))
from avatar_assets.fringe_export import omit_unused_fringe_tangents
from avatar_assets.fringe_fidelity import restore_authored_fringe


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with path.open('x') as handle:handle.write(payload)


if __name__=='__main__':
    started=time.monotonic();request=json.loads(Path(sys.argv[1]).read_text());out=ROOT/request['output']
    if out.exists():raise ValueError('Output already exists')
    out.mkdir();save(out/'request.json',request)
    try:
        for row in request['sources']:
            if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Pinned input/source changed: '+row['path'])
        source=ROOT/request['source']['path']
        if request['omit_first']:
            report=omit_unused_fringe_tangents(source,out/'tangent-omitted.glb',['SourceColoredFringe'])
            save(out/'omission-report.json',report);source=out/'tangent-omitted.glb'
        report=restore_authored_fringe(source,out/'model.glb',ROOT/request['helper']['path'],ROOT/request['native_report']['path'],
            experimental=True,expected_hashes={'source':digest(source),'correspondence':request['helper']['sha256'],
                                               'native_report':request['native_report']['sha256']})
        save(out/'repair-report.json',report)
        for row in request['sources']:
            if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Pinned input/source changed during execution')
        save(out/'result.json',{'status':'produced_experimental_derivative_pending_independent_audit',
            'model_sha256':digest(out/'model.glb'),'elapsed_seconds':time.monotonic()-started,
            'quality_approved':False,'body_attachment_approved':False,'native_run_completed':False})
    except Exception as error:
        save(out/'result.json',{'status':'failed','error':str(error),'elapsed_seconds':time.monotonic()-started})
        (out/'failure.txt').write_text(traceback.format_exc());raise
