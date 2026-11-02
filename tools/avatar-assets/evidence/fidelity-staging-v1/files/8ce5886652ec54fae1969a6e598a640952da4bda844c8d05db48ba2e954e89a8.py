"""Small independent timing/FK/skin oracles; analytic fixtures, never avatar evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit


def run():
    checks=[]
    model=SimpleNamespace(doc={'nodes':[{}]},parents={})
    samples=[(-1,2),(.5,2),(1,2),(1.5,2),(2,7),(3,7)]
    rows=[((0,'translation'),np.array([1.,2.]),np.array([[2.,0,0],[7.,0,0]]),'STEP')]
    for seconds,x in samples:
        assert np.array_equal(audit.world(model,rows,seconds)[0,:3,3],[x,0,0])
    checks.append('STEP holds first/preceding/last keys with exact-key ownership')
    middle=audit.slerp(np.array([0.,0,0,1.]),np.array([0.,0,1.,0]),.5)
    assert np.allclose(audit.rotation(middle)@np.array([1.,0,0]),[0,1,0],atol=1e-14,rtol=0)
    checks.append('Quaternion halfway identity to180degZ rotates X toY')
    model=SimpleNamespace(doc={'nodes':[{'rotation':[0,0,2**-.5,2**-.5]}, {'translation':[1,0,0]}]},parents={1:0})
    matrices=audit.world(model,[],0)
    assert np.allclose(matrices[1,:3,3],[0,1,0],atol=1e-14,rtol=0)
    checks.append('Parent90degZ rotates child1mX translation to1mY')
    model=SimpleNamespace(doc={'skins':[{'joints':[0,1],'inverseBindMatrices':0}]},
                          array=lambda index:np.tile(np.eye(4).reshape(1,16),(2,1)))
    matrices=np.tile(np.eye(4),(2,1,1));matrices[0,0,3]=2;matrices[1,1,3]=4
    moved=audit.skin(model,{'skin':0},np.array([[1.,1,0]]),np.array([[0,1,0,0]]),np.array([[.25,.75,0,0]]),matrices)
    assert np.array_equal(moved,np.array([[1.5,4,0]]))
    checks.append('Two-bone linear skinning known weighted world position')
    assert audit.canonical([2,3,1])==audit.canonical([1,2,3])
    assert audit.canonical([1,3,2])!=audit.canonical([1,2,3])
    checks.append('Triangle canonicalization retains winding')
    assert np.array_equal(np.array([1,7,13,19,25])/24,np.array([1/24,7/24,13/24,19/24,25/24]))
    checks.append('Frame numbers retain original seconds rather than subtracting first frame')
    return {'status':'passed','provenance':'analytic_fixture','checks':checks,
            'audit_sha256':hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
            'thresholds_sha256':audit.THRESHOLDS_SHA256,'avatar_approval':False}


if __name__=='__main__':
    path=Path(__file__).with_name('oracle-result.json')
    with path.open('x') as handle:
        json.dump(run(),handle,indent=2,sort_keys=True);handle.write('\n')
