"""Independent original-tolerance check of actual native and GLB normals."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources/host'))
from avatar_assets.three_d import _GLB, validate_render_attributes


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def save(name,data):
    with (HERE/name).open('x') as handle:
        json.dump(data,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')


def main():
    config=json.loads((HERE/'config.json').read_text())
    before=json.loads((HERE/'native-before-export.json').read_text())
    after=json.loads((HERE/'native-after-export.json').read_text())
    before_by_name={r['name']:r for r in before['observations']}
    after_by_name={r['name']:r for r in after['observations']}
    source_cases={r['id']:r for r in config['cases']}
    model=_GLB(HERE/'model.glb');rows=[]
    for node in model.doc['nodes']:
        if 'mesh' not in node:continue
        name=node['name'];first=before_by_name[name];last=after_by_name[name]
        case=source_cases[first['case']]
        assert all(key not in node for key in ('matrix','rotation','translation','scale','skin'))
        mesh=model.doc['meshes'][node['mesh']]
        assert len(mesh['primitives'])==1
        primitive=mesh['primitives'][0];attrs=primitive['attributes']
        assert primitive.get('mode',4)==4 and set(attrs)=={'POSITION','NORMAL'}
        positions=np.asarray(model.accessor(attrs['POSITION'],'VEC3',(5126,)))
        normals=np.asarray(model.accessor(attrs['NORMAL'],'VEC3',(5126,)))
        indices=np.asarray(model.accessor(primitive['indices'],'SCALAR',(5121,5123,5125))).reshape(-1,3)
        intended=np.asarray(case['normals_blender'])
        expected_positions=np.asarray(case['positions_blender_m'])[:,[0,2,1]].copy();expected_positions[:,2]*=-1
        expected_normals=intended[:,[0,2,1]].copy();expected_normals[:,2]*=-1
        assert positions.shape==normals.shape==(3,3) and indices.shape==(1,3)
        ids=[]
        for position in positions:
            candidates=np.flatnonzero(np.max(abs(expected_positions-position),axis=1)<=config['position_tolerance_m'])
            assert len(candidates)==1
            ids.append(int(candidates[0]))
        assert set(ids)=={0,1,2}
        triangle=tuple(ids[int(i)] for i in indices[0]);assert triangle in ((0,1,2),(1,2,0),(2,0,1))
        native=np.asarray(first['corner_normals']);post=np.asarray(last['corner_normals'])
        limit=config['normal_component_limit']
        native_error=np.max(abs(native-intended),axis=1)
        export_error=np.max(abs(normals-expected_normals[ids]),axis=1)
        row={'name':name,'case':case['id'],'method':first['method'],'attribute':first['attribute'],
            'native_before_after_exact':bool(np.array_equal(native,post)),
            'native_max_component_error':float(native_error.max()),'native_over_original_limit':int(np.count_nonzero(native_error>limit)),
            'export_max_component_error':float(export_error.max()),'export_over_original_limit':int(np.count_nonzero(export_error>limit)),
            'position_max_error_m':float(np.max(abs(positions-expected_positions[ids]))),
            'position_correspondence_unique':True,'oriented_triangle_equal':True,
            'original_normal_component_limit':limit,'native_normals':native.tolist(),'export_normals':normals.tolist(),
            'helper_normals_gltf':expected_normals[ids].tolist()}
        rows.append(row)
    assert len(rows)==18 and set(r['name'] for r in rows)==set(before_by_name)
    render=validate_render_attributes(HERE/'model.glb');save('render-attributes.json',render)
    legacy=[r for r in rows if r['method']=='legacy-short2']
    floated=[r for r in rows if r['method']!='legacy-short2']
    controls=all(r['native_over_original_limit']>0 and r['export_over_original_limit']>0 for r in legacy)
    floats=all(r['native_over_original_limit']==0 and r['export_over_original_limit']==0 and r['native_before_after_exact'] for r in floated)
    result={'status':'passed_native_normal_storage_component' if controls and floats else 'unsupported_native_normal_storage_component',
        'legacy_control_reproduces_failure_all_six':controls,'both_float_methods_pass_all_six':floats,
        'model_sha256':model.hash,'config_sha256':sha(HERE/'config.json'),'observations':rows,
        'normal_threshold_changed':False,'rendering_performed':False,'avatar_quality_accepted':False,
        'limits':['Tiny exact-witness mesh normals only; no skin, rig, action, tangent or full-avatar test.',
            'Weight pruning is unrelated and is not corrected by float normal storage.',
            'No production/default promotion or installed exporter patch.']}
    save('verification.json',result)
    print(json.dumps({key:result[key] for key in ('status','legacy_control_reproduces_failure_all_six','both_float_methods_pass_all_six','model_sha256')}))
    if not controls or not floats:raise ValueError('Native storage control or float-vector gate failed')


if __name__=='__main__':
    try:main()
    except Exception as error:
        save('checker-failure.json',{'type':type(error).__name__,'message':str(error),'accepted':False})
        raise
