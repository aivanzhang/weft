"""Analytic installed-code lineage check with mocked RNA reads; not a native export."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent


def functions(path, names, scope, class_name=None):
    tree=ast.parse(path.read_text())
    body=tree.body if class_name is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name).body
    chosen=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in chosen}==set(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),scope)


def main():
    constants=SimpleNamespace(ComponentType=SimpleNamespace(Float=5126,UnsignedShort=5123,UnsignedByte=5121),
        DataType=SimpleNamespace(Scalar='SCALAR',Vec2='VEC2',Vec3='VEC3',Vec4='VEC4',Mat4='MAT4'))
    scope={'np':np,'gltf2_io_constants':constants}
    functions(HERE/'sources/conversion.py',('get_component_type','get_data_type','get_numpy_type'),scope)
    conversion=SimpleNamespace(**{k:scope[k] for k in ('get_component_type','get_data_type','get_numpy_type')})
    scope['gltf2_blender_conversion']=conversion
    functions(HERE/'sources/primitive_extract.py',('__get_layer_attribute','__set_regular_attribute'),scope,'PrimitiveCreator')
    functions(HERE/'sources/gltf2_blender_utils.py',('fast_structured_np_unique',),scope)
    ids=np.arange(100000,dtype=np.int32)
    assert np.array_equal(ids,ids.astype(conversion.get_numpy_type('INT')).astype(np.int32))
    assert conversion.get_component_type('INT')==5126 and conversion.get_data_type('INT')=='SCALAR'
    native_ids=np.array([99996,99997,99998,99999],dtype=np.int32)
    corner_vertices=np.array([0,1,2,0,2,3,0,1,2],dtype=np.uint32)
    # Vertex0 appears on a UV seam; the third triangle duplicates the first.
    dots=np.zeros(9,dtype=[('vertex_index',np.uint32),('_AUTHOR_VERTEX_ID0',np.float32),('TEXCOORD_00',np.float32)])
    dots['vertex_index']=corner_vertices
    dots['TEXCOORD_00']=[0,.2,.5,.9,.5,.3,0,.2,.5]
    class Data:
        def foreach_get(self,name,out):
            assert name=='value';out[:]=native_ids
    owner=SimpleNamespace(blender_mesh=SimpleNamespace(vertices=[None]*4,attributes=[SimpleNamespace(data=Data())]),
        dots=dots,export_settings={'gltf_loose_edges':False,'gltf_loose_points':False},attributes={})
    attr={'blender_domain':'POINT','blender_data_type':'INT','blender_attribute_index':0,'len':1,'type':np.float32,
          'gltf_attribute_name':'_AUTHOR_VERTEX_ID'}
    scope['__get_layer_attribute'](owner,attr)
    assert np.array_equal(dots['_AUTHOR_VERTEX_ID0'],native_ids[corner_vertices])
    rows=[]
    for label,selections in (('shared',[np.arange(9)]),('separate-materials',[np.arange(3),np.arange(3,6),np.arange(6,9)])):
        for primitive,indices in enumerate(selections):
            distinct,inverse=scope['fast_structured_np_unique'](dots[indices].copy(),return_inverse=True)
            scope['__set_regular_attribute'](owner,distinct,attr)
            exported=owner.attributes['_AUTHOR_VERTEX_ID']['data'].ravel()
            assert np.array_equal(exported[inverse],native_ids[corner_vertices[indices]])
            assert np.array_equal(exported,native_ids[distinct['vertex_index']])
            rows.append({'path':label,'primitive':primitive,'source_corners':len(indices),'split_vertices':len(distinct),
                'exported_ids':exported.astype(int).tolist(),'corner_lineage_restored':True,
                'component_type':owner.attributes['_AUTHOR_VERTEX_ID']['component_type'],
                'accessor_type':owner.attributes['_AUTHOR_VERTEX_ID']['data_type']})
    result={'status':'analytic_installed_code_lineage_passed','source_kind':'Deterministic mocked RNA fixture, not native glTF',
        'integer_id_range_tested':[0,99999],'all_ids_exact_float32_roundtrip':True,'seam_material_duplicate_paths':rows,
        'native_calls':0,'skin_or_animation_verified':False,
        'limits':['Real Blender RNA conversion, evaluated-mesh preservation, actual skins/actions and export flags still need a tiny native fixture.',
            'Custom IDs do not prevent pruning; they identify original source rows for a separately verified correction.']}
    with (HERE/'analytic-result.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':result['status'],'cases':len(rows),'native_calls':0}))


if __name__=='__main__':main()
