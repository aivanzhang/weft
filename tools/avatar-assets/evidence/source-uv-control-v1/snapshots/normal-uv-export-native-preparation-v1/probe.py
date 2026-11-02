"""Read-only real-scene normal-UV guard proof; no export/render/generation."""
import hashlib, json, sys
from pathlib import Path
import bpy
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'sources/scripts'))
from blender_normal_uv_export import assert_normal_uv_export
from normal_uv_export import NormalUVExportError
OUT=Path(sys.argv[sys.argv.index('--')+1])
def write(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def digest(data): return hashlib.sha256(data).hexdigest()
def values(collection,attribute,width,dtype):
    data=np.empty(len(collection)*width,dtype=dtype)
    collection.foreach_get(attribute,data)
    return digest(data.tobytes())
def fingerprint(objects):
    rows=[]
    for obj in objects:
        mesh=obj.data
        mats=[]
        for slot in obj.material_slots:
            mat=slot.material
            if mat is None: mats.append(None); continue
            nodes=[]; links=[]
            if mat.use_nodes:
                for n in mat.node_tree.nodes:
                    record={'name':n.name,'type':n.type,'mute':n.mute}
                    for key in ('uv_map','space','is_active_output','projection','from_instancer'):
                        if hasattr(n,key): record[key]=getattr(n,key)
                    record['inputs']=[(s.name,str(getattr(s,'default_value',None))) for s in n.inputs]
                    if n.type=='TEX_IMAGE' and n.image:
                        im=n.image
                        record['image']={'name':im.name,'source':im.source,'dirty':im.is_dirty,
                            'packed_sha256':digest(im.packed_file.data) if im.packed_file else None}
                    nodes.append(record)
                links=[(l.from_node.name,l.from_socket.name,l.to_node.name,l.to_socket.name,l.is_valid) for l in mat.node_tree.links]
            mats.append({'name':mat.name,'use_nodes':mat.use_nodes,'nodes':nodes,'links':links})
        rows.append({'name':obj.name,'positions':values(mesh.vertices,'co',3,np.float32),
            'vertex_normals':values(mesh.vertices,'normal',3,np.float32),
            'corner_normals':values(mesh.corner_normals,'vector',3,np.float32),
            'indices':values(mesh.loops,'vertex_index',1,np.int32),
            'material_indices':values(mesh.polygons,'material_index',1,np.int32),
            'uvs':[{'name':u.name,'sha256':values(u.data,'uv',2,np.float32),'render':u.active_render} for u in mesh.uv_layers],
            'active_uv':mesh.uv_layers.active_index,'materials':mats})
    return {'sha256':digest(json.dumps(rows,sort_keys=True,allow_nan=False).encode()),'objects':rows}
scene=ROOT/'outputs/panda-uv2-staged-control-v1/prepared-source.blend'
bpy.ops.wm.open_mainfile(filepath=str(scene),use_scripts=False)
receipt=json.loads((ROOT/'outputs/panda-uv2-staged-control-v1/prepared-finish-report.json').read_text())['fur_undercoat']
owners=json.loads((ROOT/'outputs/panda-uv2-staged-control-v1/source-skin.json').read_text())['owners']
objects=[bpy.data.objects[row['name']] for row in owners]
before=fingerprint(objects)
accepted=assert_normal_uv_export(objects,receipt,report_path=OUT/'authored-normal-uv-export.json')
after=fingerprint(objects)
write('authored-read-only.json',{'before':before,'after_sha256':after['sha256'],'unchanged':before==after})
assert before==after
assert accepted['status']=='supported'
assert len(accepted['authored_targets'])==3
assert any(row['admission']=='coherent_uv0' for row in accepted['materials'])
assert not any(row['normal'] and row['normal']['normal_uv_index']==1 for row in accepted['materials'])
bpy.ops.wm.read_factory_settings(use_empty=True)
source=ROOT/'outputs/panda-shoulder-uv-repair-v2/model.glb'
bpy.ops.import_scene.gltf(filepath=str(source))
objects=sorted([o for o in bpy.context.scene.objects if o.type=='MESH'],key=lambda o:o.name)
before=fingerprint(objects)
try:
    assert_normal_uv_export(objects,report_path=OUT/'original-normal-uv-export.json')
    raise AssertionError('Unreceipted source UV1 was accepted')
except NormalUVExportError as error:
    rejected=error.report
    assert rejected['status']=='unsupported'
    assert rejected['errors'][0]['code']=='nonzero_normal_uv_without_receipt'
after=fingerprint(objects)
write('original-read-only.json',{'before':before,'after_sha256':after['sha256'],'unchanged':before==after})
assert before==after
write('result.json',{'status':'passed','scope':'Actual native read-only admission, not export or appearance acceptance',
    'accepted_authored_targets':len(accepted['authored_targets']),
    'accepted_vertices':accepted['vertices'],'accepted_triangles':accepted['triangles'],
    'rejection_code':rejected['errors'][0]['code'],'native_state_unchanged':True,
    'new_provider_calls':0,'new_pipeline_role_calls':0,'quality_approved':False,
    'public_fringe_enabled':False,'image_pixels_read_by_guard':False})
