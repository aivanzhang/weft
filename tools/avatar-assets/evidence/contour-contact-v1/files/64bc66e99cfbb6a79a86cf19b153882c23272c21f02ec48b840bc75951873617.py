"""One exact-ID-bound, tip-position-only experimental regional-fin derivative."""
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources'))
from fringe_flow import reshape_fringe
from avatar_assets.fringe_export import _array, _strict_json
from avatar_assets.safety import sha256, require
from avatar_assets.three_d import _GLB, validate_render_attributes

START=time.monotonic()
PREP=json.loads((HERE/'prepared.json').read_text())
OUT=Path(PREP['output'])
PATHS={key:Path(row['path']) for key,row in PREP['inputs'].items()}


def deadline():
    require(time.monotonic()-START<27,'Internal27-second production limit exceeded')


def arrays(path, maximum):
    with zipfile.ZipFile(path) as archive:
        rows=archive.infolist()
        require(len(rows)<=16 and len({r.filename for r in rows})==len(rows)
                and sum(r.file_size for r in rows)<=maximum,'Unsafe source NPZ')
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key] for key in archive.files}


def axis(points):
    result=points[:,[0,2,1]].copy();result[:,2]*=-1
    return result


def canonical(values):
    return Counter(min((int(a),int(b),int(c)),(int(b),int(c),int(a)),(int(c),int(a),int(b))) for a,b,c in values)


def produce():
    require(all(sha256(PATHS[key])==row['sha256'] for key,row in PREP['inputs'].items()),'Frozen input changed')
    require(all(sha256(HERE/row['snapshot'])==row['sha256'] for row in PREP['sources']),'Frozen implementation changed')
    require(OUT.is_dir() and not (OUT/'model.glb').exists(),'Require fresh derivative output')
    source,raw=_GLB(PATHS['model']),_GLB(PATHS['raw'])
    design=json.loads(PATHS['design'].read_text());report=json.loads(PATHS['fringe_report'].read_text())
    skin_report=json.loads(PATHS['skin_report'].read_text())
    skin=arrays(PATHS['skin'],8*1024**2);helper=arrays(PATHS['correspondence'],8*1024**2)
    require(skin_report['npz_sha256']==sha256(PATHS['skin']),'Source-skin snapshot hash differs')
    material=report['material']['material']
    owners=[o for o in skin_report['owners'] if o['materials']==[material]]
    require(len(owners)==1,'Fin source owner is ambiguous')
    owner=owners[0];base_owners=[o for o in skin_report['owners'] if o is not owner]
    require(len(base_owners)==1,'This source requires one proven base emitter')
    base=base_owners[0]
    body=skin['positions'][base['vertex_start']:base['vertex_start']+base['vertex_count']]
    low,span=body.min(axis=0),np.ptp(body,axis=0)
    require(source.doc['bufferViews']==raw.doc['bufferViews'],'Final/raw buffer storage differs')
    selected=[];total_vertices=total_triangles=0;base_vertices=base_triangles=0
    for mi,mesh in enumerate(source.doc['meshes']):
        for pi,primitive in enumerate(mesh['primitives']):
            attrs=primitive['attributes'];position=source.doc['accessors'][attrs['POSITION']]
            index=source.doc['accessors'][primitive['indices']]
            require(primitive.get('mode',4)==4 and index['count']%3==0,'Invalid triangle primitive')
            total_vertices+=position['count'];total_triangles+=index['count']//3
            m=source.doc['materials'][primitive['material']]['name']
            if m==material:selected.append((mi,pi,primitive))
            else:base_vertices+=position['count'];base_triangles+=index['count']//3
    require(len(selected)==1 and total_vertices<=100000 and total_triangles<=100000,'Fin selection or global bounds differ')
    mi,pi,primitive=selected[0]
    require(source.doc['meshes'][mi]['name']==owner['mesh_name'],'Fin mesh identity differs')
    require(len([n for n in source.doc['nodes'] if n.get('mesh')==mi and n.get('name')==owner['name']])==1,'Fin node identity differs')
    original=raw.doc['meshes'][mi]['primitives'][pi]
    require(original['indices']==primitive['indices'] and original['material']==primitive['material'],'Final/raw topology binding differs')
    aid=primitive['attributes']['POSITION'];require(aid==original['attributes']['POSITION'],'Position accessor identity differs')
    id_values=_array(raw,original['attributes']['_AUTHOR_VERTEX_ID'],'SCALAR',{5126}).ravel()
    require(np.array_equal(id_values,np.floor(id_values)),'Noninteger native source ID')
    ids=id_values.astype(np.int64);local=ids-owner['vertex_start']
    require(np.array_equal(np.sort(local),np.arange(owner['vertex_count'])),'Fin native IDs are not unique complete ownership')
    native_map=helper['native_vertex_to_helper_vertex']
    require(native_map.shape==(owner['vertex_count'],) and native_map.dtype.kind in 'iu'
            and np.array_equal(native_map,helper['triangles'].ravel()),'Native/helper winding map differs')
    mapping=native_map[local]
    before=_array(source,aid,'VEC3',{5126});raw_points=_array(raw,aid,'VEC3',{5126})
    require(before.tobytes()==raw_points.tobytes(),'Final/raw fin positions differ')
    require(np.array_equal(before,axis(skin['positions'][ids]))
            and np.array_equal(skin['positions'][ids],helper['positions'][mapping]),'Exact ID/helper position chain differs')
    root_ids=helper['root_vertex_ids']
    require(root_ids.min()>=0 and root_ids.max()<len(body)
            and np.array_equal(helper['positions'].reshape(-1,4,3)[:,:2],body[root_ids]),
            'Recorded fin roots do not match their exact base source vertex IDs')
    old_indices=_array(source,primitive['indices'],'SCALAR',{5121,5123,5125},300000).ravel()
    raw_indices=_array(raw,original['indices'],'SCALAR',{5121,5123,5125},300000).ravel()
    require(np.array_equal(old_indices,raw_indices),'Final/raw triangle indices differ')
    native_triangles=skin['triangles'][owner['triangle_start']:owner['triangle_start']+owner['triangle_count']]
    require(canonical(ids[old_indices].reshape(-1,3))==canonical(native_triangles),'Oriented source-ID topology differs')
    core={key:helper['uv_gltf' if key=='uv' else key] for key in ['positions','normals','uv','triangles','joints','weights']}
    reshaped=reshape_fringe(core,low=low,span=span,base_direction=design['groom']['direction'],
        base_comb=design['groom']['comb'],groom_flow=design['groom_flow'],base_length_m=.0045,
        source_export_vertices=base_vertices,source_export_triangles=base_triangles)
    (OUT/'regional-response.json').write_text(json.dumps(reshaped['report'],indent=2)+'\n')
    desired=axis(reshaped['positions'])[mapping]
    tip=mapping%4>=2
    require(desired[~tip].tobytes()==before[~tip].tobytes(),'Root position bytes changed')
    descriptor=source.doc['accessors'][aid];view_id=descriptor['bufferView'];view=source.doc['bufferViews'][view_id]
    require(descriptor['componentType']==5126 and descriptor['type']=='VEC3' and descriptor.get('byteOffset',0)==0
            and 'sparse' not in descriptor and view.get('byteStride',12)==12 and view['byteLength']==before.nbytes,
            'Selected POSITION requires exclusive contiguous float32 storage')
    start,end=view.get('byteOffset',0),view.get('byteOffset',0)+view['byteLength']
    require(sum(a.get('bufferView')==view_id for a in source.doc['accessors'])==1,'Position view has accessor aliases')
    for i,v in enumerate(source.doc['bufferViews']):
        if i!=view_id:require(v.get('byteOffset',0)+v['byteLength']<=start or v.get('byteOffset',0)>=end,'Position view overlaps another view')
    references=[(i,j,key) for i,m in enumerate(source.doc['meshes']) for j,p in enumerate(m['primitives'])
                for key,value in p['attributes'].items() if value==aid]
    require(references==[(mi,pi,'POSITION')],'Position accessor has other live references')
    require(all(p['indices']!=aid for m in source.doc['meshes'] for p in m['primitives'])
            and all(s.get('inverseBindMatrices')!=aid for s in source.doc.get('skins',[]))
            and all(s[key]!=aid for a in source.doc.get('animations',[]) for s in a['samplers'] for key in ('input','output'))
            and all(i['bufferView']!=view_id for i in source.doc.get('images',[])),
            'Selected position storage also drives non-position data')
    with PATHS['model'].open('rb') as stream:
        prefix=stream.read(20);json_size=struct.unpack_from('<I',prefix,12)[0]
        require(_strict_json(stream.read(json_size))==source.doc,'Ambiguous original JSON')
        stream.seek(28+json_size+len(source.binary));padding=stream.read(3)
    binary=bytearray(source.binary);binary.extend(padding)
    destination=np.ndarray(before.shape,dtype='<f4',buffer=binary,offset=start)
    destination[tip]=desired[tip]
    require(binary[:start]==source.binary[:start] and binary[end:len(source.binary)]==source.binary[end:],
            'Non-position binary data changed')
    require(destination[~tip].tobytes()==before[~tip].tobytes(),'Position roots changed after encoding')
    require(destination.tobytes()==desired.tobytes(),'Stored tip positions differ')
    old_bytes=before.view(np.uint8);new_bytes=destination.view(np.uint8)
    changed_bytes=int(np.count_nonzero(old_bytes!=new_bytes))
    changed_rows=int(np.count_nonzero(np.any(before!=destination,axis=1)))
    updated=deepcopy(source.doc)
    updated['accessors'][aid]['min']=destination.min(axis=0).tolist()
    updated['accessors'][aid]['max']=destination.max(axis=0).tolist()
    expected=deepcopy(updated);expected['accessors'][aid]=source.doc['accessors'][aid]
    require(expected==source.doc,'Non-allowlisted JSON changed')
    encoded=json.dumps(updated,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode();encoded+=b' '*(-len(encoded)%4)
    length=28+len(encoded)+len(binary)
    require(length<=64*1024**2,'Candidate exceeds64MiB')
    deadline()
    with (OUT/'model.glb').open('xb') as stream:
        stream.write(struct.pack('<4sII',b'glTF',2,length));stream.write(struct.pack('<II',len(encoded),0x4e4f534a))
        stream.write(encoded);stream.write(struct.pack('<II',len(binary),0x004e4942));stream.write(binary)
    np.savez_compressed(OUT/'regional-helper.npz',**{key:reshaped[key] for key in core})
    audit={'schema_version':1,'status':'produced_tip_only_candidate','source_sha256':source.hash,
        'raw_id_source_sha256':raw.hash,'model_sha256':sha256(OUT/'model.glb'),'source_vertices':skin_report['vertex_count'],
        'exported_vertices':total_vertices,'triangles':total_triangles,'base_exported_vertices':base_vertices,
        'fins':report['cards'],'owner':owner,'position_accessor':aid,'position_buffer_view':view_id,
        'position_range_bytes':[start,end],'tip_rows':int(tip.sum()),'root_rows':int((~tip).sum()),
        'changed_position_rows':changed_rows,'changed_position_bytes':changed_bytes,
        'old_positions_sha256':hashlib.sha256(before.tobytes()).hexdigest(),
        'new_positions_sha256':hashlib.sha256(destination.tobytes()).hexdigest(),
        'source_global_ids_sha256':hashlib.sha256(ids.astype('<u4').tobytes()).hexdigest(),
        'native_to_helper_ids_sha256':hashlib.sha256(mapping.astype('<u4').tobytes()).hexdigest(),
        'raw_final_positions_and_indices_exact_before_change':True,'native_ids_complete_unique_and_position_bound':True,
        'oriented_native_id_triangles_exact':True,'all_original_roots_byte_exact':True,
        'all_non_tip_binary_bytes_including_padding_exact':True,
        'json_changes':[f'/accessors/{aid}/min',f'/accessors/{aid}/max'],
        'unchanged':['body geometry','root positions','normals','tangents','UVs','weights','joints','images','materials','nodes','skins','animations'],
        'full_native_emitter_bounds':{'low':low.tolist(),'span':span.tolist()},
        'experimental':True,'quality_approved':False,'motion_verified':False,'provider_calls':0,'native_calls':0,
        'limitation':'Root positions/skin/animations remain exact, but changed fin tips still need independent motion/appearance review; no geometry-contact repair or density acceptance.'}
    del source,raw,binary,destination,old_bytes,new_bytes,skin,helper,reshaped,core
    gc.collect();deadline()
    validation=validate_render_attributes(OUT/'model.glb')
    (OUT/'render-attributes.json').write_text(json.dumps(validation,indent=2)+'\n')
    audit['render_attributes_status']=validation['status']
    audit['inputs_unchanged']=all(sha256(PATHS[key])==row['sha256'] for key,row in PREP['inputs'].items())
    require(audit['inputs_unchanged'],'Original input changed')
    audit['elapsed_seconds']=time.monotonic()-START;deadline()
    (OUT/'byte-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({key:audit[key] for key in ['status','model_sha256','exported_vertices','triangles','tip_rows','changed_position_bytes','elapsed_seconds']}))


try:
    produce()
except Exception as error:
    (OUT/'failure.json').write_text(json.dumps({'status':'rejected','type':type(error).__name__,'message':str(error)},indent=2)+'\n')
    raise
