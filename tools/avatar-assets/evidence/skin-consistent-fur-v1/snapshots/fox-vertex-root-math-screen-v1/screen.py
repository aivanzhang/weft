"""One-hot placement screen on all retained sampled fox triangles, without a derivative."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.three_d import _GLB, _animation_tracks, _pose, _world_matrices

START = time.monotonic()
RESULT = {'schema_version': 1, 'status': 'running', 'appearance_accepted': False,
          'new_avatar_created': False, 'native_provider_AI_calls': 0}


def save(name, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(text)


def require(ok, text):
    if not ok:
        raise ValueError(text)


def array(glb, index, kind, types, indices=False):
    return np.asarray(glb.accessor(index, kind, types, count_limit=300000 if indices else 100000))


def quantiles(values):
    return {name:float(np.quantile(values,q)) for name,q in [('min',0),('median',.5),('p95',.95),('max',1)]}


def main():
    plan = json.loads((HERE/'prepared.json').read_text())
    for row in plan['inputs']:
        require(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256'], 'Input changed '+row['path'])
    require(json.loads((ROOT/'outputs/fox-material-fur-independent-audit-v2/native-source-identity.json').read_text())['equal'],
            'Original native vertex-ID correspondence not proved')
    prior = ROOT/plan['old_audit']
    root_records = json.loads((prior/'roots.json').read_text())['samples']
    old_motion = json.loads((prior/'motion.json').read_text())
    with np.load(prior/'sampled-motion.npz',allow_pickle=False) as old:
        old_component_errors = abs(old['actual_roots']-old['source_surface_minus_inset']).max(axis=3)
    old_failed = (old_component_errors>plan['fixed_motion_gate_m']).any(axis=(0,1))
    require(len(root_records) == plan['root_count'] and int(old_failed.sum()) == 21, 'Sealed sample/witness count mismatch')
    glb = _GLB(ROOT/plan['source']);doc=glb.doc;nodes=doc['nodes']
    owners = [i for i,n in enumerate(nodes) if 'mesh' in n]
    require(len(owners)==1, 'Source mesh owner is ambiguous')
    owner=nodes[owners[0]];skin=doc['skins'][owner['skin']]
    names=[nodes[i]['name'] for i in skin['joints']]
    require(len(names)==len(set(names))==20, 'Named source skeleton changed')
    inverse=array(glb,skin['inverseBindMatrices'],'MAT4',(5126,)).reshape(-1,4,4).transpose(0,2,1)
    wanted={r['material'] for r in root_records};rows={};vo=fo=0
    for primitive in doc['meshes'][owner['mesh']]['primitives']:
        name=doc['materials'][primitive['material']]['name'];attrs=primitive['attributes']
        nv=doc['accessors'][attrs['POSITION']]['count'];nf=doc['accessors'][primitive['indices']]['count']//3
        if name in wanted:
            rows[name]={'points':array(glb,attrs['POSITION'],'VEC3',(5126,)),
                'weights':array(glb,attrs['WEIGHTS_0'],'VEC4',(5121,5123,5126)),
                'joints':array(glb,attrs['JOINTS_0'],'VEC4',(5121,5123)).astype(np.int32),
                'faces':array(glb,primitive['indices'],'SCALAR',(5121,5123,5125),True).reshape(-1,3).astype(np.int32),
                'vertex_offset':vo,'face_offset':fo}
        vo+=nv;fo+=nf
    points=[];joints=[];weights=[];source_ids=[];placement=[];all_distances=[]
    for index,record in enumerate(root_records):
        row=rows[record['material']];face=record['source_polygon']-row['face_offset']
        ids=row['faces'][face];global_ids=ids+row['vertex_offset']
        require(global_ids.tolist()==record['source_vertices'], 'Root triangle identity changed')
        p=row['points'][ids];w=row['weights'][ids];j=row['joints'][ids]
        bary=np.asarray(record['barycentric']);old_root=bary@p
        distances=np.linalg.norm(p-old_root,axis=1)
        one_hot=np.eye(3);anchored=one_hot@p
        require(np.array_equal(anchored,p), 'One-hot placement altered exact source positions')
        points.extend(p);joints.extend(j);weights.extend(w);source_ids.extend(global_ids);all_distances.append(distances)
        placement.append({'sample':index,'material':record['material'],'old_root_index':record['root_index'],
            'source_polygon':record['source_polygon'],'old_barycentric':record['barycentric'],
            'source_vertex_ids':global_ids.tolist(),'distance_from_old_baryroot_m':distances.tolist(),
            'nearest_corner':int(np.argmin(distances)), 'old_attachment_failed':bool(old_failed[index]),
            'candidate_named_weights':[{names[k]:float(v) for k,v in zip(ji,wi) if v>0} for ji,wi in zip(j,w)]})
    points=np.asarray(points);joints=np.asarray(joints);weights=np.asarray(weights);source_ids=np.asarray(source_ids)
    require(points.shape==(1152,3) and weights.shape==(1152,4), 'All three corners must be included for every root')
    sorted_names=sorted(names);remap=np.array([sorted_names.index(n) for n in names])
    named_weights=np.zeros((len(points),len(names)))
    for column in range(4):
        np.add.at(named_weights,(np.arange(len(points)),remap[joints[:,column]]),weights[:,column])
    require(np.array_equal(named_weights.sum(axis=1),weights.sum(axis=1)), 'Named influence copy altered total weight')
    active=(weights>0).sum(axis=1);distances=np.asarray(all_distances)
    save('placements.json',{'samples':placement,'all_candidate_displacement_m':quantiles(distances),
        'nearest_vertex_displacement_m':quantiles(distances.min(axis=1)),
        'old_failed_nearest_vertex_displacement_m':quantiles(distances[old_failed].min(axis=1)),
        'changed_root_reconstruction':True,'note':'Distances are from the old un-inset barycentric surface root. A nearest corner is descriptive only; no allocation algorithm or avatar was produced.'})
    parents={c:i for i,n in enumerate(nodes) for c in n.get('children',[])}
    tracks={name:t for name,t,_ in _animation_tracks(glb,nodes,set(skin['joints']))}
    require(set(tracks)=={x['clip'] for x in old_motion['clips']}, 'Source clips changed')
    named_order=[names.index(n) for n in sorted_names];homogeneous=np.column_stack((points,np.ones(len(points))))
    clips=[];pose_errors=[];max_error=0.
    for clip in [x['clip'] for x in old_motion['clips']]:
        samples=[];clip_errors=[]
        for second in plan['sample_seconds']:
            world=_world_matrices(nodes,parents,_pose(tracks[clip],second))
            matrices=np.array([np.asarray(world[i]).reshape(4,4).T for i in skin['joints']])@inverse
            source_pose=np.zeros_like(points)
            for column in range(4):
                source_pose+=np.einsum('nij,nj->ni',matrices[joints[:,column],:3],homogeneous)*weights[:,column,None]
            anchor_pose=np.einsum('nb,bij,nj->ni',named_weights,matrices[named_order,:3],homogeneous)
            errors=abs(anchor_pose-source_pose).max(axis=1).reshape(384,3)
            max_error=max(max_error,float(errors.max()));clip_errors.append(errors)
            samples.append({'second':second,'max_component_error_m':float(errors.max()),
                'old_failed_roots_all_corner_max_error_m':float(errors[old_failed].max()),
                'potential_anchors_over_fixed_gate':int((errors>plan['fixed_motion_gate_m']).sum())})
        clips.append({'clip':clip,'samples':samples});pose_errors.append(clip_errors)
    np.savez_compressed(HERE/'measurements.npz',source_vertex_ids=source_ids.reshape(384,3),
        source_positions=points.reshape(384,3,3),source_joint_slots=joints.reshape(384,3,4),
        unchanged_source_weights=weights.reshape(384,3,4),old_failed_root_mask=old_failed,
        old_attachment_component_errors=old_component_errors,placement_distances=distances,
        one_hot_motion_component_errors=np.asarray(pose_errors),seconds=np.asarray(plan['sample_seconds']))
    save('motion.json',{'clips':clips,'fixed_gate_m':plan['fixed_motion_gate_m'],'maximum_error_m':max_error,
        'methods':'Reference: original four indexed influences. Candidate: one-hot source vertex with unchanged weights, remapped through bone names and evaluated by an independent dense weighted transform. No root-weight averaging or normalization.'})
    save('old-failure-witnesses.json',{'samples':[placement[i] for i in np.flatnonzero(old_failed)],
        'count':21,'all_three_potential_anchors_evaluated':True,'all_mixed_weights_retained':True})
    RESULT.update(status='passed_numerical_source_vertex_screen' if max_error<=plan['fixed_motion_gate_m'] else 'failed',
        source_sha256=glb.hash,sampled_roots=384,potential_anchors=1152,unique_source_vertex_ids=int(len(np.unique(source_ids))),
        potential_anchors_with_mixed_weights=int((active>1).sum()),weight_support_histogram={str(i):int((active==i).sum()) for i in range(1,5)},
        source_weight_sum_max_deviation=float(abs(weights.sum(axis=1)-1).max()),unchanged_source_weight_values=True,
        clips=4,times_per_clip=9,maximum_source_motion_component_error_m=max_error,
        old_failed_roots_covered=21,nearest_vertex_displacement_m=quantiles(distances.min(axis=1)),
        all_candidate_displacement_m=quantiles(distances),same_root_reconstruction=False,
        limits=['Algebraic/source-data numerical screen, not native/export proof','No actual new roots allocated or avatar created',
                'Does not prove density, spacing, coverage, native hair motion or appearance'])
    require(max_error<=plan['fixed_motion_gate_m'],'Source-vertex motion exceeded unchanged gate')


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        RESULT.update(status='failed',exception={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        RESULT.update(elapsed_seconds=time.monotonic()-START,
            process_high_water_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save('result.json',RESULT)
