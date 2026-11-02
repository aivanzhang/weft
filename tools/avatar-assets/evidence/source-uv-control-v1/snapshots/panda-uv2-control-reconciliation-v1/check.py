"""One CPU replay of saved native inputs, without Blender or a new bake."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import warnings
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def save(name, value):
    with (OUT/name).open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class SumProducts(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),
                attr='sum',ctx=ast.Load()), args=[ast.BinOp(left=node.left,op=ast.Mult(),right=node.right)],
                keywords=[ast.keyword(arg='axis',value=ast.Constant(value=-1))]),node)
        return node


def stats(weights):
    return {'shape':list(weights.shape),'finite':bool(np.isfinite(weights).all()),
        'minimum':float(weights.min()),'maximum':float(weights.max()),
        'row_sum_max_error':float(np.abs(weights.sum(axis=1)-1).max()),
        'max_nonzero_influences':int((weights>0).sum(axis=1).max()),
        'float64_sha256':hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest()}


def main():
    start=time.monotonic();prep=json.loads((OUT/'prepared.json').read_text())
    for row in prep['inputs']:
        require(sha(ROOT/row['path'])==row['sha256'],'Changed input '+row['path'])
    cfg=json.loads((ROOT/prep['native_config']).read_text())
    control=json.loads((ROOT/prep['prepared_control']).read_text())
    selection=json.loads((ROOT/prep['selection']).read_text())
    plan=json.loads((ROOT/prep['plan']).read_text())
    source_audit=json.loads((ROOT/prep['source_audit']).read_text())
    differences={key:{'native':cfg.get(key),'prepared':control.get(key)} for key in sorted(set(cfg)|set(control)) if cfg.get(key)!=control.get(key)}
    require(set(differences)<= {'id'},'Unexpected control/config difference: '+','.join(differences))
    require('normal_continuity' not in cfg,'Normal transfer was reintroduced')
    path=ROOT/prep['arrays']
    with zipfile.ZipFile(path) as archive:
        rows=archive.infolist()
        require(len(rows)==11 and len({r.filename for r in rows})==11,'Unexpected NPZ entries')
        require(sum(r.file_size for r in rows)<=32*1024**2,'NPZ decoded size exceeds32MiB')
        require(all(not r.is_dir() and r.filename.endswith('.npy') and '/' not in r.filename for r in rows),'Unsafe NPZ member')
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)=={'vertices','faces','uv0','uv1','selected','material_ids','vertex_mask','loops','uv0_tangents','uv1_tangents','corner_normals'},'NPZ schema changed')
        points=archive['vertices'];faces=archive['faces'];chosen=archive['selected'];mats=archive['material_ids'];mask=archive['vertex_mask']
    require(points.shape==(51376,3) and points.dtype==np.float32 and np.isfinite(points).all(),'Unexpected native points')
    require(faces.shape==(62581,3) and faces.min()>=0 and faces.max()<len(points),'Unexpected native faces')
    require(chosen.dtype==bool and chosen.shape==(len(faces),) and np.array_equal(chosen,mask[faces].all(axis=1)),'Native selection mask mismatch')
    material_rows=selection['materials'];by_name={row['material']:row for row in material_rows}
    coverage=[]
    for row in material_rows:
        selected_ids=np.flatnonzero(chosen&(mats==row['slot']))
        require(int(np.sum(mats==row['slot']))==row['faces'] and len(selected_ids)==row['selected_faces'],'Material count mismatch')
        if len(selected_ids):
            planned=next(r for r in plan['regions'] if r['region_id']==row['slot'])
            require(planned['selected_faces']==len(selected_ids) and planned['selection_sha256']==hashlib.sha256(selected_ids.astype('<i4').tobytes()).hexdigest(),'Plan selection binding mismatch')
        coverage.append({'material':row['material'],'slot':row['slot'],'faces':row['faces'],
                         'selected_faces':len(selected_ids),'all_selected':len(selected_ids)==row['faces']})
    source_uv1_names=[r['material'] for r in source_audit['material_primitives']
        if r['texture_routes'].get('normalTexture',{}).get('texCoord',0)==1]
    hazard=[]
    for name in source_uv1_names:
        row=by_name[name];unselected=np.flatnonzero((mats==row['slot'])&~chosen)
        require(not len(unselected) and row['slot'] in plan['atlas_regions'],'Source UV1 normal faces would survive export')
        hazard.append({'material':name,'slot':row['slot'],'source_normal_uv':1,'faces':row['faces'],
            'unselected_faces':len(unselected),'planned_authored_normal_uv':2,
            'replacement_material':name+'__short_pile_adaptive_v1'})
    require(plan['projected_export_vertex_ceiling']==99154 and plan['additional_export_vertices']==482,'Unexpected bound')
    code=(ROOT/prep['rig_helper']).read_text();native_ns={'__name__':'saved_rig_numeric_replay'}
    exec(compile(code,prep['rig_helper'],'exec'),native_ns)
    tree=ast.parse(code);count=sum(isinstance(n,ast.BinOp) and isinstance(n.op,ast.MatMult) for n in ast.walk(tree))
    require(count==2,'Unexpected rig projection expression')
    tree=SumProducts().visit(tree);ast.fix_missing_locations(tree)
    shadow_ns={'__name__':'sum_products_numeric_oracle'}
    exec(compile(tree,'diagnostic_sum_products_oracle','exec'),shadow_ns)
    low,high=points.min(axis=0).astype(float),points.max(axis=0).astype(float)
    rig=cfg['authored_rig']; span=high-low
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        actual,proof=native_ns['skin_weights'](points,low,high,rig)
    with np.errstate(all='raise'):
        oracle,oracle_proof=shadow_ns['skin_weights'](points,low,high,rig)
    error=float(np.abs(actual-oracle).max())
    require(np.allclose(actual,oracle,atol=1e-12,rtol=0) and proof==oracle_proof,'BLAS/non-BLAS numeric mismatch')
    algebra=[]
    for bone in rig['bones']:
        head=np.asarray(bone['head'])*span+low;tail=np.asarray(bone['tail'])*span+low;direction=tail-head
        numerator=np.sum((points.astype(float)-head)*direction,axis=1);denominator=float(np.sum(direction*direction))
        require(np.isfinite(numerator).all() and denominator>0,'Invalid segment projection operands')
        algebra.append({'bone':bone['name'],'direction_squared_length':denominator,
            'numerator_min_max':[float(numerator.min()),float(numerator.max())],
            'unclipped_fraction_max_absolute':float(np.abs(numerator/denominator).max())})
    worst=np.argsort(-np.max(np.abs(actual-oracle),axis=1),kind='stable')[:32]
    witness={'points':points[worst],'native_vertex_indices':worst.astype('<i4'),
        'matmul_weights':actual[worst],'sum_products_weights':oracle[worst],'bounds_min':low,'bounds_max':high}
    np.savez_compressed(OUT/'rig-witness.npz',**witness)
    save('rig-config.json',rig)
    source_log=(ROOT/prep['native_log']).read_text()
    flags=[line.strip() for line in source_log.splitlines() if 'RuntimeWarning:' in line]
    report={'schema_version':1,'status':'control_config_and_numeric_replay_passed',
        'source_sha256':cfg['source_sha256'],'native_config_sha256':sha(ROOT/prep['native_config']),
        'prepared_control_sha256':sha(ROOT/prep['prepared_control']),'config_differences':differences,
        'all_other_config_fields_exact':True,'normal_transfer_omitted_control_not_guard_relaxation':True,
        'coverage_from_actual_saved_arrays':coverage,'source_uv1_normal_coverage':hazard,
        'planned_export_vertices':99154,'additional_adaptive_seams':482,'remaining_vertices':846,
        'warning_reconciliation':{'native_warnings':flags,'local_numpy':np.__version__,'local_python':sys.version,
            'local_replay_warnings':[{'category':w.category.__name__,'message':str(w.message),'line':w.lineno} for w in captured],
            'actual_native_point_count':len(points),'actual_authored_bones':len(rig['bones']),
            'matmul_weights':stats(actual),'non_blas_sum_products_weights':stats(oracle),
            'max_absolute_weight_difference':error,'all_layer_coverage_counts_equal':proof==oracle_proof,
            'rig_proof':proof,'segment_projection_operand_bounds':algebra,
            'conclusion':'Actual saved points/config produce finite normalized weights matching independent sum-of-products projection; native warning flags alone are not numeric failure evidence.',
            'limits':['Replay uses host NumPy, not Blender bundled NumPy/BLAS; the native warning cause is not proven.',
                'Saved arrays are float32 post-authoring positions. Original transient pre-round native rig inputs and weights were not captured.',
                'This checks initial authored skin math, not part replacement, later constraints, motion deformation or visual quality.']},
        'required_staged_export_assertion':{
            'before_bake':'Every live triangle whose original Normal Map uses nonzero source UV must be selected and assigned an authored atlas payload. Here slot4 requires146/146, no exception.',
            'before_export':'No live polygon may retain a source UV1 Normal Map route after undercoat assignment; unused original material slots are allowed.',
            'replacement_binding':{'material':'body_surface__contact_uv_dark_v1__short_pile_adaptive_v1','selected_faces':146,'normal_texcoord':2,
                'also_require':['exact authored UV index/name/hash','exact new PNG hash','mikk-unshared-triangle-v-up-v2','MR links remain source UV1']},
            'after_export':'Reject any live primitive normalTexture.texCoord=1. Bind only generated UV2 target names, actual face counts and embedded PNG hashes before targeted tangent correction.',
            'scope':'Valid only for the measured complete replacement route; not a general source UV1 tangent preservation solution.'},
        'inputs_unchanged':all(sha(ROOT/r['path'])==r['sha256'] for r in prep['inputs']),
        'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':round(time.monotonic()-start,6),
        'native_calls':0,'paid_calls':0,'quality_approved':False}
    require(report['inputs_unchanged'],'Changed source evidence')
    save('report.json',report)
    print(json.dumps({'status':report['status'],'config_differences':list(differences),'weight_max_error':error,
        'weights':stats(actual),'local_warning_count':len(captured),'native_warning_count':len(flags),
        'elapsed_seconds':report['elapsed_seconds'],'self_peak_rss_bytes':report['self_peak_rss_bytes']}))


if __name__=='__main__':main()
