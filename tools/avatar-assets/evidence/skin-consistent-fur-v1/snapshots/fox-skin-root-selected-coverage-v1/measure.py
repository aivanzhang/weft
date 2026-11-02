"""Measure selected binding locations, independently of exported skin/image review."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.monotonic()
spec=importlib.util.spec_from_file_location('census_helpers',OUT/'sources/reference_census.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
require=helpers.require


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(name,value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def archive(path,count):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=512*1024,'Invalid bounded archive')
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist())==3 and set(z.namelist())=={'triangle_index.npy','barycentric.npy','root_inset.npy'}
                and sum(i.file_size for i in z.infolist())<=512*1024,'Invalid archive members')
    with np.load(path,allow_pickle=False) as z:
        values={k:z[k] for k in z.files}
    for key,shape,dtype in [('triangle_index',(count,),'<i4'),('barycentric',(count,3),'<f8'),('root_inset',(count,3),'<f8')]:
        a=values[key]
        require(a.shape==shape and a.dtype.str==dtype and np.isfinite(a).all(),'Invalid '+key+' schema')
    b=values['barycentric'];inset=values['root_inset']
    require(np.all(b>=0) and np.all(abs(b.sum(axis=1)-1)<=1e-6),'Invalid barycentric rows')
    require(np.all(abs(inset)<=.0005+1e-12) and np.all(np.linalg.norm(inset,axis=1)<=.0005+1e-12),'Invalid root inset')
    return values


def main():
    prepared=load(OUT/'prepared.json')
    for item in prepared['inputs']:
        p=ROOT/item['path']
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes']
                and sha(p)==item['sha256'],'Input changed: '+item['path'])
    candidate=ROOT/'outputs/fox-skin-consistent-fur-finish-v1'
    prior=ROOT/'outputs/fox-skin-root-coverage-preflight-v1'
    original=ROOT/'outputs/fox-material-fur-finish-v1'
    finish=load(candidate/'finish-report.json');config=load(candidate/'finish-config.json')
    groom=load(candidate/'material-groom/report.json');old_groom=load(original/'material-groom/report.json')
    require(groom['body_before']==old_groom['body_before'],'Candidate source signature changed')
    require(groom['correspondence']==old_groom['correspondence'],'Candidate source face correspondence changed')
    require(sha(candidate/'material-groom/correspondence.npz')==groom['correspondence']['sha256'],'Current face archive differs')
    require(load(candidate/'material-coat.json')==load(original/'material-coat.json'),'Material allocation changed')
    census=load(prior/'report.json')
    require(config['source_sha256']==census['source_sha256'],'Classification source differs')
    proof=load(ROOT/'outputs/fox-vertex-root-coverage-preflight-v1/source-binding.json')
    grid=load(ROOT/'outputs/fox-vertex-root-coverage-preflight-v1/spatial-grid.json')
    with np.load(prior/'classification.npz',allow_pickle=False) as z: classification={k:z[k] for k in z.files}
    with np.load(candidate/'material-groom/correspondence.npz',allow_pickle=False) as z:
        face_ids={k:z[k] for k in z.files if k.endswith('_face_ids')}
    doc,binary=helpers.glb(Path(config['source']))
    parts=finish['portable_fur'];require(len(parts)==3 and sum(p['cards'] for p in parts)==3000,'Expected three unchanged parts')
    rows,component_rows,grid_rows,selected_arrays=[],[],[],{}
    all_positions=[];archive_paths=[]
    for index,part in enumerate(parts):
        material=part['source_material'];previous=next(x for x in census['materials'] if x['material']==material)
        require(part['cards']==previous['requested_tufts'] and part['source_material_slot']==previous['material_slot']
                and part['root_summary']['root_placement']=='skin-consistent-v1','Part contract changed')
        receipt=part['root_summary']['binding_archive']
        path=candidate/receipt['path'];archive_paths.append(receipt['path'])
        require(path.parent==candidate/'portable-root-bindings' and path.name.startswith('PortableCoat_')
                and receipt['schema_version']==1 and receipt['roots']==part['cards']
                and receipt['space']=='blender_local_z_up_meters'
                and receipt['root_inset_convention']=='surface_root_minus_bottom_edge_midpoint'
                and path.stat().st_size==receipt['bytes'] and sha(path)==receipt['sha256'],'Unbound archive')
        values=archive(path,part['cards']);ordinals=values['triangle_index'];b=values['barycentric']
        bound=next(x for x in proof['primitive_ranges'] if x['material']==material)
        primitive=doc['meshes'][0]['primitives'][bound['primitive']]
        p=helpers.read_array(doc,binary,primitive['attributes']['POSITION'])
        p=np.column_stack((p[:,0],-p[:,2],p[:,1])).astype('<f4')
        f=helpers.read_array(doc,binary,primitive['indices']).ravel().astype('<i4').reshape(-1,3)
        selected_faces=face_ids[f'part_{index:02d}_face_ids'];f=f[selected_faces-bound['face_start']]
        require(np.all(ordinals>=0) and np.all(ordinals<len(f)),'Binding triangle escaped material domain')
        source_triangles=selected_faces[ordinals]
        uniform=np.isin(source_triangles,classification[f'part_{index:02d}_uniform_source_faces'])
        varying=np.isin(source_triangles,classification[f'part_{index:02d}_varying_source_faces'])
        require(np.all(uniform!=varying),'A selected face has missing/ambiguous classification')
        require([int(uniform.sum()),int(varying.sum())]==previous['domain_counts'],'Actual domain counts differ')
        require(np.all(b[uniform]>0),'Uniform interior draw is not interior')
        require(np.all((b[varying]==0)|(b[varying]==1)) and np.all((b[varying]==1).sum(axis=1)==1),
                'Varying root is not an exact source vertex')
        vertex_local=f[ordinals[varying],np.argmax(b[varying],axis=1)]
        vertex_global=vertex_local+bound['vertex_start']
        require(len(np.unique(vertex_global))==len(vertex_global) and
                np.isin(vertex_global,classification[f'part_{index:02d}_varying_source_vertices']).all(),
                'Repeated or ineligible selected source vertex ID')
        positions=np.sum(p[f[ordinals]].astype(float)*b[:,:,None],axis=1)
        all_positions.append(positions)
        unique,inv,multiplicity=np.unique(positions[varying],axis=0,return_inverse=True,return_counts=True)
        aliases=[]
        for group in np.flatnonzero(multiplicity>1):
            hits=np.flatnonzero(inv==group)
            aliases.append({'source_vertices':vertex_global[hits].tolist(),
                'binding_rows':np.flatnonzero(varying)[hits].tolist(),'position_blender_xyz_m':unique[group].tolist()})
        labels=helpers.components(f,len(p));root_labels=labels[f[ordinals,0]]+bound['vertex_start']
        components=[]
        for c in previous['components']:
            hit=root_labels==c['source_vertex_component']
            components.append({'source_vertex_component':c['source_vertex_component'],
                'authored_bone_support':c['authored_bone_support'],
                'eligible_area_m2':c['uniform_area_m2']+c['varying_area_m2'],
                'expected_uniform_roots':c['uniform_fractional_quota'],'expected_vertex_roots':c['varying_fractional_quota'],
                'actual_roots':int(hit.sum()),'actual_uniform_roots':int((hit&uniform).sum()),
                'actual_vertex_roots':int((hit&varying).sum()),
                'root_minus_area_quota':float(hit.sum()-c['uniform_fractional_quota']-c['varying_fractional_quota']),
                'old_vertex_only_saturated':c['old_vertex_only_saturated'],
                'old_vertex_only_capacity':c['old_vertex_only_capacity']})
        require(sum(x['actual_roots'] for x in components)==part['cards'],'Component count incomplete')
        component_rows.append({'material':material,'components':components})
        low,span=np.asarray(grid['low_blender_xyz_m']),np.asarray(grid['span_m'])
        cells=np.minimum(((positions-low)/span*6).astype(int),5)
        require(np.all(cells>=0),'Root outside preserved body bounds')
        cells=np.ravel_multi_index(cells.T,(6,6,6));count=np.bincount(cells,minlength=216)
        vcount=np.bincount(cells[varying],minlength=216)
        cell_lookup={tuple(x['cell']):x for x in next(x['cells'] for x in grid['materials'] if x['material']==material)}
        cell_report=[]
        for cell in range(216):
            xyz=tuple(map(int,np.unravel_index(cell,(6,6,6))));oldcell=cell_lookup.get(xyz,{})
            if oldcell or count[cell]:
                cell_report.append({'cell':xyz,'actual_roots':int(count[cell]),'actual_vertex_roots':int(vcount[cell]),
                    'centroid_assigned_area_m2':oldcell.get('centroid_assigned_area_m2',0),
                    'nominal_area_roots':oldcell.get('fractional_area_quota',0)})
        grid_rows.append({'material':material,'cells':cell_report})
        zero=[x for x in components if x['actual_roots']==0]
        rows.append({'material':material,'total_roots':part['cards'],'uniform_roots':int(uniform.sum()),
            'vertex_anchors':int(varying.sum()),'unique_selected_vertex_ids':len(vertex_global),
            'unique_selected_vertex_positions':len(unique),'selected_coincident_position_groups':len(aliases),
            'coincident_excess_anchors':len(vertex_global)-len(unique),
            'maximum_selected_position_multiplicity':int(multiplicity.max()) if len(multiplicity) else 0,
            'selected_aliases':aliases,'components_total':len(components),
            'components_with_roots':len(components)-len(zero),
            'zero_root_component_area_m2':sum(x['eligible_area_m2'] for x in zero),
            'zero_root_components_with_expected_at_least_one':sum(x['expected_uniform_roots']+x['expected_vertex_roots']>=1 for x in zero),
            'maximum_absolute_component_quota_deviation':max(abs(x['root_minus_area_quota']) for x in components),
            'formerly_saturated_components':[x for x in components if x['old_vertex_only_saturated']]})
        selected_arrays[f'part_{index:02d}_surface_positions']=positions
        selected_arrays[f'part_{index:02d}_source_faces']=source_triangles.astype('<i4')
        selected_arrays[f'part_{index:02d}_is_vertex']=varying
        selected_arrays[f'part_{index:02d}_vertex_ids']=vertex_global.astype('<i4')
        require(time.monotonic()-START<25,'Internal placement deadline')
    actual_paths={str(x.relative_to(candidate)) for x in (candidate/'portable-root-bindings').iterdir()}
    require(actual_paths==set(archive_paths) and len(archive_paths)==len(set(archive_paths)),'Extra/missing root archives')
    positions=np.concatenate(all_positions)
    require(len(positions)==3000,'Root count changed')
    with (OUT/'selected-placement.npz').open('xb') as stream:np.savez(stream,**selected_arrays)
    save('components.json',{'materials':component_rows})
    save('spatial-grid.json',{'grid':[6,6,6],
        'qualifier':'Area by original triangle centroid, roots by surface position; not raster footprint or clipped surface coverage.',
        'materials':grid_rows})
    result={'schema_version':1,'status':'measured','candidate_model_sha256':prepared['model_sha256'],
        'materials':rows,'total_roots':3000,'total_uniform_roots':sum(x['uniform_roots'] for x in rows),
        'total_vertex_anchors':sum(x['vertex_anchors'] for x in rows),
        'all_root_distinct_positions':len(np.unique(positions,axis=0)),
        'source_body_proof_reused':True,'selector_rerun':False,'quality_approved':False,
        'limitations':['Measures saved source-root bindings; separate audit must prove their exported geometry/skin correspondence.',
            'No final GLB attributes, images or skin motion were decoded or evaluated here.',
            'Component counts and root spacing do not prove visible full-coat coverage; zero-root tiny components can share adjacent surfaces.',
            'Authored bone names are role labels, not recovered anatomy.'],
        'elapsed_seconds':time.monotonic()-START,
        'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    save('report.json',result)
    print(json.dumps({'status':'measured','materials':rows},sort_keys=True))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),
                            'elapsed_seconds':time.monotonic()-START})
        raise
