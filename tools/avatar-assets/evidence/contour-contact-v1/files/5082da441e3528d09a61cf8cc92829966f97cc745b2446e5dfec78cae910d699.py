"""Read-only fin profile/projection measurements; no native or visibility verdict."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time
import zipfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source'))
from avatar_assets.runner import resident_bytes
from groom_flow import prepare_flow, evaluate_prepared_flow
import os

START = time.monotonic()
PREP = json.loads((HERE/'prepared.json').read_text())
PATHS = {key:Path(row['path']) for key,row in PREP['inputs'].items()}


def require(condition, message):
    if not condition:raise ValueError(message)


def deadline():
    require(time.monotonic()-START < 27, 'Internal27-second bound exceeded')
    require(resident_bytes(os.getpid()) <= 512*1024**2, 'Internal512MiB sampled RSS bound exceeded')


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def stats(value):
    value=np.asarray(value,dtype=float)
    if not value.size:return {'count':0}
    require(np.isfinite(value).all(),'Nonfinite metric')
    return {'count':int(value.size),**dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(value,[0,.1,.5,.9,1]))))}


def load_npz(path, maximum):
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        require(len(entries) <= 16 and len({row.filename for row in entries})==len(entries)
                and sum(row.file_size for row in entries)<=maximum,'Unsafe NPZ bounds')
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def normalize(v):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    require(np.all(length>1e-10),'Degenerate vector')
    return v/length


def nearest(points, selected=None):
    ids=np.arange(len(points)) if selected is None else np.flatnonzero(selected)
    if len(ids)<2:return np.empty(0)
    target=points[ids]; result=[]
    for start in range(0,len(ids),128):
        deadline(); stop=min(start+128,len(ids))
        delta=target[start:stop,None]-target[None]
        squared=np.einsum('ijk,ijk->ij',delta,delta)
        squared[np.arange(stop-start),np.arange(start,stop)]=np.inf
        result.extend(np.sqrt(squared.min(axis=1)))
    return np.asarray(result)


def groups(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return np.column_stack((np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def camera(boxes, offset, bounds):
    low,high=boxes.min(axis=0),boxes.max(axis=0)
    require(np.allclose(low,bounds['min'],rtol=0,atol=2e-7)
            and np.allclose(high,bounds['max'],rtol=0,atol=2e-7),'Saved preview bounds differ')
    center=(low+high)*.5; back=normalize(np.array(offset,dtype=float))
    right=normalize(np.cross([0.,0,1.],back));up=normalize(np.cross(back,right))
    basis=np.column_stack((right,up,back)); projected=(boxes-center)@basis
    scale=float(np.ptp(projected[:,:2],axis=0).max()*1.18)
    return basis,scale


require(all(sha(PATHS[key])==row['sha256'] for key,row in PREP['inputs'].items()),'Input pin changed')
report=json.loads(PATHS['fringe_report'].read_text()); native=json.loads(PATHS['native_inspection'].read_text())
skin_report=json.loads(PATHS['skin_report'].read_text())
skin=load_npz(PATHS['skin_npz'],8*1024**2); fins=load_npz(PATHS['fringe_npz'],8*1024**2)
require(native['source_sha256']==PREP['inputs']['preview_model']['sha256'],'Preview source binding differs')
positions=fins['positions'].astype(float).reshape(-1,4,3); count=len(positions)
require(count==report['cards']==3600,'Unexpected pinned fin count')
require(np.isfinite(positions).all() and len(skin['positions'])<=100000,'Unsafe point arrays')
uv=fins['uv_gltf'].astype(float).reshape(count,4,2)
normals=fins['normals'].astype(float).reshape(count,4,3)
outward=normalize(normals[:,:2].sum(axis=1))
roots=positions[:,:2].mean(axis=1); vectors=positions[:,2]-positions[:,0]
length=np.linalg.norm(vectors,axis=1); direction=normalize(vectors)
spans=np.linalg.norm(positions[:,1]-positions[:,0],axis=1)
lean=direction-outward*np.sum(direction*outward,axis=1)[:,None]
lean_length=np.linalg.norm(lean,axis=1)
require(np.all(lean_length>1e-8),'Undefined fixed-root comb counterfactual')
lean/=lean_length[:,None]
reconstructed=normalize(outward*.85+lean*.15)
require(np.max(abs(reconstructed-direction))<2e-5,'Current fin direction cannot be reproduced')
config=json.loads(PATHS['finish_config'].read_text())
body_owner=skin_report['owners'][0]
body_points=skin['positions'][body_owner['vertex_start']:body_owner['vertex_start']+body_owner['vertex_count']]
field=prepare_flow(body_points.min(axis=0),np.ptp(body_points,axis=0),
    config['groom']['direction'],config['groom']['comb'],config['groom_flow'])
regional=[evaluate_prepared_flow(root,field) for root in roots]
regional_direction=np.asarray([row[0] for row in regional]);regional_comb=np.asarray([row[1] for row in regional])
regional_scale=np.asarray([row[2] for row in regional]);candidate_length=.0045*regional_scale
regional_lean=regional_direction-outward*np.sum(regional_direction*outward,axis=1)[:,None]
regional_lean/=np.maximum(np.linalg.norm(regional_lean,axis=1),1e-12)[:,None]
candidate_direction=normalize(outward*(1-regional_comb[:,None])+regional_lean*regional_comb[:,None])
candidate=positions.copy();candidate[:,2:]=positions[:,:2]+candidate_direction[:,None]*candidate_length[:,None,None]
with Image.open(PATHS['atlas']) as image:
    require(image.format=='PNG' and image.size==(4096,1824),'Unexpected actual atlas')
    pixels=np.asarray(image.convert('RGBA'))
require(sha(PATHS['atlas'])==report['material']['sha256'],'Atlas report binding differs')
height,width=pixels.shape[:2]
require(report['atlas']['interior_samples']==[60,28],'Unexpected cell sampling')
dest=fins['destination_for_source_fin']
cells=np.empty((count,28,60),dtype=np.uint8); rgb=np.empty((count,2,3),dtype=np.uint8)
for i,d in enumerate(dest):
    x=int(d%64)*64+2; y=int(d//64)*32+2
    cells[i]=pixels[y:y+28,x:x+60,3]
    tips=.5+np.array([-.0025,.0025])/spans[i]
    cols=np.clip((tips*60).astype(int),0,59)
    rgb[i]=pixels[y+14,x+cols,:3]
del pixels
linear=rgb.astype(float)/255
linear=np.where(linear<=.04045,linear/12.92,((linear+.055)/1.055)**2.4)
luma=linear.mean(axis=1)@np.array([.2126,.7152,.0722])
bands={'dark_luma_lt_005':luma<.05,'middle_luma_005_to_050':(luma>=.05)&(luma<.5),'bright_luma_ge_050':luma>=.5}
opaque=cells>=np.ceil(.18*255); area_fraction=opaque.mean(axis=(1,2))
row_width=opaque.sum(axis=2)/60*spans[:,None]
half_widths=[];half_groups=[];alpha_heights=[];taper=[]
for i in range(count):
    g=groups(opaque[i,14]);half_groups.append(len(g));half_widths.append((g[:,1]-g[:,0]).mean()/60*spans[i])
    rows=np.flatnonzero(opaque[i].any(axis=1));alpha_heights.append((rows[-1]-rows[0]+1)/28*length[i])
    taper.append(row_width[i,5]/row_width[i,22])
half_widths=np.asarray(half_widths);alpha_heights=np.asarray(alpha_heights)
boxes=[]
for owner in skin_report['owners']:
    pts=skin['positions'][owner['vertex_start']:owner['vertex_start']+owner['vertex_count']]
    boxes.extend(itertools.product(*zip(pts.min(axis=0),pts.max(axis=0))))
boxes=np.asarray(boxes,dtype=float)
views=[]
for name,offset in [('front',(0,-3,.05)),('three-quarter',(-1.6,-3,.1))]:
    basis,scale=camera(boxes,offset,native['bounds_blender_z_up'])
    view_dot=outward@basis[:,2]
    regions={'all':np.ones(count,dtype=bool),'front_facing':view_dot>.5,
             'near_contour_orientation':abs(view_dot)<.2,'nearly_face_on':view_dot>.9}
    for resolution in (720,256):
        pp=positions@basis[:,:2]*resolution/scale; pc=candidate@basis[:,:2]*resolution/scale
        edge=pp[:,1]-pp[:,0];lift=pp[:,2]-pp[:,0];clift=pc[:,2]-pc[:,0]
        edge_px=np.linalg.norm(edge,axis=1);height_px=np.linalg.norm(lift,axis=1)
        geometric_area=abs(edge[:,0]*lift[:,1]-edge[:,1]*lift[:,0])
        candidate_area=abs(edge[:,0]*clift[:,1]-edge[:,1]*clift[:,0])
        width_px=half_widths/spans*edge_px
        spacing_px=.005/spans*edge_px
        entries=[]
        for region,mask in regions.items():
            for band,selection in {'all':np.ones(count,dtype=bool),**bands}.items():
                selected=mask&selection
                if not selected.any():continue
                entries.append({'orientation_region':region,'luma_band':band,'fins':int(selected.sum()),
                    'fin_height_px':stats(height_px[selected]),'opaque_strip_width_half_height_px':stats(width_px[selected]),
                    'two_strip_center_spacing_px':stats(spacing_px[selected]),
                    'strip_width_below_one_px':int((width_px[selected]<1).sum()),
                    'projected_geometric_area_px2':stats(geometric_area[selected]),
                    'sum_alpha_area_proxy_px2':float((geometric_area[selected]*area_fraction[selected]).sum()),
                    'candidate_fixed_root_height_px':stats(np.linalg.norm(clift,axis=1)[selected]),
                    'candidate_sum_alpha_area_proxy_px2':float((candidate_area[selected]*area_fraction[selected]).sum()),
                    'candidate_area_ratio':float((candidate_area[selected]*area_fraction[selected]).sum()/
                         max((geometric_area[selected]*area_fraction[selected]).sum(),1e-12))})
        views.append({'view':name,'resolution':resolution,'orthographic_scale_m':scale,
                      'pixels_per_meter':resolution/scale,'basis_columns_blender_z_up':basis.tolist(),'measurements':entries})
deadline()
areas=report['geometry']['allocation']['represented_face_area_m2']
all_nearest=nearest(roots)
output={'schema_version':1,'status':'measured_read_only','profile':report['geometry']['profile'],
    'input_sha256':{key:value['sha256'] for key,value in PREP['inputs'].items()},'fins':count,
    'geometry':{'length_m':stats(length),'root_edge_span_m':stats(spans),'actual_outward_angle_degrees':stats(np.degrees(np.arccos(np.clip((direction*outward).sum(axis=1),-1,1)))),
        'nearest_root_chord_distance_m':stats(all_nearest),'represented_eligible_area_m2':areas,'roots_per_represented_m2':count/areas,
        'sqrt_area_per_root_m':float(np.sqrt(areas/count)),'atlas_tip_groups_at_half_height':dict(zip(*[list(map(int,x)) for x in np.unique(half_groups,return_counts=True)])),
        'original_and_reversed_winding':True,'single_sided_material':True},
    'alpha':{'cutoff':.18,'per_fin_alpha_occupancy':stats(area_fraction),'strip_width_half_height_m':stats(half_widths),
        'opaque_axial_height_m':stats(alpha_heights),'total_width_at_80pct_vs_20pct_height':stats(taper),
        'profile_formula':'two straight fixed x columns +/-2.5mm; radius0.8mm*(1-height/tip)^.38; tip .92..96; unchanged60x28resampling',
        'mip_or_gpu_sampling_measured':False},
    'luma_groups':{name:{'count':int(mask.sum()),'nearest_same_band_root_chord_m':stats(nearest(roots,mask)),
        'nearest_root_any_band_chord_m':stats(all_nearest[mask]),'actual_luma':stats(luma[mask])} for name,mask in bands.items()},
    'views':views,'candidate':{'status':'proposed_not_built','new_opt_in_policy':'shared-regional-fin-response-v2',
        'base_length_m':.0045,'length_rule':'base_length_m * existing groom_flow length_scale evaluated at each fixed root midpoint',
        'comb_rule':'existing groom_flow comb and direction at the same root; no species/color coordinates or fitted thresholds',
        'regional_comb':stats(regional_comb),'regional_length_scale':stats(regional_scale),
        'fin_length_m':stats(candidate_length),'outward_angle_degrees':stats(np.degrees(np.arccos(np.clip((candidate_direction*outward).sum(axis=1),-1,1)))),
        'roots_atlas_colors_materials_masks_body_weights_unchanged':True,'fins_unchanged':count,'geometry_counts_unchanged':True,
        'projected_export_ceiling':report['budget']['projected_vertices'],'projected_triangles':report['budget']['projected_triangles'],
        'measurement_scope':'Counterfactual fixed-root projection only. Future versioned policy applies regional response after unchanged selection; no candidate model/render exists.',
        'rationale':'Shorter regional fins reduce contour reach while the already-authored native comb exposes more projected area on nearly face-on surfaces. This reuses previously discarded regional length/comb, not added density.'},
    'limitations':['Projected area is a summed unoccluded alpha-area proxy, not visible fur/body coverage. No depth raster, overlap, lighting, GPU mip or temporal antialiasing measured.',
        'Luminance bands sample existing atlas source colors at two nominal tips, not total eligible dark/bright surface area. Counts alone cannot prove color-biased root allocation.',
        'Nearest root distance is a3D chord, not geodesic distance; folded surfaces can be close.',
        'Actual source contact/cap defects remain separate; no masks, geometry, colors or normal maps are changed.',
        '720and256 use the exact rest-camera recipe;256is an analytic projected display size, not a new native image.',
        'No proposed setting is quality accepted. Native material detail remains lighting-dependent and not volumetric fur.'],
    'inputs_unchanged':all(sha(PATHS[key])==row['sha256'] for key,row in PREP['inputs'].items()),
    'elapsed_seconds':time.monotonic()-START,'self_sampled_rss_bytes':resident_bytes(os.getpid()),'native_calls':0,'provider_calls':0,'quality_approved':False}
require(output['inputs_unchanged'],'Input changed during diagnosis')
(HERE/'diagnosis.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'fins':count,'length_m':output['geometry']['length_m'],'nearest_root_m':output['geometry']['nearest_root_chord_distance_m'],
    'alpha_occupancy':output['alpha']['per_fin_alpha_occupancy'],'elapsed_seconds':output['elapsed_seconds'],'self_sampled_rss_bytes':output['self_sampled_rss_bytes']}))
