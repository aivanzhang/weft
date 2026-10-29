"""Construction-only silhouette fits and explicitly conditional cuboid dimensions."""

from copy import deepcopy
import math

import cv2
import numpy as np

from weft.spatial_rigid import corners, validate_scene


def silhouette_overlap(mask, source_box, projected_vertices):
    raster=np.asarray(mask)
    box=np.asarray(source_box,dtype=float)
    vertices=np.asarray(projected_vertices,dtype=float)
    if (raster.ndim!=2 or not 1<=raster.size<=1_000_000 or not np.isfinite(raster).all()
            or not np.isin(raster,[0,1,255]).all() or not np.any(raster)):
        raise ValueError('Silhouette must be a nonempty bounded binary raster')
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[:2]<0).any() or (box[2:]>10000).any() or (box[2:]<=box[:2]).any():
        raise ValueError('Source box requires finite positive image extents')
    if vertices.ndim!=2 or vertices.shape[1:]!=(2,) or not 3<=len(vertices)<=8 or not np.isfinite(vertices).all() or (np.abs(vertices)>20000).any():
        raise ValueError('Projected primitive requires 3–8 bounded image vertices')
    rows,cols=raster.shape
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        local=(vertices-box[:2])*[cols,rows]/(box[2:]-box[:2])
    if not np.isfinite(local).all() or (np.abs(local)>1_000_000).any():
        raise ValueError('Silhouette comparison exceeds its sampling-grid bound')
    hull=cv2.convexHull(local.astype(np.float32)).reshape(-1,2).astype(float)
    if len(hull)<3 or abs(cv2.contourArea(hull.astype(np.float32)))<1e-6:
        raise ValueError('Projected primitive has no silhouette area')
    low=np.floor(np.minimum(hull.min(axis=0),[0,0])).astype(int)
    high=np.ceil(np.maximum(hull.max(axis=0),[cols,rows])).astype(int)
    width,height=high-low
    if width*height>1_000_000:
        raise ValueError('Silhouette comparison exceeds its sampling-grid bound')
    x=np.arange(low[0],high[0],dtype=float)[None,:]+.5
    y=np.arange(low[1],high[1],dtype=float)[:,None]+.5
    inside=np.ones((height,width),dtype=bool)
    next_vertex=np.roll(hull,-1,axis=0)
    if np.sum(hull[:,0]*next_vertex[:,1]-hull[:,1]*next_vertex[:,0])<0:
        hull=hull[::-1]
    for start,end in zip(hull,np.roll(hull,-1,axis=0)):
        delta=end-start
        inside &= delta[0]*(y-start[1])-delta[1]*(x-start[0])>=-1e-8
    source=np.zeros_like(inside)
    source[-low[1]:rows-low[1],-low[0]:cols-low[0]]=raster!=0
    intersection=int(np.count_nonzero(source & inside));union=int(np.count_nonzero(source | inside))
    return {'iou':intersection/union,'intersection_cells':intersection,'union_cells':union,
        'source_cells':int(source.sum()),'projected_cells':int(inside.sum()),'grid_cells':int(width*height),
        'sampling':'Exact convex inclusion at original mask-cell centers; not continuous area or measured shape'}


def fit_initial_width(scene, object_id, projection, mask, source_box, *, width_factors=None):
    result=validate_scene(scene)
    if not isinstance(projection,dict) or projection.get('scope')!='construction':
        raise ValueError('Primitive fitting accepts only explicit construction projections')
    matrix=np.asarray(projection.get('matrix_px'),dtype=float)
    if matrix.shape!=(2,4) or not np.isfinite(matrix).all() or (np.abs(matrix)>100000).any():
        raise ValueError('Primitive fitting requires a bounded affine image projection')
    matches=[obj for obj in result['objects'] if obj['id']==object_id]
    if len(matches)!=1 or matches[0]['shape']!='box' or matches[0]['role']=='static':
        raise ValueError('Initial-width fitting requires one moving cuboid')
    obj=matches[0];initial_width=obj['size'][0]
    factors=list(np.linspace(.6,1.2,31) if width_factors is None else width_factors)
    if not 3<=len(factors)<=61 or any(type(v) not in (float,int,np.float64) or not math.isfinite(v) or not .25<=v<=2 for v in factors):
        raise ValueError('Use 3–61 finite bounded width alternatives')
    candidates=[]
    for factor in factors:
        candidate=deepcopy(obj);candidate['size'][0]=float(initial_width*factor)
        if not .01<=candidate['size'][0]<=.6:
            raise ValueError('Width alternative exceeds the primitive contract')
        pixels=np.column_stack([corners(candidate),np.ones(8)])@matrix.T
        score=silhouette_overlap(mask,source_box,pixels)
        candidates.append({'factor':float(factor),'width':candidate['size'][0],'overlap':score,
                           'projected_bounds_px':[*pixels.min(axis=0).tolist(),*pixels.max(axis=0).tolist()]})
    selected=min(candidates,key=lambda row:(-row['overlap']['iou'],abs(row['factor']-1),row['factor']))
    obj['size'][0]=selected['width']
    obj['provenance']['initial_silhouette_width']={'scope':'construction_only','selected':deepcopy(selected),
        'unchanged':['center','height','depth','orientation','mass','materials','camera'],
        'physical_validation':'unverified'}
    result['assumptions'].append('One local cuboid width fits the initial projected silhouette; height/contact and depth remain separate hypotheses.')
    report={'scope':'initial_projection_fit_only','object_id':object_id,'initial_width':initial_width,
        'selected':selected,'alternatives':candidates,'depth_identified':False,'metric_geometry_verified':False}
    return validate_scene(result),report


def conditional_box_depth(width, height, *, reported_volume_ml, normalized_gravity, physical_gravity=9.81):
    for value in (width,height,reported_volume_ml,normalized_gravity,physical_gravity):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('Conditional depth requires positive finite dimensions, volume and gravity')
    if not .01<=width<=.6 or not .01<=height<=.6 or not 1<=reported_volume_ml<=1000 or not .1<=normalized_gravity<=20 or not 9<=physical_gravity<=10:
        raise ValueError('Conditional cuboid inputs exceed the explicit envelope')
    scale=physical_gravity/normalized_gravity
    depth=reported_volume_ml*1e-6/(width*height*scale**3)
    if not .01<=depth<=.6:
        raise ValueError('Conditional depth lies outside the cuboid profile')
    return {'depth_scene_units':depth,'conditional_meters_per_scene_width':scale,
        'reported_volume_ml':reported_volume_ml,'physical_gravity_assumption':physical_gravity,
        'normalized_gravity_hypothesis':normalized_gravity,'metric_calibration_verified':False,
        'scope':'Uniform cuboid at the assumed gravity/scale; not an independent measurement of depth'}
